from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
import pandas as pd
import os
import numpy as np
#from scipy.spatial import distance
from sklearn.cluster import KMeans


from participants_utils import generate_n_spikes_using_variance
import utils

import yaml

@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('calculation')  

    def run(self):

        #Load configuration file
        print("loading config")
        config = os.path.join(os.getcwd(), "mnt", "input", "config.yaml")
        config = yaml.load(open(config), Loader=yaml.FullLoader)

        self.store(key="config", value=config)
        print(config)

        dataFile = os.path.join(os.getcwd(), "mnt", "input", "data.csv")

        sample_data = pd.read_csv(dataFile, sep="\t")
        print(sample_data.head())
        sample_data = sample_data.T.to_numpy()

        print("shape of sample_data: ", sample_data.shape)


        _dict = {"data": sample_data}#, "id": id_column}
        self.store(key="data", value=_dict)
        
        return 'calculation'  


@app_state('calculation')
class CalculationState(AppState):

    def register(self):
        self.register_transition('terminal')
        self.register_transition('AggregateSpikePoints', role=Role.COORDINATOR)  
        self.register_transition('recieveSpikePoints')

    def run(self):

        data = self.load("data")
        data = data["data"]


        #overwrite this method for different spike point generation
        #or simply write another method and call it here with new if
        def gen_spikepoints(data, N=10, method="random_centroids"):
            print(f"generating using method: {method}")
            if method == "centroids":
                centroids = KMeans(n_clusters=N).fit(data).cluster_centers_
                #centroids = AffinityPropagation().fit(data).cluster_centers_
                return centroids
            
            if method == "random":
                return generate_n_spikes_using_variance(
                    dataset=data,
                    variance=0.9,
                    no_of_spikes=N,
                )
            
            if method == "random_centroids":
                ## concatenate both methods
                c_spikes = gen_spikepoints(data, N=N // 2, method="centroids")
                r_spikes = gen_spikepoints(data, N=N // 2, method="random")
                return np.concatenate([c_spikes, r_spikes], axis=0)
        
        #read SpikePoint generation method & cluster_n from config
        self.load("config")
        method = self.load("config")["spike_point_generation"]["generation_method"]
        n_clusters = self.load("config")["spike_point_generation"]["n_clusters"]

        #generate spike points
        spike_points = gen_spikepoints(data, N=n_clusters, method=method)

        #send spike points to coordinator
        self.send_data_to_coordinator(spike_points, 
                                      send_to_self=True, 
                                      use_smpc=False, 
                                      use_dp=False)
        if self.is_coordinator:
            return "AggregateSpikePoints"
        else:
            return "recieveSpikePoints"
        
@app_state("recieveSpikePoints")
class recieveSpikePointsState(AppState):
    def register(self):
        self.register_transition('shareLSDM')

        self.register_transition('terminal')
    def run(self):
        if self.is_coordinator:
            #for some reason send to self does not work as expected in Broadcast
            #so we need to load the data from the coordinator 
            spikepoints = self.load("merged_array")
        else:
            spikepoints = self.await_data(n = 1, unwrap=True, is_json=False)

        print(f"shape of spikepoints: {spikepoints.shape}")

        #calulate LSDM (S x N)
        #for each spike point calculate distance to all data points
        own_data = self.load("data")
        own_data = own_data["data"]

        lsdm = utils.euclidean_distances(own_data, spikepoints)       
        print("Shape of LSDM", lsdm.shape)
#

        self.store(key="lsdm", value=lsdm)

        #perform regression for the PEDM generation later on
        reg = utils.regression_per_client(own_data, lsdm)
        
        print(f"reg: {reg}")
        
        self.store(key="reg", value=reg)


        return "shareLSDM"



@app_state('shareLSDM')
class shareLSDM(AppState):
        def register(self):
            self.register_transition('terminal')
        def run(self):
            
            data = self.load("data")
            lsdm = self.load("lsdm")
            reg = self.load("reg")

            #send a dictionary with lsdm and reg to the coordinator
            obj= {"lsdm": lsdm, "reg": reg}

            print("sending data to coordinator")
            self.send_data_to_coordinator(obj, 
                                    send_to_self=True, 
                                    use_smpc=False, 
                                    use_dp=False)
            if self.is_coordinator:

                print("start gathering")
                agg_lsdm= self.gather_data(use_smpc=False, use_dp=False)
                print("end gathering")
                print("aggregated lsdms:", len(agg_lsdm))
                #print(f"agg_lsdm: {agg_lsdm}")
                agg_lsdms   = [x["lsdm"] for x in agg_lsdm]
                #print(f"shape of agg_lsdms: {agg_lsdms.shape}")
                agg_regs    = [x["reg"] for x in agg_lsdm]
                #print(f"shape of agg_regs: {agg_regs.shape}")

                merged_lsdm = np.concatenate(agg_lsdms, axis=0)
                print(f"shape of merged_lsdm: {merged_lsdm.shape}")

                num_rows = merged_lsdm.shape[0]

                #fedm should have shape N x N (N = number of samples in all participants)
                fedm = utils.euclidean_distances(merged_lsdm)
                print("Shape of FEDM", fedm.shape)
                np.save(os.path.join(os.getcwd(), "mnt", "output", "fedm.npy"), fedm)

                Mx, Mc = utils.construct_global_Mx_Cx_matrix(agg_regs)
                print("Shape of Mx and Cx", Mx.shape, Mc.shape)

                pedm = utils.calc_pred_dist_matrix(
                    fedm=fedm,
                    global_Mx=Mx,
                    global_Cx=Mc
                )
                # make pedm symmetric
                pedm = (pedm + pedm.T) / 2
                print("Shape of PEDM", pedm.shape)
                #save PEDM to file

                df = pd.DataFrame(pedm)
                df.to_csv(os.path.join(os.getcwd(), "mnt", "output", "PEDM.csv"), index=False)
                np.save(os.path.join(os.getcwd(), "mnt", "output", "PEDM.npy"), pedm)




            return "terminal"
        

@app_state('AggregateSpikePoints')
class AggregateSpikePointsState(AppState):

    def register(self):
        self.register_transition('terminal')
        self.register_transition('recieveSpikePoints')
        # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):

        aggData = self.gather_data(use_smpc=False, use_dp=False)
        merged_array = np.concatenate(aggData, axis=0)
        
        #BEGIN: printing and saving: 
        print(f"shape of merged_array: {merged_array.shape}")
        #save aggData to numpy file
        np.save(os.path.join(os.getcwd(), "mnt", "output", "SpikePoints.npy"), merged_array)
        #END: printing and saving


        #redistribute to participants
        self.store(key="merged_array", value=merged_array)
        self.broadcast_data(merged_array, send_to_self=False    , use_dp=False )


        return "recieveSpikePoints"



