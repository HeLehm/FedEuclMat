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
        print("Loading config...")
        config = os.path.join(os.getcwd(), "mnt", "input", "config.yaml")
        config = yaml.load(open(config), Loader=yaml.FullLoader)
        config = config["fed_euclidean_matrix"]

        self.store(key="config", value=config)
        print("Loading data...")
        dataFileName = config["data_file"]
        dataFile = os.path.join(os.getcwd(), "mnt", "input", dataFileName)

        sample_data = pd.read_csv(dataFile, sep=config["seperator"])
        sample_ids = sample_data.columns
        sample_ids = [self._app.id + "_" + _id for _id in sample_ids]
        sample_data = sample_data.T.to_numpy()

        _dict = {"data": sample_data, "sample_ids": sample_ids}
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
            print(f"generating spike points using method: {method}")
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
                num_c_spikes = N // 2
                num_r_spikes = N - num_c_spikes
                c_spikes = gen_spikepoints(data, N=num_c_spikes, method="centroids")
                r_spikes = gen_spikepoints(data, N=num_r_spikes, method="random")
                return np.concatenate([c_spikes, r_spikes], axis=0)
        
        #read SpikePoint generation method & cluster_n from config
        self.load("config")
        method = self.load("config")["spike_point_generation"]["generation_method"]
        num_spikepoints = self.load("config")["spike_point_generation"]["num_spikepoints"]

        #generate spike points
        print("Generating spikes...")
        spike_points = gen_spikepoints(data, N=num_spikepoints, method=method)

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


        #calulate LSDM (S x N)
        #for each spike point calculate distance to all data points
        own_data = self.load("data")
        own_data = own_data["data"]

        print("Calculating lsdm...")
        lsdm = utils.euclidean_distances(own_data, spikepoints)       
#

        self.store(key="lsdm", value=lsdm)

        #perform regression for the PEDM generation later on
        print("Calculating reg...")
        reg = utils.regression_per_client(own_data, lsdm)
        
        self.store(key="reg", value=reg)


        return "shareLSDM"



@app_state('shareLSDM')
class shareLSDM(AppState):
        def register(self):
            self.register_transition('terminal')
        def run(self):
            
            lsdm = self.load("lsdm")
            reg = self.load("reg")
            sample_ids = self.load("data")["sample_ids"]

            #send a dictionary with lsdm and reg to the coordinator
            obj= {"lsdm": lsdm, "reg": reg, "sample_ids": sample_ids}

            print("Sending data to coordinator...")
            self.send_data_to_coordinator(obj, 
                                    send_to_self=True, 
                                    use_smpc=False, 
                                    use_dp=False)
            if self.is_coordinator:
                print("start gathering fedm and reg")
                list_gathered_data = self.gather_data(use_smpc=False, use_dp=False)
                print("end gathering, saving files")
                #print(f"agg_lsdm: {agg_lsdm}")
                agg_lsdms   = [x["lsdm"] for x in list_gathered_data]
                #print(f"shape of agg_lsdms: {agg_lsdms.shape}")
                agg_regs    = [x["reg"] for x in list_gathered_data]
                #print(f"shape of agg_regs: {agg_regs.shape}")
                agg_ids     = [x["sample_ids"] for x in list_gathered_data]


                merged_lsdm = np.concatenate(agg_lsdms, axis=0)
                merged_ids = np.concatenate(agg_ids, axis=0)

                num_rows = merged_lsdm.shape[0]

                #fedm should have shape N x N (N = number of samples in all participants)
                fedm = utils.euclidean_distances(merged_lsdm)

                Mx, Mc = utils.construct_global_Mx_Cx_matrix(agg_regs)

                pedm = utils.calc_pred_dist_matrix(
                    fedm=fedm,
                    global_Mx=Mx,
                    global_Cx=Mc
                )
                # make pedm symmetric
                pedm = (pedm + pedm.T) / 2

                #save fedm to file
                df = pd.DataFrame(fedm)
                df.columns = merged_ids
                df.set_index(pd.Index(merged_ids), inplace=True)
                df.to_csv(os.path.join(os.getcwd(), "mnt", "output", "FEDM.csv"), index=True)
                np.save(os.path.join(os.getcwd(), "mnt", "output", "FEDM.npy"), fedm)

                #save PEDM to file
                df = pd.DataFrame(pedm)
                df.columns = merged_ids
                df.set_index(pd.Index(merged_ids), inplace=True)
                df.to_csv(os.path.join(os.getcwd(), "mnt", "output", "PEDM.csv"), index=True)
                np.save(os.path.join(os.getcwd(), "mnt", "output", "PEDM.npy"), pedm)


            return "terminal"
        

@app_state('AggregateSpikePoints')
class AggregateSpikePointsState(AppState):

    def register(self):
        self.register_transition('terminal')
        self.register_transition('recieveSpikePoints')
        # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        print("Aggregating spike points...")
        aggData = self.gather_data(use_smpc=False, use_dp=False)
        merged_array = np.concatenate(aggData, axis=0)
        
        #BEGIN: printing and saving: 
        #save aggData to numpy file
        print("Final spike points shape: ", merged_array.shape)
        print("Final spike points: ", merged_array)
        np.save(os.path.join(os.getcwd(), "mnt", "output", "SpikePoints.npy"), merged_array)
        #END: printing and saving


        #redistribute to participants
        self.store(key="merged_array", value=merged_array)
        self.broadcast_data(merged_array, send_to_self=False, use_dp=False )


        return "recieveSpikePoints"



