from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
import pandas as pd
import os
import numpy as np
#from scipy.spatial import distance
from sklearn.cluster import KMeans

import utils

import yaml
# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('calculation')  
        # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        # Checkout our documentation for help on how to implement an app
        # https://featurecloud.ai/assets/developer_documentation/getting_started.html

        #Load configuration file
        print("loading config")
        config = os.path.join(os.getcwd(), "mnt", "input", "config.yaml")
        config = yaml.load(open(config), Loader=yaml.FullLoader)
        self.store(key="config", value=config)
        print(config)
        print("self:", self.__dict__)
        dataFile = os.path.join(os.getcwd(), "mnt", "input", "data.csv")
        print(dataFile)
        print(self.id)
        sample_data = pd.read_csv(dataFile, sep="\t")
        sample_data = sample_data.T.to_numpy()
        print("shape of sample_data: ", sample_data.shape)


        _dict = {"data": sample_data}#, "id": id_column}
        self.store(key="data", value=_dict)
        
        return 'calculation'  
        # This means we are done. If the coordinator transitions into the 
        # 'terminal' state, the whole computation will be shut down.


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
        def gen_spikepoints(data):
            from sklearn.cluster import AffinityPropagation
            return AffinityPropagation().fit(data).cluster_centers_
        spike_points = gen_spikepoints(data)
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
            spikepoints = self.load("merged_array")
        else:
            spikepoints = self.await_data(n = 1, unwrap=True, is_json=False)

        print(f"shape of spikepoints: {spikepoints.shape}")
        #calulate LSDM (S x N)
        # for each spike point calculate distance to all data points
        own_data = self.load("data")
        own_data = own_data["data"]

        lsdm = utils.euclidean_distances(own_data, spikepoints)       
        print("Shape of LSDM", lsdm.shape, lsdm.shape)
#

        self.store(key="lsdm", value=lsdm)

        reg = utils.regression_per_client(own_data, lsdm)
        
        print(f"reg: {reg}")
        
        self.store(key="reg", value=reg)


        return "shareLSDM"



@app_state('shareLSDM')
class shareLSDM(AppState):
        def register(self):
            self.register_transition('terminal')
        def run(self):
            #share LSDM
            data = self.load("data")
            #ids = data["id"]
            lsdm = self.load("lsdm")
            reg = self.load("reg")
            obj= {"lsdm": lsdm, "reg": reg}# "ids": ids}

            print("sending data to coordinator")
            self.send_data_to_coordinator(obj, 
                                    send_to_self=True, 
                                    use_smpc=False, 
                                    use_dp=False)
            if self.is_coordinator:

                print("start gathering")
                agg_lsdm= self.gather_data(use_smpc=False, use_dp=False)
                print("end gathering")

                print(f"agg_lsdm: {agg_lsdm}")
                agg_lsdms   = [x["lsdm"] for x in agg_lsdm]
                #agg_ids = [id_ for dictionary in agg_lsdm for id_ in dictionary["ids"]]
                print(agg_lsdms)
                #print(f"shape of agg_lsdms: {agg_lsdms.shape}")
                agg_regs    = [x["reg"] for x in agg_lsdm]
                print(agg_regs)
                #print(f"shape of agg_regs: {agg_regs.shape}")

                merged_lsdm = np.concatenate(agg_lsdms, axis=0)
                print(f"shape of merged_lsdm: {merged_lsdm.shape}")

                num_rows = merged_lsdm.shape[0]

                #fedm should have shape N x N (N = number of samples in all participants) so currently 200 x 200 
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

                #print(f"aggIDS : {agg_ids}")
                df = pd.DataFrame(pedm)
                #df.insert(0, 'ID', agg_ids)
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
        # Checkout our documentation for help on how to implement an app
        # https://featurecloud.ai/assets/developer_documentation/getting_started.html
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



