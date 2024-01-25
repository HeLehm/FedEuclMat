from FeatureCloud.app.engine.app import AppState, app_state, Role
from sklearn.cluster import KMeans
import time
import numpy as np
import pandas as pd
import os
import numpy as np
#from scipy.spatial import distance
from sklearn.cluster import KMeans

import utils


#For each participant: generate spike points of their dataset (different generation methods possible)

#Share generated spike-in points for all participants to coordinator with size of their respective dataset

#Coordinator concatenates all generated spike-in points received and broadcast the points to all participant(s)

#Each participant construct a pairwise distance matrix between their data points 
# and the spike points which can be denoted as Local Spike Distance Matrices(LSDMs)

#Coordinator concatenates all the LSDM(s) from each participant(s)

#Coordinator constructs FEDM by getting pairwise euclidean distance between
#each points of the concatenated LSDM(s) shared by all participant(s)


@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('generate_spike_points')  

    def run(self):
        # Checkout our documentation for help on how to implement an app
        # https://featurecloud.ai/assets/developer_documentation/getting_started.html
        dataFile = os.path.join(os.getcwd(), "mnt", "input", "part1.tsv")
        print(dataFile)
        print(self.id)
        sample_data = pd.read_csv(dataFile, sep="\t")
        
        #drop first column
        sample_data = sample_data.drop(sample_data.columns[0], axis=1)
        #replace data loading here
        
        sample_data = sample_data.to_numpy()
        print(f"shape of sample_data: {sample_data.shape}")
        #sample_data = np.random.rand(100, 1200)

        self.store(key="data", value=sample_data)
        
        return 'generate_spike_points'  


@app_state('generate_spike_points')
class GenerateSPState(AppState):

    def register(self):
        self.register_transition('terminal')
        self.register_transition('AggregateSpikePoints', role=Role.COORDINATOR)  
        self.register_transition('recieveSpikePoints')
        # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        data = self.load("data")
        #run kmeans on data
        #import sklearn
        #from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=10, random_state=0).fit(data)
        print(f"local Kmeans finished: {kmeans.cluster_centers_.shape}")
        self.send_data_to_coordinator(kmeans.cluster_centers_, 
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

        lsdm = np.zeros((own_data.shape[0], spikepoints.shape[0]))
        for i in range(own_data.shape[0]):
            for j in range(spikepoints.shape[0]):
                lsdm[i,j] = np.linalg.norm(own_data[i] - spikepoints[j])

        #diff = own_data[:, np.newaxis, :] - data[np.newaxis, :, :]
#
        ## Euclidean distance: square the difference, sum across dimensions, and take the square root
        #distances = np.sqrt(np.sum(diff**2, axis=2))
        distances_file = os.path.join(os.getcwd(), "mnt", "output", "LSDM.txt")
        print(distances_file)
        with open(distances_file, "w") as f:
            f.write(str(lsdm) )

        print(f"shape of lsdm: {lsdm.shape}")
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
            lsdm = self.load("lsdm")
            reg = self.load("reg")
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
                print(f"agg_lsdm: {agg_lsdm}")
                agg_lsdms   = [x["lsdm"] for x in agg_lsdm]
                print(agg_lsdms)
                #print(f"shape of agg_lsdms: {agg_lsdms.shape}")
                agg_regs    = [x["reg"] for x in agg_lsdm]
                print(agg_regs)
                #print(f"shape of agg_regs: {agg_regs.shape}")

                merged_lsdm = np.concatenate(agg_lsdms, axis=0)
                print(f"shape of merged_lsdm: {merged_lsdm.shape}")
                #save mergedLSDM to file
                resFile = os.path.join(os.getcwd(), "mnt", "output", "mergedLSDM.txt")
                with open(resFile, "w") as f:
                    f.write(str(merged_lsdm) )
                
                num_rows = merged_lsdm.shape[0]
                #fedm should have shape N x N (N = number of samples in all participants) so currently 200 x 200 
                #fedm = np.concatenate(agg_lsdm)
                fedm= merged_lsdm
                print(f"shape of fedm: {fedm.shape}")

                #save FEDM to file
                resFile = os.path.join(os.getcwd(), "mnt", "output", "FEDM.txt")
                with open(resFile, "w") as f:
                    f.write(str(fedm) )
                Mx, Mc = utils.construct_global_Mx_Cx_matrix(agg_regs)
                print("Shape of Mx and Cx", Mx.shape, Mc.shape)
                pedm = utils.calc_pred_dist_matrix(
                    fedm=fedm,
                    global_Mx=Mx,
                    global_Cx=Mc
                )
                print("Shape of PEDM", pedm.shape)
                #save PEDM to file
                resFile = os.path.join(os.getcwd(), "mnt", "output", "PEDM.txt")
                with open(resFile, "w") as f:
                    f.write(str(pedm) )
                




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
        print(f"shape of merged_array: {merged_array.shape}")
        #save aggData to file
        resFile = os.path.join(os.getcwd(), "mnt", "output", "aggData.txt")
        with open(resFile, "w") as f:
            f.write(str(aggData) )



        #figure out shape of aggData
        #redistribute to participants
        self.store(key="merged_array", value=merged_array)
        self.broadcast_data(merged_array, send_to_self=False    , use_dp=False )


        return "recieveSpikePoints"



