from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
import pandas as pd
import os
import numpy as np
#from scipy.spatial import distance
from sklearn.cluster import KMeans

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
        #dataFile = os.path.join(os.getcwd(), "mnt", "input", "data.csv")
        #data = pd.read_csv(dataFile)
        #replace data loading here
        

        sample_data = np.random.rand(100, 1200)

        self.store(key="data", value=sample_data)
        
        return 'calculation'  
        # This means we are done. If the coordinator transitions into the 
        # 'terminal' state, the whole computation will be shut down.


@app_state('calculation')
class CalculationState(AppState):

    def register(self):
        self.register_transition('terminal')
        self.register_transition('AggregateSpikePoints', role=Role.COORDINATOR)  
        self.register_transition('recieveSpikePoints')
        # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        # Checkout our documentation for help on how to implement an app
        # https://featurecloud.ai/assets/developer_documentation/getting_started.html
        data = self.load("data")
        #run kmeans on data
        #import sklearn
        #from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
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
            data = self.load("merged_array")
        else:
            data = self.await_data(n = 1, unwrap=True, is_json=False)
        #calulate LSDM (S x N)
        # for each spike point calculate distance to all data points
        own_data = self.load("data")
        #either implement wiht loop
        #or better with one function
        #for spikePoint in data:
        #    for dataPoint in own_data:
#
        #        #calculate distance
        #        #store distance
        #        pass
        diff = own_data[:, np.newaxis, :] - data[np.newaxis, :, :]

        # Euclidean distance: square the difference, sum across dimensions, and take the square root
        distances = np.sqrt(np.sum(diff**2, axis=2))
        distances_file = os.path.join(os.getcwd(), "mnt", "output", "LSDM.txt")
        with open(distances_file, "w") as f:
            f.write(str(distances) )

        print(f"shape of lsdm: {distances.shape}")
        self.store(key="lsdm", value=distances)


        #construct PEDM
        # something with regression !
        #share results:
        #self.send_data_to_coordinator(kmeans.cluster_centers_, 
        #                              send_to_self=True, 
        #                              use_smpc=False, 
        #                              use_dp=False)

        return "shareLSDM"



@app_state('shareLSDM')
class shareLSDM(AppState):
        def register(self):
            self.register_transition('terminal')
        def run(self):
            #share LSDM
            lsdm = self.load("lsdm")
            print("sending data to coordinator")
            self.send_data_to_coordinator(lsdm, 
                                    send_to_self=True, 
                                    use_smpc=False, 
                                    use_dp=False)
            if self.is_coordinator:

                print("start gathering")
                agg_lsdm= self.gather_data(use_smpc=False, use_dp=False)
                print("end gathering")

                merged_lsdm = np.concatenate(agg_lsdm, axis=0)
                #save mergedLSDM to file
                resFile = os.path.join(os.getcwd(), "mnt", "output", "mergedLSDM.txt")
                with open(resFile, "w") as f:
                    f.write(str(merged_lsdm) )
                
                num_rows = merged_lsdm.shape[0]
                fedm = np.zeros((num_rows, num_rows))
                print(f"shape of merged_lsdm: {merged_lsdm.shape}")
                print(f"shape of fedm: {fedm.shape}")
                for i in range(num_rows):
                    for j in range(num_rows):
            
                        fedm[i, j] = np.linalg.norm(agg_lsdm[i] - agg_lsdm[j])

                #save FEDM to file
                resFile = os.path.join(os.getcwd(), "mnt", "output", "FEDM.txt")
                with open(resFile, "w") as f:
                    f.write(str(fedm) )



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



        #calulate LSDM

        #construct PEDM


@app_state('wrapup')
class wrapupState(AppState):
    def register(self):
        self.register_transition('terminal')
    def run(self):
        aggData = self.gather_data(use_smpc=False, use_dp=False)
        return "terminal"
        #perform warp up