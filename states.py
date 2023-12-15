from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
import pandas as pd
import os
from scipy.spatial import distance

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
        dataFile = os.path.join(os.getcwd(), "mnt", "input", "data.csv")
        data = pd.read_csv(dataFile)
        #replace data loading here
        

        self.store(key="data", value=data)
        
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
        self.send_data_to_coordinator(kmeans.cluster_centers_, 
                                      send_to_self=True, 
                                      use_smpc=False, 
                                      use_dp=False)
        if self.is_coordinator:
            return "AggregateSpikePoints"
        else:
            return "terminal"
        
@app_state("recieveSpikePoints")
class recieveSpikePointsState(AppState):
    def register(self):
        self.register_transition('terminal')
    def run(self):
        data = await_data(n: int = 1, unwrap=True, is_json=False)
        #calulate LSDM (S x N)
        # for each spike point calculate distance to all data points
        own_data = load("data")
        #either implement wiht loop
        #or better with one function
        for spikePoint in data:
            for dataPoint in own_data:

                #calculate distance
                #store distance
                pass

        #construct PEDM
        # something with regression !
        #share results:
        self.send_data_to_coordinator(kmeans.cluster_centers_, 
                                      send_to_self=True, 
                                      use_smpc=False, 
                                      use_dp=False)

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

        #figure out shape of aggData
        #redistribute to participants
        broadcast_data(data, send_to_self=True    , use_dp=False )


        return "recieveSpikePoints"



        #calulate LSDM

        #construct PEDM


@app_state('wrapup')
class wrapupState(AppState)
    def register(self):
        self.register_transition('terminal')
    def run(self):
        aggData = self.gather_data(use_smpc=False, use_dp=False)
        #perform warp up