from FeatureCloud.app.engine.app import AppState, app_state, Role
from sklearn.cluster import KMeans
import time
import numpy as np
import pandas as pd
import os


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
        # TODO: review how to read in data
        dataFile = os.path.join(os.getcwd(), "mnt", "input", "data.csv")
        data = pd.read_csv(dataFile)
        self.store(key="data", value=data)
        
        return 'generate_spike_points'  


@app_state('generate_spike_points')
class GenerateSPState(AppState):

    def register(self):
        self.register_transition('share_spike_points')
        self.register_transition('concatenate_spike_points', role=Role.COORDINATOR)  

    def run(self):
        data = self.load("data")
        kmeans = KMeans(n_clusters=2).fit(data)
        clusterCenters = kmeans.cluster_centers_
        #? Im supposed to send the size of the dataset as well
        #? Why? We dont do anything with it
        self.store(key="clusterCenters", value=clusterCenters)
        self.send_data_to_coordinator(clusterCenters)

        if self.is_coordinator:
            return "concatenate_spike_points"
        else:
            return "construct_lsdm"

@app_state('concatenate_spike_points')
class ConcatenateSPState(AppState):

    def register(self):
        self.register_transition('construct_lsdm')

    def run(self):
        # TODO: concatinate SP and broadcast 
        allClusterCenters = self.gather_data()
        concatenatedCenters = np.concatenate(allClusterCenters)
        self.log(concatenatedCenters)
        self.store(key="concatenatedCenters", value=concatenatedCenters)
        self.broadcast_data(concatenatedCenters)
        return 'construct_lsdm'


@app_state('construct_lsdm')
class ConstructLSDMState(AppState):

    def register(self):
        self.register_transition('share_lsdm')
        self.register_transition('concatenate_lsdm', role=Role.COORDINATOR)  

    def run(self):
        if self.is_coordinator:
            concatenatedCenters = self.load("concatenatedCenters")
        else:
            concatenatedCenters = self.await_data(n=1)

        data = self.load(data)

        # TODO: construct lsdm


        if self.is_coordinator:
            return "concatenate_lsdm"
        else:
            return "share_lsdm"



@app_state('share_lsdm')
class ShareLSDMState(AppState):

    def register(self):
        self.register_transition('terminal')

    def run(self):
        # TODO: share generated lsdm

        return "terminal"


@app_state('concatenate_lsdm')
class ConcatenateLSDMState(AppState):

    def register(self):
        self.register_transition('construct_fedm')

    def run(self):
        # TODO: concatinate lsdm
        
        return 'construct_fedm'


@app_state('construct_fedm')
class ConstructFEDMState(AppState):

    def register(self):
        self.register_transition('terminal')

    def run(self):
        # TODO: construct fedm
        
        return 'terminal'