from FeatureCloud.app.engine.app import AppState, app_state, Role
from sklearn.cluster import KMeans
import time
import numpy as np
import pandas as pd
import os


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
        kmeans = KMeans(n_clusters=2).fit(data)

        #! Question: Do we have to fear data races?
        #! ie. wait with transistion until coordinator is done with its task?
        if self.is_coordinator:
            return "concatenate_spike_points"
        else:
            return "share_spike_points"
        

@app_state('share_spike_points')
class ShareSPState(AppState):

    def register(self):
        self.register_transition('construct_lsdm')

    def run(self):
        self.send_data_to_coordinator(kmeans.cluster_centers_)

        #! Question: Do we have to fear data races?
        #! ie. wait with transistion until coordinator is done with its task?
        return "construct_lsdm"


@app_state('concatenate_spike_points')
class ConcatenateSPState(AppState):

    def register(self):
        self.register_transition('construct_lsdm')

    def run(self):
        # TODO: concatinate SP and broadcast 
        return 'construct_lsdm'


@app_state('construct_lsdm')
class ConstructLSDMState(AppState):

    def register(self):
        self.register_transition('share_lsdm')
        self.register_transition('concatenate_lsdm', role=Role.COORDINATOR)  

    def run(self):
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