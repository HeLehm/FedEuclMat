from typing import Union
from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
import pandas as pd
import numpy as np
import os

from sklearn.cluster import KMeans
K = 8

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('spikepoint_generation')  
        # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        # init data
        data = np.random.randn(100, 32)
        # we will need to know our data in differnet states (just our data)
        self.store("data", data)
        return "spikepoint_generation"


@app_state('spikepoint_generation')
class SpikepointGenerationStage(AppState):

    def register(self):
        self.register_transition('spikepoint_broadcast', role=Role.COORDINATOR)
        self.register_transition('spikepoint_distance_computation', role=Role.PARTICIPANT)


    def run(self):
        #Algorithm 1: Generate spike points based on centroid(s) of each participant’s dataset (For now)
        # 1. Load data
        data = self.load("data")
        # 2. perfom k-means clustering to get centroids (spkie points)
        k_means = KMeans(n_clusters=K)
        k_means.fit(data)
        centroids = k_means.cluster_centers_
        
        # send centroids to coordinator
        self.send_data_to_coordinator(centroids, 
                                      send_to_self=True, 
                                      use_smpc=False, 
                                      use_dp=False)

        self.log(
            "Participant generated spike points and sent to coordinator."
        )

        if self.is_coordinator:
            return "spikepoint_broadcast"
        
        return "spikepoint_distance_computation"

        

@app_state('spikepoint_broadcast')
class SpikePointBroadcastState(AppState):
    # This will only be reaached by ccodinator

    def register(self):
        self.register_transition('spikepoint_distance_computation', role=Role.COORDINATOR)

    def run(self):

        # get all data sent to us by participants
        data = self.gather_data(use_smpc=False, 
                                      use_dp=False)
        self.log(
            "Coordinator received spike points from all participants."
        )
        # combine to one array
        spikepoints = np.concatenate(data)

        # send out to all participants as spyke points
        self.broadcast_data(spikepoints, 
                            send_to_self=True,
                            use_dp=False)

    

@app_state('spikepoint_distance_computation')
class SpikepointDistanceStage(AppState):

    def register(self):
        self.register_transition('DistanceMatrixCalculation', role=Role.COORDINATOR)
        self.register_transition('terminal', role=Role.PARTICIPANT)

    def run(self):
        # get the data sent to us by coordinator
        spikepoints = self.await_data()

        # Local Spike Distance Matrices(LSDMs) are computed by each participant
        # 1. Load data
        data = self.load("data")
        # 2. Compute LSDM i.e. distance of each data point to each spike point
        #    using Euclidean distance
        LSDM = np.zeros((data.shape[0], spikepoints.shape[0]))
        for i in range(data.shape[0]):
            for j in range(spikepoints.shape[0]):
                LSDM[i,j] = np.linalg.norm(data[i] - spikepoints[j])
        # 3. Send LSDM to coordinator
        self.send_data_to_coordinator(LSDM, 
                                      send_to_self=True, 
                                      use_smpc=False, 
                                      use_dp=False)
        if self.is_coordinator:
            return "DistanceMatrixCalculation"
        
        return "terminal"
    

@app_state('DistanceMatrixCalculation')
class DistanceMatrixCalculationState(AppState):
    
        def register(self):
            self.register_transition('terminal', role=Role.COORDINATOR)
    
        def run(self):
            # get all data sent to us by participants
            data = self.gather_data(use_smpc=False,
                                          use_dp=False)
            
            # NOTE: what order????

            # combine to one array
            # we know that number of spike points is the same for all participants
            FEDM = np.concatenate(data)

            # save to file
            resFile = os.path.join(os.getcwd(), "FEDM.csv")
            np.savetxt(resFile, FEDM, delimiter=",")

            return "terminal"