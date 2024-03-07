# Description
Feature Cloud implementation for Shihab Ullahs Master Thesis ”Calculating pairwise euclidean distance matrix for horizontally partitioned data in federated learning environment"


# Input
A tab seperated csv file for each participant called """data.csv""". """create_splits.py""" can help in creating datasplits for local testing.
Additionally a config.yaml is expected in the shared directory in the form of config.yaml that is provided in this directory.

# Output
Saves the different Matrices as .npy files, the final PEDM is also saved as csv.

# Workflow

1. Clone Repository

2. Start contorller 
```
featurecloud controller start

```
3. Build the app 
```
featurecloud app build <name> <name> latest
```
4. Input data into data directory or using the create_splits.py
5. Run the build using the FeatureCloud.ai frontend

In utils.py a local execution is provided and visualize.ipynb allows for visualization of the algorithm. These however perform the calculations locally only simulating the federated setting by splitting the data. 