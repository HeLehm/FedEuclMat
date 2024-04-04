# Description
Feature Cloud implementation for Shihab Ullahs Master Thesis
”Calculating pairwise euclidean distance matrix for horizontally partitioned data in federated learning environment".
This app approximates the euclidean distance between all data points from all clients and returns the distance matrix 
of all data points from all clients.

# Input
A csv file and a config file called `config.yaml`. The format and name of the csv file 
are given in `config.yaml`.
The csv file is expected to have features as rows and samples as columns.
Additional, sample ids must be given in the first row, while no feature names should 
be given. An example would be a csv file in the following format:
```
sample1,sample2
20,23
0.5,0.3
123,150
```

# Config
A `config.yaml` file must be given with the input data. This file contains the 
following variables:
```yaml
fed_euclidean_matrix:
  data_file: "data.csv"         # The file containing the data, see Input
                                # for more information about the format
  seperator: "\t"               # The seperator used in the data file, use \t for tabs
  spike_point_generation:
    num_spikepoints: 20         # The number of spike points to generate
                                # more spike points is not always better 
                                # due to the curse of dimensionality 
                                # if you're unsure, use 20
    generation_method: "random" # either centroids, random or random_centroids
                                # if you're unsure, use random
                                # centroids: generate spike points around centroids
                                # random: generate uniformly distributed random spike points
                                # random_centroids: generate half of the spike points using 
                                #   the random method and the other half using the centroids method.
                                #   In case of an uneven number of spike points, the random method
                                #   will be used for the extra spike point.
```

# Output
Only the coordinator produces output and the following output is generated:
1. FEDM.npy         # The fedm matrix as a numpy object 
2. FEDM.csv         # The fedm matrix as a csv file
3. PEDM.csv         # The pedm matrix as a csv file 
4. PEDM.npy         # The pedm matrix as a numpy object
5. SpikePoints.npy  # The generated spike points as a numpy object

Please note the following about the sample_ids given:

The sample ids provided are used, however they are expanded by the randomly generated client_id that can also be seen in the logs.
That means that e.g. the sample_ids `sample1` and `sample2` in the final files will be e.g.
`a5e05d9e41b43935_sample1` and `a5e05d9e41b43935_sample2`. 
This ensures that there are no collisions between sample names of different clients.

# Utils
In utils.py a local execution of the algorithm is provided and visualize.ipynb allows for visualization of the algorithm. 
These however perform the calculations locally only simulating the federated setting by splitting the data. 

# Calculate federated euclidean matric locally using this app
Please refer to the [FeatureCloud Documentation](https://featurecloud.ai/assets/developer_documentation/index.html), specifically [4. Test your application with Testbed](https://featurecloud.ai/assets/developer_documentation/getting_started.html#application-development).