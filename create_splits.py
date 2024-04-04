import os
import pandas as pd
import numpy as np 
import shutil
### Variables
# path to the data file to be split
basefolder = os.path.dirname(os.path.abspath(__file__))
filename = "A.n_genes=500,m=4,std=1,overlap=no.exprs_z.tsv"
datapath = os.path.join(basefolder, "simulated", "A", filename)

# number of participants to split the data into
num_participants = 4

### Functions
def split_data(data_path, n_participants):
    print(f"opening {data_path}")
    data = pd.read_csv(data_path, sep="\t", index_col=0)
    data = data.T.to_numpy()
    split_data = np.split(data, n_participants)

    #transpose back to retain original data structure
    transposed_splits = [split.T for split in split_data]
    return transposed_splits
    
def save_in_respective_folders(splits, controller_path):
    for i, split in enumerate(splits):
        path = os.path.join(controller_path, f"client{i+1}")
        # create the corresponding client folder
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, "data.csv")
        split_df = pd.DataFrame(split)
        split_df.to_csv(f"{path}", sep="\t", index=False)
        # copy the file config.yaml to the directory of path
        config_filename = "config.yaml"
        shutil.copy(os.path.join(basefolder, config_filename), 
                    os.path.join(os.path.dirname(path), config_filename))
        print(f"Finished client{i+1}")

### Main
if __name__=="__main__":
    splits = split_data(datapath, num_participants)
    save_in_respective_folders(splits, os.path.join(basefolder, "data"))
