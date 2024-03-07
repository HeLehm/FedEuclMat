import pandas as pd
import numpy as np 
def split_data(data_path, n_participants):
    data = pd.read_csv(data_path, sep="\t", index_col=0)
    data = data.T.to_numpy()
    split_data = np.split(data, n_participants)

    #transpose back to retain original data structure
    transposed_splits = [split.T for split in split_data]
    return transposed_splits
    
def save_in_respective_folders(splits, controller_path):
    for i, split in enumerate(splits):
        path = controller_path + f"/input{i+1}/data.csv"
        
        split_df = pd.DataFrame(split)
        print(split_df)
        split_df.to_csv(f"{path}", sep="\t", index=False)
        print(f"Participant data saved in {path}")

if __name__=="__main__":
    data = "PATH/TO/simulated/A/A.n_genes=500,m=4,std=1,overlap=no.exprs_z.tsv"
    splits = split_data(data, 4)
    save_in_respective_folders(splits, "PATH/TO/data")
