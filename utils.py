import numpy as np
from typing import Literal, Tuple, List

from scipy.linalg import block_diag
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression, HuberRegressor, TheilSenRegressor

__all__ = ["regression_per_client", "construct_global_Mx_Cx_matrix", "calc_pred_dist_matrix", "euclidean_distances"]


def regression_per_client(
        data: np.ndarray,
        lsdm: np.ndarray,
        regressor: Literal["Huber", "Linear", "TheilSen"]= "Huber",
) -> Tuple[float, float]:
    """
    USED BY PARTICIPANTS
    Perform regression to correct FEDM. Get coefficient and intercept of regression model

    Args:
        data (n-D array):
            participants datapoint
            shape: (n_samples, n_features)   
        lsdm (n-D array):
            participants LSDM
            generated like: euclidean_distances(data, global_spikepoints)
            shape: (n_samples, n_spikepoints)
        regressor (str): Type of regressor to be used. Default value is "Huber".
                         Values can be "Huber", "Linear" or "TheilSen" respectively
    
    Returns:
        Tuple: coefficient, intercept, particpants data length
    """
    # get number of datapoints in participant's dataset
    data_len = data.shape[0]
    # euclidean distance of datapoints to itself
    euc_dist_data = euclidean_distances(data).flatten() # shape: (n_samples*n_samples,)
    # flatten FEDM to 1-D array
    local_fed_dist = euclidean_distances(lsdm).flatten().reshape((-1,1)) # shape: (n_samples*n_samples,1)
    # fit the regression model
    if regressor == "Huber":
        model = HuberRegressor().fit(local_fed_dist, euc_dist_data)
    elif regressor == "Linear":
        model = LinearRegression().fit(local_fed_dist, euc_dist_data)
    elif regressor == "TheilSen":
        model = TheilSenRegressor().fit(local_fed_dist, euc_dist_data)
    else:
        raise ValueError(f"Invalid regressor type, {regressor}. Please use 'Huber', 'Linear' or 'TheilSen'")

    return model.coef_.item(), model.intercept_, data_len
    

def construct_global_Mx_Cx_matrix(
        regresion_outputs: List[Tuple[float, float, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    USED BY COORDINATOR
    Return two matrices containing the slopes and intercepts for each datapoints of all participants

    Args:
        regresion_outputs (list):
            List of tuples containing the slope, intercept and data length of each participant
            I.e. [(slope1, intercept1, data_len1), (slope2, intercept2, data_len2), ...]
            This is the output of regression_per_client() function
    Returns:
        global_Mx (n-D array) : Matrix containing the slopes for each datapoints of all participants
        global_Cx (n-D array) : Matrix containing the intercepts for each datapoints of all participants
    """
    
    coef_intersects = np.array([
        (coef, intercept) for coef, intercept, _ in regresion_outputs
    ])
    dataset_lens = [
        data_len for _, _, data_len in regresion_outputs
    ]

    Mi,Ci = np.split(coef_intersects, 2, axis=1)
    arrayMi=Mi.flatten()
    arrayCi=Ci.flatten()
    
    Mi_avg=np.average(arrayMi)
    Ci_avg=np.average(arrayCi)
    
    #Placing the respective Mi of each datapoints and getting Mx matrix
    global_Mx = block_diag(*[np.full((i, i), c) for c, i in zip(arrayMi, dataset_lens)])
    #Placing the respective Ci of each datapoints and getting Cx matrix
    global_Cx = block_diag(*[np.full((i, i), c) for c, i in zip(arrayCi, dataset_lens)])
    
    # The zeroes in global slopes and constants matrix are replaced by Mi_avg and Ci_avg respectively 
    # They are used to calculate the predicted distance for cross-sectional data
    # For example: distance(a1,b5) where a1 and b5 belongs to different datasets
    global_Mx[global_Mx == 0] = Mi_avg
    global_Cx[global_Cx == 0] = Ci_avg
    return global_Mx, global_Cx


def calc_pred_dist_matrix(fedm, global_Mx, global_Cx):
    """
    USED BY COORDINATOR
    Return the distance matrix calculated from the slope and intercept returned from regression and after applying to FEDM

    Args:
        fedm (n-D array) :
            Matrix containing the distance of the pairwise distances between each element of all the participants' LSDM
        global_Mx (n-D array) :
            Matrix containing the slopes for each datapoints of all participants (construct_global_Mx_Cx_matrix output [0])
        global_Cx (n-D array) :
            Matrix containing the intercepts for each datapoints of all participants (construct_global_Mx_Cx_matrix output [1])

    Returns:
        PEDM (n-D array) : Distance matrix calculated from the slope and intercept returned from regression and after applying to FEDM
    """
    # TODO: I dont know if this is the fedm or the value we passed here was the fedm
    fedm = euclidean_distances(fedm)

    # Multiply the slope with each value of FEDM followed by adding intercept with it 
    # to minimize error of the distance between two points
    PEDM=np.add(
        np.multiply(global_Mx, fedm),
        global_Cx
    )
    #Fill the diagonal value to 0 as distance between same point is 0
    np.fill_diagonal(PEDM, 0)
    return PEDM


if __name__ == "__main__":
    # EXAMPLE USAGE
    from sklearn.cluster import AffinityPropagation

    n_participants = 2
    feature_dim = 32
    n_samples_per_participant = 200
    full_data = np.random.rand(n_participants * n_samples_per_participant, feature_dim)

    true_euc_dist = euclidean_distances(full_data)

    D1, D2 = np.array_split(full_data, 2)

    def gen_spikepoints(data):
        return AffinityPropagation().fit(data).cluster_centers_
    
    # EACH PARTICIPANT
    spikpoints_D1 = gen_spikepoints(D1)
    spikpoints_D2 = gen_spikepoints(D2)
    print("Shape of Spikepoints", spikpoints_D1.shape, spikpoints_D2.shape)
    # SEND SPIKEPOINTS TO COORDINATOR

    # CONSTRUCTOR
    generated_spikes = np.concatenate((spikpoints_D1, spikpoints_D2))
    print("Shape of concatinated Spikes", generated_spikes.shape)
    # SEND SPIKEPOINTS TO PARTICIPANTS

    # EACH PARTICIPANT
    lsdm1 = euclidean_distances(D1, generated_spikes)
    lsdm2 = euclidean_distances(D2, generated_spikes)
    print("Shape of LSDM", lsdm1.shape, lsdm2.shape)
    reg_1 = regression_per_client(D1, lsdm1)
    reg_2 = regression_per_client(D2, lsdm2)
    # SEND LSDM AND REG VALUES TO COORDINATOR

    # CONSTRUCTOR
    fedm= np.concatenate((lsdm1, lsdm2))
    print("Shape of FEDM", fedm.shape)
    Mx, Mc = construct_global_Mx_Cx_matrix([reg_1, reg_2])
    print("Shape of Mx and Cx", Mx.shape, Mc.shape)

    pedm = calc_pred_dist_matrix(
        fedm=fedm,
        global_Mx=Mx,
        global_Cx=Mc
    )
    print("Shape of PEDM", pedm.shape)

    # CALCULATE ERROR
    print("MAE: ", np.mean(np.abs(true_euc_dist - pedm)))