"""
PV optimization on DNS dataset of Xu using an encoder-decoder architecture (Kamila Zdybal)
Version: Tools to visualize and sample the data
Author: Grégoire Corlùy (gregoire.stephane.corluy@ulb.be)
Date: September 2024
Python version: 3.10.10
"""

import numpy as np
from scipy.io import loadmat
from PCAfold import DensityEstimation, KReg, compute_normalized_variance, normalized_variance_derivative, cost_function_normalized_variance_derivative, plot_normalized_variance_derivative
import matplotlib.pyplot as plt
from pandas import read_csv
from torch import cat
from utils import get_dataset
from os import listdir
from os.path import isfile, join
import pandas as pd


def smart_sampling_data(data, idx_sure, idx_unsure, seed):
    """Smart sampling with all the indices where we are sure, are kept.
    And random sampling is done one indices we are not sure.

    Args:
        data (numpy.ndarray): Matrix of the state space.
        idx_sure (list[int]): List of indices that have to be sampled for sure.
        idx_unsure (list[int]): List of indices that have to be sampled.
        seed (int): Seed for reproducibility.

    Returns:
        numpy.ndarray: Sampled array.
    """

    N = np.sum(idx_sure)

    print(N)

    data = np.array(data)
    idx_sure = np.array(idx_sure)
    idx_unsure = np.array(idx_unsure)

    data_sure = data[idx_sure]

    np.random.seed(seed)
    indices_unsure = np.flatnonzero(idx_unsure)
    idx_unsure_sampled = np.random.choice(indices_unsure, size=N, replace=False)
    data_unsure_sampled = data[idx_unsure_sampled]

    sampled_data = np.concatenate((data_sure, data_unsure_sampled))

    return sampled_data


def load_array(species_name, data_path = "data-files/", idx_layer = 10, scaled = False):
    """Load array of a species for a specific layer of the dataset.

    Args:
        species_name (str): Name of the species to visualize.
        data_path (str, optional): Path to data files. Defaults to "data-files/".
        idx_layer (int, optional): Index of the layer to be considered. Defaults to 10.
        scaled (bool, optional): Indicates if the data has to be scaled between 0 and 1. Defaults to False.

    Returns:
        numpy.ndarray: Numpy array of the chosen species.
    """
    file_name = species_name + "_3D_slice.mat"
    #load the mat file
    data = loadmat(data_path + file_name)

    #remove the .mat suffix and load the data as an array
    data_matrix = np.array(data[file_name.removesuffix('.mat')])

    data_layer = data_matrix[idx_layer,0].flatten()

    if(scaled):
        min_data = np.min(data_layer)
        max_data = np.max(data_layer)

        data_layer = (data_layer-min_data)/(max_data-min_data)

    return data_layer

def load_matrix(data_path, species_name, idx_layer):
    """Load matrix of a species for a specific layer of the dataset.

    Args:
        data_path (str): Path to data files.
        species_name (str): Name of the species to visualize.
        idx_layer (int): Index of the layer to be considered.

    Returns:
        numpy.ndarray: Numpy maptrix of the chosen species.
    """
    file_name = species_name + "_3D_slice.mat"
    #load the mat file
    data = loadmat(data_path + file_name)

    #remove the .mat suffix and load the data as an array
    data_matrix = np.array(data[file_name.removesuffix('.mat')])

    data_matrix = data_matrix[idx_layer,0]

    return data_matrix[0]



def Compute_density(data_path, x_element, y_element, chosen_layer, nbr_neighbours, file_type = "mat", filenameT = None, filenameF = None):
    """Compute the data density in the mixture fraction - temperature manifold using the k-th nearest neighbor.

    Args:
        data_path (str):  Path to data files
        x_element (str): Name of the species to visualize used as first variable of the manifold.
        y_element (str): Name of the species to visualize used as second variable of the manifold.
        chosen_layer (int): Index of the chosen layer.
        nbr_neighbours (int): Number of neighbours considered to compute the data density.
        file_type (str, optional): Type of the file containing the species data. Defaults to "mat".
        filenameT (str, optional): Filename of the temperature dataset. Defaults to None.
        filenameF (str, optional): Filename of the mixture fraction dataset. Defaults to None.

    Returns:
        np.ndarray: Returns the data density in every point of the manifold.
    """

    if(file_type.lower() == "mat"):
        array_x = load_array( x_element, data_path, chosen_layer, True)
        array_y = load_array(y_element, data_path, chosen_layer, True)
    elif(file_type.lower() == "csv"):
        data_temp = read_csv(data_path + filenameT, header = None)
        data_mf = read_csv(data_path + filenameF, header = None)
        array_x = data_temp.iloc[:,0].to_numpy()
        array_y = data_mf.iloc[:,0].to_numpy()

        #rescale the data
        min_datax = np.min(array_x)
        max_datax = np.max(array_x)
        min_datay = np.min(array_y)
        max_datay = np.max(array_y)

        array_x = (array_x-min_datax)/(max_datax-min_datax)
        array_y = (array_y-min_datay)/(max_datay-min_datay)

    data_mf_T = np.column_stack((array_x, array_y))
    density_estimation = DensityEstimation(data_mf_T, n_neighbors=nbr_neighbours)
    data_density = density_estimation.kth_nearest_neighbor_density()

    return data_density

def visualize_manifold_color(data_path, x_species, y_species, chosen_species_as_color, chosen_layer, file_type = "mat", filenameT = None, filenameF = None, data_color = None):  
    """Visualize manifold choosing the variables defining the manifold and the variable colouring the manifold.

    Args:
        data_path (str):  Path to data files.
        x_species (str): Species on the x-axis of the manifold.
        y_species (str): Species on the y-axis of the manifold.
        chosen_species_as_color (str): Species to color the manifold.
        chosen_layer (int): Index of the chosen layer in the dataset.
        file_type (str, optional): Type of the dataset. Defaults to "mat".
        filenameT (str, optional): Filename for the temperature dataset. Defaults to None.
        filenameF (str, optional): Filename for the mixture fraction dataset. Defaults to None.
        data_color (numpy.ndrarray, optional): Data for the color of the manifold. Defaults to None.
    """

    if(file_type.lower() == "mat"):
        array_x = load_array(data_path, x_species, chosen_layer)
        array_y = load_array(data_path, y_species, chosen_layer)
        array_color = load_array(data_path, chosen_species_as_color, chosen_layer)
        
    elif(file_type.lower() == "csv"):
        data_temp = read_csv(data_path + filenameT, header = None)
        data_mf = read_csv(data_path + filenameF, header = None)
        array_x = data_temp.iloc[:,0].to_numpy()
        array_y = data_mf.iloc[:,0].to_numpy()
        array_color = data_color

    plt.scatter(array_x, array_y, s=1, c = array_color, cmap='viridis')
    plt.colorbar(label=f'{chosen_species_as_color}')
    plt.xlabel(f'{x_species}')
    plt.ylabel(f'{y_species}')
    plt.title(f'{y_species} vs {x_species} - layer {chosen_layer}')

    plt.show()

def visualize_species_layer(data_path, species_name, idx_layer):
    """Visualize the 2D layer for one species.

    Args:
        data_path (str): Path to data files.
        species_name (str): Name of the species to visualize.
        idx_layer (int): Index of the chosen layer.
    """
    
    matrix_species = load_matrix(data_path, species_name, idx_layer)

    plt.imshow(matrix_species, cmap='viridis')

    plt.colorbar()
    plt.title(f"Layer {idx_layer} of {species_name}")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

def compute_Kreg(   output_idx,
                    idx_species_removed,
                    input_scaling,
                    input_species_scaling,
                    input_species_bias,
                    range_mf,
                    model,
                    path_data,
                    dataset_type = "low",
                    idx_species_removed_source = None):

    """Compute the MSE of the kernel regression on the validation dataset using different number of neighbours.
    The QOI's are reconstructed given the mixture fraction and the progress variable.

    Args:
        output_idx (list[int]): List of indices indicating the species to be predicted.  
        idx_species_removed (list[int]): Indices of species removed from the state space.  
        input_scaling (float): Range of the scaling to be applied on mixture fraction and progress variable.  
        input_species_scaling (str): Name of the scaling to be applied.  
        input_species_bias (float): Centering applied on mixture fraction and progress variable.  
        range_mf (float): Range of mixture fraction values to be considered.  
        model (PV_autoencoder): Trained encoder-decoder defining the optimized progress variable.  
        path_data (str): Path to data files. 
        dataset_type (str, optional): Type of dataset to be used. Defaults to "low" referring to the sampled DNS dataset.  
        idx_species_removed_source (list of int, optional): Indices of species removed from the source terms. Defaults to None.  

    Returns:
        list[float]: Return the average MSE and its standard deviation.
    """

    neighbours = [5, 10, 15, 20, 25]
    seed = 9

    mse_values_model = np.zeros(len(neighbours))

    if(idx_species_removed_source is None):
        idx_species_removed_source = idx_species_removed


    input, output = get_dataset(path_data + f"Xu-state-space-{dataset_type}.csv", path_data + f"Xu-state-space_source-{dataset_type}.csv",
                            path_data + f"Xu-T-{dataset_type}.csv", path_data + f"Xu-mf-{dataset_type}.csv",
                            output_idx, idx_species_removed, input_scaling, input_species_scaling,
                            input_species_bias, range_mf, idx_species_removed_source=idx_species_removed_source)

    PV = model.get_PV(input)
    output = model.get_source_PV(output, input_species_scaling)

    f_PV = cat((input[:,-1].unsqueeze(1), PV), dim = 1)

    min_f_PV = f_PV.min(dim=0, keepdim=True)[0]  # Minimum values for each column
    max_f_PV = f_PV.max(dim=0, keepdim=True)[0]  # Maximum values for each column

    f_PV_scaled = (f_PV - min_f_PV) / (max_f_PV - min_f_PV)

    min_output = output.min(dim=0, keepdim=True)[0]  # Minimum values for each column
    max_output = output.max(dim=0, keepdim=True)[0]  # Maximum values for each column

    output_scaled = (output - min_output) / (max_output - min_output)

    #create training and validation datasets
    np.random.seed(seed)
    nbr_observations = f_PV_scaled.shape[0]
    indices = np.arange(nbr_observations)
    nbr_train = int(nbr_observations*0.8)
    sampled_indices = np.random.choice(indices, size=nbr_train, replace=False)
    validation_indices = np.setdiff1d(indices, sampled_indices)

    input_model = f_PV_scaled.detach().numpy()
    output_model = output_scaled.detach().numpy()

    for j, neighbour in enumerate(neighbours):

        query = input_model[validation_indices,:]

        kernel_model = KReg(input_model[sampled_indices, :], output_model[sampled_indices, :])
        predicted_model = kernel_model.predict(query, 'nearest_neighbors_isotropic', n_neighbors=neighbour)

        squared_error_model = (predicted_model - output_model[validation_indices,:]) ** 2
        mse_model = np.mean(squared_error_model)
        mse_values_model[j] = mse_model

    avg_mse = np.mean(mse_values_model)
    std_mse = np.std(mse_values_model)

    return [avg_mse, std_mse]

def compute_avg(costs):
    """Compute the Root mean square of all QoI costs.

    Args:
        costs (list[float]): List of all QoI costs.

    Returns:
        float: Root mean square of all QoI costs.
    """

    n = len(costs)
    sum = np.sum(costs**2)
    return 1/n*np.sqrt(sum)

def compute_cost(   output_idx,
                    idx_species_removed,
                    input_scaling,
                    input_species_scaling,
                    input_species_bias,
                    range_mf,
                    depvar_names_species,
                    depvar_names_idx,
                    PV_dim,
                    model,
                    id, path_data):
    
    """Computes the cost of the mixture fraction-progress variable manifold using the PCAfold library.

    Args:
        output_idx (list[int]): List of indices indicating the species to be predicted.  
        idx_species_removed (list[int]): Indices of species removed from the state space. 
        input_scaling (float): Range of the scaling to be applied on mixture fraction and progress variable.  
        input_species_scaling (str): Name of the scaling to be applied.  
        input_species_bias (float): Centering applied on mixture fraction and progress variable.  
        range_mf (float): Range of mixture fraction values to be considered.  
        depvar_names_species (list[str]): List of species names for which the cost has to be computed.
        depvar_names_idx (list[int]): List of indices for which the cost has to be computed.
        PV_dim (int): Indicates the dimension of the progress variable.
        model (PV_autoencoder): Trained encoder-decoder defining the optimized PV
        id (str): id of the model.
        path_data (str):  Path to data files.

    Returns:
        float: Compute root mean square of all QoI costs.
    """

    print("start compute cost")
    penalty_function = 'log-sigma-over-peak'
    start_bw = -6
    end_bw = 2
    nbr_points_bw = 100
    bandwidth_values = np.logspace(start_bw, end_bw, nbr_points_bw)
    power = 1
    vertical_shift = 1

    depvar_names = depvar_names_species + depvar_names_idx[-(PV_dim+1):]

    print("import dataset")
    #get the input (PV and f) and the output (interested Yi, T and source terms) data
    input, output = get_dataset(path_data + "Xu-state-space-low.csv", path_data + "Xu-state-space_source-low.csv", path_data + "Xu-T-low.csv", path_data + "Xu-mf-low.csv", output_idx, idx_species_removed, input_scaling, input_species_scaling, input_species_bias, range_mf)
    PV = model.get_PV(input)
    PV_f = cat((PV, input[:, -1].reshape(-1, 1)), dim = 1) #reshape to be (5200,1) instead of (52000)
    output = model.get_source_PV(output, input_species_scaling)

    #scale every column of the PV_f tensor between 0 and 1
    min_vals = PV_f.min(dim=0, keepdim=True).values
    max_vals = PV_f.max(dim=0, keepdim=True).values
    PV_f_scaled = (PV_f - min_vals) / (max_vals - min_vals)

    indepVars = PV_f_scaled.detach().numpy()
    depVars = output.detach().numpy()

    print("compute variance data")
    variance_data = compute_normalized_variance(indepVars,
                                                    depVars,
                                                    depvar_names=depvar_names,
                                                    bandwidth_values=bandwidth_values)
    np.save(f"costs/variance_{id}-bw_{start_bw}_{end_bw}_{nbr_points_bw}.npy", variance_data)

    print("compute costs")
    costs = cost_function_normalized_variance_derivative(variance_data,
                                                        penalty_function=penalty_function,
                                                        power=power,
                                                        vertical_shift=vertical_shift,
                                                        norm=None)
    np.save(f"costs/costs_{id}-bw_{start_bw}_{end_bw}_{nbr_points_bw}-p_{power}-ver_sh_{vertical_shift}.npy", costs)

    (derivative, bandwidth_values, max_derivative) = normalized_variance_derivative(variance_data)

    plt = plot_normalized_variance_derivative(variance_data)
    plt.savefig(f"costs/plot_Dhat_{id}-bw_{start_bw}_{end_bw}_{nbr_points_bw}-p_{power}-ver_sh_{vertical_shift}.png")

    cost = compute_avg(np.array(costs))

    return cost

def load_DNS_data(data_path = "data-files/", nbr_rows = 1536, nbr_cols = 1024, layer = 10, nbr_species = 21):
    """Load the different arrays composing the autoignition dataset.

    Args:
        data_path (str, optional):  Path to data files.. Defaults to "data-files/".
        nbr_rows (int, optional): Number of rows in the 2D layer. Defaults to 1536.
        nbr_cols (int, optional): Number of columns in the 2D layer. Defaults to 1024.
        layer (int, optional): Index of the chosen layer. Defaults to 10.
        nbr_species (int, optional): Number of species in the dataset. Defaults to 21.

    Returns:
        data_state_space (numpy.ndarray): State space values from the dataset.  
        data_state_space_source (numpy.ndarray): Source term values corresponding to the dataset state space.  
        data_mf (numpy.ndarray): Mixture fraction values from the dataset.  
        data_T (numpy.ndarray): Temperature values from the dataset.  
        state_space_names (list[str]): Names of the state space variables in the dataset.
    """

    files = [i for i in listdir(data_path) if isfile(join(data_path + i))]

    state_space_names = []
    state_space_source_names = [] #to check it is the same order as the state space
    Not_species = ["T","U","V","W","X","Z"]
    data_state_space = np.zeros((nbr_rows*nbr_cols,nbr_species))
    data_state_space_source = np.zeros((nbr_rows*nbr_cols,nbr_species))
    data_mf = np.zeros((nbr_rows*nbr_cols,1))
    data_T = np.zeros((nbr_rows*nbr_cols,1))

    counter_state_space = 0
    counter_state_space_source = 0

    for file in files:
        if file.endswith('.mat'):

            data = loadmat(data_path + file)

            filename = file.removesuffix('.mat')
            file_data = np.array(data[filename])
            

            #if name of the file is not starting with an uppercase
            if(not file[0] == "R" and file[0].isupper() and file[0] not in Not_species):
                state_space_names.append(file.removesuffix('_3D_slice.mat'))
                flattened_data = file_data[layer, 0][0, :, :].flatten()
                data_state_space[:,counter_state_space] = flattened_data

                counter_state_space+=1

            elif(file[0] == "R"):
                state_space_source_names.append(file.removesuffix('_3D_slice.mat'))
                flattened_data = file_data[layer, 0][0, :, :].flatten()
                data_state_space_source[:,counter_state_space_source] = flattened_data

                counter_state_space_source+=1

            elif(file[0] == "Z"):

                flattened_data = file_data[layer, 0][0, :, :].flatten()
                data_mf[:,0] = flattened_data
                
            elif(file[0] == "T"):

                flattened_data = file_data[layer, 0][0, :, :].flatten()
                data_T[:,0] = flattened_data
                

    return data_state_space, data_state_space_source, data_mf, data_T, state_space_names

def load_0D_data(filename_autoignition = "isochoric-adiabatic-closed-HR-H2-air-lin_Z_0.015_0.035_100-T0_900-",
                 path_data_autoignition = "data-files/autoignition/", path_data = "data-files/"):
    """Load the different arrays composing the autoignition dataset.

    Args:
        filename_autoignition (str, optional): Filename of the autoignition dataset. Defaults to "isochoric-adiabatic-closed-HR-H2-air-lin_Z_0.015_0.035_100-T0_900-".
        path_data_autoignition (str, optional): Path to the autoignition dataset. Defaults to "data-files/autoignition/".
        path_data (str, optional): Path to the DNS dataset. Defaults to "data-files/".

    Returns:
        mixture_fractions_train (numpy.ndarray): Mixture fraction values for the training dataset.  
        mixture_fractions_test (numpy.ndarray): Mixture fraction values for the test dataset.  
        state_space_names (list[str]): Names of the state space variables.  
        state_space_train (pandas.Dataframe): State space values for the training dataset.  
        state_space_source_train (pandas.Dataframe): Source term values corresponding to the training state space.  
        time_train (numpy.ndarray): Time values associated with the training dataset.  
        state_space_train_np (ndanumpy.ndarrayrray): Numpy array representation of the training state space.  
        state_space_source_train_np (numpy.ndarray): Numpy array representation of the training state space source terms.  
        state_space_names_DNS (list[str]): Names of the state space variables from the DNS dataset. 
    """

    #create all the datasets
    mixture_fractions_train = np.loadtxt(f"{path_data_autoignition}{filename_autoignition}mixture-fraction.csv") #1 x nbr_timesteps
    mixture_fractions_test = np.loadtxt(f"{path_data_autoignition}{filename_autoignition}mixture-fractions-test-trajectories.csv") #1 x nbr_test_trajectories
    state_space_names = np.genfromtxt(f"{path_data_autoignition}{filename_autoignition}state-space-names.csv", delimiter=",", dtype=str)
    state_space_train = pd.read_csv(f"{path_data_autoignition}{filename_autoignition}state-space.csv", names = state_space_names)
    state_space_source_train = pd.read_csv(f"{path_data_autoignition}{filename_autoignition}state-space-sources.csv", names = state_space_names)
    time_train = np.loadtxt(f"{path_data_autoignition}{filename_autoignition}time.csv") #1 x nbr_timesteps

    state_space_names_DNS = np.genfromtxt(f"{path_data}Xu-state-space-names.csv", delimiter=",", dtype=str)

    #create a np array in the format for the DNS dataset/optimized PV
    state_space_train_np = state_space_train[state_space_names_DNS].to_numpy()
    state_space_source_train_np = state_space_source_train[state_space_names_DNS].to_numpy()

    return mixture_fractions_train, mixture_fractions_test, state_space_names, state_space_train, state_space_source_train, time_train, state_space_train_np, state_space_source_train_np, state_space_names_DNS.tolist()

def compute_costs_from_varianceFile(name_variance, path_variance = "data-files/costs/", penalty_function = 'log-sigma-over-peak', power = 4, vertical_shift = 1):
    """Compute the costs based on the variance file.

    Args:
        name_variance (str): Name of the variance file.
        path_variance (str, optional): Path to the variance file. Defaults to "data-files/costs/".
        penalty_function (str, optional): Type of penalty applied to the derivatives. Defaults to 'log-sigma-over-peak'.
        power (int, optional): Power applied for the penalty. Defaults to 4.
        vertical_shift (int, optional): Vertical shift applied for the penalty. Defaults to 1.

    Returns:
        np.ndarray: Array containing the cost of every QoI.
    """
    variance = np.load(f"{path_variance}{name_variance}", allow_pickle=True).item()

    (derivative, bandwidth_values, max_derivative) = normalized_variance_derivative(variance)

    costs = cost_function_normalized_variance_derivative(   variance,
                                                            penalty_function=penalty_function,
                                                            power=power,
                                                            vertical_shift=vertical_shift,
                                                            norm=None)
    
    return costs

def compute_derivative_from_varianceFile(name_variance, path_variance = "data-files/costs/", penalty_function = 'log-sigma-over-peak', power = 4, vertical_shift = 1):
    """Returns the derivatives, bandwith value and the bandwidth value with the highest derivative based on the variance file.

    Args:
        name_variance (str): Name of the variance file.
        path_variance (str, optional): Path to the variance file. Defaults to "data-files/costs/".
        penalty_function (str, optional): Type of penalty applied to the derivatives. Defaults to 'log-sigma-over-peak'.
        power (int, optional): Power applied for the penalty. Defaults to 4.
        vertical_shift (int, optional): Vertical shift applied for the penalty. Defaults to 1.

    Returns:
        derivative (np.ndarray): Array containing the derivative at each bandwidth value.
        bandwidth_values (np.ndarray): Array of bandwidth values at which the derivatives were computed.
        max_derivative (float): Bandwidth value at which the highest derivative is reached.
    """
    variance = np.load(f"{path_variance}{name_variance}", allow_pickle=True).item()

    (derivative, bandwidth_values, max_derivative) = normalized_variance_derivative(variance)

    return derivative, bandwidth_values, max_derivative