import os
from pandas import read_csv, concat
import csv
import numpy as np
import pickle
import logging
from models import *
import pandas as pd
from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
from loader import *
from PCAfold import normalized_variance_derivative, cost_function_normalized_variance_derivative

"""
PV optimization on DNS dataset of Xu using an encoder-decoder architecture (Kamila Zdybal)
Version: Tools for the training
Author: Grégoire Corlùy (gregoire.stephane.corluy@ulb.be)
Date: September 2024
Python version: 3.10.10
"""

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    datefmt='%Y-%m-%d %H:%M:%S'
)


#################################
#Create directories and filenames
#################################

class create_dirs:
    """ Creates directories for saving trained models and training information """

    def __init__(self, optimizer, epo, lr, current_time, training_id):
        self.optimizer = optimizer
        self.epo = epo
        self.lr = lr
        self.formatted_date = current_time.strftime('%d%b%Y')
        self.formatted_time = current_time.strftime('%Hh%M')
        self.train_info_path = 'train-info/trained-models.csv'
        self.training_id = training_id

        self.path_out = "out/"
        self.path_curve = "curves/"
        self.path_metadata = "metadata/"

        self.training_name = 'Xu-AE-opt_{}-epo_{}-lr_{}-date_{}-hour_{}_{}'.format(
            self.optimizer, 
            self.epo, 
            self.lr,
            self.formatted_date,
            self.formatted_time,
            self.training_id)
            
        self.dirout = f'{self.training_name}_model.pth'
        
        self.dircurves = f'{self.training_name}_curves.csv'

        self.dirMetadata = f'{self.training_name}_metadata.pkl'

    def create(self, train_headers):
        """Create necessary paths to save the different file types.

        Args:
            train_headers (str): Name of the model used which is included inside the filenames.
        """
        
        if not os.path.exists(self.path_out):
            os.makedirs(self.path_out)

        if not os.path.exists('train-info/'):
            os.makedirs('train-info/')
        
        if not os.path.exists(self.path_curve):
            os.makedirs(self.path_curve)

        if not os.path.exists(self.path_metadata):
            os.makedirs(self.path_metadata)
        
        if not os.path.exists(self.train_info_path):
            #add headers to the file if the file is not existing yet
            with open(self.train_info_path, mode='w', newline='') as file: #wrie or overwrite document
                writer = csv.DictWriter(file, fieldnames=train_headers)
                writer.writeheader()

    def save_model(self, state):
        """Save the weights of the trained model.

        Args:
            state (dict): Dictionary containing the weights of the model.
        """

        torch.save(state, '{}{}'.format(self.path_out, self.dirout))

    def load_model(self, model_params):
        """Load the model given model hyperparameters.

        Args:
            model_params (dict): Hyperparameters of the model.

        Returns:
            PV_autoencoder: Trained encoder-decoder.
        """

        model_reloaded = PV_autoencoder(**model_params)
        model_reloaded.load_state_dict(torch.load('{}{}'.format(self.path_out, self.dirout), weights_only=False))

        return model_reloaded

    def save_train_info_model(self, data_train_info):
        """Save all the information concerning the training of the encoder-decoder.

        Args:
            data_train_info (dict): Dictionary with all the information about the metadata.
        """

        with open(self.train_info_path, mode='a', newline='') as file:  #append data with "a"
            writer = csv.DictWriter(file, fieldnames=data_train_info.keys())
            writer.writerow(data_train_info)

    def save_train_val_curves(self, train_curve, val_curve):
        """Save the training and validation curves of the encoder-decoder training.

        Args:
            train_curve (numpy.array): Encoder loss over the complete training.
            val_curve (numpy.array): Validation loss over the complete training.
        """

        with open(f'{self.path_curve}{self.dircurves}', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(train_curve)  #training curve
            writer.writerow(val_curve)  #validation curve

    def save_metadata(self, metadata):
        """Save the metadata of trained encoder-decoder.

        Args:
            metadata (dict): Dictionary with all the information about the metadata.
        """
        
        with open(f'{self.path_metadata}{self.dirMetadata}', 'wb') as f:
            pickle.dump(metadata, f)
    
    def modify_dirout(self, epo: int):
        """Modify the filename of the encoder-decoder

        Args:
            epo (int): Current epoch of the training

        """

        self.dirout = f'{self.training_name}_epo{epo}_model.pth'

        return None

######################################
#Handle input and output species lists
######################################

class Species:
    """
        Set of functions to convert name of species to indices
    """
    def __init__(self, path_data, file_species_names = "Xu-state-space-names.csv"):
        self.list_species = []

        #create a list with all the species names
        with open(path_data + file_species_names, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.list_species.append(row[0])

    def get_idx_of_species(self, species_name):
        """Get the index of a specific species given its name.

        Args:
            species_name (list[str]): List of species names.

        Returns:
            list[int]: List of indices corresponding to species to remove.
        """
        
        try:
            idx_species_removed = self.list_species.index(species_name)
        except ValueError:
            logging.warning(f"'{species_name}' is not in the list.")

        return idx_species_removed
    
    def get_idx_from_list_species(self, list_species_name):
        """Get the indices of a list of species given their names.

        Args:
            list_species_name (list[str]): List of species names

        Returns:
            list[int]: Get indices of species in list
        """

        list_idx = []

        for species_name in list_species_name:
            list_idx.append(self.get_idx_of_species(species_name))

        return list_idx
    
    def get_list_species(self):
        """Get list of species names.

        Returns:
            str: List of species names
        """

        return self.list_species


###################################
#Get data for training and plotting
###################################

def get_data(path_data_state_space, path_data_source, path_data_temp, path_data_mf, generator, perc_val, output_idx, list_idx_species_removed, input_scaling, scaling_mf, output_scaling, temperature_output, header = 'infer', list_idx_species_removed_source = None):
    """Get a train and validation input and output tensors for the chemical species and temperature
    
        Input = all chemical species + mixture fraction (last row)
        
        Output = chosen chemical species (with output_idx); temperature and all the source terms. PV-source has to be added afterwards given the source terms.
    
        Remark: data is given in torch.float64 format. This function is a short version of the get_dataloader function.
                Compared to get_dataset function, here the scaling of the output variables is already done and split into training and validation tensors.

    Args:
        path_data_state_space (str): Path to the data state space.
        path_data_source (str): Path to the data state space source.
        path_data_temp (str): Path to the temperature dataset.
        path_data_mf (str): Path to the mixture fraction dataset.
        generator (torch.Generator): Generator for reproducibility of the encoder-decoder initialization.
        perc_val (float): Float between 0 and 1 indicating the percentage for the validation dataset.
        output_idx (list[int]): List of species to include in the output dataset.
        list_idx_species_removed (list[int]):  List of indices corresponding to the species to be removed.
        input_scaling (str): Scaling for the input.
        scaling_mf (float): Range to scale the mixture fraction.
        output_scaling (str): Scaling of the output.
        temperature_output (bool): Indicate if the temperature should be added to the output dataset.
        header (str, optional): Indicates if the dataset contains a header or not. Defaults to 'infer'.
        idx_species_removed_source (list[int], optional): List of indices corresponding to the species to be removed for the source term. Defaults to None.

    Returns:
            train_input (torch.Tensor): Input tensor for the training dataset.  
            train_output (torch.Tensor): Output tensor for the training dataset.  
            val_input (torch.Tensor): Input tensor for the validation dataset.  
            val_output (torch.Tensor): Output tensor for the validation dataset.  
            dataset_size (int): Total number of samples in the dataset before splitting.  
            input_species_scaling (torch.Tensor): Scaling factors applied to input species features.  
            input_species_bias (torch.Tensor): Bias values used during input rescaling.  
    """

    #load the data
    data_state_space = read_csv(path_data_state_space, header = header)
    data_state_space_source = read_csv(path_data_source, header = header)
    data_temp = read_csv(path_data_temp, header = header)
    data_mf = read_csv(path_data_mf, header = header)

    #selected output species
    data_output_species = data_state_space.iloc[:,output_idx]

    if(list_idx_species_removed_source is None):
        list_idx_species_removed_source = list_idx_species_removed

    #remove one species from the dataframe
    data_state_space = data_state_space.drop(data_state_space.columns[list_idx_species_removed], axis=1)
    data_state_space_source = data_state_space_source.drop(data_state_space_source.columns[list_idx_species_removed_source], axis=1)

    #combine the data for input and output
    data_input = concat([data_state_space, data_mf], axis=1)

    if(temperature_output):
        data_output = concat([data_output_species, data_temp, data_state_space_source], axis=1)
        nbr_feat_rescale = len(output_idx)+1 #all species and temperature
    else:
        data_output = concat([data_output_species, data_state_space_source], axis=1)
        nbr_feat_rescale = len(output_idx) #all species without temperature

    input = data_input.iloc[:,:].values
    output = data_output.iloc[:, :].values #contains only the species that do not change, the PV reaction rate has to be added during the training phase

    #convert to PyTorch tensors
    input_tensor = torch.tensor(input)
    output_tensor = torch.tensor(output)
    
    #determine number of training and validation samples
    dataset_size = input_tensor.size(0)
    train_size = int((1-perc_val) * dataset_size)

    #shuffle the indices
    indices = torch.randperm(dataset_size, generator=generator)

    #split in training and validation indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    #create input and output tensors for both training and validation
    train_input, val_input = input_tensor[train_indices], input_tensor[val_indices]
    train_output, val_output = output_tensor[train_indices], output_tensor[val_indices]


    ##############
    #RESCALE input
    ##############
    if(input_scaling=="0to1" or input_scaling == "std"):
        mins = train_input[:, :-1].min(dim=0, keepdim=True)[0]

        if(input_scaling == "0to1"):
            maxs = train_input[:, :-1].max(dim=0, keepdim=True)[0]
            input_species_scaling = maxs - mins
        elif(input_scaling == "std"):
            input_species_scaling = train_input[:, :-1].std(dim=0, unbiased = False) #compute the population standard deviation (N, unbiased False)
        input_species_bias = mins

        #rescale the selected training inputs
        train_input[:, :-1] = (train_input[:, :-1] - mins) / input_species_scaling

        #rescale the selected validation inputs
        val_input[:, :-1] = (val_input[:, :-1] - mins) / input_species_scaling

    elif(input_scaling=="-1to1"):
        mins = train_input[:, :-1].min(dim=0, keepdim=True)[0]
        maxs = train_input[:, :-1].max(dim=0, keepdim=True)[0]

        input_species_scaling = maxs - mins
        input_species_bias = mins

        #rescale the selected training inputs
        train_input[:, :-1] = 2*(train_input[:, :-1] - mins) / input_species_scaling -1

        #rescale the selected validation inputs
        val_input[:, :-1] = 2*(val_input[:, :-1] - mins) / input_species_scaling -1

    elif(input_scaling=="pareto"):
        input_species_scaling = train_input[:, :-1].std(dim=0, unbiased = False)
        nbr_columns = train_input[:, :-1].shape[1]
        input_species_bias = torch.zeros(1, nbr_columns)

        #rescale the selected training inputs
        train_input[:, :-1] = (train_input[:, :-1]) / input_species_scaling

        #rescale the selected validation inputs
        val_input[:, :-1] = (val_input[:, :-1]) / input_species_scaling

    elif(input_scaling=="None"):
        nbr_columns = train_input[:, :-1].shape[1]
        input_species_scaling = torch.ones(1, nbr_columns)
        input_species_bias = torch.zeros(1, nbr_columns)

    else:
        raise ValueError("get_data: input scaling not recognized")

    ###########
    #RESCALE f
    ###########
    if(scaling_mf is not None):
        min_mf = train_input[:,-1].min()
        max_mf = train_input[:,-1].max()

        #scale mf between -0.5 and 0.5, and apply it also to the validation dataset
        #also a range of 1 like PV then
        train_input[:, -1] = (train_input[:, -1] - min_mf) / (max_mf - min_mf) - 0.5
        val_input[:, -1] = (val_input[:, -1] - min_mf) / (max_mf - min_mf) - 0.5

        train_input[:, -1] = train_input[:, -1]*scaling_mf
        val_input[:, -1] = val_input[:, -1]*scaling_mf

    elif(scaling_mf == "None"):
        logging.warning("No scaling for the mixture fraction feature at the input.")

    else:
        raise ValueError("get_data: mixture fraction scaling not recognized")

    ###############
    #RESCALE OUTPUT
    ###############
    #get min and max value of the species and temperature for the training dataset
    if(output_scaling=="-1to1"):
        mins = train_output[:, :nbr_feat_rescale].min(dim=0, keepdim=True)[0]
        maxs = train_output[:, :nbr_feat_rescale].max(dim=0, keepdim=True)[0]

        #rescale the selected training outputs
        train_output[:, :nbr_feat_rescale] = 2 * (train_output[:, :nbr_feat_rescale] - mins) / (maxs - mins) - 1

        #rescale the selected validation outputs
        val_output[:, :nbr_feat_rescale] = 2 * (val_output[:, :nbr_feat_rescale] - mins) / (maxs - mins) - 1
    else:
        raise ValueError("get_data: output scaling not recognized")

    return train_input, train_output, val_input, val_output, dataset_size, input_species_scaling, input_species_bias

def get_dataset(path_data_state_space, path_data_source, path_data_temp, path_data_mf,
                output_idx, idx_species_removed, input_scaling_name, input_scaling, input_bias,
                range_mf = "None", header = 'infer', idx_species_removed_source = None):
    """Get the complete dataset in tensor format.

    Args:
        path_data_state_space (str): Path to the data state space.
        path_data_source (str): Path to the data state space source.
        path_data_temp (str): Path to the temperature dataset.
        path_data_mf (str): Path to the mixture fraction dataset.
        output_idx (list[int]): List of species to include in the output dataset.
        idx_species_removed (list[int]): List of indices corresponding to the species to be removed.
        input_scaling_name (str): Name of the scaling applied to the input.
        input_scaling (float): Scaling for the input.
        input_bias (flaot): Centering used for the preprocessing of the input.
        range_mf (str, optional): Range to scale the mixture fraction. Defaults to "None".
        header (str, optional): Indicates if the dataset contains a header or not. Defaults to 'infer'.
        idx_species_removed_source (list[int], optional): List of indices corresponding to the species to be removed for the source term. Defaults to None.

    Returns:
        input_tensor (torch.Tensor): Input tensor for training the encoder-decoder.  
        output_tensor (torch.Tensor): Output tensor for training the encoder-decoder. 
    """

    #load the data
    data_state_space = read_csv(path_data_state_space, header = header)
    data_state_space_source = read_csv(path_data_source, header = header)
    data_temp = read_csv(path_data_temp, header = header)
    data_mf = read_csv(path_data_mf, header = header)

    if(range_mf != "None"):
        min_mf = data_mf.min().values[0]
        max_mf = data_mf.max().values[0]

        data_mf = (data_mf - min_mf) / (max_mf - min_mf) * (range_mf) - range_mf/2

    #selected output species
    data_output_species = data_state_space.iloc[:,output_idx]

    if(idx_species_removed_source is None):
        idx_species_removed_source = idx_species_removed

    #remove one species from the dataframe
    data_state_space = data_state_space.drop(data_state_space.columns[idx_species_removed], axis=1)
    data_state_space_source = data_state_space_source.drop(data_state_space_source.columns[idx_species_removed_source], axis=1)

    #combine the data for input and output
    data_input = concat([data_state_space, data_mf], axis=1)
    data_output = concat([data_output_species, data_temp, data_state_space_source], axis=1)

    input = data_input.iloc[:,:].values
    output = data_output.iloc[:, :].values #contains only the species that do not change, the PV reaction rate has to be added during the training phase

    #convert to PyTorch tensors
    input_tensor = torch.tensor(input)
    output_tensor = torch.tensor(output)

    if(input_scaling_name == "0to1" or input_scaling_name == "std"):
        input_tensor[:, :-1] = (input_tensor[:, :-1]-input_bias)/input_scaling
    elif(input_scaling_name == "-1to1"):
        input_tensor[:, :-1] = 2*(input_tensor[:, :-1]-input_bias)/input_scaling -1
    elif(input_scaling_name == "None"):
        logging.info("No input scaling")
    else:
        logging.warning("Input scaling not recognized")

    return input_tensor, output_tensor

def get_dataset_from_np(np_state_space, np_state_space_source, np_temp, np_mf, output_idx, idx_species_removed):
    """Get a the complete dataset in tensor format. Input and output tensor still need to be further converted to get the progress variable and progress variable source terms.

    Args:
        np_state_space (numpy.ndarray): State space dataset in numpy format.
        np_state_space_source (numpy.ndarray): State space source term dataset in numpy format.
        np_temp (numpy.ndarray): Temperature dataset in numpy format.
        np_mf (numpy.ndarray): Mixture fraction dataset in numpy format.
        output_idx (list[int]): List of species to include in the output dataset.
        idx_species_removed (list[int]): List of indices corresponding to the species to be removed.

    Returns:
        input_tensor (torch.Tensor): Input tensor for training the encoder-decoder.  
        output_tensor (torch.Tensor): Output tensor for training the encoder-decoder.    
    """

    #load the data
    data_state_space = pd.DataFrame(np_state_space)
    data_state_space_source = pd.DataFrame(np_state_space_source)
    data_temp = pd.DataFrame(np_temp)
    data_mf = pd.DataFrame(np_mf)

    #selected output species
    data_output_species = data_state_space.iloc[:,output_idx]

    #remove one species from the dataframe
    data_state_space = data_state_space.drop(data_state_space.columns[idx_species_removed], axis=1)
    data_state_space_source = data_state_space_source.drop(data_state_space_source.columns[idx_species_removed], axis=1)

    #combine the data for input and output
    data_input = concat([data_state_space, data_mf], axis=1)
    data_output = concat([data_output_species, data_temp, data_state_space_source], axis=1)

    input = data_input.iloc[:,:].values
    output = data_output.iloc[:, :].values #contains only the species that do not change, the PV reaction rate has to be added during the training phase

    #convert to PyTorch tensors
    input_tensor = torch.tensor(input)
    output_tensor = torch.tensor(output)

    return input_tensor, output_tensor

def get_dataset_from_np_scaled(np_state_space, np_state_space_source, np_temp, np_mf, output_idx, idx_species_removed, input_scaling, mf_scaling, path_data_state_space, path_data_mf):
    """Get a the complete dataset in tensor format.

    Args:
        np_state_space (numpy.ndarray): State space dataset in numpy format.
        np_state_space_source (numpy.ndarray): State space source term dataset in numpy format.
        np_temp (numpy.ndarray): Temperature dataset in numpy format.
        np_mf (numpy.ndarray): Mixture fraction dataset in numpy format.
        output_idx (list[int]): List of species to include in the output dataset.
        idx_species_removed (list[int]): List of indices corresponding to the species to be removed.
        input_scaling (str): Type of scaling used for the input.
        mf_scaling (float): Scaling of the mixture fraction.
        path_data_state_space (str): Path to data state space dataset.
        path_data_mf (str): Path to mixture fraction dataset

    Returns:
        input_tensor (torch.Tensor): Input tensor for training the encoder-decoder.  
        output_tensor (torch.Tensor): Output tensor for training the encoder-decoder.  
    """

    #load the data
    data_state_space = pd.DataFrame(np_state_space)
    data_state_space_source = pd.DataFrame(np_state_space_source)
    data_temp = pd.DataFrame(np_temp)
    data_mf = pd.DataFrame(np_mf)

    data_state_space_scaler = read_csv(path_data_state_space)
    data_mf_scaler = read_csv(path_data_mf)

    #scaling of input
    if(input_scaling == "0to1"):
        for i in range(data_state_space.shape[1]):
            min_val = data_state_space_scaler.iloc[:, i].min()
            max_val = data_state_space_scaler.iloc[:, i].max()
            
            #Scalte 0 to 1
            data_state_space.iloc[:, i] = (data_state_space.iloc[:, i] - min_val) / (max_val - min_val)

            print(data_state_space.iloc[:, i].min())
            print(data_state_space.iloc[:, i].max())

    if(mf_scaling == "-05to05"):
        min_val = data_mf_scaler.iloc[:, 0].min()
        max_val = data_mf_scaler.iloc[:, 0].max()
        print(min_val)
        print(max_val)
        
        #Scalte 0 to 1
        data_mf.iloc[:, 0] = (data_mf.iloc[:, 0] - min_val) / (max_val - min_val)

    #selected output species
    data_output_species = data_state_space.iloc[:,output_idx]

    #remove one species from the dataframe
    data_state_space = data_state_space.drop(data_state_space.columns[idx_species_removed], axis=1)
    data_state_space_source = data_state_space_source.drop(data_state_space_source.columns[idx_species_removed], axis=1)

    #combine the data for input and output
    data_input = concat([data_state_space, data_mf], axis=1)
    data_output = concat([data_output_species, data_temp, data_state_space_source], axis=1)

    input = data_input.iloc[:,:].values
    output = data_output.iloc[:, :].values #contains only the species that do not change, the PV reaction rate has to be added during the training phase

    #convert to PyTorch tensors
    input_tensor = torch.tensor(input)
    output_tensor = torch.tensor(output)

    return input_tensor, output_tensor

def loadFullData(data_path, layer):
    
    """Load the complete dataset from a layer and return the separate numpy arrays.

    Args:
        data_path (str): Path to data files.
        idx_layer (int): Index of the layer to be considered.

    Returns:
        state_space_names (list[str]): Names of the state space variables.  
        data_state_space (numpy.ndarray): State space values from the dataset.  
        data_state_space_source (numpy.ndarray): Source term values corresponding to the state space  
        data_T (numpy.ndarray): Temperature values from the dataset.  
        data_mf (numpy.ndarray): Mixture fraction values from the dataset.
    """

    files = [i for i in listdir(data_path) if isfile(join(data_path + i))]

    nbr_species = 21
    nbr_rows = 1536
    nbr_cols = 1024

    state_space_names = []
    state_space_source_names = [] #to check it is the same order as the state space
    Not_species = ["T","U","V","W","X","Z"]
    state_space = np.zeros((1,nbr_species))
    data_state_space = np.zeros((nbr_rows*nbr_cols,nbr_species))
    data_state_space_source = np.zeros((nbr_rows*nbr_cols,nbr_species))
    data_mf = np.zeros((nbr_rows*nbr_cols,1))
    data_T = np.zeros((nbr_rows*nbr_cols,1))

    counter_state_space = 0
    counter_state_space_source = 0

    for file in files:
        if file.endswith('.mat'):
            print(f"Busy with {file}")

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
                
                print()
                print("mass fraction data done")
                print()

            elif(file[0] == "T"):

                flattened_data = file_data[layer, 0][0, :, :].flatten()
                data_T[:,0] = flattened_data
                
                print()
                print("Temperature data done")
                print()

    return state_space_names, data_state_space, data_state_space_source, data_T, data_mf

###########
#Rescalings
###########

def rescale_PVsource(data, nbr_PV):
    """Rescale indicated variable between -1 and 1.

    Used for the source PV.

    Args:
        data (torch.Tensor): Output dataset containing the PV source terms.
        nbr_PV (int): Number of progress variables.

    Returns:
        torch.Tensor: Returns the output dataset with the PV source terms rescaled.
    """

    #get the feature to rescale
    features = data[:, -nbr_PV:]
    
    #initialize holder for rescaled features
    rescaled_features = []
    
    #rescale every feature
    for i in range(nbr_PV):
        column = features[:, i]
        min_val = column.min()
        max_val = column.max()
        rescaled_column = 2 * (column - min_val) / (max_val - min_val) - 1
        rescaled_features.append(rescaled_column.unsqueeze(1))
    
    # Concatenate all rescaled columns together along the last dimension
    rescaled_features = torch.cat(rescaled_features, dim=1)
    
    # Concatenate the non-PV part of the data with the rescaled PV features
    data = torch.cat((data[:, :-nbr_PV], rescaled_features), dim=1)
    
    return data

################
#Optimizer tools
################

def get_optimizer(model_parameters, optimizer_name, learning_rate, alpha = None, momentum = None):

    """Get the optimizer to train the encoder-decoder.

    Args:
        model_parameters (dict): Dictionary with the hyperparameters of the encoder-decoder.
        optimizer_name (str): Optimizer name.
        learning_rate (float): Learning rate for training the encoder-decoder
        alpha (Float, optional): Alpha variable for the RMSprop optimizer. Defaults to None.
        momentum (Float, optional): Momentum variable for the RMSprop optimizer. Defaults to None.

    Raises:
        ValueError: Not all variables provided for the RMSprop.
        ValueError: The provided optimizer name is not part of the ones defined in the function.

    Returns:
        torch.optim: Optimizer for the encoder-decoder training.
    """

    if(optimizer_name.lower()=="adam"):
        optimizer = torch.optim.Adam(model_parameters, lr=learning_rate)
    elif(optimizer_name.lower()=="rmsprop"):
        if(alpha is None or momentum is None):
            raise ValueError("get_optimizer: both alpha and momentum should be defined for RMSprop optimizer")
        optimizer = torch.optim.RMSprop(model_parameters, lr=learning_rate, alpha = alpha, momentum = momentum)
    else:
        raise ValueError("get_optimizer: optimizer_name not in the list")
    
    return optimizer

def get_loss_criterion(loss_name, lambda_reg=1):
    """Get the loss criterion for training the encoder-decoder.

    Args:
        loss_name (str): Name of the loss.
        lambda_reg (float): Trade-off between the MSE and the orthogonalization term.

    Raises:
        ValueError: Error in case the provided loss names does not correspond to one of the ones defined in the function.

    Returns:
        Union[nn.Module, Callable]: Either a PyTorch loss module (e.g., nn.MSELoss) or 
        a custom loss function that computes and returns a scalar torch.Tensor.
    """
    
    if(loss_name.lower()=="mse"):
        loss_criterion = nn.MSELoss()

    elif(loss_name.lower() == "mse_orth_W_pv1_pv2"):

        """
        MSE for the reconstruction with a regularization term promoting orthogonality between the weights of PV1 and PV2.

        Pay attention: this loss function only works correctly in case the first PV is not fixed or if one PV is fixed, then without scaling.
        The loss here does not the scaling which would have an influence with the first PV fixed.
        """

        def custom_loss(predictions, targets, model):
            #MSE for reconstruction error
            mse = nn.MSELoss()(predictions, targets)
            
            #regularization term on the encoder weights
            PV1_PV2_orth_reg = torch.abs(torch.dot(model.encoder_species.weight[0],model.encoder_species.weight[1]))
            
            # Combine the two losses
            loss = mse + lambda_reg * PV1_PV2_orth_reg
            return loss
        
        loss_criterion = custom_loss
    
    elif(loss_name.lower() == "mse_orth_pv1_pv2"):

        """
        MSE for the reconstruction with a regularization term promoting orthogonality between PV1 and PV2.

        Pay attention: this loss function only works correctly in case the first PV is not fixed or if one PV is fixed, then without scaling.
        The loss here does not the scaling which would have an influence with the first PV fixed.
        """

        def custom_loss(predictions, targets, model, input):
            #MSE for reconstruction error
            mse = nn.MSELoss()(predictions, targets)
            
            PV = model.get_PV(input)

            #regularization term on the encoder weights
            PV1_PV2_orth_reg = torch.abs(torch.dot(PV[:,0], PV[:,1]))
            
            # Combine the two losses
            loss = mse + lambda_reg * PV1_PV2_orth_reg
            return loss
        
        loss_criterion = custom_loss
    
    elif(loss_name.lower() == "mse_orth_pv123"):

        """
        MSE for the reconstruction with a regularization term promoting orthogonality between PV1, PV2 and PV3.

        Pay attention: this loss function only works correctly in case the first PV is not fixed or if one PV is fixed, then without scaling.
        The loss here does not the scaling which would have an influence with the first PV fixed.
        """

        def custom_loss(predictions, targets, model, input):
            #MSE for reconstruction error
            mse = nn.MSELoss()(predictions, targets)
            
            PV = model.get_PV(input)

            #regularization term on the encoder weights
            PV123_orth_reg = torch.abs(torch.dot(PV[:,0], PV[:,1])) + torch.abs(torch.dot(PV[:,0], PV[:,2])) + torch.abs(torch.dot(PV[:,1], PV[:,2]))
            
            # Combine the two losses
            loss = mse + lambda_reg * PV123_orth_reg
            return loss
        
        loss_criterion = custom_loss
    
    elif(loss_name.lower() == "mse_orth_pv1234"):

        """
        MSE for the reconstruction with a regularization term promoting orthogonality between PV1, PV2, PV3 and PV4.

        Pay attention: this loss function only works correctly in case the first PV is not fixed or if one PV is fixed, then without scaling.
        The loss here does not the scaling which would have an influence with the first PV fixed.
        """

        def custom_loss(predictions, targets, model, input):
            #MSE for reconstruction error
            mse = nn.MSELoss()(predictions, targets)
            
            PV = model.get_PV(input)

            #regularization term on the encoder weights
            PV1234_orth_reg = torch.abs(torch.dot(PV[:,0], PV[:,1])) + torch.abs(torch.dot(PV[:,0], PV[:,2])) + torch.abs(torch.dot(PV[:,0], PV[:,3])) + torch.abs(torch.dot(PV[:,1], PV[:,2])) + torch.abs(torch.dot(PV[:,1], PV[:,3])) + torch.abs(torch.dot(PV[:,2], PV[:,3]))
            
            # Combine the two losses
            loss = mse + lambda_reg * PV1234_orth_reg
            return loss
        
        loss_criterion = custom_loss


    else:
        raise ValueError("get_loss_criterion: loss_name not in the list")
    
    return loss_criterion

def cosine_decay(alpha, epo, tot_epo):
    """Cosine decay learning rate. Start at the initial learning rate and ends at initial learning rate times alpha.
    After tot_epo, the learning rate is constant and equal to initial learning rate times alpha.
    Alpha is the multiplier for the final learning rate.

    Args:
        alpha (float): Indicates the decrease of the learning rate compared to the initial learning rate.
        epo (int): Current epoch.
        tot_epo (int): Epoch at which the final learning rate is reached.

    Returns:
        float: Current learning rate
    """

    myEpo = np.min([epo,tot_epo])

    return 0.5*(1-alpha)*(1+np.cos(np.pi*myEpo/tot_epo))+alpha

def load_PV(optimized_PV, data_state_space, data_state_space_source, state_space_names, scaled_PV = False, filename_optimized_PV = "", weight_inversion = False):
    """Load the optimized PV and its source term.

    Args:
        optimized_PV (bool): Indicating if the PV is the optimized or heuristic one.
        data_state_space (numpy.ndarray): Dataset of the state space.
        data_state_space_source (numpy.ndarray): Dataset of the state space source terms.
        state_space_names (list[str]): List of the species in the state space.
        scaled_PV (bool, optional): Indicates if the PV has to be scaled. Defaults to False.
        filename_optimized_PV (str, optional): Metadata filename for the optimized PV. Defaults to "".
        weight_inversion (bool, optional): Indicates if the definition of the PV has to be inversed. Defaults to False.

    Returns:
        PV (numpy.ndarray): Array containing the optimized PV.
        PV_source (numpy.ndarray): Array containing the optimized PV source terms.
        id_model (str): Id of the model.
    """
    
    if(optimized_PV):
        filename_metadata = filename_optimized_PV + "_metadata.pkl"
        path_metadata = "metadata/"
        filename_species_names = "Xu-state-space-names.csv"
        path_data = "data-files/"

        loader = loadData(filename_species_names, path_metadata, filename_metadata)
        idx_species_removed = loader.metadata["list idx species removed source"] if loader.metadata["dataset_type"]=="autoignition_augm" else loader.metadata["idx species removed"]
        model = loader.loadModel()
        id_model = loader.metadata["Training_id"]

        # inverse the PV definition
        if(weight_inversion):
            with torch.no_grad():  # Ensures we do not track gradients for this operation
                model.encoder_species.weight.mul_(-1)

        PV = model.get_PV(torch.from_numpy(np.delete(data_state_space, idx_species_removed, axis=1))).detach().numpy()
        if(scaled_PV):
            PV = (PV - PV.min())/(PV.max() - PV.min())
        PV_source = model.get_PV(torch.from_numpy(np.delete(data_state_space_source, idx_species_removed, axis=1))).detach().numpy()
        
    else:
        idx_H2O = state_space_names.index("H2O")
        idx_H2 = state_space_names.index("H2")
        idx_O2 = state_space_names.index("O2")

        PV = data_state_space[:, idx_H2O] - data_state_space[:, idx_H2] - data_state_space[:, idx_O2]
        if(scaled_PV):
            PV = (PV - PV.min())/(PV.max() - PV.min())
        PV_source = data_state_space_source[:, idx_H2O] - data_state_space_source[:, idx_H2] - data_state_space_source[:, idx_O2]

        id_model = "Heuristic PV"

    return PV, PV_source, id_model

def get_variance_data(name_variance, path_variance = "data-files/costs/"):
    """Get the variance data given its filename.

    Args:
        name_variance (str): Filename containing the Dhat variance of each QoI.
        path_variance (str, optional): Path of where the variance files are saved. Defaults to "data-files/costs/".

    Returns:
        derivative (dict): Dictionary of Dhat for each variable.
        bandwidth_values (numpy.ndarray): Array of the bandwidth values at which the QoI was assessed.
        max_derivative (dict): Dictionaly of max Dhat for each variable.
    """

    variance = np.load(f"{path_variance}{name_variance}", allow_pickle=True).item()

    (derivative, bandwidth_values, max_derivative) = normalized_variance_derivative(variance)

    return derivative, bandwidth_values, max_derivative

def get_costs(name_variance, path_variance = "data-files/costs/", penalty_function = "log-sigma-over-peak", power = 4, vertical_shift = 1):

    """Compute costs of QoIs given the Dhat variance values.

    Args:
        name_variance (str): Filename containing the Dhat variance of each QoI.
        path_variance (str, optional): Path of where the variance files are saved. Defaults to "data-files/costs/".
        penalty_function (str, optional): Type of penalty used to compute the cost. Defaults to "log-sigma-over-peak".
        power (float, optional): Power used for the cost function indicating how much the small variance is amplified. Defaults to 4.
        vertical_shift (float, optional): Vertical shift for the computation of the cost. Defaults to 1.

    Returns:
        list[float]: List of the QoI costs.
    """
    
    variance = np.load(f"{path_variance}{name_variance}", allow_pickle=True).item()

    costs = cost_function_normalized_variance_derivative(variance,
                                                        penalty_function=penalty_function,
                                                        power=power,
                                                        vertical_shift=vertical_shift,
                                                        norm=None)
    
    return costs