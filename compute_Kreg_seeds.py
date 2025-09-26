"""
Compute the cost of manifolds parametrized by the different optimized PVs using the cost function of Kamila Zdybał with the PCAfold library.
Author: Grégoire Corlùy (gregoire.stephane.corluy@ulb.be)
Date: November 2024
Python version: 3.10.10
"""

from PCAfold import KReg
import numpy as np
from utils import *
from models import *
from loader import *
import logging
import sys

logging.basicConfig(level=logging.CRITICAL)

#name of the file
if len(sys.argv) > 1:
    filename_base = sys.argv[1]  #argument to the script
else:
    print("No name provided to the script.")
    exit()

dataset_type = "autoignition"

for i in range(0,10):
    filename = filename_base + str(i)
    filename_metadata = filename + "_metadata.pkl"
    path_metadata = "metadata/"
    filename_species_names = "Xu-state-space-names.csv"
    path_data = "data-files/"

    neighbours = [5, 10, 15, 20, 25]
    seed = 9

    mse_values_model = np.zeros(len(neighbours))

    loader = loadData(filename_species_names, path_metadata, filename_metadata)
    output_idx = loader.metadata["output idx Kreg"] if loader.metadata["dataset_type"].startswith("autoignition_augm") else loader.metadata["output species idx"]
    idx_species_removed = loader.metadata["list idx species removed source"] if loader.metadata["dataset_type"].startswith("autoignition_augm") else loader.metadata["idx species removed"]
    input_scaling = loader.metadata["input scaling"]
    input_species_scaling = loader.metadata["input species scaling"]
    input_species_bias = loader.metadata["input species bias"]
    range_mf  = loader.metadata["range_mf"]
    model = loader.loadModel()
    id = loader.metadata["Training_id"]

    input, output = get_dataset(path_data + f"Xu-state-space-{dataset_type}.csv", path_data + f"Xu-state-space_source-{dataset_type}.csv",
                                path_data + f"Xu-T-{dataset_type}.csv", path_data + f"Xu-mf-{dataset_type}.csv",
                                output_idx, idx_species_removed, input_scaling, input_species_scaling,
                                input_species_bias, range_mf)

    PV = model.get_PV(input)
    output = model.get_source_PV(output, input_species_scaling)

    f_PV = torch.cat((input[:,-1].unsqueeze(1), PV), dim = 1)

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
    # Sample 40,000 unique indices
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

    print(f" - {round(avg_mse*10000,3)} (\u00B1 {round(std_mse*10000, 3)}) - ")