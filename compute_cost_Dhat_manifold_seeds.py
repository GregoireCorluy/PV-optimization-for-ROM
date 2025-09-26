"""
Compute the MSE to reconstruct the QoIs given the mixture fraction-PV manifold obtained from the encoder-decoder.
Author: Grégoire Corlùy (gregoire.stephane.corluy@ulb.be)
Date: November 2024
Python version: 3.10.10
"""


from utils import *
from loader import *

from PCAfold import compute_normalized_variance, normalized_variance_derivative, cost_function_normalized_variance_derivative, plot_normalized_variance_derivative
import numpy as np
import matplotlib.pyplot as plt
import sys

logging.disable(logging.CRITICAL)


#name of the file
if len(sys.argv) > 1:
    filename_base = sys.argv[1]  #argument to the script
else:
    print("No name provided to the script.")
    exit()

def compute_avg(costs):
    n = len(costs)
    sum = np.sum(costs**2)
    return 1/n*np.sqrt(sum)

print(filename_base)

for i in range(0,10):
    filename = filename_base + str(i)

    path_data = "data-files/"
    filename_metadata = filename + "_metadata.pkl"
    filename_species_names = "Xu-state-space-names.csv"
    path_metadata = "metadata/"

    dataset_type = "autoignition"

    penalty_function = 'log-sigma-over-peak'
    start_bw = -6
    end_bw = 2
    nbr_points_bw = 100
    bandwidth_values = np.logspace(start_bw, end_bw, nbr_points_bw)
    power = 4
    vertical_shift = 1

    loader = loadData(filename_species_names, path_metadata, filename_metadata)
    output_idx = [1, 2, 3, 5, 10, 15, 16, 18, 19]  #loader.metadata["output species idx"]
    idx_species_removed = loader.metadata["list idx species removed source"] if loader.metadata["dataset_type"].startswith("autoignition_augm") else loader.metadata["idx species removed"]
    input_scaling = loader.metadata["input scaling"]
    input_species_scaling = loader.metadata["input species scaling"]
    input_species_bias = loader.metadata["input species bias"]
    range_mf  = loader.metadata["range_mf"]
    depvar_names_species = ['H2O2', 'H2O', 'H2', 'HO2', 'N2O', 'NO2', 'NO', 'O2', 'OH'] #loader.metadata["list species output"]
    depvar_names_idx = loader.metadata["output idx Kreg"] if loader.metadata["dataset_type"]=="autoignition_augm" else loader.metadata["output elements"]
    if(loader.metadata["dataset_type"].startswith("autoignition_augm")):
        if(loader.metadata["Temperature at output"]):
            depvar_names_idx.append("T")
        for i in range(1,1+loader.metadata["PV dim"]):
            depvar_names_idx.append(f"PV{i}")
    PV_dim = loader.metadata["PV dim"]
    depvar_names = depvar_names_species + depvar_names_idx[-(PV_dim+1):]
    model = loader.loadModel()
    id = loader.metadata["Training_id"]

    #get the input (PV and f) and the output (interested Yi, T and source terms) data
    input, output = get_dataset(path_data + f"Xu-state-space-{dataset_type}.csv", path_data + f"Xu-state-space_source-{dataset_type}.csv",
                                path_data + f"Xu-T-{dataset_type}.csv", path_data + f"Xu-mf-{dataset_type}.csv", output_idx,
                                idx_species_removed, input_scaling, input_species_scaling, input_species_bias, range_mf)
    PV = model.get_PV(input)
    PV_f = torch.cat((PV, input[:, -1].reshape(-1, 1)), dim = 1) #reshape to be (5200,1) instead of (52000)
    output = model.get_source_PV(output, input_species_scaling)

    #scale every column of the PV_f tensor between 0 and 1
    min_vals = PV_f.min(dim=0, keepdim=True).values
    max_vals = PV_f.max(dim=0, keepdim=True).values
    PV_f_scaled = (PV_f - min_vals) / (max_vals - min_vals)

    indepVars = PV_f_scaled.detach().numpy()
    depVars = output.detach().numpy()

    variance_data = compute_normalized_variance(indepVars,
                                                    depVars,
                                                    depvar_names=depvar_names,
                                                    bandwidth_values=bandwidth_values)
    np.save(f"costs/variance_{id}-bw_{start_bw}_{end_bw}_{nbr_points_bw}-dataset_{dataset_type}.npy", variance_data)

    costs = cost_function_normalized_variance_derivative(variance_data,
                                                        penalty_function=penalty_function,
                                                        power=power,
                                                        vertical_shift=vertical_shift,
                                                        norm=None)
    np.save(f"costs/costs_{id}-bw_{start_bw}_{end_bw}_{nbr_points_bw}-p_{power}-ver_sh_{vertical_shift}-dataset_{dataset_type}.npy", costs)

    (derivative, bandwidth_values, max_derivative) = normalized_variance_derivative(variance_data)

    plt = plot_normalized_variance_derivative(variance_data)
    plt.savefig(f"costs/plot_Dhat_{id}-bw_{start_bw}_{end_bw}_{nbr_points_bw}-p_{power}-ver_sh_{vertical_shift}-dataset_{dataset_type}.png")

    print(f"{np.round(compute_avg(np.array(costs)),5)}")