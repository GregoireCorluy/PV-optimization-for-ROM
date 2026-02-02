from utils import *
from loader import *

logging.disable(logging.CRITICAL)

from PCAfold import compute_normalized_variance, normalized_variance_derivative, cost_function_normalized_variance_derivative, plot_normalized_variance_derivative
import numpy as np
import matplotlib.pyplot as plt

#Compute the cost for the PV of Xu via a model where the encoder is modified

def compute_avg(costs):
    n = len(costs)
    sum = np.sum(costs**2)
    return 1/n*np.sqrt(sum)

filename = "Xu-AE-opt_adam-epo_100000-lr_0.001-date_04Nov2024-hour_16h20_Tr31b"

path_data = "data-files/"
filename_metadata = filename + "_metadata.pkl"
filename_species_names = "Xu-state-space-names.csv"
path_metadata = "metadata/"

dataset_type = "low"

penalty_function = 'log-sigma-over-peak'
start_bw = -6
end_bw = 2
nbr_points_bw = 100
bandwidth_values = np.logspace(start_bw, end_bw, nbr_points_bw)
power = 4
vertical_shift = 1
sample_norm_var = False
sample_norm_range = False

loader = loadData(filename_species_names, path_metadata, filename_metadata)
output_idx = loader.metadata["output species idx"]
idx_species_removed = loader.metadata["idx species removed"]
input_scaling = loader.metadata["input scaling"]
input_species_scaling = loader.metadata["input species scaling"]
input_species_bias = loader.metadata["input species bias"]
range_mf  = loader.metadata["range_mf"]
depvar_names_species = loader.metadata["list species output"]
depvar_names_idx = loader.metadata["output elements"]
PV_dim = loader.metadata["PV dim"]
depvar_names = depvar_names_species + depvar_names_idx[-(PV_dim+1):]
model = loader.loadModel()
id = loader.metadata["Training_id"]

weights_Xu = torch.tensor([1, -1, 0, -1, 0, 0])

with torch.no_grad():
    model.encoder_species.weight.copy_(weights_Xu)

print("Updated Weights:\n", model.encoder_species.weight)

print("Get dataset")
#get the input (PV and f) and the output (interested Yi, T and source terms) data
input, output = get_dataset(path_data + f"Xu-state-space-{dataset_type}.csv", path_data + f"Xu-state-space_source-{dataset_type}.csv", path_data + f"Xu-T-{dataset_type}.csv", path_data + f"Xu-mf-{dataset_type}.csv", output_idx, idx_species_removed, input_scaling, input_species_scaling, input_species_bias, range_mf)
PV = model.get_PV(input)
PV_f = torch.cat((PV, input[:, -1].reshape(-1, 1)), dim = 1) #reshape to be (5200,1) instead of (52000)
output = model.get_source_PV(output, input_species_scaling)

#scale every column of the PV_f tensor between 0 and 1
min_vals = PV_f.min(dim=0, keepdim=True).values
max_vals = PV_f.max(dim=0, keepdim=True).values
PV_f_scaled = (PV_f - min_vals) / (max_vals - min_vals)

min_vals_output = output.min(dim=0, keepdim=True).values
max_vals_output = output.max(dim=0, keepdim=True).values
output_scaled = (output - min_vals_output) / (max_vals_output - min_vals_output)

indepVars = PV_f_scaled.detach().numpy()
depVars = output_scaled.detach().numpy()

print("Compute the variance")
variance_data = compute_normalized_variance(indepVars,
                                                depVars,
                                                depvar_names=depvar_names,
                                                bandwidth_values=bandwidth_values,
                                                compute_sample_norm_range=sample_norm_range,
                                                compute_sample_norm_var=sample_norm_var)
print("Computing the variance is fininshed")
np.save(f"costs/variance_Xu-bw_{start_bw}_{end_bw}_{nbr_points_bw}-dataset_{dataset_type}{'-sample-norm-var' if sample_norm_var else ''}{'-sample-norm-range' if sample_norm_range else ''}.npy", variance_data)

print("Compute the costs")
costs = cost_function_normalized_variance_derivative(variance_data,
                                                    penalty_function=penalty_function,
                                                    power=power,
                                                    vertical_shift=vertical_shift,
                                                    norm=None)
np.save(f"costs/costs_Xu-bw_{start_bw}_{end_bw}_{nbr_points_bw}-p_{power}-ver_sh_{vertical_shift}-dataset_{dataset_type}{'-sample-norm-var' if sample_norm_var else ''}{'-sample-norm-range' if sample_norm_range else ''}.npy", costs)

print("Compute the derivatives")
(derivative, bandwidth_values, max_derivative) = normalized_variance_derivative(variance_data)

print("Plot the derivatives")
plt = plot_normalized_variance_derivative(variance_data)
plt.savefig(f"costs/plot_Dhat_Xu-bw_{start_bw}_{end_bw}_{nbr_points_bw}-p_{power}-ver_sh_{vertical_shift}-dataset_{dataset_type}{'-sample-norm-var' if sample_norm_var else ''}{'-sample-norm-range' if sample_norm_range else ''}.png")
#plt.show()

print(f"Cost of PV_Xu for dataset {dataset_type}")
print(f"{np.round(compute_avg(np.array(costs)),3)}")