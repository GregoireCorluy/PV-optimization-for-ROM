"""
PV optimization on DNS dataset of Xu using an encoder-decoder architecture (Kamila Zdybal)
Version: Sped up version without using the dataloader during the training and using manual batches - train multiple times the same NN with different seeds
Author: Grégoire Corlùy (gregoire.stephane.corluy@ulb.be)
Date: September 2024
Python version: 3.10.10
"""


########
#Imports
########

from utils import *
import time
import numpy as np
import copy
import logging
from datetime import datetime
from tools import *


###########
#Parameters
###########

path_data = 'data-files/'
training_nbr = "0D-2s"
layer = 10
dataset_type = "autoignition_augm"
dataset_Kreg = "autoignition"
max_epo = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"My device: {device}")
optimizer_name = "adam"
output_scaling = "-1to1"
loss_name = "MSE"
lambda_reg = 1
PV_rescaling_init = True
PV_rescaling_batch = True
lr = 0.025
nbr_species = 21
PV_dim = 1
perc_val = 0.2 #percentage of validation data
species_removed = ["N2H3","N2"]
other_species_removed = ['logH2O2-10','logH2O-10','logH2-10','logHO2-10','logN2O-10','logNO2-10','logNO-10',
                        'logO2-10','logOH-10','logH2O2-20','logH2O-20','logH2-20','logHO2-20','logN2O-20','logNO2-20','logNO-20','logO2-20','logOH-20']
total_species_removed = species_removed + other_species_removed
init_algo_enc = "Normal"
init_algo_dec = "Normal"
init_enc = 0.05 #standard deviation for initialization of the encoder weights
init_dec = 0.05
current_time = datetime.now()
decoder_layers = [0, 10, 10] #decoder architecture
activation_function = "tanh"
activation_function_output = "tanh"
list_species_output = ['logH2O2-20','logH2O-20','logH2-20','logHO2-20','logN2O-20','logNO2-20','logNO-20','logO2-20','logOH-20']
list_species_output_Kreg = ['H2O2', 'H2O', 'H2', 'HO2', 'N2O', 'NO2', 'NO', 'O2', 'OH']
learning_rate_decay = "Cosine"
cosine_alpha = 0.01
cosine_decay_steps = 100000
optimizer_alpha = 0.9
optimizer_momentum = 0.3
range_mf = 1 #from -x/2 to x/2
scale_PV = 0.001
input_scaling = "None"
Temperature_output = True
auto_species_scaling = True
init_species_scaling_range = (1.0, 2.0)
epo_show_loss = 10000
header_data = 'infer'
bool_compute_Kreg = True
always_rescale_PV = False

nbr_seeds = 10

####################################
#Set name of file with species names
####################################

if(dataset_type.startswith("autoignition_augm")):
    file_species_names = f"Xu-state-space-names-{dataset_type}.csv"
else:
    file_species_names = "Xu-state-space-names.csv"

MSE_vals = np.zeros(nbr_seeds)
MSE_kr_vals = []

for my_seed in range(nbr_seeds):

    training_id = f"Tr{training_nbr}_s{my_seed}"

    ###############################
    #Initialization of the training
    ###############################

    epo = 1
    training_loss_list = np.zeros(max_epo)
    validation_loss_list = np.zeros(max_epo)
    best_validation_loss = np.inf
    epo_best_model = 0


    #############################
    #Set seed for reproducibility
    #############################

    generator = torch.Generator()
    generator.manual_seed(my_seed)


    ###############################
    #Set input and output idx lists
    ###############################

    my_species = Species(path_data, file_species_names)
    list_idx_species_removed = my_species.get_idx_from_list_species(total_species_removed)
    output_idx = my_species.get_idx_from_list_species(list_species_output)

    my_species_source = Species(path_data)
    list_idx_species_removed_source = my_species_source.get_idx_from_list_species(species_removed)
    output_idx_Kreg = my_species.get_idx_from_list_species(list_species_output_Kreg)

    all_output = copy.deepcopy(output_idx)
    if(Temperature_output):
        all_output.append("T")
    for i in range(1,1+PV_dim):
        all_output.append(f"PV{i}")
    output_dim = len(all_output) #species + Temperature and Source PV

    variable_headers = ["Training_id", "training_name","model_name", "curve_name", "metadata_name", "data_layer", "nbr datapoints", "max epo", "epo best", "optimizer",
                        "output scaling", "loss", "PV-rescaling init", "PV-rescaling batch", "lr", "seed", "batch size", "nbr species",
                        "nbr input species", "PV dim", "perc val data", "species removed", "idx species removed", "algo init w_enc", "algo init w_dec", "std init w_enc", "std init w_dec",
                        "date", "hour", "output dim", "list species output", "output species idx","output elements", "extra decoder layers neurons", "Training time", "dataset_type", "learning rate decay", 
                        "cosine alpha", "cosine decay steps", "optimizer alpha", "optimizer momentum", "range_mf", "scale PV", "input scaling", "Temperature at output",
                        "auto species scaling", "init species scaling range", "model_params", "input species scaling", "input species bias", "activation function", "activation function output",
                        "best valid value", "avg_std MSE Kreg", "avg_std MSE Kreg", "cost manifold", "dataset_Kreg", "list idx species removed source", "output idx Kreg", "list idx species removed Kreg",
                        "always rescale PV", "lambda reg"]


    ##############
    #Load the data
    ##############

    logging.info("Load the data")
    train_input, train_output, val_input, val_output, datapoints, input_species_scaling, input_species_bias = get_data(path_data + "Xu-state-space-" + dataset_type + ".csv",
                                                    path_data + "Xu-state-space_source-" + dataset_type + ".csv",
                                                    path_data + "Xu-T-" + dataset_type + ".csv",
                                                    path_data + "Xu-mf-" + dataset_type + ".csv",
                                                    generator, perc_val,
                                                    output_idx,
                                                    list_idx_species_removed,
                                                    input_scaling, range_mf, output_scaling, Temperature_output,
                                                    header_data, list_idx_species_removed_source)

    nbr_training_datapoints = train_input.size(0)
    batch_size = nbr_training_datapoints

    nbr_input_species = train_input.size(1)-1 #all except f (last column)

    train_input, train_output = train_input.to(device), train_output.to(device)
    val_input, val_output = val_input.to(device), val_output.to(device)


    #################################
    #Create directories and filenames
    #################################

    dirs = create_dirs(optimizer = optimizer_name,
                    epo = max_epo,
                    lr = lr,
                    current_time = current_time,
                    training_id = training_id)
    dirs.create(variable_headers)


    ###############
    #Load the model
    ###############

    logging.info("Load the model")
    model_params = {"nbr_species": nbr_input_species,
                    "PV_dim": PV_dim,
                    "output_dim": output_dim,
                    "decoder_layers": decoder_layers,
                    "auto_scaling": auto_species_scaling,
                    "activation_function": activation_function,
                    "activation_function_output": activation_function_output}
    model = PV_autoencoder(**model_params)
    model.to(device)

    #initialize the weights of the model
    model.initialize_model_weights(generator, init_enc, init_dec, init_species_scaling_range)

    #scale the weights to have the PV having a range of 1
    if(PV_rescaling_init):    
        model.rescale_encoder_data(train_input, scale_PV)
        logging.info("PV rescaled")

    ##############################
    #Intialize the optimizer tools
    ##############################

    optimizer = get_optimizer(model.parameters(), optimizer_name, lr, optimizer_alpha, optimizer_momentum)  #torch.optim.Adam(model.parameters(), lr=lr)
    loss_criterion = get_loss_criterion(loss_name, lambda_reg=lambda_reg)
    scheduler = torch.optim.lr_scheduler.LambdaLR( #cosine decay learning rate scheduler
    optimizer, lr_lambda=lambda epoch: cosine_decay(cosine_alpha, epoch, cosine_decay_steps)
    )


    ###############
    #Start training
    ###############

    logging.info("Start of the training")
    start_time = time.time()
    while(epo<=max_epo):
        #add criterion stop model
        
        training_loss = 0
        validation_loss = 0

        if(PV_rescaling_batch and epo>1):
            model.rescale_encoder_data(train_input, scale_PV, always_rescale = always_rescale_PV)

        #Perform the minibatching
        torch.manual_seed(epo) #seed value is the epoch number
        indices = torch.randperm(nbr_training_datapoints) #shuffle the indices
        split_indices = torch.split(indices, batch_size) #split in subtensors for the batches
        
        #Start training
        for batch_idx in split_indices:

            output_model = model(train_input[batch_idx,:])
            
            #prepare the output batch, create PV source and scale it between -1 and 1
            batch_output_mod = model.get_source_PV(train_output[batch_idx,:], input_species_scaling)
            
            #rescale source PV between -1 and 1
            batch_output_rescaled = rescale_PVsource(batch_output_mod, PV_dim) #rescale the last feature
        
            #get MSE loss
            loss = loss_criterion(output_model, batch_output_rescaled, model, train_input[batch_idx,:])
            #loss = loss_criterion(output_model, batch_output_rescaled)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += len(batch_idx)*loss.detach().cpu().numpy()

        #############
        #End training
        #############

        #################
        #Begin validation
        #################

        output_model = model(val_input)

        #prepare the output batch, create PV source and scale it between -1 and 1
        batch_output_mod = model.get_source_PV(val_output, input_species_scaling)

        #rescale source PV between -1 and 1
        batch_output_rescaled = rescale_PVsource(batch_output_mod, PV_dim)

        #get MSE loss
        loss = loss_criterion(output_model, batch_output_rescaled, model, val_input)
        #loss = loss_criterion(output_model, batch_output_rescaled)

        validation_loss += loss.detach().cpu().numpy()
        ###############
        #End validation
        ###############
        
        #########################
        #Post-processing of epoch
        #########################

        #save training and validation loss
        training_loss_list[epo-1] = training_loss/nbr_training_datapoints #weighted average of all the training losses
        validation_loss_list[epo-1] = validation_loss
        
        #checkpoint save model, in case it is a better model
        if(validation_loss < best_validation_loss):
            dirs.save_model(model.state_dict())
            best_validation_loss = validation_loss
            epo_best_model = epo
        
        epo += 1
        scheduler.step() #next learning rate in the scheduler

        if(epo%epo_show_loss==0):
            logging.info(f"Current epoch: {epo} - Validation loss (1e-04): {np.round(validation_loss_list[epo-2]*10000,1)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Training finished - {np.round(elapsed_time,2)} seconds")
    #############
    #End training
    #############

    MSE_vals[my_seed] = best_validation_loss

    ###########################
    #Assess model's performance
    ###########################

    best_model = dirs.load_model(model_params)
    avg_std_MSE_Kreg = [-1, -1]

    if(bool_compute_Kreg):

        logging.info("Start computing the Kreg performance")

        #determine which idx to remove for the source terms
        #needed when there the dataset for training is different than the dataset for Kreg
        if(dataset_Kreg == "autoignition_augm"):
            list_idx_species_removed_source_Kreg = list_idx_species_removed_source
            list_idx_species_removed_Kreg = list_idx_species_removed
        else:
            list_idx_species_removed_source_Kreg = list_idx_species_removed_source
            list_idx_species_removed_Kreg = list_idx_species_removed_source
        
        avg_std_MSE_Kreg = compute_Kreg(output_idx_Kreg, list_idx_species_removed_Kreg, input_scaling, input_species_scaling,
                                    input_species_bias, range_mf, best_model, path_data, dataset_type=dataset_Kreg,
                                    idx_species_removed_source=list_idx_species_removed_source_Kreg)
        
    MSE_kr_vals.append(avg_std_MSE_Kreg)

    ################
    #Post-processing
    ################

    variable_data = {"Training_id": training_id, "training_name":dirs.training_name,"model_name": dirs.dirout, "curve_name": dirs.dircurves, "metadata_name": dirs.dirMetadata,"data_layer": layer, "nbr datapoints": datapoints, "max epo": max_epo, "epo best": epo_best_model, "optimizer": optimizer_name,
                        "output scaling": output_scaling, "loss": loss_name, "PV-rescaling init": PV_rescaling_init, "PV-rescaling batch": PV_rescaling_batch ,"lr": lr, "seed": my_seed, "batch size": batch_size, "nbr species": nbr_species,
                        "nbr input species": nbr_input_species, "PV dim": PV_dim, "perc val data": perc_val, "species removed": total_species_removed, "idx species removed": list_idx_species_removed,  "algo init w_enc": init_algo_enc, "algo init w_dec": init_algo_dec, "std init w_enc": init_enc, "std init w_dec": init_dec,
                        "date": dirs.formatted_date, "hour": dirs.formatted_time, "output dim": output_dim, "list species output": list_species_output, "output species idx": output_idx, "output elements": all_output, "extra decoder layers neurons": decoder_layers, "Training time": elapsed_time, "dataset_type": dataset_type,
                        "learning rate decay": learning_rate_decay, "cosine alpha": cosine_alpha, "cosine decay steps": cosine_decay_steps, "optimizer alpha": optimizer_alpha, "optimizer momentum": optimizer_momentum, "range_mf": range_mf,
                        "scale PV": scale_PV, "input scaling": input_scaling, "Temperature at output": Temperature_output, "auto species scaling": auto_species_scaling, "init species scaling range": init_species_scaling_range, "model_params": model_params, "input species scaling": input_species_scaling, "input species bias": input_species_bias,
                        "activation function": activation_function, "activation function output": activation_function_output, "best valid value": best_validation_loss, "avg_std MSE Kreg":avg_std_MSE_Kreg, "dataset_Kreg": dataset_Kreg, "list idx species removed source": list_idx_species_removed_source, "output idx Kreg": output_idx_Kreg,
                        "list idx species removed Kreg": list_idx_species_removed_Kreg, "always rescale PV": always_rescale_PV, "lambda reg": lambda_reg}

    dirs.save_train_info_model(variable_data)
    dirs.save_train_val_curves(training_loss_list, validation_loss_list)
    dirs.save_metadata(variable_data)

    logging.info(f"Training seed {my_seed} finished with MSE {np.round(best_validation_loss*10000,2)}")

for seed in range(nbr_seeds):
    print(f"Seed {seed}: {np.round(MSE_vals[seed]*10000,2)} - {round(MSE_kr_vals[seed][0]*10000,3)} (\u00B1 {round(MSE_kr_vals[seed][1]*10000, 3)}) - ")

best_seed = np.argmin(MSE_vals)
print(f"Best seed: {best_seed} - MSE: {np.round(MSE_vals[best_seed]*10000,2)} - {round(MSE_kr_vals[best_seed][0]*10000,3)} (\u00B1 {round(MSE_kr_vals[best_seed][1]*10000, 3)}) - ")

logging.info("Script finished")