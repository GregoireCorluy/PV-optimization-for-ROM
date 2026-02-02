from utils import *
from loader import *
from ANN_regression import *
import re

logging.disable(logging.CRITICAL)

filename_model = "Xu-AE-opt_adam-epo_100000-lr_0.025-date_24Mar2025-hour_10h50_Tr0D-2s_s2"

#Xu-AE-opt_adam-epo_100000-lr_0.025-date_14Mar2025-hour_14h23_Tr0D-2q_s0
#"Xu-AE-opt_RMSprop-epo_100000-lr_0.025-date_28Jan2025-hour_17h43_Tr35a_s3"

path_data = "data-files/"
filename_metadata = filename_model + "_metadata.pkl"
filename_species_names = "Xu-state-space-names.csv"
path_metadata = "metadata/"

dataset_type = "autoignition"

if(dataset_type == "low"):
    header_data = None
else:
    header_data = "infer"

my_seed = 7

generator = torch.Generator()
generator.manual_seed(my_seed)

max_epo = 500
std_weights = 0.05
learning_rate = 0.01 #0.025
cosine_alpha = 0.01
cosine_decay_steps = 500
epo_show_loss = 10000
perc_val = 0.1
nbr_input = 2
nbr_output = 11
neuron_layers = [11, 21, 21]
loss = "MSE"
scaleManifold = True

loader = loadData(filename_species_names, path_metadata, filename_metadata)
output_idx = loader.metadata["output idx Kreg"] if loader.metadata["dataset_type"].startswith("autoignition_augm") else loader.metadata["output species idx"]
idx_species_removed = loader.metadata["list idx species removed source"] if loader.metadata["dataset_type"].startswith("autoignition_augm") else loader.metadata["idx species removed"]
input_scaling = loader.metadata["input scaling"]
input_species_scaling = loader.metadata["input species scaling"]
input_species_bias = loader.metadata["input species bias"]
range_mf  = loader.metadata["range_mf"]
depvar_names_species = loader.metadata["list species output"]
depvar_names_idx = loader.metadata["output elements"]
PV_dim = loader.metadata["PV dim"]
depvar_names = depvar_names_species + depvar_names_idx[-(PV_dim+1):]
model_PV = loader.loadModel()
id = loader.metadata["Training_id"]
nbr_input_species = loader.metadata["nbr input species"]

species = Species(path_data)
list_all_species = species.get_list_species()
list_input_species = [item for idx, item in enumerate(list_all_species) if idx not in idx_species_removed]


order_species_removed = []

path_state = path_data + f"Xu-state-space-{dataset_type}.csv"

list_MSE_val = []
list_MSE_PVsource_val = []


#############
### Tools ###
#############

def cosine_decay(alpha, epo, tot_epo):
    """
    Cosine decay learning rate. Start at the initial learning rate and ends at initial learning rate times alpha.
    After tot_epo, the learning rate is constant and equal to initial learning rate times alpha.
    Alpha is the multiplier for the final learning rate.
    """

    myEpo = np.min([epo,tot_epo])

    return 0.5*(1-alpha)*(1+np.cos(np.pi*myEpo/tot_epo))+alpha

class LogMSELoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(LogMSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        log_y_pred = torch.log(y_pred + self.epsilon + 1) #+1 to avoid negative values
        log_y_true = torch.log(y_true + self.epsilon + 1)
        loss = torch.mean((log_y_pred - log_y_true) ** 2)
        return loss
    
class RelativeErrorLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(RelativeErrorLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_true - y_pred) / (torch.abs(y_true) + self.epsilon))


for i in range(nbr_input_species): #nbr_input_species):

    #Update the model with the threshold

    final_weights = copy.deepcopy(model_PV.encoder_species.weight.detach().numpy()[0])

    print(final_weights)

    if(model_PV.auto_scaling):
        final_weights *= model_PV.scaling_weights.detach().numpy()

    #get index spcies removed
    my_species = Species(path_data)
    
    list_species = my_species.get_list_species()
    for index in sorted(idx_species_removed, reverse=True):
        list_species.pop(index)

    #get mass fractions of species
    state_space = read_csv(path_state, header = header_data)
    state_space = state_space.drop(state_space.columns[idx_species_removed], axis=1)
    state_space_matrix = state_space.to_numpy()

    wi_Yi = state_space_matrix*np.abs(final_weights) #|w_i|*Y_i
    max_wi_Yi = np.max(wi_Yi, axis = 0) #take max of every species

    #selection of values greater than zero
    #CHECK that none of the weights in the PV is already zero!
    mask = max_wi_Yi > 0

    #Find the index with the minimum max wi Yi which is not zero
    index_species_min_max_wi_Yi = np.argmin(max_wi_Yi[mask])

    #Get the index back in the original array
    index_species_min_max_wi_Yi = np.where(mask)[0][index_species_min_max_wi_Yi]
    
    if(i>0): #do not simplify for the first iteration to have the first value with all input species
        with torch.no_grad():
            model_PV.encoder_species.weight[0,index_species_min_max_wi_Yi] = 0
            order_species_removed.append(list_input_species[index_species_min_max_wi_Yi])


    ##########################
    #Compute the manifold cost
    ##########################
    
    #get the input (PV and f) and the output (interested Yi, T and source terms) data
    input, output = get_dataset(path_data + f"Xu-state-space-{dataset_type}.csv", path_data + f"Xu-state-space_source-{dataset_type}.csv",
                                path_data + f"Xu-T-{dataset_type}.csv", path_data + f"Xu-mf-{dataset_type}.csv",
                                output_idx, idx_species_removed, input_scaling, input_species_scaling, input_species_bias, range_mf)
    

    PV = model_PV.get_PV(input)
    f_PV = torch.cat((input[:, -1].reshape(-1, 1), PV), dim = 1) #reshape to be (5200,1) instead of (52000)
    output = model_PV.get_source_PV(output, input_species_scaling)

    
    if(scaleManifold):
        mf = f_PV[:, 0]
        PV = f_PV[:, 1]

        PV_scaled = torch.zeros_like(PV)

        unique_mf = torch.unique(mf)

        for mf_val in unique_mf:
            mask = mf == mf_val
            PV_group = PV[mask]

            PV_min = PV_group.min()
            PV_max = PV_group.max()

            if PV_max > PV_min:
                PV_scaled[mask] = 2 * (PV_group - PV_min) / (PV_max - PV_min) - 1
            else:
                PV_scaled[mask] = 0.0

    f_PV = torch.stack((mf, PV_scaled), dim=1)

    ###################
    #Compute the NN MSE
    ###################


    min_output = output.min(dim=0, keepdim=True)[0]  # Minimum values for each column
    max_output = output.max(dim=0, keepdim=True)[0]  # Maximum values for each column

    output_scaled = (output - min_output) / (max_output - min_output)

    #create training and validation datasets
    np.random.seed(my_seed)
    nbr_observations = f_PV.shape[0]
    indices = np.arange(nbr_observations)
    nbr_train = int(nbr_observations*(1-perc_val))

    sampled_indices = np.random.choice(indices, size=nbr_train, replace=False)
    validation_indices = np.setdiff1d(indices, sampled_indices)

    train_input = f_PV[sampled_indices, :].detach()
    train_output = output[sampled_indices, :].detach()

    val_input = f_PV[validation_indices, :].detach()
    val_output = output[validation_indices, :].detach()

    #scale the input features between 0 and 1
    mins_input = train_input[:, :].min(dim=0, keepdim=True)[0]
    maxs_input = train_input[:, :].max(dim=0, keepdim=True)[0]
    train_input_scaling = maxs_input - mins_input

    #rescale the selected training inputs, rescale now between -0.5 and 0.5
    train_input[:, :] = (train_input[:, :] - mins_input) / train_input_scaling - 0.5

    #rescale the selected validation inputs
    val_input[:, :] = (val_input[:, :] - mins_input) / train_input_scaling - 0.5


    #rescale the output between -1 and 1
    mins_output = train_output[:, :].min(dim=0, keepdim=True)[0]
    maxs_output = train_output[:, :].max(dim=0, keepdim=True)[0]

    #rescale the selected training outputs
    train_output[:, :] = 2 * (train_output[:, :] - mins_output) / (maxs_output - mins_output) - 1

    #rescale the selected validation outputs
    val_output[:, :] = 2 * (val_output[:, :] - mins_output) / (maxs_output - mins_output) - 1

    batch_size = 1000 #train_input.shape[0]
    nbr_training_datapoints = train_input.shape[0]

    ###############################
    #Initialization of the training
    ###############################

    epo = 1
    training_loss_list = np.zeros(max_epo)
    validation_loss_list = np.zeros(max_epo)
    best_validation_loss = np.inf
    epo_best_model = 0

    #####################
    #Initialize the model
    #####################

    model_NN = ANN_regression(nbr_input, nbr_output, neuron_layers)
    model_NN.initialize_model_weights(generator, std_weights)

    model_params = {"nbr_input": nbr_input,
                    "nbr_output": nbr_output,
                    "neuron_layers": neuron_layers}

    optimizer = torch.optim.Adam(model_NN.parameters(), lr=learning_rate)
    if(loss == "MSE"):
        loss_criterion = nn.MSELoss()
    elif(loss == "logMSE"):
        loss_criterion = LogMSELoss()
    elif(loss == "relError"):
        loss_criterion = RelativeErrorLoss()

    scheduler = torch.optim.lr_scheduler.LambdaLR( #cosine decay learning rate scheduler
                    optimizer, lr_lambda=lambda epoch: cosine_decay(cosine_alpha, epoch, cosine_decay_steps)
                    )

    smallest_training_loss = np.inf
    smallest_validation_loss = np.inf
    epo_best_model = -1


    ################
    #Train the model
    ################

    while(epo<=max_epo):

        training_loss = 0
        validation_loss = 0

        #Perform the minibatching
        torch.manual_seed(epo) #seed value is the epoch number
        indices = torch.randperm(nbr_training_datapoints) #shuffle the indices
        split_indices = torch.split(indices, batch_size) #split in subtensors for the batches
        
        #Start training
        for batch_idx in split_indices:

            output_model_train = model_NN(train_input[batch_idx,:])
        
            #get MSE loss
            loss_train = loss_criterion(output_model_train, train_output[batch_idx,:])
            
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            training_loss += len(batch_idx)*loss_train.detach().cpu().numpy()

        #############
        #End training
        #############

        #################
        #Begin validation
        #################

        output_model_val = model_NN(val_input)

        #get MSE loss
        loss_val = loss_criterion(output_model_val, val_output)

        validation_loss += loss_val.detach().cpu().numpy()
        ###############
        #End validation
        ###############
        
        #########################
        #Post-processing of epoch
        #########################

        #save training and validation loss
        training_loss_list[epo-1] = training_loss/nbr_training_datapoints #weighted average of all the training losses
        validation_loss_list[epo-1] = validation_loss
        
        epo += 1
        scheduler.step() #next learning rate in the scheduler

        if(epo%epo_show_loss==0):
            print(f"Current epoch: {epo} - Training loss (1e-04): {np.round(training_loss_list[epo-2]*10000,1)} - Validation loss (1e-04): {np.round(validation_loss_list[epo-2]*10000,1)}")
        
        if(validation_loss < smallest_validation_loss):
            smallest_training_loss = training_loss/nbr_training_datapoints
            smallest_validation_loss = validation_loss
            epo_best_model = epo
            MSE_PVsource = loss_criterion(output_model_val[:, -1], val_output[:, -1]).detach().numpy()

    print(smallest_validation_loss)
    print(MSE_PVsource)

    list_MSE_val.append(smallest_validation_loss)
    list_MSE_PVsource_val.append(MSE_PVsource)


clear_id_model = re.sub(r'_s\d', '', id)
np.save(f'data-files/MSE_NN/MSE_val_NN_{clear_id_model}_{dataset_type}_scaleManifold{scaleManifold}_PV_simplification.npy', list_MSE_val)
np.save(f'data-files/MSE_NN/MSE_PVsource_val_NN_{clear_id_model}_{dataset_type}_s{my_seed}_lr{learning_rate}_scaleManifold{scaleManifold}_PV_simplification.npy', list_MSE_PVsource_val)