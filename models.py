"""
PV optimization on hydrogen datasets using an encoder-decoder architecture (Kamila Zdybal).
Version: Definition of the model and the tools linked to it.
Author: Grégoire Corlùy (gregoire.stephane.corluy@ulb.be)
Date: September 2024
Python version: 3.10.10
"""


import torch
import torch.nn as nn
import copy

class PV_autoencoder(nn.Module):
    """
        Encoder-decoder architecture to optimize the PV defintiion given a combustion data.

        Input: All the mass fractions of the species and the mixture fraction.

        Output: The selected species, the temperature and the PV source term.

        Remark: Input data should be in torch.float64 format.
    """

    def __init__(   self, nbr_species, PV_dim, output_dim, decoder_layers,
                    auto_scaling = False, activation_function = "tanh",
                    activation_function_output = "tanh", fixed_PV = False, **kwargs):

        super(PV_autoencoder, self).__init__()

        self.fixed_PV = fixed_PV
        self.PV_dim = PV_dim
        
        self.nbr_species = nbr_species
        n_decoder = copy.deepcopy(decoder_layers)
        n_decoder.append(0) #add the zero for the output layer
        
        self.auto_scaling = auto_scaling
        #Automatic scaling of the input species
        if(self.auto_scaling):
            self.scaling_weights = nn.Parameter(torch.ones(self.nbr_species, dtype=torch.float64))

        #PV definition with all the species
        self.encoder_species = nn.Linear(self.nbr_species, PV_dim, bias = False, dtype=torch.float64)

        #Store the different decoder layers
        self.decoder = nn.ModuleList()

        self.decoder.append(nn.Linear(PV_dim+1, output_dim + n_decoder[0], dtype=torch.float64))

        for layer_idx in range(len(n_decoder)-1):
            self.decoder.append(nn.Linear(output_dim + n_decoder[layer_idx], output_dim + n_decoder[layer_idx+1], dtype=torch.float64))
        
        if(activation_function.lower() == "tanh"):
            self.activation_function = torch.nn.Tanh()
        elif(activation_function.lower() == "relu"):
            self.activation_function = torch.nn.ReLU()
        elif activation_function.lower() == "sigmoid":
            self.activation_function = torch.nn.Sigmoid()

        if(activation_function_output.lower() == "tanh"):
            self.activation_function_output = torch.nn.Tanh()
        elif activation_function_output.lower() == "sigmoid":
            self.activation_function_output = torch.nn.Sigmoid()


    def forward(self, input_data):
        #split the input into species and mixture fraction
        
        mf = input_data[:, self.nbr_species].unsqueeze(1)
        PV = self.get_PV(input_data)

        #Concatenate the PV and the mixture fraction horizontally to get the latent space
        latent = torch.cat((PV, mf), dim=1)
        
        #decoder layer
        x = self.activation_function(self.decoder[0](latent))
        
        #make x pass through all the layers of the decoder
        for layer in self.decoder[1:-1]:
            x = self.activation_function(layer(x))

        x = self.activation_function_output(self.decoder[-1](x))

        return x

    def get_PV(self, input_data):
        """Get the PV-value using the input data and the encoder.
        
        Remark: Can also be used to get the source PV, by giving the source terms as input data.

        Args:
            input_data (torch.Tensor): Tensor of species' mass fractions.

        Returns:
            torch.Tensor: Tensor of the progress variable obtained with the encoder-decoder.
        """

        species = input_data[:, :self.nbr_species] 

        #scale the species
        if(self.auto_scaling):
            species_scaled = species * self.scaling_weights  #+ self.scaling_biases

        #Combine the species to get the PV
        if(not self.fixed_PV): #in case there is no fixed PV, use the species or scaled species for all the rows
            if(not self.auto_scaling):
                PV = self.encoder_species(species)
            elif(self.auto_scaling):
                PV = self.encoder_species(species_scaled)
        elif(self.fixed_PV): #in case the first PV is fixed
            if(not self.auto_scaling):
                PV = self.encoder_species(species)
            elif(self.auto_scaling): #do not apply the scaling to the row of the fixed PV which is already scaled
                PV_fixed = self.encoder_species(species)[:,0].unsqueeze(1) #keep the first column
                PV_others  = self.encoder_species(species_scaled)[:, 1:] #keep all the columns except the first one

                PV = torch.cat([PV_fixed, PV_others], dim = 1) #stack different columns together

        return PV

    def initialize_model_weights(self, generator, init_enc, init_dec, init_scaling = (1.0, 2.0), weights_first_PV = "None"):
        """Initialize the encoder and decoder weights randomly according to a normal distribution.

        Args:
            generator (torch.Generator): Generator for reproducibility of the encoder-decoder initialization.
            init_enc (float): Standard deviation of the initial weight distribution for the encoder.
            init_dec (float): Standard deviation of the initial weight distribution for the decoder.
            init_scaling (tuple, optional): Range of weights for the scaling layer. Defaults to (1.0, 2.0).
            weights_first_PV (torch.Tensor, optional): Weights of the first PV if first PV is fixed. Defaults to "None".
        """

        #encoder initialization
        #all weights equal to one
        nn.init.normal_(self.encoder_species.weight, mean=0.0, std=init_enc, generator = generator)

        #set the encoder weights for the first PV in case the first PV is fixed
        if(self.fixed_PV):
            with torch.no_grad():
                self.encoder_species.weight[0] = weights_first_PV

        #decoder initialization
        #weights random, method has still to be investigated
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):  # Check if the layer is of type nn.Linear
                nn.init.normal_(layer.weight, mean=0.0, std=init_dec, generator = generator)  # Initialize weights with normal distribution
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # Initialize bias to zero
        
        if(self.auto_scaling):
            low, high = init_scaling
            nn.init.uniform_(self.scaling_weights, a=low, b=high, generator = generator)

        return None

    def rescale_encoder_data(self, input_data, scale_PV, always_rescale = False):
        """Rescale the encoder to get a desired PV range if the PV range is below the scale_PV value.
        Second version of rescale_encoder function where an input tensor is used instead of a dataloader.

        Args:
            input_data (torch.Tensor): Input tensor containing species for the PV and the mixture fraction.
            scale_PV (float): Scalar indicating the range to which the PV should be scaled
            always_rescale (bool, optional): Indicates if the PV should be rescaled independent of the current PV range. Defaults to False.
        """

        all_PV = self.get_PV(input_data)

        scaling_factor = torch.max(all_PV, dim = 0).values-torch.min(all_PV, dim = 0).values

        #scale for every PV separately
        #rescale now only when the range is too small, do nothing when range large enough
        for i in range(scaling_factor.numel()):
            if(scaling_factor[i]<scale_PV):
                with torch.no_grad():
                    if(self.fixed_PV and i>0 or not self.fixed_PV):
                        self.encoder_species.weight.data[i,:] /= (scaling_factor[i]/scale_PV)
            elif(always_rescale):
                with torch.no_grad():
                    if(self.fixed_PV and i>0 or not self.fixed_PV):
                        self.encoder_species.weight.data[i,:] /= (scaling_factor[i]/scale_PV)
        return None

    def get_source_PV(self, batch, input_species_scaling):
        """Remove the source terms from the batch and add the PV source term.

        Args:
            batch (torch.Tensor): Batch containing the QoIs and species' source terms.
            input_species_scaling (torch.Tensor): Array containing the scaling to be applied to every species.

        Returns:
            torch.Tensor: Output batch tensor containing the PV source terms next to the QoIs.
        """

        source_terms = batch[:, -self.nbr_species:]/input_species_scaling
        PV_source = self.get_PV(source_terms)

        batch_without_source_terms = batch[:,:-self.nbr_species]
        batch_with_PV_source = torch.cat((batch_without_source_terms, PV_source), dim=1)

        return batch_with_PV_source
    
    def get_mf_PV_PVsource(self, input, output):
        """Get the mixture fraction, the optimized progress variable (PV) and PV source term.

        Args:
            input (torch.Tensor): Input tensor containing input species for PV and mixture fraction.
            output (torch.Tensor): Output tensor containing the QoIs and species' source terms.

        Returns:
            mf (numpy.ndarray): Array of mixture fractions.
            PV (numpy.ndarray): Array of progress variables.
            PVsource (numpy.ndarray): Array of PV source terms.
        """

        PV = self.get_PV(input).detach().numpy()
        mf = input[:,-1].numpy()
        PVsource = self.get_PV(output[:,-self.nbr_species:]).detach().numpy()

        return mf, PV, PVsource
    
    def get_mf_PV_species(self, input, speciesIdx):
        """Get the mixture fraction, PV and chosen species from the input tensor.

        Args:
            input (torch.Tensor): Input tensor containing the species' mass fractions and the mixture fraction.
            speciesIdx (int): Index of the species to be selected.

        Returns:
            mf (numpy.ndarray): Array of the mixture fraction.
            PV (numpy.ndarray): Array of the optimized PV.
            species (numpy.ndarray): Array of the selected species' mixture fractions.
        """

        PV = self.get_PV(input).detach().numpy()
        mf = input[:,-1].numpy()
        species = input[:, speciesIdx].numpy()

        return mf, PV, species
    
    def get_scaling_weights(self):
        """Return the weights of the encoder-decoder's scaling layer.

        Returns:
            torch.Tensor: Weights of the encoder-decoder's scaling layer.
        """

        return self.scaling_weights.detach()
    
    def get_encoder_weights(self):
        """Returns the encoder weights of the encoder-decoder.

        Returns:
            torch.Tensor: Weights of the encoder-decoder's encoder layer.
        """

        return self.encoder_species.weight.detach()

    def get_scaled_encoder_weights(self):
        """Get the scaled encoder weights of the encoder-decoder, corresponding to w_i * s_i.

        Returns:
            torch.Tensor: Scaled encoder weights.
        """
        
        if(self.auto_scaling):
            return self.scaling_weights.detach()*self.encoder_species.weight.detach()
        else:
            print("Warning: model has no scaling layer. Only encoder weights returned")
            return self.encoder_species.weight.detach()

    def get_total_encoder_weights(self, npy = False):
        """Get the total encoder weights corresponding to w_i if no scaling layer and s_i*w_i if a scaling layer is present.

        Args:
            npy (bool, optional): Boolean indicating if output should be in torch.Tensor or numpy format. Defaults to False.

        Returns:
            torch.Tensor: Total encoder weights corresponding to w_i for encoder-decoders without scaling layer and s_i*w_i for encoder-decoders without scaling layer.
        """

        if(self.auto_scaling):
            total_weights = self.scaling_weights.detach()*self.encoder_species.weight.detach()
        else:
            total_weights = self.encoder_species.weight.detach()
        
        if(npy):
            total_weights = total_weights.numpy()
            
        return total_weights
    
    def reset_first_PV(self, weights_first_PV):
        """
        As it is not possible to fix partially a tensor,
        it has to be done manually after every weight update by resetting the encoder weights to the old ones.
        """

        if(self.fixed_PV):
            with torch.no_grad():
                self.encoder_species.weight[0] = weights_first_PV

        return None