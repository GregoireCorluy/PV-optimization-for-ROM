"""
ANN regression model
Author: Grégoire Corlùy (gregoire.stephane.corluy@ulb.be)
Date: February 2025
Python version: 3.10.10
"""


import torch
import torch.nn as nn
import copy

class ANN_regression(nn.Module):
    """
        General ANN model for regression tasks
    """

    def __init__(   self, nbr_input, nbr_output, neuron_layers, activation_function = "tanh",
                    activation_function_output = "tanh"):
        
        """
        Args:
            nbr_input (int): Input dimension. Corresponds to the number of neurons for the input layer.
            nbr_output (int): Output dimension. Corresponds to the number of neurons for the output layer.
            neuron_layers (list of int): List of the number of additional neurons compared to the output dimension for every hidden layer.
            activation_function (str): String defining which activation function to use in the hidden layers.
            activation_function_output (str): String defining which activation function to use at the output layer.
        """

        super(ANN_regression, self).__init__()

        self.nbr_input = nbr_input
        self.nbr_output = nbr_output
        self.neuron_layers = copy.deepcopy(neuron_layers)
        self.neuron_layers.append(0) #add the zero for the output layer

        #Store the different layers
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.nbr_input, self.nbr_output + self.neuron_layers[0], dtype=torch.float64))

        for layer_idx in range(len(self.neuron_layers)-1):
            self.layers.append(nn.Linear(self.nbr_output + self.neuron_layers[layer_idx], self.nbr_output + self.neuron_layers[layer_idx+1], dtype=torch.float64))
        
        # define activation function of hidden layers
        if(activation_function.lower() == "tanh"):
            self.activation_function = torch.nn.Tanh()
        elif(activation_function.lower() == "relu"):
            self.activation_function = torch.nn.ReLU()
        elif activation_function.lower() == "sigmoid":
            self.activation_function = torch.nn.Sigmoid()

        # define activation function of output layer
        if(activation_function_output.lower() == "tanh"):
            self.activation_function_output = torch.nn.Tanh()
        elif activation_function_output.lower() == "sigmoid":
            self.activation_function_output = torch.nn.Sigmoid()

    def forward(self, x):

        # hidden layers
        for layer in self.layers[:-1]:
            x = self.activation_function(layer(x))
        
        # output layer
        x = self.activation_function_output(self.layers[-1](x))

        return x
    
    def initialize_model_weights(self, generator, init_layer):
        """Initialize randomly the weights in the different layers of the neural network.

        Args:
            generator (torch.Generator): Random number generator for reproducibility.
            init_layer (float): Standard deviation used to initialize randomly the weights accordinat to a normal distribution.

        """

        #decoder initialization
        #weights random, method has still to be investigated
        for layer in self.layers:
            if isinstance(layer, nn.Linear):  # Check if the layer is of type nn.Linear
                nn.init.normal_(layer.weight, mean=0.0, std=init_layer, generator = generator)  # Initialize weights with normal distribution
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # Initialize bias to zero

        return None