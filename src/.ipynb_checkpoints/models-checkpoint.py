#!/usr/bin/env python

# Imports
import numpy as np
import torch
import torch.nn as nn

def activation(name):
    """Matches the activation function name with the actual Pytorch function.
    
    Args:
        name(str): Name of the activation function.
    
    Returns:
        act(fun): Pytorch activation function.
    """
    
    activations = {'tanh': nn.Tanh(),
                   'relu': nn.ReLU(inplace=True),
                   'leakyrelu': nn.LeakyReLU(inplace=True),
                   'sigmoid': nn.Sigmoid()
                  }
    
    if name.lower() in activations:
        return activations[name.lower()]
    else: 
        raise ValueError(f'Activation function "{name}" not implemented, available activations are {list(activations.keys())}.')

class NN(nn.Module):
    """Feed-Forward Neural Network (NN).

    Attributes:
        layers(list): Architecture of the NN.
        act_name(str): Activation function to use after each layer.
        depth(int): Maximum depth of the NN.
        nn(nn.Sequential): Sequential container of the NN.
    """
    
    def __init__(self, nn_params):
        """Inits the NN architecture with given parameters."""
        super().__init__()
        
        # Unpack parameters
        self.layers = nn_params['layers']
        self.act_name = nn_params['act_name']
        
        # Define parameters
        self.depth = len(layers) - 1
        
        # Create architecture
        self.nn = nn.Sequential()
        
        # Add layers until output layer
        for n in range(self.depth - 1):
            self.nn.add_module(f"layer_{n}", nn.Linear(layers[n], layers[n + 1]))
            self.nn.add_module(f"act_{n}", activation(self.act_name))
        
        # Output layer (excluding activation)
        self.nn.add_module(f"layer_{n + 1}", nn.Linear(layers[n + 1], layers[n + 2]))

    def forward(self, x):
        """Feed-forward the inputs through the network."""
        return self.nn(x)