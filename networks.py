#!/usr/bin/env python

# %% Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# %% Functions

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

def get_loss(name):
    """Matches the loss function name with the actual Pytorch function.

    Args:
        name(str): Name of the loss function.

    Returns:
        loss(fun): Pytorch losss function.
    """

    loss_funs = {'MSE': nn.MSELoss(),
                 'MAE': nn.L1Loss(),
                }

    if name in loss_funs:
        return loss_funs[name]
    else: 
        raise ValueError(f'Loss function "{name}" not implemented, available loss functions are {list(loss_funs.keys())}.')
    
def gradients(outputs, inputs):
    """ Returns gradients of outputs w.r.t. inputs, retains graph for higher order derivatives """
    return torch.autograd.grad(outputs.sum(), inputs, retain_graph=True, create_graph=True)[0]

# %% Classes

class PyroNN(nn.Module):
    """ Vanilla Feed-Forward Neural Network (NN) """
    
    def __init__(self, nn_params):
        super().__init__()
        
        # Unpack parameters
        layers = nn_params['layers']
        activation = nn_params['activation']
        lr = nn_params['learning_rate']
        opt = nn_params['optimizer']
        
        loss_fun = nn_params['loss_function']
        
        # Set activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
            
        # Create architecture
        depth = len(layers) - 1
        self.nn = nn.Sequential()
        for n in range(depth - 1):
            self.nn.add_module(f"layer_{n}", nn.Linear(layers[n], layers[n + 1]))
            self.nn.add_module(f"act_{n}", self.activation)
        self.nn.add_module(f"layer_{n + 1}", nn.Linear(layers[n + 1], layers[n + 2]))
                   
        # Loss function
        if loss_fun == 'MSE':
            self.loss_fun = nn.MSELoss()
        elif loss_fun == 'MAE':
            self.loss_fun = nn.L1Loss()

    def forward(self, x):
        return self.nn(x)
    
    def loss_(self, y_hat, y):
        return self.loss_fun(y_hat, y)

class NN(nn.Module):
    """ Vanilla Feed-Forward Neural Network (NN) """
    
    def __init__(self, nn_params):
        super().__init__()
        
        # Unpack parameters
        layers = nn_params['layers']
        activation = nn_params['activation']
        lr = nn_params['learning_rate']
        opt = nn_params['optimizer']
        
        loss_fun = nn_params['loss_function']
        
        # Set activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
            
        # Create architecture
        depth = len(layers) - 1
        self.nn = nn.Sequential()
        for n in range(depth - 1):
            self.nn.add_module(f"layer_{n}", nn.Linear(layers[n], layers[n + 1]))
            self.nn.add_module(f"act_{n}", self.activation)
        self.nn.add_module(f"layer_{n + 1}", nn.Linear(layers[n + 1], layers[n + 2]))
                   
        # Loss function
        if loss_fun == 'MSE':
            self.loss_fun = nn.MSELoss()
        elif loss_fun == 'MAE':
            self.loss_fun = nn.L1Loss()

    def forward_nn(self, x):
        return self.nn(x)
    
    def loss_nn(self, y_hat, y):
        return self.loss_fun(y_hat, y)
    
class PINN(NN):
    """ Physics-Informed Neural Network (NN) """
    
    def __init__(self, nn_params, data_params):
        super().__init__(nn_params)
        
        # Unpack equation parameters
        self.m = torch.tensor([data_params['inertia']])
        self.d = torch.tensor([data_params['damping']])
        self.B = torch.tensor([data_params['susceptance']])

    def forward_pinn(self, p, t):
        
        # Forward pass
        u_hat = self.forward_nn(torch.cat([p, t], dim=1))
        # First derivative
        dudt = gradients(u_hat, t)
        print(dudt)
        # Second derivative
        dudtt = gradients(dudt, t)
        print(dudtt)
        # Physics term
        f = self.m * dudtt + self.d * dudt + self.B * torch.sin(u_hat) - p
        
        return u_hat, f
    
    def loss_pinn(self, u_hat, u, p_type, f, w1=1, w2=1):

        # Angle loss
        u_filtered = u_hat * p_type
        MSE_u = self.loss_nn(u_filtered, u)
        # Physics loss
        MSE_f = torch.mean(f**2)
        # Total loss
        MSE = w1*MSE_u + w2*MSE_f
        
        return MSE_u, MSE_f, MSE

    
# %% Testing

if __name__ == '__main__':
    
    from cli import cli

    args, general, params, nn_params = cli()
    
    # @profile
    # def test():
    #     x1 = torch.tensor([[1.]])
    #     x2 = torch.tensor([[2.]], requires_grad=True)
    #     x = torch.cat([x1, x2], dim=1)
    #     w = 5*x
    #     y = w**3
    #     g1 = gradients(y, x2)
    #     g2 = gradients(g1, x2)
        
    # test()
    
    print('NN\n')
    x1 = torch.tensor([[1.],
                       [0.9],
                       [0.8]], requires_grad=True)
    x2 = torch.tensor([[0.5],
                       [0.1],
                       [0.4]])
    y = torch.tensor([[0.8]])
    x = torch.cat([x1, x2], dim=1)
    
    net = NN(nn_params)
    y_hat = net.forward_nn(x)
    loss = net.loss_nn(y_hat, y)
    loss.backward()
    print(x1.grad)
    print(x2.grad)
    
    print(f"\n-----------------------------\n")
    
    print('PINN\n')
    x = torch.tensor([[1., 0.5],
                      [0.9, 0.1],
                      [0.8, 0.4]])
    x1 = x.select(1,0)
    x2 = x.select(1,1)
    x2.requires_grad_(True)
    y = torch.tensor([[0.8]])
    
    x = torch.tensor([[1., 0.5],
                      [0.9, 0.1],
                      [0.8, 0.4]])
    x1 = x.select(1,0)
    x2 = x.select(1,1)
    x2.requires_grad_(True)
    y = torch.tensor([[0.8]])
    net = PINN(nn_params, params)
    optimizer = optim.Adam(net.parameters())
    optimizer.zero_grad()
    u_hat, f = net.forward_pinn(x1.view(-1,1), x2.view(-1,1))
    loss_u, loss_f, loss = net.loss_pinn(u_hat, y, torch.zeros_like(u_hat), f)
    loss.backward()
    print(x1.grad)
    print(x2.grad)
    # optimizer.step()
    
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    #     for _ in range(100):
    #         net.optimizer.zero_grad()
    #         u_hat, f = net.forward_pinn(x1.view(-1,1), x2.view(-1,1))
    #         loss_u, loss_f, loss = net.loss_pinn(u_hat, y, torch.zeros_like(u_hat), f)
    #         loss.backward()
    #         net.optimizer.step()
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

# %%
