#!/usr/bin/env python

# %% Imports
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
# Local
from cli import cli
from data import create_dataset, init_dataset
from networks import NN, PINN
from plots import plot_sol
from utils import set_seed_everywhere

# %% Functions

def train_NN(model, train, test, nn_params, data_params):
    
    # Unpack parameters
    n_data = data_params['n_data']
    opt = nn_params['optimizer']
    batch_size = nn_params['batch_size']
    n_epochs = nn_params['n_epochs']
    lr = nn_params['learning_rate']
    wd = nn_params['weight_decay']
    
    # Initialize the optimizer
    if opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Unpack matrices and transform to tensor
    X_train, y_delta_train, y_omega_train, _ = train
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_delta_train, dtype=torch.float32)
    
    # Select from matrix and convert to chunks 
    p = X.select(1,0)
    t = X.select(1,1)
    p_batch = torch.split(p, batch_size, dim=0)
    t_batch = torch.split(t, batch_size, dim=0)
    y_batch = torch.split(y, batch_size, dim=0)
    
    # Output stats
    train_loss = []
    
    for epoch in tqdm(range(n_epochs), unit='epoch'):
        
        # Activate train mode
        model.train()
        
        batch_loss = []
                
        for i in range(len(p_batch)):

            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            
            # Forward pass
            y_hat = model.forward_nn(torch.cat([p_batch[i].view((-1,1)), t_batch[i].view((-1,1))], dim=1))

            # Compute the loss
            loss = model.loss_nn(y_hat, y_batch[i])

            # Backward pass through the network
            loss.backward()

            # Update the weights
            optimizer.step()
            
            # Save train loss
            batch_loss.append(loss.detach().cpu().item())
        
        # Pack epoch losses
        train_loss.append(np.mean(batch_loss))
    
    # Save models and data
    torch.save(model.state_dict(), PATH_MODELS + NAME + f'_{n_epochs}_{n_data}.pth')
    np.savez(PATH_DATA + NAME + f'_{n_epochs}_{n_data}', train_loss=train_loss)
    
    return train_loss
    
def train_PINN(model, train, test, nn_params, data_params, weight=False):
    
    # Unpack parameters
    n_data = data_params['n_data']
    n_coll = data_params['n_collocation']
    opt = nn_params['optimizer']
    batch_size = nn_params['batch_size']
    n_epochs = nn_params['n_epochs']
    lr = nn_params['learning_rate']
    wd = nn_params['weight_decay']
    
    # Initialize the optimizer
    if opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Unpack matrices and transform to tensor
    X_train, y_delta_train, y_omega_train, _ = train
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_delta_train, dtype=torch.float32)
    
    # Select from matrix and convert to chunks 
    p = X.select(1,0)
    t = X.select(1,1)
    dp = X.select(1,2)
    t.requires_grad_(True)
    p_batch = torch.split(p, batch_size, dim=0)
    t_batch = torch.split(t, batch_size, dim=0)
    dp_batch = torch.split(dp, batch_size, dim=0)
    y_batch = torch.split(y, batch_size, dim=0)
    
    if weight:
        w2 = n_data / (n_data + n_coll)
    else:
        w2 = 1
    
    # Output stats
    train_loss = []
    
    for epoch in tqdm(range(n_epochs), unit='epoch'):
        
        # Activate train mode
        model.train()
        
        mse_loss, mse_u_loss, mse_f_loss = [], [], []
                
        for i in range(len(p_batch)):

            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            
            # Forward pass
            u_hat, f = model.forward_pinn(p_batch[i].view((-1,1)), t_batch[i].view((-1,1)))
            
            # Compute the loss
            loss_u, loss_f, loss = model.loss_pinn(u_hat, y_batch[i], dp_batch[i], f, w2)

            # Backward pass through the network
            loss.backward(retain_graph=True)

            # Update the weights
            optimizer.step()
            
            # Save train loss
            mse_loss.append(loss.detach().cpu().item())
            mse_u_loss.append(loss_u.detach().cpu().item())
            mse_f_loss.append(loss_f.detach().cpu().item())
        
        # Pack epoch losses
        train_loss.append([np.mean(mse_loss), np.mean(mse_u_loss), np.mean(mse_f_loss)])
    
    # Save models and data
    torch.save(model.state_dict(), PATH_MODELS + NAME + f'_{n_epochs}_{n_data}_{n_coll}.pth')
    np.savez(PATH_DATA + NAME + f'_{n_epochs}_{batch_size}_{n_data}_{n_coll}', train_loss=train_loss)
    
    return train_loss

#%% Testing

if __name__ == '__main__':
    
    args, general, params, nn_params = cli()
    params['t_span'] = (params['t_min'], params['t_max'])
    params['p_span'] = (params['p_min'], params['p_max'])
    
    data = create_dataset(params)
    train, trainc, test = init_dataset(data, params)
    
    #model_nn = NN(nn_params)
    
    #results_nn = train_NN(model_nn, train, test, nn_params, data_params)
    
    model_pinn = PINN(nn_params, params)

    results_pinn = train_PINN(model_pinn, trainc, test, nn_params, params)

# %%
