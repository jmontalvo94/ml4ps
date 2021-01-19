#!/usr/bin/env python

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from glob import glob
from datetime import datetime

# Local
from cli import cli
from data import create_dataset, init_dataset
from networks import NN, PINN
from plots import plot_sol
from train import train_NN, train_PINN
from utils import set_seed_everywhere

# %% Initialization

args, general, params, nn_params = cli()
SEED = general['seed']
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
NAME = args.name + '_' + timestamp if args.name is not None else timestamp
PATH_MODELS = args.path_models
PATH_DATA = args.path_data
MODEL_TYPE = 'PINN'

# Operations on the GPU if available
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
    
device = torch.device(DEVICE)

set_seed_everywhere(SEED)

# %% Data

params['t_span'] = (params['t_min'], params['t_max'])
params['p_span'] = (params['p_min'], params['p_max'])
data = create_dataset(params)
train, trainc, test = init_dataset(data, params, transformation=None)

# %% Train

model_nn = NN(nn_params)

results_NN = train_NN(model_nn, train, test, nn_params, params, args, noise=0.01)

# model_pinn = PINN(nn_params, params)

# results_PINN = train_PINN(model_pinn, trainc, test, nn_params, params, args)
# %% Evaluation

if MODEL_TYPE == 'NN':
    model = NN(nn_params)
    models = glob(f'{PATH_MODELS}NN_*')
elif MODEL_TYPE == 'PINN':
    model = PINN(nn_params, params)
    models = glob(f'{PATH_MODELS}PINN_*')
    
#%% Visualization

idx = 0
for m in models:
    n = m.split('/')[1].split('.')[0]
    model.load_state_dict(torch.load(m, map_location=device))

    print(f'{m}')
    plt.plot(np.linspace(params['t_min'], params['t_max'], params['n_data']), data[2][idx].reshape(-1,1), label='Real')
    plt.plot(np.linspace(params['t_min'], params['t_max'], params['n_data']), model.forward_nn(torch.tensor(data[0][idx][:,:2], dtype=torch.float32)).detach().numpy(), 'r--', label='Prediction')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [$\delta$]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'images/{n}.png')
    plt.show()
# %%
