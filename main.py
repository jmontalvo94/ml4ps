#!/usr/bin/env python

# %% Imports
import numpy as np
import torch
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
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
NAME = args.name + '_' + timestamp if args.name is not None else timestamp
PATH_MODELS = args.path_models
PATH_DATA = args.path_data
SEED = general['seed']

set_seed_everywhere(SEED)

# %% Data

params['t_span'] = (params['t_min'], params['t_max'])
params['p_span'] = (params['p_min'], params['p_max'])
data = create_dataset(params)
train, trainc, test = init_dataset(data, params)

# %% Train

model_nn = NN(nn_params)

results_NN = train_NN(model_NN, train, test, nn_params, data_params)

model_pinn = PINN(nn_params, data_params)

results_PINN = train_PINN(model_PINN, trainc, test, nn_params, data_params)