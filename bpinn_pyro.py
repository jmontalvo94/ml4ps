#!/usr/bin/env python

# %% Imports

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, HMC
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.optim import Adam, ClippedAdam
from pyro.infer import Predictive

from networks import PyroNN

#%% Functions
def gradients(outputs, inputs):
    return torch.autograd.grad(outputs.sum(), inputs, retain_graph=True, create_graph=True)[0]
    
def model_pinn_elevated(nn_params, X, u=None, F=None):

    m = 0.15 # angular inertia
    d = 0.15 # damping coefficient
    B = 0.2 # susceptance [pu]
    P = 0.1 # disturbance [pu]

    p = X[:,0].reshape((-1,1))
    t = X[:,1].reshape((-1,1))
    t.requires_grad_(True)

    # Initialize model
    torch_model = PyroNN(nn_params)
    
    # Convert the PyTorch neural net into a Pyro model with priors
    priors = {} # Priors for the neural model
    for name, par in torch_model.named_parameters():     # Loop over all neural network parameters
        priors[name] = dist.Normal(torch.zeros(*par.shape), torch.ones(*par.shape)).independent(1) # Each parameter has a N(0, 1) prior
    
    bayesian_model = pyro.random_module('bayesian_model', torch_model, priors) # Make this model and these priors a Pyro model
    sampled_model = bayesian_model()                                           # Initialize the model
    
    # The generative process
    with pyro.plate("observations"):
        
        # Forward pass
        nn_pred = sampled_model(torch.cat([p, t], dim=1))
        u = pyro.sample("obs1", dist.Normal(nn_pred, 1.), obs=u)
        
        # First derivative
        dudt = gradients(nn_pred, t)
        
        # Second derivative
        dudtt = gradients(dudt, t)
        
        # Physics term
        f = m * dudtt + d * dudt + B * torch.sin(u) - P
        
        F = pyro.sample("obs2", dist.Normal(f, 1.), obs=F)
        
    return F

#%% Testing

if __name__ == '__main__':
    
    from cli import cli
    from data import create_dataset, init_dataset
    
    args, general, params, nn_params = cli()
    params['t_span'] = (params['t_min'], params['t_max'])
    params['p_span'] = (params['p_min'], params['p_max'])
    n_data = params["n_data"]
    n_coll = params['n_collocation']
    
    data = create_dataset(params)
    train, trainc, test = init_dataset(data, params)
    X_u, X_f, y_delta, y_omega = data
    X_train, y_delta_train, y_omega_train, trf_params = train
    
# %%
    X_u = X_u.reshape((-1,3))
    y_delta = y_delta.reshape((-1,1))
    
    idx = 5
    idx_test = 20
    X = X_u[idx*n_data:idx*n_data+n_data,0:2]
    y = y_delta[idx*n_data:idx*n_data+n_data]
    X_f = X_f[:n_coll,0:2]
    X_test = X_u[idx_test*n_data:idx_test*n_data+n_data,0:2]
    y_test = y_delta[idx_test*n_data:idx_test*n_data+n_data]
    F_train = np.concatenate([np.zeros_like(y), np.zeros((X_f.shape[0], 1))], axis=0)
    y_train = np.concatenate([y, np.zeros((X_f.shape[0], 1))])
    X_train = np.concatenate([X, X_f], axis=0)
    
    print(X.shape, X_f.shape, X_train.shape, y_train.shape, F_train.shape)
    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    F = torch.tensor(F_train, dtype=torch.float32)
    
    print(X.shape, y.shape, F.shape)
    
# %%

    # Run inference in Pyro
    model = model_pinn_elevated(nn_params, X, y, F)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=100, num_chains=1)
    mcmc.run(nn_params, X, y, F)

    # Show summary of inference results
    mcmc.summary()
# %%

# Define guide function
guide = AutoDiagonalNormal(model_pinn_elevated)

# Reset parameter values
pyro.clear_param_store()

# Define the number of optimization steps
n_steps = 100000

# Setup the optimizer
adam_params = {"lr": 0.01}
optimizer = Adam(adam_params)

# Setup the inference algorithm
elbo = Trace_ELBO(num_particles=1)
svi = SVI(model_pinn_elevated, guide, optimizer, loss=elbo)

# Do gradient steps
for step in range(n_steps):
    elbo = svi.step(nn_params, X, y, F)
    if step % 500 == 0:
        print("[%d] ELBO: %.1f" % (step, elbo))

# %%
from pyro.infer import Predictive

# Make predictions for test set
predictive = Predictive(model_pinn_elevated, guide=guide, num_samples=1000,
                        return_sites=("obs1", "obs2", "_RETURN"))
samples = predictive(nn_params, X_test)
# %%
