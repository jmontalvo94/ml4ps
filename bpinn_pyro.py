#!/usr/bin/env python

# %% Imports

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import seaborn as sns
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, HMC
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.optim import Adam, ClippedAdam
from pyro.infer import Predictive


#%% Classes

class BNN(PyroModule):
    def __init__(self, nn_params):
        super().__init__()
        h_in = nn_params['layers'][0]
        h1 = nn_params['layers'][1]
        h2 = nn_params['layers'][2]
        h_out = nn_params['layers'][-1]
        self.fc1 = PyroModule[nn.Linear](h_in, h1)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([h1, h_in]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([h1]).to_event(1))
        self.fc2 = PyroModule[nn.Linear](h1, h2)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([h2, h1]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([h2]).to_event(1))
        self.fc3 = PyroModule[nn.Linear](h2, h_out)
        self.fc3.weight = PyroSample(dist.Normal(0., 1.).expand([h_out, h2]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., 1.).expand([h_out]).to_event(1))
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Model_BPINN(PyroModule):
    def __init__(self, nn_params, data_params):
        super().__init__()
        self.net = BNN(nn_params)  # this is a PyroModule
        # self.net_prec = PyroSample(dist.Gamma(100.0, 1.0))
        # self.net_scale = 1.0 / torch.sqrt(self.net_prec)
        # self.f_prec = PyroSample(dist.Gamma(100.0, 1.0))
        # self.f_scale = 1.0 / torch.sqrt(self.f_prec)
        self.net_scale = 0.001
        self.f_scale = 0.001
        self.m = data_params['inertia']
        self.d = data_params['damping']
        self.B = data_params['susceptance']

    def forward(self, X, u=None, f=None):
        
        p = X.select(1,0).view(-1,1)
        t = X.select(1,1).view(-1,1)
        t.requires_grad_(True)
        
        mu = self.net(torch.cat([p, t], dim=1))
        dudt = self.grad1(mu, t)
        dudtt = self.grad1(dudt, t)
        # print(mu, dudt, dudtt)
        # mu = mu.detach()
        # dudt = dudt.detach()
        # print(mu, dudt, dudtt)
        
        sigma = self.net_scale
        f_sigma = self.f_scale
        
        with pyro.plate("observations", X.shape[0]):
            u_hat = pyro.sample("obs1", dist.Normal(mu, sigma).to_event(1), obs=u)
            # print(u_hat)
            # dudt = self.grad1(u_hat, t)
            # dudtt = self.grad2(dudt, t)
            # print(u_hat, dudt, dudtt)
            # u_hat = u_hat.detach()
            # dudt = dudt.detach()
            # print(u_hat, dudt, dudtt)
            f_mu = self.m * dudtt + self.d * dudt + self.B * torch.sin(u_hat) - p
            # print(f_mu)
            f_hat = pyro.sample("obs2", dist.Normal(f_mu, f_sigma).to_event(1), obs=f)
        return u_hat, f_hat
    
    def grad1(self, outputs, inputs):
        return torch.autograd.grad(outputs.sum(), inputs, retain_graph=True, create_graph=True)[0]
    
    def grad2(self, outputs, inputs):
        return torch.autograd.grad(outputs.sum(), inputs)[0]
    

#%% Testing

if __name__ == '__main__':
    
    DATA_DIR = 'data/'
    
    from cli import cli
    from data import create_dataset, init_dataset
    
    args, general, params, nn_params = cli()
    params['t_span'] = (params['t_min'], params['t_max'])
    params['p_span'] = (params['p_min'], params['p_max'])
    n_data = params["n_data"]
    n_coll = params['n_collocation']
    
    data = create_dataset(params)
    train, trainc, test = init_dataset(data, params)
    train_idx, trainc_idx, test_idx = init_dataset(data, params, sample=False)
    X_u, X_f, y_delta, y_omega = data
    X_trainc, y_delta_trainc, y_omega_trainc, trf_params_trainc = trainc
    X_test, y_delta_test, y_omega_test, trf_params_test = test
    X_train_idx, y_delta_train_idx, y_omega_train_idx, trf_params_train_idx = train_idx
    X_test_idx, y_delta_test_idx, y_omega_test_idx, trf_params_test_idx = test_idx
    
    idx_data = np.where(X_trainc[:,2] == 1.)[0]
    mask_data = np.zeros(len(X_trainc), dtype=bool)
    mask_data[idx_data] = True
    
    X_train_data = torch.tensor(X_trainc[mask_data,:2], dtype=torch.float32)
    X_train_coll = torch.tensor(X_trainc[~mask_data,:2], dtype=torch.float32)
    y_train_data = torch.tensor(y_delta_trainc[mask_data], dtype=torch.float32)
    y_train_coll = torch.tensor(y_delta_trainc[~mask_data], dtype=torch.float32)
    F_data = torch.zeros_like(y_train_data)
    F_coll = torch.zeros_like(y_train_coll)
    
    idx = 0
    X_selected = torch.tensor(X_train_idx[idx*n_data:idx*n_data+n_data,:2], dtype=torch.float32)
    y_selected = torch.tensor(y_delta_train_idx[idx*n_data:idx*n_data+n_data,:], dtype=torch.float32)

#%% Data for BNN

# X_train = torch.tensor(X_trainc[:,:2], dtype=torch.float32)
# y_train = torch.tensor(y_delta_trainc, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_delta_test, dtype=torch.float32)
# F = torch.zeros(y_train.shape[0], 1)


# %% SVI

model = Model_BPINN(nn_params, params)

# Define guide function
guide = AutoDiagonalNormal(model)

# Reset parameter values
pyro.clear_param_store()

# Define the number of optimization steps
n_steps = 10000

# Setup the optimizer
adam_params = {"lr": 0.001}
optimizer = Adam(adam_params)

# Setup the inference algorithm
elbo = Trace_ELBO(num_particles=1)
svi = SVI(model, guide, optimizer, loss=elbo)

# Do gradient steps
for step in range(n_steps):
    elbo = svi.step(X_selected, y_selected, F_data)
    # if step % 2 == 0:
    #     elbo = svi.step(X_selected, y_selected, F_data)
    # else:
    #     elbo = svi.step(X_train_coll, f=F_coll)
    if step % 500 == 0:
        print("[%d] ELBO: %.1f" % (step, elbo))

# %% Prediction

# Make predictions for test set
predictive_svi = Predictive(model, guide=guide, num_samples=1000,
                        return_sites=["obs1", "obs2"])
samples_svi = predictive_svi(X_selected)
pred_mean = samples_svi["obs1"].detach().mean(axis=0).squeeze(-1)
pred_std = samples_svi["obs1"].detach().std(axis=0).squeeze(-1)

# %% Visualize

sns.distplot([dist.Normal(pyro.get_param_store()['AutoDiagonalNormal.loc'][0].item(), pyro.get_param_store()['AutoDiagonalNormal.scale'][0].item()).sample() for _ in range(1000)])
plt.show()

plt.plot(pred_mean, label='Prediction')
plt.plot(y_selected, label='Real')
plt.fill_between(torch.arange(len(pred_mean)), pred_mean + pred_std, pred_mean - pred_std, alpha=0.5)
plt.xlabel("Time [s]")
plt.ylabel("δ [rad]")
plt.legend()
plt.show()

# %% MCMC

# Run inference in Pyro
model = Model_BNN(nn_params)
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=100, num_chains=1)
mcmc.run(X_selected, y_selected)

# Show summary of inference results
mcmc.summary()
    
# %% Prediction

# Samples
posterior_samples = mcmc.get_samples()

# Prediction
predictive_mcmc = Predictive(model, posterior_samples=mcmc.get_samples(), num_samples=1000, return_sites=["obs"])
samples_mcmc = predictive(X_selected)
pred_mean = samples_mcmc["obs"].detach().mean(axis=0).squeeze(-1)
pred_std = samples_mcmc["obs"].detach().std(axis=0).squeeze(-1)

#%% Save

import pickle

with open(DATA_DIR + NAME + '.pickle', 'wb') as handle:
    pickle.dump(samples_mcmc, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% Visualize

sns.distplot(posterior_samples['net.fc1.weight'][:,0,0])
plt.show()

sns.displot(x=posterior_samples['net.fc1.weight'][:,0,0], y=posterior_samples['net.fc1.weight'][:,0,1], kind='kde')
plt.plot(posterior_samples['net.fc1.weight'][:,0,0], posterior_samples['net.fc1.weight'][:,0,1])
plt.show()

plt.plot(pred_mean)
plt.plot(y_selected)
plt.fill_between(torch.arange(len(pred_mean)), pred_mean + pred_std, pred_mean - pred_std)
plt.xlabel("Time [s]")
plt.ylabel("δ [rad]")
plt.show()
