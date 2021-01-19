#!/usr/bin/env python

# %% Imports
import os
import time

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import seaborn as sns
from jax import grad, vmap, jacrev, jacfwd, hessian, value_and_grad
from jax.config import config
from numpyro import handlers
from numpyro.infer import HMC, MCMC, NUTS
from numpyro.util import set_platform

#%% Testing functions

def masked_model(x, y, data_type):
    with numpyro.plate('data'):
        with handlers.mask(mask=data_type):
            Y = numpyro.sample("Y", dist.Normal(x, 1.), obs=y)
    return Y

def test_fun(p, t, w1, w2):
    X = jnp.concatenate([p,t], axis=1)
    z1= nonlin(jnp.matmul(X, w1))
    z2 = jnp.matmul(z1, w2)
    return z2.squeeze()

def test_fun2(p, t, w1, w2):
    X = jnp.hstack([p,t])
    z1= jnp.matmul(X, w1)
    z2 = jnp.matmul(z1, w2)
    return jnp.reshape(z2,())

def test_fun3(X, w1, w2):
    z1= jnp.matmul(X, w1)
    z2 = jnp.matmul(z1, w2)
    return z2

def test_fun4(p, t, w1, w2):
    X = jnp.stack((p, t), axis=1)
    z1= jnp.matmul(X, w1)
    z2 = jnp.matmul(z1, w2)
    return z2.squeeze()

def test_fun5(p, t, w1, w2):
    X = jnp.hstack([p,t])
    z1= nonlin(jnp.matmul(X, w1))
    z2 = jnp.matmul(z1, w2)
    return jnp.reshape(z2, ())

# %% Functions

def nonlin(x):
    return jnp.tanh(x)

def model_bnn(p, t, w1, b1, w2, b2, w3, b3):
    X = jnp.hstack([p,t]) # jnp.concatenate((p, t), axis=1)
    z1 = nonlin(jnp.matmul(X, w1) + jnp.transpose(b1))
    z2 = nonlin(jnp.matmul(z1, w2) + jnp.transpose(b2))
    z3 = jnp.matmul(z2, w3) + jnp.transpose(b3)
    return jnp.reshape(z3, ()) # z3.squeeze()

# first_grad = jacfwd(model_bnn, argnums=1)
# second_grad = hessian(model_bnn, argnums=1)
mu_grad = vmap(value_and_grad(model_bnn, argnums=1), (0,0,None,None,None,None,None,None),0)
second_grad = vmap(grad(grad(model_bnn, argnums=1), argnums=1), (0,0,None,None,None,None,None,None),0)

def model_bpinn(p, t, Y, F, data_type, D_H, u_sigma=None, f_sigma=None, sigma_w=1):

    m = 0.15
    d = 0.15
    B = 0.2
    
    D_X, D_Y = 2, 1
    
    # sample first layer
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), sigma_w*jnp.ones((D_X, D_H))))
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros((D_H, 1)), sigma_w*jnp.ones((D_H, 1))))
    # sample second layer
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), sigma_w*jnp.ones((D_H, D_H))))
    b2 = numpyro.sample("b2", dist.Normal(jnp.zeros((D_H, 1)), sigma_w*jnp.ones((D_H, 1))))
    # sample final layer
    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, D_Y)), sigma_w*jnp.ones((D_H, D_Y))))
    b3 = numpyro.sample("b3", dist.Normal(jnp.zeros((D_Y, 1)), sigma_w*jnp.ones((D_Y, 1))))

    u_mu, dudt = mu_grad(p, t, w1, b1, w2, b2, w3, b3)
    dudtt = second_grad(p, t, w1, b1, w2, b2, w3, b3)
    
    # prior on the observation noise
    if u_sigma is None:
        prec_u = numpyro.sample("prec_u", dist.Gamma(3.0, 1.0))
        u_sigma = 1.0 / jnp.sqrt(prec_u)
    if f_sigma is None:
        prec_f = numpyro.sample("prec_f", dist.Gamma(3.0, 1.0))
        f_sigma = 1.0 / jnp.sqrt(prec_f)

    # observe data
    with numpyro.plate('observations', p.shape[0]):
        with handlers.mask(mask=data_type):
            u_hat = numpyro.sample("Y", dist.Normal(u_mu, u_sigma), obs=Y)
        f_mu = m * dudtt + d * dudt + B * jnp.sin(u_mu) - p # Forcing physics-term, always=0
        f_hat = numpyro.sample("F", dist.Normal(f_mu, f_sigma), obs=F)
    
    return u_mu, f_mu
    
    
def run_inference(model, args, rng_key, p, t, Y, F, data_type, D_H, u_sigma=None, f_sigma=None, sigma_w=1):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(kernel, args['num_warmup'], args['num_samples'], num_chains=args['num_chains'],
                progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
    mcmc.run(rng_key, p, t, Y, F, data_type, D_H, u_sigma, f_sigma, sigma_w)
    mcmc.print_summary()
    print('\nMCMC elapsed time:', time.time() - start)
    return mcmc.get_samples()


# helper function for prediction
def predict(model, rng_key, samples, p, t, D_H, data_type):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model_trace = handlers.trace(model).get_trace(p=p, t=t, Y=None, F=None, D_H=D_H, data_type=data_type)
    return model_trace['Y']['value']

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
    SEED = general['seed']
    
    data = create_dataset(params)
    X_u, X_f, y_delta, y_omega = data
    train, trainc, test = init_dataset(data, params, transformation=None)
    train_idx, trainc_idx, test_idx = init_dataset(data, params, sample=False, transformation=None)
    X_train, y_delta_train, y_omega_train, trf_params_train = train
    X_trainc, y_delta_trainc, y_omega_trainc, trf_params_trainc = trainc
    X_test, y_delta_test, y_omega_test, trf_params_test = test
    X_train_idx, y_delta_train_idx, y_omega_train_idx, trf_params_train_idx = train_idx
    X_test_idx, y_delta_test_idx, y_omega_test_idx, trf_params_test_idx = test_idx
    
    idx_data = np.where(X_trainc[:,2] == 1.)[0]
    mask_data = np.zeros(len(X_trainc), dtype=bool)
    mask_data[idx_data] = True
    
#%% Data for BNN

idx = 0
X_selected = jnp.array(X_train_idx[idx*n_data:idx*n_data+n_data,:2])
y_selected = jnp.array(y_delta_train_idx[idx*n_data:idx*n_data+n_data,:]).squeeze()
X = jnp.array(X_train[:,:2])
y = jnp.array(y_delta_train)
X_test = jnp.array(X_test)
y_test = jnp.array(y_delta_test)

p_selected = X_selected[:,0]
t_selected = X_selected[:,1] # / params['t_max'] 

F = jnp.zeros_like(y_selected)

X_train = jnp.array(X_trainc[:,:2])
p_selected = X_train[:,0]
t_selected = X_train[:,1] # / params['t_max']
y_selected = jnp.array(y_delta_trainc.squeeze())
mask = jnp.array(mask_data, dtype=bool)
F = jnp.zeros_like(y_selected)

#%% Data for BNN


# X_train = torch.tensor(X_trainc[:,:2], dtype=torch.float32)
# y_train = torch.tensor(y_delta_trainc, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_delta_test, dtype=torch.float32)
# mask = torch.tensor(mask_data.reshape(-1,1), dtype=torch.bool)
# F = torch.zeros(y_train.shape[0], 1)

# %% MCMC

D_H = 20
args = {'num_samples': 2000,
    'num_warmup': 500,
    'num_chains': 1}

# Inference
rng_key, rng_key_predict = random.split(random.PRNGKey(SEED))
samples = run_inference(model_bpinn, args, rng_key, p_selected, t_selected, y_selected, F, mask, D_H, u_sigma=0.001, f_sigma=0.001)

#%% Export samples

import pickle 

with open('data/samples_2000_500_prec.pickle', 'wb') as handle:
    pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
with open('data/samples1100.pickle', 'rb') as handle:
    samples = pickle.load(handle)


#%% Inference visualization

sns.distplot(samples['w1'][:,0,0])
plt.show()

sns.displot(x=samples['w1'][:,0,0], y=samples['w1'][:,0,1], kind='kde')
plt.plot(samples['w1'][:,0,0], samples['w1'][:,0,1], alpha=0.5)
plt.show()

# %% Predictions

vmap_args = (samples, random.split(rng_key_predict, args['num_samples'] * args['num_chains']))
predictions = vmap(lambda samples, rng_key: predict(model_bpinn, rng_key, samples, p_selected, t_selected, D_H))(*vmap_args)

# preds = predictions[..., 0]
pred_mean = jnp.mean(predictions, axis=0)
pred_std = jnp.std(predictions, axis=0)
percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)


#%% Prediction visualization

plt.plot(pred_mean, label='Prediction')
plt.plot(y_selected, label='Real')
plt.fill_between(np.arange(len(pred_mean)), pred_mean + pred_std, pred_mean - pred_std, alpha=0.5)
#plt.fill_between(np.arange(len(pred_mean)), percentiles[0], percentiles[1], alpha=0.5)
plt.xlabel("Time [s]")
plt.ylabel("δ [rad]")
plt.legend()
plt.show()


# %%
