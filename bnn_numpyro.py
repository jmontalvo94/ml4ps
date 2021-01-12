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
from jax import grad, vmap
from jax.config import config
from numpyro import handlers
from numpyro.infer import HMC, MCMC, NUTS
from numpyro.util import set_platform

# %% Functions

def nonlin(x):
    return jnp.tanh(x)

def model_bnn(X, Y, D_H, sigma_obs=None, sigma_w=1):

    D_X, D_Y = X.shape[1], 1

    # sample first layer (we put unit normal priors on all weights)
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), sigma_w*jnp.ones((D_X, D_H))))  # D_X D_H
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros((D_H, 1)), sigma_w*jnp.ones((D_H, 1))))  # D_H 1
    z1 = nonlin(jnp.matmul(X, w1) + jnp.transpose(b1))   # N D_H  <= first layer of activations

    # sample second layer
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), sigma_w*jnp.ones((D_H, D_H))))  # D_H D_H
    b2 = numpyro.sample("b2", dist.Normal(jnp.zeros((D_H, 1)), sigma_w*jnp.ones((D_H, 1))))  # D_H 1
    z2 = nonlin(jnp.matmul(z1, w2) + jnp.transpose(b2))  # N D_H  <= second layer of activations

    # sample final layer of weights and neural network output
    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, D_Y)), sigma_w*jnp.ones((D_H, D_Y))))  # D_H D_Y
    b3 = numpyro.sample("b3", dist.Normal(jnp.zeros((D_Y, 1)), sigma_w*jnp.ones((D_Y, 1))))  # D_H 1
    z3 = jnp.matmul(z2, w3) + jnp.transpose(b3)  # N D_Y  <= output of the neural network
    
    # prior on the observation noise
    if sigma_obs is None:
        prec_obs = numpyro.sample("prec_obs", dist.Gamma(100.0, 1.0))
        sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    with numpyro.plate('observations', D_X):
        numpyro.sample("Y", dist.Normal(z3, sigma_obs), obs=Y)
    
    
def run_inference(model, args, rng_key, X, Y, D_H, sigma_obs=None):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(kernel, args['num_warmup'], args['num_samples'], num_chains=args['num_chains'],
                progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
    mcmc.run(rng_key, X, Y, D_H, sigma_obs)
    mcmc.print_summary()
    print('\nMCMC elapsed time:', time.time() - start)
    return mcmc.get_samples()


# helper function for prediction
def predict(model, rng_key, samples, X, D_H, sigma_obs=None):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H, sigma_obs=sigma_obs)
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
    train, trainc, test = init_dataset(data, params)
    train_idx, trainc_idx, test_idx = init_dataset(data, params, sample=False)
    X_train, y_delta_train, y_omega_train, trf_params_train = train
    X_test, y_delta_test, y_omega_test, trf_params_test = test
    X_train_idx, y_delta_train_idx, y_omega_train_idx, trf_params_train_idx = train_idx
    X_test_idx, y_delta_test_idx, y_omega_test_idx, trf_params_test_idx = test_idx
    
#%% Data for BNN

idx = 0
X_selected = jnp.array(X_train_idx[idx*n_data:idx*n_data+n_data,:2])
y_selected = jnp.array(y_delta_train_idx[idx*n_data:idx*n_data+n_data,:])
X = jnp.array(X_train[:,:2])
y = jnp.array(y_delta_train)
X_test = jnp.array(X_test)
y_test = jnp.array(y_delta_test)

# %% MCMC

D_H = 20
args = {'num_samples': 2000,
    'num_warmup': 500,
    'num_chains': 1}

# Inference
rng_key, rng_key_predict = random.split(random.PRNGKey(SEED))
samples = run_inference(model_bnn, args, rng_key, X_selected, y_selected, D_H)

#%% Inference visualization

plt.plot(samples['prec_obs'])
plt.show()

sns.distplot(samples['prec_obs'])
plt.show()

sns.distplot(samples['w1'][:,0,0])
plt.show()

sns.displot(x=samples['w1'][:,0,0], y=samples['w1'][:,0,1], kind='kde')
plt.plot(samples['w1'][:,0,0], samples['w1'][:,0,1], alpha=0.5)
plt.show()

# %% Predictions

vmap_args = (samples, random.split(rng_key_predict, args['num_samples'] * args['num_chains']))
predictions = vmap(lambda samples, rng_key: predict(model_bnn, rng_key, samples, X_selected, D_H))(*vmap_args)

preds = predictions[..., 0]
pred_mean = jnp.mean(preds, axis=0)
pred_std = jnp.std(preds, axis=0)
percentiles = np.percentile(preds, [5.0, 95.0], axis=0)

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
