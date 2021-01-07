#!/usr/bin/env python

# %% Imports
from jax import vmap, grad
import jax.numpy as jnp
import jax.random as random
from jax.config import config

import os
import time
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, HMC, NUTS
from numpyro.util import set_platform

# %% Functions

def nonlin(x):
    return jnp.tanh(x)

def model_bpinn_inner(X, Y, D_X, D_Y, sigma_obs=1, sigma_w=5):

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

    # Observe data
    y = numpyro.sample("Y", dist.Normal(z3, sigma_obs), obs=Y)
    
    return y
        
def model_bpinn(X, Y, F, D_H, sigma_obs=1, sigma_w=5):

    D_X, D_Y = X.shape[1], 1

    m = 0.15 # angular inertia
    d = 0.15 # damping coefficient
    B = 0.2 # susceptance [pu]
    p = 0.1 # disturbance [pu]

    # Gradients
    y = model_bpinn_inner(X, Y, D_X, D_Y)
    grad_uhat = grad(model_bpinn_inner)
    #t = X[:,1]
    dudt = grad_uhat(X, Y, D_X, D_Y)
    dudtt = grad(dudt)
    
    # Observe data
    f = numpyro.sample("F", m * dudtt + d * dudt + B * jnp.sin(u_hat) - p, obs=F)
    
    return f
    
    
def run_inference_bpinn(model, args, rng_key, X, y, F, D_H):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(kernel, args['num_warmup'], args['num_samples'], num_chains=args['num_chains'],
                progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
    mcmc.run(rng_key, X, Y, F, D_H)
    mcmc.print_summary()
    print('\nMCMC elapsed time:', time.time() - start)
    return mcmc.get_samples()


# helper function for prediction
def predict(model, rng_key, samples, X, D_H):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, F=None, D_H=D_H)
    return model_trace['Y']['value']
#%% Testing

if __name__ == '__main__':
    
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
    X = jnp.array(X_u[idx*n_data:idx*n_data+n_data,0:2])
    y = jnp.array(y_delta[idx*n_data:idx*n_data+n_data])
    X_f = jnp.array(X_f[:n_coll,0:2])
    X_test = jnp.array(X_u[idx_test*n_data:idx_test*n_data+n_data,0:2])
    y_test = jnp.array(y_delta[idx_test*n_data:idx_test*n_data+n_data])
    F_train = jnp.concatenate([jnp.ones_like(y), jnp.zeros((X_f.shape[0], 1))], axis=0)
    y_train = jnp.concatenate([y, jnp.zeros((X_f.shape[0], 1))])
    X_train = jnp.concatenate([X, X_f], axis=0)
    
    print(X.shape, X_f.shape, X_train.shape, y_train.shape, F_train.shape)

# %%
    D_H = 20
    args = {'num_samples': 1000,
        'num_warmup': 100,
        'num_chains': 1}

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(SEED))
    samples = run_inference(model_bpinn, args, rng_key, X_train, y_train, F_train, D_H)
    
# %%
