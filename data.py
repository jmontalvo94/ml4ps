#!/usr/bin/env python

#%% Imports

import numpy as np
import math
from pyDOE import lhs
from scipy import integrate
from sklearn.model_selection import train_test_split
from cli import cli

#%% Functions

def swing_equation(t, x, *args):
    r""" Swing equation expressed as a system of ODEs.
    
    Given by the formula:
    
    .. math::
        m_i \ddot{\delta} + d_i \dot{\delta} + B_{ij} V_i V_j sin(\delta) - P_i = 0
    
    Args:
        t: Time
        x: State vector [delta_i, omega_i]
        m: Inertia of the machine
        d: Damping coefficient
        B: Bus susceptance matrix
        P: Power injection or retrieval
    
    Returns:
        x_new: Updated state vector [delta_i, omega_i]
    """
    # Unpack parameters
    m, d, B, P = args
    state_delta = x[0]
    state_omega = x[1]

    # Computing the non-linear term in the swing equation sum_j (B_ij sin(delta_i - delta_j))
    delta_i = state_delta
    delta_j = np.zeros_like(delta_i)
    delta_ij = np.sin(delta_i - delta_j)
    connectivity_vector = np.sum(np.multiply(B, delta_ij))

    # Preallocate memory
    state_delta_new = np.zeros_like(state_delta)
    state_omega_new = np.zeros_like(state_omega)
    
    # Update states
    state_delta_new = state_omega
    state_omega_new = (1 / m) * (P - d * state_omega - connectivity_vector)

    return np.array([state_delta_new, state_omega_new])

def solve_swing_eq(params):
    """Solves the swing equation with pre-defined parameters.
    
    Args:
        params: Dictionary containing relevant parameters.
    
    Returns:
        states: Solution of the swing equation (angle)
    """

    # Unpack parameters
    n_data = params['n_data']
    m = params['inertia']
    d = params['damping']
    B = params['susceptance']
    P = params['power']
    delta_0 = np.array([params['delta_0']])
    omega_0 = np.array([params['omega_0']])
    t_span = params['t_span'] # (t_min, t_max)
    
    # Define analysis time period
    t_eval = np.linspace(t_span[0], t_span[1], n_data)
    
    # Pack initial state
    initial_states = np.concatenate([delta_0, omega_0])
    
    # Solve ODE
    ode_solution = integrate.solve_ivp(swing_equation,
                                       t_span=t_span,
                                       y0=initial_states,
                                       args=[m, d, B, P],
                                       t_eval=t_eval)
    
    # Extract solution only
    states = ode_solution.y
    
    return states

def create_dataset(params):
    """Creates a dataset based on solving the swing equation on given parameters.
    
    Args:
        params: Dictionary containing relevant parameters.
    
    Returns:
        X: Inputs  [x_power, x_time, type]
        Y_delta: Target [delta]
        Y_omega: Secondary target [omega]
    """
        
    # Unpack parameters
    n_data = params['n_data']
    n_collocation = params['n_collocation']
    p_min, p_max = params['p_span']
    t_min, t_max = params['t_span']
    
    # Define upper and lower bounds
    lb = np.array([p_min, t_min])
    ub = np.array([p_max, t_max])
    
    # Define span of power disturbance    
    p_span = np.linspace(p_min, p_max, n_data)
    t_span = np.linspace(t_min, t_max, n_data).reshape((-1,1))
    
    # Preallocate datasets in memory
    X_u = np.ones((n_data, n_data, 3)) # [p_span, t_span, ones]
    X_f = np.zeros((n_collocation, 3)) # [p_span, t_span, zeros]
    y_delta = np.empty((n_data, n_data)) # [p_span, t_span, delta]
    y_omega = np.empty((n_data, n_data)) # [p_span, t_span, omega]
    
    # Solve for each power disturbance and save to matrices
    for idx, P in enumerate(p_span):
        params['power'] = P
        sol = solve_swing_eq(params)
        X_u[idx][:, :2] = np.concatenate([np.repeat(P, n_data).reshape((-1,1)), t_span], axis=1)
        y_delta[idx] = sol[0]
        y_omega[idx] = sol[1]
    
    # Create collocation points matrix
    X_f[:, :2] = lb + (ub - lb) * lhs(2, n_collocation)

    return X_u, X_f, y_delta, y_omega

def init_dataset(data, data_params, split=0.8, sample=True, transformation='standardize', noise=None):
    """Inits the Dataset, depends on collocation flag.
    
    Args:
    inputs(np.ndarray): Matrix of inputs.
    targets(np.ndarray): Matrix of targets.
    data_params(Dict): Includes the parameters of the dataset.
    collocation(bool): Indicates if the matrices should be sliced _
        to only include data points or also collocation points.
    """
    
    # Unpack parameters
    X_u, X_f, y_delta, y_omega = data
    n_u = data_params['n_data']
    n_f = data_params['n_collocation']
    
    # Prepare train/test split
    idx_split = int(np.floor(X_u.shape[0]*split))
    idx_total = np.random.permutation(X_u.shape[0]) # permute trajectories (dim0)
    idx_left = idx_total[:idx_split]
    idx_test = idx_total[idx_split:]
    
    # Train/test split
    X_u_left = X_u[idx_left]
    y_delta_left = y_delta[idx_left]
    y_omega_left = y_omega[idx_left]
    X_u_test = X_u[idx_test]
    y_delta_test = y_delta[idx_test]
    y_omega_test = y_omega[idx_test]
    
    # Reshape
    X_train = X_u_left.reshape((-1,3))
    y_delta_train = y_delta_left.reshape((-1,1))
    y_omega_train = y_omega_left.reshape((-1,1))
    X_test = X_u_test.reshape((-1,3))
    y_delta_test = y_delta_test.reshape((-1,1))
    y_omega_test = y_omega_test.reshape((-1,1))
    
    # Copy to keep real trajectories
    y_delta_train_real = y_delta_train.copy()
    y_omega_train_real = y_omega_train.copy()
    
    # Sample
    if sample:
        idx_train = np.random.choice(X_train.shape[0], n_u, replace=False)
        X_train = X_train[idx_train]
        y_delta_train = y_delta_train[idx_train]
        y_omega_train = y_omega_train[idx_train]
    
    # Concatenate with collocation points
    X_trainc = np.vstack((X_train, X_f))
    y_delta_trainc = np.vstack((y_delta_train, np.zeros((n_f, 1))))
    y_omega_trainc = np.vstack((y_omega_train, np.zeros((n_f, 1))))

    # Add noise to trajectories
    if noise is not None:
        y_delta_train = y_delta_train + np.random.normal(0., noise, (y_delta_train.shape[0], 1))
        y_omega_train = y_omega_train + np.random.normal(0., noise, (y_delta_train.shape[0], 1))
        
    # Transform
    if transformation == 'standardize':
        mu_train = X_train.mean(axis=0)[:2]
        mu_trainc = X_trainc.mean(axis=0)[:2]
        std_train = X_train.std(axis=0)[:2]
        std_trainc = X_trainc.std(axis=0)[:2]
        X_train[:,:2] = (X_train[:,:2] - mu_train) / std_train
        X_trainc[:,:2] = (X_trainc[:,:2] - mu_trainc) / std_trainc
        X_test[:,:2] = (X_test[:,:2] - mu_train) / std_train
        trf_params = ((mu_train, std_train), (mu_trainc, std_trainc))
    elif transformation == 'normalize':
        max_train = X_train.max(axis=0)[:2]
        max_trainc = X_trainc.max(axis=0)[:2]
        X_train[:,:2] = X_train[:,:2] / max_train
        X_trainc[:,:2] = X_trainc[:,:2] / max_trainc
        X_test = X_test / max_train
        trf_params = (max_train, max_trainc)
    else:
        trf_params = None
        
    # Shuffle again because of the collocation points
    if sample:
        idx_shuffle = np.random.permutation(X_trainc.shape[0])
        X_trainc = X_trainc[idx_shuffle]
        y_delta_trainc = y_delta_trainc[idx_shuffle]
        y_omega_trainc = y_omega_trainc[idx_shuffle]
 
    return ((X_train, y_delta_train, y_omega_train, trf_params), 
            (X_trainc, y_delta_trainc, y_omega_trainc, trf_params),
            (X_test, y_delta_test, y_omega_test, trf_params))

#%% Testing

if __name__ == '__main__':

    args, general, params, nn_params = cli()
    params['t_span'] = (params['t_min'], params['t_max'])
    params['p_span'] = (params['p_min'], params['p_max'])
    n_data = params['n_data']
    
    data = create_dataset(params)
    train, trainc, test = init_dataset(data, params)
    train_idx, trainc_idx, test_idx = init_dataset(data, params, sample=False, transformation=None)
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
    
    #%% Plotting

    #plt.plot(np.linspace(params['t_min'], params['t_max'], 101), real, alpha=0.5)
    plt.scatter(np.linspace(params['t_min'], params['t_max'], params['n_data']), y_selected)
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [$\delta$]')

