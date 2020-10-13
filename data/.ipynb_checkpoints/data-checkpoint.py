#!/usr/bin/env python

import numpy as np
import math
from scipy import integrate

def swing_equation(t, state_variable, m, d, B, power):
    """
    system of first order ordinary differential equations (swing equation)
    :param state_variable: state vector consisting of delta_i and omega_i
    :param m: inertia of the machine
    :param d: damping coefficient
    :param B: bus susceptance matrix
    :param power: power injection or retrieval
    :return: updated state variable
    """
    # Split the state variable into delta and omega
    state_delta = state_variable[0]
    state_omega = state_variable[1]

    # Computing the non-linear term in the swing equation sum_j (B_ij sin(delta_i - delta_j))
    delta_i = state_delta
    delta_j = np.zeros_like(delta_i)
    delta_ij = np.sin(delta_i - delta_j)
    connectivity_vector = np.sum(np.multiply(B, delta_ij))

    # Preallocate states
    state_delta_new = np.zeros_like(state_delta)
    state_omega_new = np.zeros_like(state_omega)
    
    # Update states
    state_delta_new = state_omega
    state_omega_new = 1 / m * (power - d * state_omega - connectivity_vector)

    return np.array([state_delta_new, state_omega_new])

def solve_swing_eq(params):
    """solves the swing equation with pre-defined parameters"""

    # Unpack parameters
    m = params['inertia']
    d = params['damping']
    B = params['susceptance']
    power = params['power']
    delta_0 = params['delta_0']
    omega_0 = params['omega_0']
    t_span = params['t_span'] # (t_min, t_max)
    n_data = params['n_data']
    
    # Define analysis time period
    t_eval = np.linspace(t_span[0], t_span[1], n_data)
    
    # Pack initial state
    initial_states = np.concatenate([delta_0, omega_0])
    
    # Solve ODE
    ode_solution = integrate.solve_ivp(swing_equation,
                                       t_span=t_span,
                                       y0=initial_states,
                                       args=[m, d, B, power],
                                       t_eval=t_eval)
    
    # Transpose solution
    states = ode_solution.y
    
    return states

def create_dataset(params):
    """creates dataset based on solving the swing equation on given parameters."""
    
    # Unpack parameters
    p_min, p_max = params['p_span']
    t_min, t_max = params['t_span']
    n_data = params['n_data']
    
    # Define span of power disturbance    
    p_span = np.linspace(p_min, p_max, n_data)
    t_span = np.linspace(t_min, t_max, n_data).reshape((-1,1))
    
    # Preallocate dataset in memory
    X = np.empty((n_data, n_data, 2)) # [trajectory, [power, t_span], 2]
    Y_delta = np.empty((n_data, n_data, 1)) # [trajectory, delta, 1]
    Y_omega = np.empty((n_data, n_data, 1)) # [trajectory, omega, 1]
    
    # Solve for each power disturbance and save to matrices
    for idx, power in enumerate(p_span):
        params['power'] = power
        sol = solve_swing_eq(params)
        X[idx] = np.concatenate([np.repeat(power, n_data).reshape((-1,1)), t_span], axis=1)
        Y_delta[idx] = sol[0].reshape((-1,1))
        Y_omega[idx] = sol[1].reshape((-1,1))
    
    return X, Y_delta, Y_omega


if __name__ == '__main__':

    # Parameters
    n_data = 101 # number of data points per trajectory
    # n_collocation = 80 # number of collocation points

    m = 0.15 # angular inertia
    d = 0.15 # damping coefficient
    B = 0.2 # susceptance [pu]
    delta_0 = np.array([0]) # initial angle [rad]
    omega_0 = np.array([0]) # initial angular speed [rad/s]

    p_min = 0.08 # per unit
    p_max = 0.18
    p_span = (p_min, p_max)

    t_min = 0
    t_max = 10 # [seconds]
    t_span = (t_min, t_max)

    params = {'n_data': n_data,
              'inertia': m,
              'damping': d,
              'susceptance': B,
              'delta_0': delta_0,
              'omega_0': omega_0,
              'p_span': p_span,
              't_span': t_span
             }
    
    X, Y_delta, Y_omega = create_dataset(params)