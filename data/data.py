#!/usr/bin/env python

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate
from pyDOE import lhs
from torch.utils import data

class Dataset(data.Dataset):
    """Creates a Dataset depending on the given matrices.
    
    Attributes:
        inputs(np.ndarray): Sliced matrix of inputs.
        p_type(np.ndarray): Sliced matrix of point type (from inputs).
        targets(np.ndarray): Sliced matrix of targets.
        n_u(int): Number of data points.
        n_f(int): Number of collocation points.
    
    Returns:
        X(np.ndarray): Input observation at idx.
        y(np.ndarray): Target observation at idx.
        p_type(np.ndarray): Point type at idx
    """
    def __init__(self, data_params, collocation=True):
        """Inits the Dataset, depends on collocation flag.
        
        Args:
        inputs(np.ndarray): Matrix of inputs.
        targets(np.ndarray): Matrix of targets.
        data_params(Dict): Includes the parameters of the dataset.
        collocation(bool): Indicates if the matrices should be sliced _
            to only include data points or also collocation points.
        """
        super().__init__()
        
        X, y_delta, y_omega = self.create_dataset(data_params)
        
        if collocation:
            self.inputs = X[:, 0:2]
            self.p_type = X[:, -1].reshape((-1,1))
            self.targets = y_delta
        else:
            self.inputs = X[:n_data*n_data, 0:2]
            self.p_type = X[:n_data*n_data, -1].reshape((-1,1))
            self.targets = y_delta

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.targets)

    def __getitem__(self, idx):
        """Retrieve inputs and targets at the given index."""
        X = self.inputs[idx]
        p_type = self.p_type[idx]
        y = self.targets[idx]

        return X, p_type, y
    
    def create_dataset(self, params):
        """Creates a dataset based on solving the swing equation on given parameters.

        Args:
            params: Dictionary containing relevant parameters.

        Returns:
            X: Inputs  [x_power, x_time, type]
            Y_delta: Target [delta]
            Y_omega: Secondary target [omega]
        """

        # Unpack parameters
        p_min, p_max = params['p_span']
        t_min, t_max = params['t_span']
        n_data = params['n_data']
        n_collocation = params['n_collocation']
        seed = params['seed']
        
        n_total = n_data + n_collocation

        # TODO move to main()?
        np.random.seed(seed)

        # Define upper and lower bounds
        lb = np.array([p_min, t_min])
        ub = np.array([p_max, t_max])

        # Define span of power disturbance    
        p_span = np.linspace(p_min, p_max, n_data)
        t_span = np.linspace(t_min, t_max, n_data).reshape((-1,1))

        # Preallocate datasets in memory
        X_u = np.ones((n_data, n_data, 3)) # [trajectory, [power, t_span, ones], 3]
        X_f = np.zeros((n_collocation, 3)) # [power, t_span, zeros], 3]
        y_delta = np.zeros((n_data*n_data+n_collocation, 1)) # [delta]
        y_omega = np.zeros((n_data*n_data+n_collocation, 1)) # [omega]

        # Solve for each power disturbance and save to matrices
        for idx, P in enumerate(p_span):
            params['power'] = P
            sol = self.solve_swing_eq(params)
            X_u[idx][:, :2] = np.concatenate([np.repeat(P, n_data).reshape((-1,1)), t_span], axis=1)
            y_delta[idx] = sol[0].reshape((-1,1))
            y_omega[idx] = sol[1].reshape((-1,1))

        # Create collocation points matrix
        X_f[:, :2] = lb + (ub - lb) * lhs(2, n_collocation)

        # Stack matrices
        X = np.vstack((X_u.reshape((-1, 3)), X_f))
        y_delta = np.vstack((y_delta.reshape((-1,1)), np.zeros((n_collocation, 1)))) # include collocation points
        y_omega = np.vstack((y_omega.reshape((-1,1)), np.zeros((n_collocation, 1)))) # include collocation points

        return X, y_delta, y_omega # reshaped
    
    def solve_swing_eq(self, params):
        """Solves the swing equation with pre-defined parameters.

        Args:
            params: Dictionary containing relevant parameters.

        Returns:
            states: Solution of the swing equation (angle)
        """

        # Unpack parameters
        m = params['inertia']
        d = params['damping']
        B = params['susceptance']
        P = params['power']
        delta_0 = np.array([params['delta_0']])
        omega_0 = np.array([params['omega_0']])
        t_span = params['t_span'] # (t_min, t_max)
        n_data = params['n_data']

        # Define analysis time period
        t_eval = np.linspace(t_span[0], t_span[1], n_data)

        # Pack initial state
        initial_states = np.concatenate([delta_0, omega_0])

        # Solve ODE
        ode_solution = integrate.solve_ivp(self.swing_equation,
                                           t_span=t_span,
                                           y0=initial_states,
                                           args=[m, d, B, P],
                                           t_eval=t_eval)

        # Extract solution only
        states = ode_solution.y

        return states
    
    def swing_equation(self, t, x, m, d, B, P):
        r""" Swing equation expressed as a system of ODEs.

        Given by the formula:

        .. math::
            m_i \ddot{\delta} + d_i \dot{\delta} + B_{ij} V_i V_j sin(\delta) - P_i = 0

        Args:
            x: State vector [delta_i, omega_i]
            m: Inertia of the machine
            d: Damping coefficient
            B: Bus susceptance matrix
            P: Power injection or retrieval

        Returns:
            x_new: Updated state vector [delta_i, omega_i]
        """
        # Split the state variable into delta and omega
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
        state_omega_new = 1 / m * (P - d * state_omega - connectivity_vector)

        return np.array([state_delta_new, state_omega_new])


if __name__ == '__main__':

    # Parameters
    n_data = 101 # number of data points per trajectory
    n_collocation = 800 # number of collocation points

    m = 0.15 # angular inertia
    d = 0.15 # damping coefficient
    B = 0.2 # susceptance [pu]
    delta_0 = 0 # initial angle [rad]
    omega_0 = 0 # initial angular speed [rad/s]

    p_min = 0.08 # minimum power [pu]
    p_max = 0.18 # maximum power [pu]
    p_span = (p_min, p_max)

    t_min = 0
    t_max = 10 # [seconds]
    t_span = (t_min, t_max)

    seed = 1

    data_params = {'n_data': n_data,
                   'n_collocation': n_collocation,
                   'inertia': m,
                   'damping': d,
                   'susceptance': B,
                   'delta_0': delta_0,
                   'omega_0': omega_0,
                   'p_span': p_span,
                   't_span': t_span,
                   'seed': seed
                  }
    
    X, Y_delta, Y_omega = create_dataset(params)