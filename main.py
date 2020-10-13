#!/usr/bin/env python

import numpy as np
from data.data import create_dataset

n_data = 101 # number of data points per trajectory
# n_collocation = 80 # number of collocation points

m = 0.15 # angular inertia
d = 0.15 # damping coefficient
B = 0.2 # susceptance [pu]
delta_0 = np.array([0]) # initial angle [rad]
omega_0 = np.array([0]) # initial angular speed [rad/s]

p_min = 0.08 # [pu]
p_max = 0.18 # [pu]
p_span = (p_min, p_max)

t_min = 0 # [s]
t_max = 10 # [s]
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