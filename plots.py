#!/usr/bin/env python

#%% Imports

import numpy as np
import math
import matplotlib.pyplot as plt
from data import swing_equation

#%% Functions

def plot_sol(sol, params):
    n_data = params['n_data']
    t_min, t_max = params['t_min'], params['t_max']
    t = np.linspace(t_min, t_max, n_data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(t, sol[0])
    ax1.set_title("Rotor angle vs. Time")
    ax1.set(xlabel="Time [s]", ylabel="δ [rad]")
    ax2.plot(t, sol[1], 'r')
    ax2.set_title("Angular speed vs. Time")
    ax2.set(xlabel="Time [s]", ylabel="ω [rad/s]")
    plt.tight_layout()

def plot_trajectory(sol):
    plt.plot(sol[0], sol[1])
    plt.title("Trajectory plot")
    plt.xlabel("δ [rad]")
    plt.ylabel("ω [rad/s]")

def plot_phase(sol, power, params):
    n_data = params['n_data']
    m, d, B = params['inertia'], params['damping'], params['susceptance']
    lim = math.ceil(sol.max())
    delta = np.linspace(-lim, lim, n_data//5)
    omega = np.linspace(-lim, lim, n_data//5)
    dlt, omg  = np.meshgrid(delta, omega)

    u, v = np.zeros(dlt.shape), np.zeros(omg.shape)

    n_i, n_j = dlt.shape

    for i in range(n_i):
        for j in range(n_j):
            x = dlt[i, j]
            y = omg[i, j]
            sol_v = swing_equation(1, np.array([x, y]), m, d, B, power)
            u[i,j] = sol_v[0]
            v[i,j] = sol_v[1]

    M = (np.hypot(u, v)) # get length of vector
    M[ M == 0] = 1. # avoid zero division errors 
    u /= M # normalize
    v /= M # normalize

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.quiver(dlt, omg, u, v, M)
    ax.plot(sol[0], sol[1], 'r')
    ax.set_title("Phase plot")
    ax.set(xlabel="δ [rad]", ylabel="ω [rad/s]")
    fig.show()
    
if __name__ == '__main__':
    
    from scipy.stats import norm
    
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(norm.ppf(0.001),
                    norm.ppf(0.999), 100)
    ax.plot(x, norm.pdf(x),
        'k-', lw=15)
    ax.axis('off')
    ax.fill_between(x, norm.pdf(x), color='orange')
    fig.savefig('normal_y.png', bbox_inches='tight', pad_inches = 0)