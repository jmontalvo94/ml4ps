#!/usr/bin/env python

#%% Imports

import numpy as np
import torch

#%% Functions

def set_seed_everywhere(seed):
    """ Set the seed for numpy, pytorch

        Args:
           seed (int): the seed to set everything to
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

