#!/usr/bin/env python

#%% Imports

import argparse
import json
from datetime import datetime

#%% Functions

def cli():
    """ Command-line interface to run the training procedure.

        Returns:
            args: arguments from CLI and JSON config file
    """

    parser = argparse.ArgumentParser(description='Run experiment with configuration from JSON file.')

    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        default='config.json',
        help='configuration file with parameters'
    )

    parser.add_argument(
        '-n',
        '--name',
        type=str,
        help='experiment name'
    )
    
    parser.add_argument(
        '-pm',
        '--path_models',
        type=str,
        default='models/',
        help='path to pre-trained models folder'
    )
    
    parser.add_argument(
        '-pd',
        '--path_data',
        type=str,
        default='data/',
        help='path to data folder'
    )

    args, unknown = parser.parse_known_args()

    # Extract config file data as dictionaries
    if args.config_file is not None:
        if '.json' in args.config_file:
            general = json.load(open(args.config_file))['GENERAL']
            data_params = json.load(open(args.config_file))['DATA_PARAMS']
            nn_params = json.load(open(args.config_file))['NN_PARAMS']

    return args, general, data_params, nn_params

#%% Testing

if __name__ == "__main__":

    all_args = cli()
    for arg in all_args:
        print(arg)
        