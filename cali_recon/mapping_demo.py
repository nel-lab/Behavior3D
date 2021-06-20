#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 01:57:57 2021

@author: jimmytabet

Mapping demo file. This script takes the model_coordinates csv file generated 
in step 2 (labeling) and creates a 3D mapping. It then takes DeepLabCut 
labeled csv files and maps common bodyparts seen across all cameras to 3D.

Things to keep in mind:
    It is imperative that the order of the DLCPaths list corresponds to the order 
    of the model variable. For reference, the user-generated camera labels are 
    printed in this script. Proper naming of the behavior movies as described 
    in step 3 (acquisition) should help this process.
    
    For example, below the camera labels were defined as 'FL', 'FR', and 'BOT'. 
    The model variable is defined in the order ['BOT','FL','FR'], so it is 
    imperative that the DLCPaths are listed in this order 
    (['bot.csv', 'front_left.csv'', 'front_right.csv'])
"""

#%% imports
from cali_recon.scripts.map import mapping
import pandas as pd

#%% setup - model options
coordPath = 'use_cases/setup_old/model_1_coordinates.csv'
# coordPath = 'model_coordinates.csv'

# print camera labels and order to help with model and DLCPaths below
model_options = pd.read_csv(coordPath).columns
model_options = [opt[:-2] for opt in model_options[:-3][::2]]
print('Cameras labels to reference when defining model and DLCPaths variables below:\n', model_options)

#%% setup - define model/DLC paths
model = ['BOT','FL','FR']

DLCPaths = ['use_cases/setup_old/bot.csv',
            'use_cases/setup_old/front_left.csv',
            'use_cases/setup_old/front_right.csv']

SVR_args = {'kernel':"poly", 'degree':2, 'C':1500}

if not set(model).issubset(model_options):
    raise ValueError(f'One or more model entry is not a valid camera label {tuple(model_options)}.')

#%% init mapping class (automatically builds model)
cal = mapping(model, coordPath, DLCPaths, **SVR_args)

#%% display calibration model results
cal.calibration_results()

#%% map DLC data to 3D (also filters data)
data = cal.map_to_3D()