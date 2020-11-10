#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:35:53 2020

@author: nel-lab
"""


from bh3D.calibration import calibration
import pandas as pd
import numpy as np

#%%
#Customize this paths dictionary for your local paths. 
pathsDict = {
    'coordPath': '/Users/jimmytabet/Desktop/NEL/Projects/Calibration/My Files/My Plotting Files/coords and dlc/model_1_coordinates.csv', 
    'DLCPath': '/Users/jimmytabet/Desktop/NEL/Projects/Calibration/My Files/My Plotting Files/coords and dlc',
    'PlotPath' : '/Users/jimmytabet/Desktop/bh3d tests'
}

cal = calibration(model = ['bot','fl','fr'], pathsDict = pathsDict)
# cal.calibration_results(save=True)

print()
# raw data
raw = cal.raw_model()
# print(raw.head())

# filtered data
new = cal.map_to_3D()
# print(filtered.head())

# load version from JN
old = pd.read_csv('/Users/jimmytabet/Desktop/NEL/Projects/Calibration/My Files/My Plotting Files/trial 17/two paws XYZ data.csv')

# any difference is result of rounding, maybe because it was read from csv file
print('max diff:', np.max(np.max(new-old)))