#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:35:53 2020

@author: nel-lab
"""


from bh3D.calibration import calibration
import pandas as pd
import numpy as np
from behavelet import wavelet_transform
import pylab as plt

#%%
#Customize this paths dictionary for your local paths. 
pathsDict = {
    'coordPath': './use_cases/setup_old/model_1_coordinates.csv', 
    'DLCPath': './use_cases/setup_old/',
    'PlotPath' : './use_cases/setup_old/'
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
means = new.mean(axis=0)
new -= means
# load version from JN
#old = pd.read_csv('./use_cases/setup_old/two paws XYZ data.csv')

# any difference is result of rounding, maybe because it was read from csv file
#print('max diff:', np.max(np.max(new-old)))
#%% wavelet analysis
freqs, power, X_new = wavelet_transform(new.values, n_freqs=25, fsample=100., fmin=1., fmax=50.)
#%%
plt.subplot(2,1,1)
plt.plot(X_new[:,:3])
plt.plot(X_new[:,10:13])
plt.subplot(2,1,2)
plt.plot(new.values[:,:1]/new.values[:,:1].max())