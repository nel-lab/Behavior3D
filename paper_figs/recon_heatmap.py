#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 13:40:26 2021

@author: jimmytabet
"""

#%% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% load data and MWT
file = '/Users/jimmytabet/Software/Behavior3D/use_cases/mapping_demo/bh3D_demo_recon.csv'

data = pd.read_csv(file)

new_cols = []
for col in data.columns:
    split = col.split('_')
    split[0] = split[0][0]
    if len(split) == 2:
        split.insert(1, 'C')
    new_cols.append(''.join(split))
    
data.columns = new_cols

data = data[[
            'LCX', 'LCY', 'LCZ',
              'LD1X', 'LD1Y', 'LD1Z',
              'LD2X', 'LD2Y', 'LD2Z',
              'LD3X', 'LD3Y', 'LD3Z',
              'LD4X', 'LD4Y', 'LD4Z',
             'RCX', 'RCY', 'RCZ',
              'RD1X', 'RD1Y', 'RD1Z',
              'RD2X', 'RD2Y', 'RD2Z',
              'RD3X', 'RD3Y', 'RD3Z',
              'RD4X', 'RD4Y', 'RD4Z'
             ]]

# from behavelet import wavelet_transform
# freqs, power, X_new = wavelet_transform(data.values, n_freqs=25, fsample=70, fmin=1, fmax=35)
    
#%% plot spectographic data
plt.imshow(data.T, aspect='auto', interpolation=None)

plt.yticks(ticks = np.arange(0, data.shape[1]),
           labels = data.columns)

plt.ylabel('Paws')

plt.xticks(ticks = np.arange(0, len(data), 70*10),
           labels = (np.arange(0, len(data)/70, 10)).astype(int))

plt.xlabel('Time (s)')

# plt.tick_params(axis='both', which='both', length=0)