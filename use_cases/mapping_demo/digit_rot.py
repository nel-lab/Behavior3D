#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 21:10:41 2021

@author: jimmytabet
"""

#%% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% angle function
def angle(df, side, digit_num):
    '''
    Calculate angle between paw center and specific digit.

    Parameters
    ----------
    df : pandas df
        Dataframe of tracked 3D coordinates.
    side : str
        'L' or 'R' to designate left or right side.
    digit_num : int
        1, 2, 3, or 4 to designate digit number.

    Returns
    -------
    degree : list
        List of angles over time.

    '''

    if side.upper() != 'R' and side.upper() != 'L':
        raise ValueError('invalid side entered: use either \'L\' or \'R\'')

    if digit_num != 1 and digit_num != 2 and digit_num != 3 and digit_num != 4:
        raise ValueError('invalid digit number entered: use either 1, 2, 3 or 4')

    base = df[[col for col in df.columns if col[0] == side.upper() and col[-3] == 'w']].to_numpy()
    digit = df[[col for col in df.columns if col[0] == side.upper() and col[-3] == str(digit_num)]].to_numpy()

    limb = digit-base
    base_yoffset = base.copy()
    base_yoffset[:,1] -= 1
    center = base_yoffset - base

    degree = []
    for v1,v2 in zip(center,limb):
        unit_v1 = v1/np.linalg.norm(v1)
        unit_v2 = v2/np.linalg.norm(v2)
        degree.append(np.arccos(np.dot(unit_v1, unit_v2))*180/np.pi)
        # degree.append(np.arccos(-unit_v2[1])*180/np.pi)

    return degree

#%% read in 3D reconstruction
data = pd.read_csv('3D_reconstruction.csv')

#%% digit rotation
# calculate digit rotation for each digit on left and right side
totl = []
totr = []
for d in range(1,5):
    totl.append(angle(data, 'l', d))
    totr.append(angle(data, 'r', d))

# calculate mean
deg_l = np.mean(totl,axis=0)
deg_r = np.mean(totr,axis=0)

# create rotation df with each digit and mean rotation for left/right
df_rot = pd.DataFrame()
for d in range(1,5):
    df_rot['Lpaw_D'+str(d)+'_rot'] = angle(data, 'l', d)

df_rot['Lpaw_mean_rot'] = deg_l

for d in range(1,5):
    df_rot['Rpaw_D'+str(d)+'_rot'] = angle(data, 'r', d)

df_rot['Rpaw_mean_rot'] = deg_r

# plot left.right paw mean rotations
plt.plot(deg_l, c='C0', label = 'Left Paw', alpha=0.7)
plt.plot(deg_r, c='C1', label = 'Right Paw', alpha=0.7)
plt.legend()
title = 'mean rot'
plt.title(title)
# optionally, zoom in on rot/walk snippet
# plt.xlim([1300,2400])

# plt.savefig(title)

#%% overlay left and right digit rotations
for i in range(4):
    plt.figure()
    plt.plot(totl[i], c='C0', label = 'Left Paw', alpha=0.7)
    plt.plot(totr[i], c='C1', label = 'Right Paw', alpha=0.7)
    plt.legend()
    title = 'D'+str(i+1)+' rot'
    plt.title(title)
    # optionally, zoom in on rot/walk snippet
    # plt.xlim([1300,2400])
    
    # plt.savefig(title)