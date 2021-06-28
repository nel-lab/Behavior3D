#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 20:52:24 2021

@author: jimmytabet
"""

#%% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

#%% digit distance functions (calculate and graph)
def digit_dist(df):
    '''
    Calculate distance from paw center to digits.

    Parameters
    ----------
    df : pandas df
        Dataframe of tracked 3D coordinates.

    Returns
    -------
    df_dist : pandas df
        Dataframe of digit distances.
    colL : list
        List of digit distance column names for left paw.
    colR : list
        List of digit distance column names for right paw.

    '''

    # seperate into paw base and digits
    baseL = [col for col in df.columns if col[0] == 'L' and col[5] != 'D']
    baseR = [col for col in df.columns if col[0] == 'R' and col[5] != 'D']
    digitL = [col for col in df.columns if col[0] == 'L' and col[5] == 'D']
    digitR = [col for col in df.columns if col[0] == 'R' and col[5] == 'D']

    # create new df and empty col lists
    df_dist = pd.DataFrame()
    colL = []
    colR = []

    # iterate over L/R paws
    for b,d,c in [[baseL, digitL, colL], [baseR, digitR, colR]]:
        # iterate over each digit in paw
        for i in range(len(digitL)//3):
            base = df[b]
            digit = df[d].iloc[:,3*i:3*i+3]
            c.append(digit.columns[0][:-2]+'_dist')
            base.columns=digit.columns = ['x','y','z']
            diff = base-digit
            df_dist[c[i]] = np.linalg.norm(diff, axis=1)
    
        df_dist[b[0][:-2]+'_sum_dist'] = df_dist[c].sum(axis=1)

    return df_dist, colL, colR

def dist_graph(df):
    '''
    Plot distance from paw center to digits.

    Parameters
    ----------
    df : pandas df
        Dataframe of digit distances to be plotted.

    Returns
    -------
    Plot of digit ditances. There are 5 subplots:
        1. cumulative digit distance of D1-D4 over time
        2-5. seperate plots of digit distance for D1-D4 over time

    '''

    # reorder: cum dist, D1-4 dist
    if df.columns[0][5] == 'D':
        cols = list(df.columns)
        cols = [cols[-1]]+cols[:-1]
        df = df[cols]
    else:
        cols = df.columns

    # set color scheme for left vs right paw
    if df.columns[0][0] == 'L':
        paw = 'C0'
        cmap = cm.Blues(np.linspace(.25,.75,4))
    elif df.columns[0][0] == 'R':
        paw = 'C1'
        cmap = cm.Oranges(np.linspace(.25,.75,4))
    else:
        raise ValueError('error, make sure digit distance is calcuated using digit_dist function')

    # plot all 5 subplots
    fig, ax = plt.subplots(5, sharex=True)
    for i in range(len(ax)):
        if i == 0:
            ax[i].plot(df.iloc[:,i], c=paw)
        else:
            ax[i].plot(df.iloc[:,i], c=cmap[i-1])

        ax[i].title.set_text(cols[i][:-5])
       
        if i > 0:
            ax[i].set_ylim([0,25])

    fig.tight_layout()
    
#%% read in 3D reconstruction
data = pd.read_csv('3D_reconstruction.csv')

#%% digit distances
df_digit_dist, colL, colR = digit_dist(data)

# overlay left and right paw cumulative distances over time
plt.plot(df_digit_dist['Lpaw_sum_dist'], c='C0', label= 'Left Paw', alpha=0.7)
plt.plot(df_digit_dist['Rpaw_sum_dist'], c='C1', label = 'Right Paw', alpha=0.7)
plt.legend()
plt.title('(Cumulative) Digit Distance from Paw Center')

# optional, zoom in on snippet where mouse is walking to see alternating left and right paw
# plt.ylim([0,30])
# plt.xlim([3500,4000])

# optionally, zoom in on rot/walk snippet
# plt.xlim([1300,2400])

# plt.savefig('mouse_filter_pawdist')

#%% overlay all digits
df_digit_dist[colL].plot()

#%% graph seperate digits and sum
Ldist = df_digit_dist[[col for col in df_digit_dist if col[0]=='L']]
Rdist = df_digit_dist[[col for col in df_digit_dist if col[0]=='R']]
dist_graph(Ldist)
dist_graph(Rdist)

#%% align with y coordinate to show swing
lpawy = data[[col for col in data.columns if col[0]=='L' and col[-3:] == 'w_Y']]
rpawy = data[[col for col in data.columns if col[0]=='R' and col[-3:] == 'w_Y']]

plt.plot(lpawy, c='k', label = 'Left Paw Y', alpha=0.7)
plt.plot(df_digit_dist['Lpaw_sum_dist'], c='C0', label='Left Paw Total Dist', alpha=0.7)

plt.plot(rpawy, c='y', label = 'Right Paw Y', alpha=0.7)
plt.plot(df_digit_dist['Rpaw_sum_dist'], c='C1', label = 'Right Paw Total Dist', alpha=0.7)
plt.legend()
plt.title('Swing and Digit Distance Correlation')
# plt.savefig('y_swing_dist')