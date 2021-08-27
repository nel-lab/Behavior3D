#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:17:49 2021

@author: jimmytabet
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

labels = []
def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
    
small = np.load('/home/nel-lab/Desktop/Jimmy/BehaviorPaper/240_320_187fps.npy')
large = np.load('/home/nel-lab/Desktop/Jimmy/BehaviorPaper/480_640_75fps.npy')

add_label(plt.violinplot(large, showmeans=True, showextrema=False), '75Hz@640x480')
add_label(plt.violinplot(small, showmeans=True, showextrema=False), '187Hz@320x240')
plt.xticks([1,2,3,4], ['camera 1', 'camera 2', 'camera 3', 'camera 4'])
plt.ylabel('Framerate (Hz)')
plt.yticks([50, 75, 100, 150, 187, 200, 250, 300])
plt.title("Acquisition Speed for 4 Camera Setup")
plt.legend(*zip(*labels))
plt.ylim([60,240])

#%% ADJUST FIG THEN SAVE
plt.savefig('/home/nel-lab/Desktop/Jimmy/BehaviorPaper/times.pdf', transparent=True, dpi=300)

#%%
print(np.quantile(small, 0.05))
print(np.quantile(large, 0.05))

#%%
import pandas as pd
print('small')
for dat in [small, large]:
    df = pd.DataFrame(dat, columns=list('1234'))
    Q1 = df.quantile(0.01)
    Q3 = df.quantile(0.99)
    IQR = Q3 - Q1
    print('outlier %', 100*(((df < (Q1)) | (df > (Q3))).sum()/len(dat)).mean())
    print('mean:', dat.mean())
    print()
    print('large')
    
#%% time acquisition
# dat = plt.violinplot(1/np.diff(times[:,:,0], axis=0), showmeans=True, showextrema=False)

# for j in range(4):
#     plt.subplot(2,2,j+1)
#     for i in movie[:50,j,:,:]:        
#         plt.cla()
#         plt.imshow(i, cmap='gray')
#         plt.pause(0.03)

# np.save('/home/nel-lab/Desktop/Jimmy/BehaviorPaper/480_640_75fps',1/np.diff(times[:,:,0], axis=0))
