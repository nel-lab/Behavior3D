#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 22:33:54 2021

@author: jimmytabet
"""

#%% imports
import cv2
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#%% define movie paths
bot = '/Users/jimmytabet/Software/Behavior3D/demo_DLC_vids/bot_raw.mp4'
fl = '/Users/jimmytabet/Software/Behavior3D/demo_DLC_vids/fl_raw.mp4'
fr = '/Users/jimmytabet/Software/Behavior3D/demo_DLC_vids/fr_raw.mp4'
recon = '/Users/jimmytabet/Software/Behavior3D/use_cases/mapping_demo/3D_animation.mp4'

#%% set up figure
fig = plt.figure()
frame = 1700

#%% get frame from each movie
for num, path in enumerate([fl, fr, bot]):
    cap = cv2.VideoCapture(path)
    cap.set(1, frame)
    ret, fr = cap.read()
    cap.release()
    
    ax = fig.add_subplot(2,2,num+1)
    ax.imshow(fr)
    ax.axis('off')
    # ax.set_yticks([])
    # ax.set_xticks([])
    

#%% clean up
axs = fig.get_axes()
fig.suptitle('3D DLC Mapping', size=20)

axs[0].set_title('Front Left Camera')
axs[1].set_title('Front Right Camera')
axs[2].set_title('Bottom Camera')
axs[3].set_title('3D Reconstruction')
fig.show()

#%% save
fig.savefig('/Users/jimmytabet/Downloads/dsa.pdf', dpi=300, transparent=True)
