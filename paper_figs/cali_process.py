#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:29:35 2021

@author: jimmytabet
"""

#%% imports
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from use_cases.mapping_demo import utils

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#%% set up figure
fig = plt.figure()
frame = 30

#%% plot 2d views
bot = f'/Users/jimmytabet/NEL/Projects/Behavior/Paper Figures/calibration/images_for_mov_recal_5_cam_1/000{frame}_frame.png'
fl = f'/Users/jimmytabet/NEL/Projects/Behavior/Paper Figures/calibration/images_for_mov_recal_5_cam_2/000{frame}_frame.png'
fr = f'/Users/jimmytabet/NEL/Projects/Behavior/Paper Figures/calibration/images_for_mov_recal_5_cam_3/000{frame}_frame.png'

# draw red star on tip of micromanipulator
for num, image in enumerate([fl, fr, bot]):
    # plt.pause(0.1)
    im = cv2.imread(image)
    ax = fig.add_subplot(2,2,num+1)
    # plt.pause(0.001)
    # plt.figure()
    ax.set_yticks([])
    ax.set_xticks([])
    ax.imshow(im)
    pt = plt.ginput()[0]
    ax.scatter(*pt, marker='*', c='r', s=75, label = '_'*num+'2D point')

#%% define wheel endpoints and radius for plotting
data = pd.read_csv('/Users/jimmytabet/Software/Behavior3D/use_cases/mapping_demo/bh3D_demo_recon.csv')

lpt = data[[col for col in data.columns if col[0] == 'L' and col[-3] == 'w']]
rpt = data[[col for col in data.columns if col[0] == 'R' and col[-3] == 'w']]
lmean = lpt.mean().to_numpy()
rmean = rpt.mean().to_numpy()
mean = np.row_stack([lmean,rmean])
mean = np.mean(mean, axis=0)
R = 216/2
L = 89
mean[0] += -1.5
mean[2] += -R+1
mean[1] += 27
pt1 = mean.copy()
pt2 = mean.copy()
pt1[0] -= L/2
pt2[0] += L/2
pt1[2] += 4

#%% plot wheel and 3d pt
# wheel
ax = fig.add_subplot(224, projection='3d')
ax.set_title('Reconstructed Point')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
X, Y, Z = utils.xyz_wheel(pt1, pt2, R)
ax.plot_surface(X, Y, Z, color='white')

# 3d pt
tp = '/Users/jimmytabet/NEL/Projects/Behavior/Paper Figures/calibration/true_pts.csv'
tp_pts = pd.read_csv(tp)

# ax.plot((-30,30), (50.8,50.8), (19,19), c='k', lw=3)
# ax.scatter(30, 50.8, 19, c='r', s=50, marker='*')
ax.plot((-tp_pts.iloc[frame,0],tp_pts.iloc[frame,0]), 
        (tp_pts.iloc[frame,1],tp_pts.iloc[frame,1]),
        (tp_pts.iloc[frame,2],tp_pts.iloc[frame,2]), c='k', lw=3)
ax.scatter(tp_pts.iloc[frame,0], tp_pts.iloc[frame,1], tp_pts.iloc[frame,2], c='b', s=75, marker='*', label='reconstructed 3D point')

ax.set_xlim([0, 50])
ax.set_ylim([0, 110])
ax.set_zlim([-10, 30])

ax.set_axis_off()

# ax.legend()

#%% clean up
axs = fig.get_axes()
# fig.legend(ncol=2, loc='upper center')
fig.suptitle('Calibration Process', size=20)

axs[0].set_title('Front Left Camera')
axs[1].set_title('Front Right Camera')
axs[2].set_title('Bottom Camera')
fig.show()

#%% save
# fig.savefig('/Users/jimmytabet/Software/Behavior3D/paper_figs/cali_process.pdf', dpi=300, transparent=True)