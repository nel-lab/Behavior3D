#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:03:49 2021

@author: jimmytabet

This script enables you to capture images that you can use to calibrate your cameras. 
The set of images for each camera and user-defined camera labels are saved as one npz file.
It is best run in blocks using the Spyder IDE, but can also be imported to a 
Jupyter Notebook or run in terminal ('python /path/to/calibration.py').

A short example is provided within this script. Paths should be updated to reflect 
the local paths of associated files in the use_cases/calibration folder of the 
Behavior3D repo.

Instructions:
    1. create realPoints.csv file with planned real 3D coordinates, should be in form:
        X   Y   Z
        0   0   0
        10  0   0
        20  0   0
        ...

    2. you may need to run the following in terminal to activate usb cameras (Linux):
        sudo chmod o+w /dev/bus/usb/001/*
        sudo chmod o+w /dev/bus/usb/002/*
        sudo chmod o+w /dev/bus/usb/003/*
        
    3. the script will walk you through the calibration snapshots (in 'capture 
       calibration snapshots' cell), but plan ahead to make sure ALL real calibration 
       coordinates can be seen in EVERY camera!
       
Note: The matplotlib backend may need to be changed. Running 

%matplotlib auto

in the IPython console usually does the trick.
"""

#%% imports
from pseyepy import Camera, Display

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#%% setup
'''
DON'T FORGET TO UPDATE PATHS!
'''


realPoints_path = 'path/to/use_cases/calibration/realPoints.csv'
# path/name of npz file to save (will contain calibration images)
cali_npz_save_path = 'path/to/use_cases/calibration/calibration_demo.npz'



num_cameras = 3
camera_fps = 70

#%% initialize connected cameras
c = Camera(list(range(num_cameras)),
           fps=[camera_fps]*num_cameras,
           resolution=[Camera.RES_LARGE]*num_cameras,
           colour=[False]*num_cameras)

#%% display cameras, exit screen when ready to start
'''
IF USING MACOS, you may need to comment this line out. See README for details.
'''

d = Display(c)

#%% initialize movie
# read in realPoints.csv to see how many frames are needed
realPoints = pd.read_csv(realPoints_path)

# init movie
frames, timestamps = c.read()
frame_size = frames[0].shape
num_frames = len(realPoints)
movie = np.zeros([num_frames, num_cameras, frame_size[0], frame_size[1]], dtype=np.uint8)

#%% capture calibration snapshots
input(f'Move to first calibration point {tuple(realPoints.iloc[0])} and press enter to begin')
# init plt.imshow for Spyder
plt.imshow([[0]], cmap='gray')
plt.pause(.1)
# for each frame...
for i in range(num_frames):
    # capture snapshot... (must be written twice)
    frames, timestamps = c.read()
    frames, timestamps = c.read()
    # add to movie...
    movie[i] = np.array(frames)
    # display to ensure it is in view...
    plt.imshow(frames[0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    # hit enter to move to next frame
    if i == num_frames-1:
        plt.title(f'Calibration point {i}: {tuple(realPoints.iloc[i])}\nPress enter to exit')
        plt.pause(0.001)
        input('Press enter to exit')
    else:
        plt.title(f'Calibration point {i}: {tuple(realPoints.iloc[i])}\nNext point: {tuple(realPoints.iloc[i+1])}')
        plt.pause(0.001)
        input(f'Move to next point {tuple(realPoints.iloc[i+1])}, then press enter to continue')
        
plt.close('all')

#%% label each camera
cam_labels = []
# init plt.imshow for Spyder
plt.imshow([[0]], cmap='gray')
plt.pause(.1)
for camera in range(num_cameras):
    plt.imshow(movie[num_frames-1,camera,:,:], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Camera {camera}')
    plt.pause(0.001)
    # input camera label
    label = input('Enter label for camera view (i.e. "cam0/1/2", "FL/FR/BOT", etc.): ')
    cam_labels.append(label)
    
plt.close('all')

#%% save movie and camera labels as npz file
np.savez(cali_npz_save_path, movie=movie, labels=cam_labels)

#%% close the camera
c.end()
