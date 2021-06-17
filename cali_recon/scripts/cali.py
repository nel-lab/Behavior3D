#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:03:49 2021

@author: jimmytabet

This script enables you to capture images that you can use to calibrate your cameras.
The set of images for each camera is saved as a seperate npz file.
It is best run in blocks via Spyder or imported to a Jupyter Notebook.
"""

#%% imports
from pseyepy import Camera, Display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% setup/instructions
'''
create realPoints.csv file with planned real X,Y,Z coordinates
    should be in form:
        X   Y   Z
        0   0   0
        0   0  10
        0   0  20
        ...

run in terminal to activate usb cameras (Linux):
    sudo chmod o+w /dev/bus/usb/001/*
    sudo chmod o+w /dev/bus/usb/002/*
    sudo chmod o+w /dev/bus/usb/003/*
    
things to keep in mind
    1. Plan ahead when condsidering your real world coordinate system
        - moving something from one peg to another is approximately 25.4mm 
        - with our micromanipulator, you have approximately 40 milimeters of horizontal freedom
            and 30 milimeters of vertical freedom
    2. regular increments make it easier to tell if you're off or missed a picture
        - Suggestions - 5, 8, or 10mm horizontally, 5, or 10mm vertically
    3. Save your planned real coordinates as an ods file with the X,Y,Z coordinates
        - see realPoints.ods as an example. 
        - make sure you have a space between the beginning of your cooridnates and the 
             top of the ods sheet
    3. Make sure to note the order of your cameras at this stage - you need to keep
        this consistent in later parts of processing
'''

realPoints_path = '/home/nel-lab/Desktop/cali_recon/realPoints.csv'
num_cameras = 3
camera_fps = 70
filenames = ['cali_test_cam0', 'cali_test_cam1', 'cali_test_cam2']
file_all = 'cali_test_all'

assert len(filenames) == num_cameras

#%% initialize connected cameras
c = Camera(list(range(num_cameras)),
           fps=[camera_fps]*num_cameras,
           resolution=[Camera.RES_LARGE]*num_cameras,
           colour=[False]*num_cameras)

#%% display cameras
d = Display(c)

#%% when finished, close the camera (may already be closed/destroyed)
d.end()

#%% read in realPoints.csv to see how many frames are needed
realPoints = pd.read_csv(realPoints_path)

#%% initializes movie
frames, timestamps = c.read()
frame_size = frames[0].shape
num_frames = len(realPoints)
mov = np.zeros([num_frames, num_cameras, frame_size[0], frame_size[1]], dtype=np.uint8)

#%% caputre calibration snapshots
input(f'Move to first calibration point {tuple(realPoints.iloc[0])} and press enter to begin')
# init plt.imshow
plt.imshow([[0]])
plt.pause(.1)
# for each frame...
for i in range(num_frames):
    # capture snapshot... (must be written twice)
    frames, timestamps = c.read()
    frames, timestamps = c.read()
    # add to mov...
    mov[i] = np.array(frames)
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

#%% check each camera below - note the order if you haven't already
# init plt.imshow
cam_labels = []
plt.imshow([[0]])
plt.pause(.1)
for camera in range(num_cameras):
    plt.imshow(mov[num_frames-1,camera,:,:], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Camera {camera}')
    plt.pause(0.001)
    label = input('Enter label for camera view (i.e. "cam0/1/2", "FL/FR/BOT", etc.): ')
    cam_labels.append(label)
    
plt.close('all')

#%% save individual movies as npz files
#for idx, fls in enumerate(filenames):
#    np.save(fls, mov[:,idx,:,:])

np.savez(file_all, movie=mov, labels=cam_labels)

#%% close the camera
c.end()

#%% (view results, for debugging only, remove later)
for i in range(num_cameras):
    for j in range(num_frames):
        plt.imshow(mov[j,i,:,:], cmap='gray')
        plt.pause(0.5)