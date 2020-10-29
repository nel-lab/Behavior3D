#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:47:31 2019

@author: williamstanford
conda activate pseye
sudo chmod o+w /dev/bus/usb/001/*
sudo chmod o+w /dev/bus/usb/002/*
"""

'''
This script enables you to capture images that you can use to calibrate your cameras 

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
from pseyepy import Camera, Display
import numpy as np
import matplotlib.pyplot as plt
import os 

path = '/home/nel-lab/Desktop/Behavior3D'
os.chdir(path)

# initialize all connected cameras
#%% input number of cameras
num_cameras = 3
c = Camera(list(range(3)), fps=[70]*num_cameras, resolution=[Camera.RES_LARGE]*num_cameras, colour=[False]*num_cameras)

#%% display movies
d = Display(c)

#%% when finished, close the camera
d.end()

#%% initializes movie
frames, timestamps = c.read()
frame_size = frames[0].shape
num_frames = 10
mov = np.zeros([num_frames, num_cameras, frame_size[0], frame_size[1]], dtype=np.uint8)

#%%
for i in range(num_frames): 
    frames, timestamps = c.read()
    frames, timestamps = c.read()
    mov[i] = np.array(frames)
    plt.imshow(mov[i,1,:,:])
    plt.title("Calibration point " + str(i))
    plt.pause(0.001)
    input("Press enter to view image")

#%% Check each camera below - note the order if you haven't already
    # Type camera order here: 
for camera in range(num_cameras):
    plt.imshow(mov[i,camera,:,:])
    plt.title("Camera " + str(camera)+", Image " + str(i))    
    plt.pause(0.001)
    input("Press enter to continue")

#%%
base_name = 'recal_3_cam_'
filenames = ['0.npz', '1.npz', '2.npz']

for idx,fls in enumerate(filenames):
    np.savez("mov_"+base_name+fls, movie=mov[:,idx,:,:])
   
    
#%%
# when finished, close the camera
c.end()