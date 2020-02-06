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

path = '/home/nel-lab/Desktop/Cynthia'
os.chdir(path)

# initialize all connected cameras
#%%
c = Camera([0, 1, 2], fps=[70,70,70], resolution=[Camera.RES_LARGE, Camera.RES_LARGE, Camera.RES_LARGE], colour=[False, False, False])

#%% display movies
d = Display(c)

#%%
# when finished, close the camera
c.end()

#%%
frames, timestamps = c.read()
num_cameras = len(frames)
frame_size = frames[0].shape
num_frames = 10
mov = np.zeros([num_frames, num_cameras, frame_size[0], frame_size[1]], dtype=np.uint8)

#%%
i = 0
#%%
i = i + 1
#%%

if i<num_frames:
    frames, timestamps = c.read()
    mov[i] = np.array(frames)
    plt.imshow(mov[i,1,:,:])
    plt.title(str(i))
    
if i<num_frames:
    frames, timestamps = c.read()
    mov[i] = np.array(frames)
    plt.imshow(mov[i,1,:,:])
    plt.title(str(i))

#%% Check each camera below - note the order if you haven't already
    # Camera order: 
#%%
plt.imshow(mov[i,0,:,:])
plt.title(str(i))
                                         #%%
plt.imshow(mov[i,1,:,:])
plt.title(str(i))
#%%
plt.imshow(mov[i,2,:,:])
plt.title(str(i))
#%%
plt.imshow(mov[i,3,:,:])
plt.title(str(i))
#%%
plt.imshow(mov[i,4,:,:])
plt.title(str(i))

#%%
base_name = 'recal_3_cam_'
filenames = ['0.npz', '1.npz', '2.npz']

for idx,fls in enumerate(filenames):
    np.savez("mov_"+base_name+fls, movie=mov[:,idx,:,:])
   
    
#%%
# when finished, close the camera
c.end()