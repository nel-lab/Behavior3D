#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 13:46:56 2021

@author: jimmytabet
"""

#%% imports and init cameras
# imports
from pseyepy import Camera, Display
import numpy as np
import time

# init cameras
num_cameras = 5
camera_fps = 50#70
num_frames = 120*50#5*60*70
c = Camera(list(range(num_cameras)),
           fps=[camera_fps]*num_cameras,
           resolution=[Camera.RES_LARGE]*num_cameras,
           colour=[False]*num_cameras)
    
#%% acquisition
# user settings
mouse = 2
trial = 2
file_name = '/home/nel-lab/Desktop/Jimmy/new_cali_cube/trial1'#f'/home/nel-lab/Desktop/Sophia/0917/mouse{mouse}_trial{trial}'

set_cam = True
record = True
quick_start = True

# set cameras
if set_cam:
    # display/edit
    d = Display(c)

# record
if record:
    # get frame size for array
    frames, timestamps = c.read()
    frame_size = frames[0].shape
    
    # init movie and times array
    movie = np.zeros([num_frames, num_cameras, frame_size[0], frame_size[1]], dtype=np.uint8)
    times = np.zeros([num_frames, num_cameras,2])
    
    # prompt to begin recording
    if not quick_start:
        input('Press enter to begin recording!')
        
    # capture video in all cameras
    print('recording...')
    s = time.time()    
    time_0 = time.time() 
    for i in range(num_frames):
        frames, timestamps = c.read()
        movie[i] = np.array(frames)
        times[i,:,0] = np.array(timestamps)
        times[i,:,1] = time.time()
    print(f'recording time = {time.time()-s}')
    
    # save npz file
    s = time.time()    
    print('saving')
    np.savez(file_name, movie=movie, times=times)
    print(f'saving time = {time.time()-s}')