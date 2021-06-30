#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 13:46:56 2021

@author: jimmytabet

This script allows for behavior movie acquisition from multiple camera angles. 
After acquiring the movies, the user will be prompted to label each camera view. 
Movies are saved as npz files for each camera with the movie and timestamps. Movies 
are also saved seperately in the user-specified format (.avi, .mp4, .mov, etc.). 
It is best run in blocks via the Spyder IDE or imported to a Jupyter Notebook.

A short example is provided within this script. Paths should be updated to reflect 
the local path of associated files in the use_cases/acquisition folder of the 
Behavior3D repo.

You may need to run the following in terminal to activate usb cameras (Linux):
    sudo chmod o+w /dev/bus/usb/001/*
    sudo chmod o+w /dev/bus/usb/002/*
    sudo chmod o+w /dev/bus/usb/003/*
    
Note: the matplotlib backend has been explicitly set to TkAgg in this script 
(line 36).
"""

#%%
from pseyepy import Camera, Display

import cv2
import numpy as np
import pandas as pd

import matplotlib
# use TKAgg backend
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#%% save_video function
def save_video(file_name, movie, f_rate):
    '''
    Convert numpy array movie to avi.

    Parameters
    ----------
    file_name : str
        Path to save video. Must include extension/format (.avi, .mp4, .mov).
    movie : numpy array
        Movie as numpy array, size should be num_frames x camera_height x camera_width.
    f_rate : int
        Frame rate of the video.

    Returns
    -------
    None. Saves video to save_path.

    '''

    # init montage video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(file_name, fourcc, f_rate, movie[0].shape[::-1])
    
    # save each frame (3 channels needed) to montage
    for frame in movie:
        video.write(cv2.merge([frame]*3))
    
    # release when done
    video.release()

#%% setup
'''
update paths to local paths in repo!
'''

# path to model_coordinates path created in step 2 (labeling)
model_coords_path = 'path/to/use_cases/labeling/model_coordinates.csv'
num_cameras = 3
camera_fps = 70
num_frames = 210

'''
camera-specific labels will be added to the base name when saving the movies
for example, '(base_name)_(label0).(video_format)',
             '(base_name)_(label1).(video_format)', etc.
'''

base_name = 'path/to/use_cases/acquisition/acquisition_demo'
video_format = 'avi' # avi, mp4, mov

#%% initialize connected cameras
c = Camera(list(range(num_cameras)),
           fps=[camera_fps]*num_cameras,
           resolution=[Camera.RES_LARGE]*num_cameras,
           colour=[False]*num_cameras)

#%% display cameras, exit screen when ready to start
d = Display(c)

#%% record from all cameras
frames, timestamps = c.read()
frame_size = frames[0].shape

# init movie array
movie = np.zeros([num_frames, num_cameras, frame_size[0], frame_size[1]], dtype=np.uint8)

# init timestamps
times = np.zeros([num_frames, num_cameras])

# capture video in all cameras
for i in range(num_frames):
    frames, timestamps = c.read()
    movie[i] = np.array(frames)
    times[i] = np.array(timestamps)
    
#%% label each camera
# print camera labels and order to help with model and DLCPaths below
old_labels = pd.read_csv(model_coords_path).columns
old_labels = [opt[:-2] for opt in old_labels[:-3][::2]]
print('Previous camera labels (for reference when labeling behavior videos):\n', old_labels)

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

#%% save videos to npz
file_names = [f'{base_name}_{lab}.npz' for lab in cam_labels]

for num, name in enumerate(file_names):
    np.savez(name, movie=movie[:,num], times=times[:,num])

#%% save videos in user-specified format
file_names = [f'{base_name}_{lab}.{video_format}' for lab in cam_labels]

for num, name in enumerate(file_names):
    save_video(file_name=name, movie=movie[:, num], f_rate=camera_fps)