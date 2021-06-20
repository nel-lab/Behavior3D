#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 13:46:56 2021

@author: jimmytabet

This script allows for behavior movie acquisition from multiple camera angles

Before starting, you may need to run the following in terminal to activate usb cameras:
    sudo chmod o+w /dev/bus/usb/001/*\n
    sudo chmod o+w /dev/bus/usb/002/*\n
    sudo chmod o+w /dev/bus/usb/003/*
"""

#%%
from pseyepy import Camera, Display
import cv2, time
import numpy as np
import matplotlib.pyplot as plt

#%% save to avi function
def save_video_avi(file_name, movie, f_rate):
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
# path to model_coordinates path created in step 2 (labeling)
model_coord_path = 'cali_recon/model_coordinates.csv'
num_cameras = 3
camera_fps = 70
num_frames = 5000
# camera-specific labels will be added to this base name when saving avi movies
# for example, '(base_name)_(label0).avi', '(base_name)_(label1).avi', etc.
base_name = 'cali_test'

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
# times = np.zeros([num_frames, num_cameras])

# capture video in all cameras
# time0 = time.time()
for i in range(num_frames):
    frames, timestamps = c.read()
    movie[i] = np.array(frames)
    # times[i] = np.array(timestamps)
    
# print(time.time()-time0)

#%% label each camera
print('Previous camera labels (for reference when labeling behavior videos):\n', np.load(model_coord_path)['labels'])

cam_labels = []
# init plt.imshow
plt.imshow([[0]])
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
# file_names = [f'{base_name}_{lab}.avi' for lab in cam_labels]

# for num, name in enumerate(file_names):
#     np.savez(name, movie=movie[:,num], times=times[:,num])

#%% save videos to avi
file_names = [f'{base_name}_{lab}.avi' for lab in cam_labels]

for num, name in enumerate[file_names]:
    save_video_avi(name, movie[:, num], f_rate=camera_fps)