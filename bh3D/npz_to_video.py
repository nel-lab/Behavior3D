#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 18:17:12 2021

@author: nel-lab
"""

import cv2

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
    
    # save each frame (3 channels needed) to montageg
    for frame in movie:
        video.write(cv2.merge([frame]*3))
    
    # release when done
    video.release()
    
    
#%%
import numpy as np

fil = '/home/nel-lab/Desktop/Sophia/0824/mouse1.3_trial1.2.npz'
movie = np.load(fil)['movie']

#%%
num_cameras = 5
camera_fps = 70
num_frames = 4*60*70

import matplotlib.pyplot as plt
cam_labels = []

plt.imshow([[0]], cmap='gray')
plt.pause(.1)

for camera in range(num_cameras):
     plt.imshow(movie[1000,camera,:,:], cmap='gray')
     plt.xticks([])
     plt.yticks([])
     plt.title(f'Camera {camera}')
     plt.pause(0.001)
     label = input('Enter label for camera view (i.e. "cam0/1/2", "FL/FR/BOT", etc.): ')
     cam_labels.append(label)
    
plt.close('all')

#%%
file_names = [f'/home/nel-lab/Desktop/Sophia/0824/mouse1.3_trial1.2_{lab}.avi' for lab in cam_labels]

for num, name in enumerate(file_names):
    save_video(file_name=name, movie=movie[:, num], f_rate=camera_fps)