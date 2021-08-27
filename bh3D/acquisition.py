#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 13:46:56 2021

@author: jimmytabet
"""

from pseyepy import Camera, Display
import cv2
import numpy as np
# import pandas as pd
import os
import matplotlib.pyplot as plt

import time

s = time.time()

mouse = 1.3
trial = 1.2

quick_start = False
set_cam = True
#}
##%% save_video function
# def save_video(file_name, movie, f_rate):
#     '''
#     Convert numpy array movie to avi.

#     Parameters
#     ----------
#     file_name : str
#         Path to save video. Must include extension/format (.avi, .mp4, .mov).
#     movie : numpy array
#         Movie as numpy array, size should be num_frames x camera_height x camera_width.
#     f_rate : int
#         Frame rate of the video.

#     Returns
#     -------
#     None. Saves video to save_path.

#     '''

#     # init montage video
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     video = cv2.VideoWriter(file_name, fourcc, f_rate, movie[0].shape[::-1])
    
#     # save each frame (3 channels needed) to montage
#     for frame in movie:
#         video.write(cv2.merge([frame]*3))
    
#     # release when done
#     video.release()

##%% setup
'''
DON'T FORGET TO UPDATE PATHS!
'''



# path to model_coordinates path created in step 2 (labeling)
# model_coords_path = 'path/to/use_cases/labeling/model_coordinates.csv'



num_cameras = 5
camera_fps = 70
num_frames = 4*60*70

'''
camera-specific labels will be added to the base name when saving the movies
for example, '(base_name)_(label0).(video_format)',
             '(base_name)_(label1).(video_format)', etc.
'''



base_name = 'path/to/use_cases/acquisition/acquisition_demo'
video_format = 'mp4' # avi, mp4, mov

##%% initialize connected cameras
c = Camera(list(range(num_cameras)),
           fps=[camera_fps]*num_cameras,
           resolution=[Camera.RES_LARGE]*num_cameras,
           colour=[False]*num_cameras)

##%% display cameras, exit screen when ready to start
'''
IF USING MACOS, you may need to comment this line out. See README for details.
'''
if set_cam:
    d = Display(c)
    
    ##%%
    d.end()

##%% record from all cameras
frames, timestamps = c.read()
frame_size = frames[0].shape

# init movie array
movie = np.zeros([num_frames, num_cameras, frame_size[0], frame_size[1]], dtype=np.uint8)

# init timestampscameras, exit screen when ready to start', '/home/nel-lab/Software/Behavior3D/bh3D/acquisi
times = np.zeros([num_frames, num_cameras,2])
if not quick_start:
    input('Press enter to begin recording!')
    
print('recording...')
    
# capture video in all cameras
time_0 = time.time() 
for i in range(num_frames):
    frames, timestamps = c.read()
    movie[i] = np.array(frames)
    times[i,:,0] = np.array(timestamps)
    times[i,:,1] = time.time()

print(f'recording time = {time.time()-s}')

##%% label each camera
# # print camera labels and order to help with model and DLCPaths below
# # old_labels = pd.read_csv(model_coords_path).columns
# # old_labels = [opt[:-2] for opt in old_labels[:-3][::2]]
# # print('Previous camera labels (for reference when labeling behavior videos):\n', old_labels)

#cam_labels = []
#init plt.imshow for Spyder
#plt.imshow([[0]], cmap='gray')
#plt.pause(.1)
#for camera in range(num_cameras):
     #plt.imshow(movie[num_frames-1,camera,:,:], cmap='gray')
     #plt.xticks([])
     #plt.yticks([])
     #plt.title(f'Camera {camera}')
     #plt.pause(0.001)
     # input camera label
     #label = input('Enter label for camera view (i.e. "cam0/1/2", "FL/FR/BOT", etc.): ')
     #cam_labels.append(label)
    
#plt.close('all')

##%%
# BASE_PATH = "~/Desktop/Sophia/mov_all"
s = time.time()

print('saving')

file_name = f'/home/nel-lab/Desktop/Sophia/0824/mouse{mouse}_trial{trial}'
np.savez(file_name, movie=movie, times=times)
print(f'saving time = {time.time()-s}')
#save_video('testsavevideo2', movie=movie, f_rate = camera_fps)
# #%% save videos to npz
#file_names = [f'{base_name}_{lab}.npz' for lab in cam_labels]

#for num, name in enumerate(file_names):
     #np.savez(name, movie=movie[:,num], times=times[:,num])

# #%% save videos in user-specified format
#file_names = [f'{base_name}_{lab}.{video_format}' for lab in cam_labels]

#for num, name in enumerate(file_names):
    #save_video(file_name=name, movie=movie[:, num], f_rate=camera_fps)
##%%
# for i in range(num_frames):
#     frames = movie[i,0,:]
#     # frames = cv2.cvtColor(frames, cv2.COLOR_RGB2BGR)
#     # frames_arr = np.array(frames)
#     plt.cla()
#     plt.imshow(frames, cmap='gray')
#     plt.pause(0.01)