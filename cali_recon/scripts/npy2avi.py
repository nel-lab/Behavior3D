#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 00:39:47 2021

@author: jimmytabet

This script converts npz files to avi.
"""

#%% imports
import numpy as np
import shutil
import os
import cv2
from datetime import date

#%% init
path = '/home/nel-lab/Desktop/cali_recon'
rootDirectory = path
# rootDirectory = 'where your video is'
os.chdir(rootDirectory)

#%% if you have npz files for each camera that you would like to convert to 
#   a video or get individual images of each frame, use this cell and modify 
#   takes to match the number of views you have 

fps = 70 
num_cameras = 3
animal = 'mouse_5_'
video_name = animal + str(date.today()).replace('-', '_') + '_120_LARGE_habituationvideomovie'
print(video_name)

#%%
for i in range(num_cameras): 
    video_name = video_name
     
    fname = video_name + str(i)+'.npz'
    with np.load(fname) as ld:
        print(list(ld.keys()))
        movie = ld['movie']


    imageDirectory = rootDirectory + '/images_for_'+ video_name + str(i)
    if os.path.isdir(imageDirectory) == False:
        os.mkdir(imageDirectory)
        
    os.chdir(imageDirectory)
    
    for idx,frame in enumerate(movie):
            cv2.imwrite('fr_'+str(idx).zfill(5) + '.png',frame)  
   
    image_folder = imageDirectory
    video_name2 =  video_name + str(i)+'.avi'
    
    #numFiles = len([img for img in os.listdir(image_folder) if img.endswith(".png")])
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(video_name2, 0, fps, (width, height))
    
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    shutil.move(video_name2, rootDirectory)
       
    video.release()
    
    os.chdir(rootDirectory)

#%%
# if you already have a folder containing the images and just need to create a video
rootDirectory = '/Applications/NoniCloud Desktop/GiovannucciLab/Results Paper/'
video_name = 'Rotating_3D_HP'
imageDirectory = rootDirectory + 'mov_frames'
os.chdir(imageDirectory)
image_folder = imageDirectory
video_name2 =  video_name + '.avi'

#numFiles = len([img for img in os.listdir(image_folder) if img.endswith(".png")])
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name2, 0, 20, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

shutil.move(video_name2, rootDirectory)
   
video.release()

os.chdir(rootDirectory)

#^^ if you converted npy to png, convert png's to avi video

#%%
#    import pdb
#    pdb.set_trace()
