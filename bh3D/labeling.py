#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 00:41:43 2021

@author: jimmytabet

This script allows user to select 2D calibration point in each camera for each frame.
User should click on same reference point in each frame/angle (for example, tip of
micromanipulator). The reference point should be visible in all cameras in each frame. 
It is best run in blocks via the Spyder IDE.

A short example is provided within this script. Paths point to associated output 
files in the use_cases/labeling folder of the Behavior3D repo. All paths are 
relative to this script's location in the Behavior3D repo.
"""

#%% imports
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#%% laebl_images function
def label_images(movie, labels, realPoints):
    '''
    Label calibration points in each camera for each frame.

    Parameters
    ----------
    movie : numpy array
        Calibration frames from file created in step 1 (calibration). Size of
        array is num_frames x num_cameras x camera_height x camera_width.
    labels : list
        Camera label/description as input in step 1 (calibration).
    realPoints : pandas df
        Dataframe (read from realPoints csv) that outline X, Y, Z coordinates 
        of calibration points.

    Returns
    -------
    coords_and_realPoints : pandas df
        Dataframe with 2D calibration points for each camera and corresponding 
        3D real point Size of dataframe is num_frames x (2*num_cameras + 3).

    '''

    # init coords
    coords = np.zeros([len(movie), 2*len(labels)])

    # loop through each camera and each frame to label calibration point
    for cam in range(len(labels)):
        for frame in range(len(movie)):
            # index into specific camera and frame
            im = movie[frame, cam, :, :]
            # show frame
            plt.imshow(im, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title(f'Cam: {labels[cam]}, Frame: {frame}')
            plt.pause(0.001)
            # save coordinate from user input
            coords[frame,2*cam:2*cam+2] = plt.ginput()[0]

    plt.close('all')
    
    # generate column labels for coords dataframe
    col_lab = []
    for lab in labels:
        col_lab.append(lab+'_x')
        col_lab.append(lab+'_y')
    
    # coords df
    coords_df = pd.DataFrame(coords, columns = col_lab)
    
    # merge coords df with realPoints df
    coords_and_realPoints = pd.concat([coords_df, realPoints], axis=1)
    
    return coords_and_realPoints
                  
#%% set up/load data
realPoints_path = '../use_cases/calibration/realPoints.csv'
cali_npz_path = '../use_cases/calibration/calibration_demo.npz'
model_coords_path = '../use_cases/labeling/model_coordinates.csv'

realPoints = pd.read_csv(realPoints_path)

# load calibration frames and camera labels from npz file generated in step 1 (calibration)
with np.load(cali_npz_path) as f:
    movie = f['movie']
    camera_labels = f['labels']

#%% label calibration points
coords = label_images(movie, camera_labels, realPoints)

#%% save as csv
coords.to_csv(model_coords_path, index=False)