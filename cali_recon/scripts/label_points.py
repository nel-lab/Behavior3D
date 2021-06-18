#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 00:41:43 2021

@author: jimmytabet

This script allows user to select 2D calibration point in each camera for each frame.
User should click on same reference point in each frame/angle (for example, tip of
micromanipulator). The reference point should be visible in all cameras in each frame.
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#%% laebl_images function
def label_images(mov, labels, realPoints):
    '''
    Label calibration points in each camera for each frame.

    Parameters
    ----------
    mov : numpy array
        Calibration frames from cali.py file. Size of array is 
        num_frames x num_cameras x camera_width x camera_height.
    labels : list
        Camera label/description as input in cali.py.
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
    coords = np.zeros([len(mov), 2*len(labels)])

    # loop through each camera and each frame to label calibration point
    for cam in range(len(labels)):
        for frame in range(len(mov)):
            # index into specific camera and frame
            im = mov[frame, cam, :, :]
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
mov_path = '/home/nel-lab/Software/Behavior3D/cali_recon/cali_test_all.npz'
realPoints_path = '/home/nel-lab/Software/Behavior3D/cali_recon/realPoints.csv'

#mov_path = '/Users/jimmytabet/Software/Behavior3D/cali_recon/cali_test_all.npz'
#realPoints_path = '/Users/jimmytabet/Software/Behavior3D/cali_recon/realPoints.csv'

realPoints = pd.read_csv(realPoints_path)

# load calibration frames and camera labels from npz file generated in cali.py
with np.load(mov_path) as f:
    mov = f['movie']
    camera_labels = f['labels']

#%% label calibration points
cord = label_images(mov, camera_labels, realPoints)
# save as csv
#cord.to_csv('/Users/jimmytabet/Software/Behavior3D/cali_recon/model_coordinates.csv', index=False)