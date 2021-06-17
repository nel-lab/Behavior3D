#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 00:41:43 2021

@author: jimmytabet
"""

'''
CLICK ON POINT TO LABEL
presents each photo, click on same point
'''

''' 
if you want to create separate models for the front and back paws, use the
code below. Since we use SVM to predict 3D locations, there cannot be any missing
points when building our model, or when processing the data output by DLC. 
'''

#%%
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd 
from pandas_ods_reader import read_ods

#%%
def label_images(start, end, views, obstructed_images, root_directory, video_name, realPoints):
    
def label_images(mov, labels, realPoints):
    
    df_coords = pd.DataFrame([])
    coords = np.zeros(len(mov), len(labels))
    for cam in range(len(labels)):
        print(f'looking at camera: {labels[cam]}')
        for frame in range(len(mov)):
            im = mov[frame, cam, :, :]
            plt.figure(1, figsize=(20,20))
            plt.imshow(im, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title(f'Cam: {labels[cam]}, Frame: {frame})

            coords[frame,cam:cam+1] = plt.ginput(1)[0] 

        
    
    
    
    
        
    os.chdir(root_directory)
    images = end - start - len(obstructed_images)
    coords = np.zeros((images+1, 2*len(views)))
    
    for j in range(len(views)):
        path = 'images_for_' + video_name + views[j]
        os.chdir(path)
        
        p = 0
        for k in range(int(end-start)):
            i = k+start
            if (np.isin(i, obstructed_images) == False):
                image = plt.imread('fr_'+str(i).zfill(5)+'.png')
                plt.figure(1, figsize=(20,20))
                plt.imshow(image, cmap='gray')
                plt.title('image ' + str(i).zfill(5) + ' cam ' + views[j])
                s1 = 2*j
                s2 = s1+2              
                coords[p,s1:s2] = plt.ginput(1)[0] 
                p = p+1

        s1 = 2*j
        s2 = s1+2
        coords[-1,s1:s2] = plt.ginput(1)[0] 

        i = 0
                    
        plt.close()            
        os.chdir(root_directory)
    
    coords_no_duplicates = pd.DataFrame(coords[1:,:])

    '''
    Process real points from ods or excel, this code will automatically remove
    rows before the starting image, after the ending image, and rows corresponding 
    to images obstructed
    '''
    
    real = pd.DataFrame([])
    zeros = pd.DataFrame([[0,0,0]], columns=["X", "Y", "Z"])
    rp = read_ods(realPoints,1, columns=["X", "Y", "Z"])
    # rp = pd.read_excel('realPoints.xlsx') # if pulling points from an excel file
    real = zeros.append(rp)
    real.reset_index(inplace = True, drop = True) 
    
    real = real.drop(real.index[end:])
    real = real.drop(real.index[0:start])
    
    k = 0  
    i = 0
    for i in range(real.shape[0]):
        for j in range(len(obstructed_images)):    
            if real.index[i-k] == obstructed_images[j]:          
                real = real.drop(real.index[i-k])
                k = k+1
    
    real.reset_index(inplace=True, drop=True)
    
    '''
    Add the two dataframes together
    '''
    coords_and_realPoints = pd.DataFrame([])
    coords_and_realPoints = coords_and_realPoints.append(coords_no_duplicates)
    coords_and_realPoints['X'] = real['X']
    coords_and_realPoints['Y'] = real['Y']
    coords_and_realPoints['Z'] = real['Z']
                
    return coords_and_realPoints
                  
#%% label calibration points
mov_path = '/home/nel-lab/Desktop/cali_recon/cali_test_all.npz'
realPoints_path = '/home/nel-lab/Desktop/cali_recon/realPoints.csv'

realPoints = pd.read_csv(realPoints_path)

with np.load(mov_path) as f:
    mov = f['movie']
    labels = f['labels']
    
camera_labels = labels
m1_views = labels#['0','1','2'] # Cameras - front left, front right, bot

m1_start = 0
m1_end = len(realPoints)

coords_m1 = label_images(m1_start, m1_end, m1_views, m1_obstructed_images, root_directory, 
                         video_name, realPointsPath)

#%%
model_1_coordinates = coords_m1
model_1_coordinates = pd.DataFrame(model_1_coordinates)
model_1_coordinates.to_csv('/home/nel-lab/Desktop/Cynthia/model_1_coordinates.csv', index=False)