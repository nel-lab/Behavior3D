#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:58:04 2019

@author: williamstanford
"""

import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd 
from pandas_ods_reader import read_ods

'''
This version of the script enables you to fill in points that can't be seen by specific
camera angles with NaN, to do this, click in the upper left hand corner within 
10 pixes of the coordinate [0,0] -> as an example [3,5] would work but not [3,15]
'''

#%%
def label_images(start, end, views, image_directory, video_name):
        
    os.chdir(root_directory)
    images = end - start
    coords = np.zeros((images+1, 2*len(views)))
    
    for j in range(len(views)):
        path = 'images_for_' + video_name + views[j]
        os.chdir(path)
        
        for k in range(int(end-start)):
            i = k+start
            image = plt.imread(str(i).zfill(5)+'_frame.png')
            plt.figure(1, figsize=(20,20))
            plt.imshow(image, cmap='gray')
            plt.title('image ' + str(i).zfill(5) + ' cam ' + views[j])
            s1 = 2*j
            s2 = s1+2              
            coords[k,s1:s2] = plt.ginput(1)[0] 
     
        # there is a bug where first image is double counted, so we built an array with 1
        # extra row index, here we fill this row with the pts from the final image
        i = end-1          
        image = plt.imread(str(i).zfill(5)+'_frame.png')
        plt.figure(1, figsize=(20,20))
        plt.imshow(image, cmap='gray')
        plt.title('image ' + str(i).zfill(5) + ' cam ' + views[j])
        s1 = 2*j
        s2 = s1+2
        coords[-1,s1:s2] = plt.ginput(1)[0] 

        i = 0
                    
        plt.close()            
        os.chdir(root_directory)
    
    coords_no_duplicates = pd.DataFrame(coords[1:,:])
    
    for x in range(coords_no_duplicates.shape[0]):
        for y in range(coords_no_duplicates.shape[1]//2):
            if (coords_no_duplicates.iloc[x,2*y] < 10) & (coords_no_duplicates.iloc[x,2*y+1] < 10):
                coords_no_duplicates.iloc[x,2*y] = 'NaN'
                
                
    return coords_no_duplicates

def Add_RealPoints(df1, df2, df3, df4, df5, views, realPoints):
    '''
    Process real points from ods or excel, this code will automatically remove
    rows before the starting image and after the ending image, make sure you're
    excel file or ods file with the real coordinates has a blank row before
    beginning the real coordinates, otherwise the row will be deleted
    '''
    
    real = read_ods(realPoints,1, columns=["X", "Y", "Z"])
    real = real.drop(real.index[end:])
    real = real.drop(real.index[0:start])
    real.reset_index(inplace=True, drop=True)
    
    '''
    Add dataframes together
    '''
    coords_and_realPoints = pd.DataFrame([])
    coords_and_realPoints['cam_0_x'] = df1.iloc[:,0]
    coords_and_realPoints['cam_0_y'] = df1.iloc[:,1]
    coords_and_realPoints['cam_1_x'] = df2.iloc[:,0]
    coords_and_realPoints['cam_1_y'] = df2.iloc[:,1]
    coords_and_realPoints['cam_2_x'] = df3.iloc[:,0]
    coords_and_realPoints['cam_2_y'] = df3.iloc[:,1]
    coords_and_realPoints['cam_3_x'] = df4.iloc[:,0]
    coords_and_realPoints['cam_3_y'] = df4.iloc[:,1]
    coords_and_realPoints['cam_4_x'] = df5.iloc[:,0]
    coords_and_realPoints['cam_4_y'] = df5.iloc[:,1]

    
    coords_and_realPoints['X'] = real['X']
    coords_and_realPoints['Y'] = real['Y']
    coords_and_realPoints['Z'] = real['Z']
                
    return coords_and_realPoints

#%%
image_directory = '/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2/Calibration_Videos2'
os.chdir(image_directory)
realPointsPath = '/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2/realPoints.ods'

video_name = 'mov_recal_5_cam_'

start = 0
end = 35

#%%
# this will get all your coordinates at once. BUT it gets very slow, instead
# do them each individually as done in fl_coords, ..., b_coords

views = ['0','1','2','3','4'] # Cameras - front left, back right, front right, back left, bot
coords = label_images(start, end, views, image_directory, 
                         video_name)
'''
if you cannot see the point you wish to label in one of the views, click in the top left corner
make sure that the coordinate values are less than 10
'''
#%%
views = ['0']
fl_coords = label_images(start, end, views, image_directory, 
                         video_name)
#%%
views =['1']
br_coords = label_images(start, end, views, image_directory, 
                         video_name)
#%%
views =['2']
fr_coords = label_images(start, end, views, image_directory, 
                         video_name)
#%%
views =['3']
bl_coords = label_images(start, end, views, image_directory, 
                         video_name)
#%%
views =['4']
b_coords = label_images(start, end, views, image_directory, 
                         video_name)
#%%
realPointsPath = '/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2/realPoints.ods'
coords = Add_RealPoints(fl_coords, br_coords, fr_coords, bl_coords, b_coords, views, realPointsPath)

#%%
root_directory = '/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2'
os.chdir(root_directory)
coords.reset_index(inplace=True, drop=True)
coords.to_csv('coords_with_NaN.csv', index=False)

#%%

''' 
if you want to create separate models for the front and back paws, see the first 
version of the script (3a-label_images_separate_model.py)
'''
