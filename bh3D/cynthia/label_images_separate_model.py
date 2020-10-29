#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' 
if you want to create separate models for the front and back paws, use the
code below. Since we use SVM to predict 3D locations, there cannot be any missing
points when building our model, or when processing the data output by DLC. 

'''
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd 
from pandas_ods_reader import read_ods

#%%
def label_images(start, end, views, obstructed_images, root_directory, video_name, realPoints):
        
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
     
        # bug where first image is double counted, so we built an array with 1
        # extra row index, here we fill this row with the pts from the final image
        i = end-1          
        image = plt.imread('fr_'+str(i).zfill(5)+'.png')
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
                  
#%%
'''
for model 1
'''
root_directory = '/home/nel-lab/Desktop/Cynthia'
os.chdir(root_directory)
realPointsPath = '/home/nel-lab/Desktop/Cynthia/real_coordinates.ods'

video_name = 'mov_mov_recal_3_cam_'

m1_start = 0
m1_end = 10
m1_obstructed_images = []
m1_views = ['0','1','2'] # Cameras - front left, front right, bot

coords_m1 = label_images(m1_start, m1_end, m1_views, m1_obstructed_images, root_directory, 
                         video_name, realPointsPath)

#%%
model_1_coordinates = coords_m1
model_1_coordinates = pd.DataFrame(model_1_coordinates)
model_1_coordinates.to_csv('/home/nel-lab/Desktop/Cynthia/model_1_coordinates.csv', index=False)

#%%
'''
for model 2 - this requires that all three cameras saw every position 
this wasn't the case for the most recent setup with 5 cameras on the wheel
you'll have to use a separate 2-camera models each of the back feet - these
models are 2a and 2b
'''
root_directory = '/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2/Calibration_Videos2'
os.chdir(root_directory)
realPointsPath = '/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2/realPoints.ods'

video_name = 'mov_recal_5_cam_'

m2_start = 15
m2_end = 70
m2_obstructed_images = [69,60,59,20,19]
m2_views = ['1','3','4'] # Cameras - back left, back right, bot

coords_m2 = label_images(m2_start, m2_end, m2_views, m2_obstructed_images, root_directory, 
                         video_name, realPointsPath)

#%%
model_2_coordinates = coords_m2
model_2_coordinates = pd.DataFrame(model_2_coordinates)
model_2_coordinates.to_csv('/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2/model_2_coordinates.csv', index=False)

#%%
'''
for model 2a
'''
root_directory = '/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2/Calibration_Videos2'
os.chdir(root_directory)
realPointsPath = '/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2/realPoints.ods'

video_name = 'mov_recal_5_cam_'

m2a_start = 15
m2a_end = 70
m2a_obstructed_images = [69,60,59,20,19]
m2a_views = ['1','4'] # camera's - back left and bottom

coords_m2a = label_images(m2a_start, m2a_end, m2a_views, m2a_obstructed_images, root_directory, video_name, realPointsPath)

#%%
model_2a_coordinates = coords_m2a
model_2a_coordinates = pd.DataFrame(model_2a_coordinates)
model_2a_coordinates.to_csv('/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2/model_2a_coordinates.csv', index=False)

#%%
'''
for model 2b
'''
root_directory = '/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2/Calibration_Videos2'
os.chdir(root_directory)
realPointsPath = '/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2/realPoints.ods'

video_name = 'mov_recal_5_cam_'

m2b_start = 15
m2b_end = 70
m2b_obstructed_images = [69,60,59,20,19]
m2b_views = ['3','4'] # camera's - back right and bottom

coords_m2b = label_images(m2b_start, m2b_end, m2b_views, m2b_obstructed_images, root_directory, video_name, realPointsPath)

#%%
model_2b_coordinates = coords_m2b
model_2b_coordinates = pd.DataFrame(model_2b_coordinates)
model_2b_coordinates.to_csv('/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2/model_2b_coordinates.csv', index=False)
