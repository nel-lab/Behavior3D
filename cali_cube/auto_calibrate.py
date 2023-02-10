#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 01:13:22 2021

@author: jimmytabet
"""

#%% ANALYZE
#%% import modules, load data, and get max pixel values (takes time!)
import numpy as np
import matplotlib.pyplot as plt

"""
Movie array:
    - shape of mov = (6000, 5, 480, 640) 480 and 640 are dimensions of 
      image for each frame
    - values in 480x640 array correspond to the pixel values/ intensity 
      in each position
    - 30000 2D arrays that are 480x640 
        -480 = x positions, 640 = y positions
        -6000 "groups", 5 arrays in each group 
    - ex: group 1 - frame 1's array of the image taken (480x640) 
        -pixel values for each position 
        -each array represents each camera view 
        
Maxs array:
    - array of max pixel values for each frame and camera:
    - shape = (6000, 5), 5 cameras, 6000 frames each 
    
"""
path = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/Behavior3D/calicube.npz'
dat = np.load(path)
mov = dat['movie']
maxs = np.max(mov, axis=(2,3))
#%% determine number of cameras used
"""
Note: this script is optimal for 3 and 5 camera set-ups 

"""
num_cameras = len(mov[0])
print(f"The number of cameras used is {num_cameras}")
#%% define recording constants
"""
fps: frames per second
d_on: set in arduino code, seconds on of light
d_off: set in arduino code, seconds off of light
t_on: frames on of light
t_off:frames off of light
t_total: total frames of light flash on/off for single flash
"""
fps = 50 
d_on = 0.150  
d_off = 0.050  
t_on = fps*d_on 
t_off = fps*d_off  
t_total = t_on+t_off 

#%% shorten movie
"""
Want to shorten movie to only when the cube is lighting up
- identify start: first frame where *a* camera's pixel is at max 
    (usually 255), single value result
- note: [0][0] can be changed to [0][1], [0][2], ect. 
- end: frame number after lighting all LEDs (8^3 * t_total)

- vertical line at end of plot signifies where the LEDs have 
 stopped lighting up, can see pause before another cycle begins;
 we are not interested in

"""
start = np.argwhere(maxs==maxs.max())[0][0] 
end = int(start+8**3*t_total)

# visualize start/end frame
plt.plot(maxs, '.')
plt.vlines([start, end], 0, 255, color='k')
plt.title("Start and End frames")
plt.xlabel("Frames")
plt.ylabel("Camera pixel")
#%% shorten mov and maxs
"""
Only including frames where pixels are lighting up (+1 frame at end)

"""
mov1 = mov[start:end+1]
maxs1 = maxs[start:end+1]
#%% deleting full movie for storage
del mov

#%% shorten movies to just peak frames
"""
- Peak frame will be every t_total frames starting at t_total/2
- Recall t_total = total frames for one "lighting up period

max_frames shape: 1D array length 512
    - Creates array ranging from 5 frames (t_total/2) to 
      5125 frames incrementing in steps of 10
    - This isolates so we are only getting the frames when the LEDs should 
      be lighting up (halfway through t_total for each LED, respectively)

max_mov shape: (512, 5, 480, 600)
    - Only the max pixel valued frames 
    
max_maxs shape: (512, 5)
    - Max pixel values for each camera
    - There are 512 max pixel values (one for each LED)
"""
max_frames = np.arange(t_total/2, t_total/2+8**3*t_total, 
                       t_total).astype(int)
max_mov = mov1[max_frames] 
max_maxs = maxs1[max_frames] 

#%% EXPLORE
"""
Note the following cells are not necessary for creating the calibration
model, but may be helpful for additional analysis of the dataset.

"""
#%% set cam and thresh for EDA
"""
- Thresh can be changed, this is a parameter that determines 
 which frames will be kept during calibration
- Cam is picking a specific camera (can be changed)
"""
cam = 0
thresh = 250

#%% create movie to check/validate
import caiman as cm

for i in range (5):
    cm_mov = cm.movie(max_mov[:,i])
    cm_mov.fr = fps
    cm_mov.save(f'/home/nel/Desktop/Cali Cube/{i}.avi')

#%% see max values for chosen cam
"""
Think about how the LED moves through the cube, there are 512 LEDs 
and each camera is going to pick up different visual.
    - Ex, top camera may not be able to see LED lighting up way at bottom, 
     low pixel val.
"""
plt.plot(max_maxs[:,cam])
plt.title(f'Max values for camera {cam}')
plt.ylabel('Max pixel value')
plt.xlabel('LED number')

#%% see unique max values
cam_max = np.unique(max_maxs[:,cam])
plt.plot(cam_max[::-1])
plt.title(f'Unique max pixel values for camera {cam}')
plt.ylabel('Max pixel value')
plt.xlabel('Unique value no.')

#%% see frames where max < thresh
"""
This cell can give an idea of if the thresh needs to be changed
 - If there are a lot of frames with bright spots, may be too low
 - Want thresh to be large enough so it eliminates the low pixel values, 
   but not too high that we can not see anything. 
 - Can change camera value as well as thresh to make sure the best one 
   is landed on after visual inspection
"""
i=0
frame = 0
while i < 10:
    if max_maxs[frame,cam] <= thresh:
        plt.imshow(max_mov[frame,cam])
        plt.title(f'Movie frames where max pixel val <{thresh} for cam {cam}')
        plt.pause(0.5)
        i += 1
    frame += 1

#%% save cm movie of frames where max < thresh
temp = np.where(max_maxs[:,cam] <= thresh)[0]
print(temp.shape)

temp_mov = cm.movie(max_mov[temp,cam])
temp_mov.fr = fps
temp_mov.save('/home/nel/Desktop/Cali Cube/temp.avi')

#%% see drop off in number of frames with max > thresh
"""
Looping through each of the cameras
- Adding the sum of the number of elements of array of the max 
  pixel values for each camera that are less than i
  - ex: if in camera 0 there are 56 max pixel values that are 
    less than 100 for i=100, append 56 to lower
    
- Plotting elements of lower against the i value being compared to
"""
for cam in range(num_cameras): 
    lower = []
    for i in np.arange(256): 
        lower.append(np.sum(max_maxs[:,cam]<=i))
        
    plt.plot(lower, np.arange(256), label = str(cam))
    plt.xlim([8**3,0])
    
plt.hlines(thresh,0,8**3, color='k', label = 'thresh value')
plt.title(f"Drop off in number of frames with max>{thresh}")
plt.ylabel("Pixel value")
plt.xlabel(f"Number of frames with max> {thresh}")
plt.legend(title = "Cameras")

#%% plot interest point if max > thresh
"""
- Looping through all max frames (512 values)
- Indexing max_mov with frame, cam is indexing with respect tp
  the 1st 2 array elements, giving the points of the image as a result
  
  - max_mov[frame, cam] is array of all the pixel values for given frame/cam 
  - max_max[frame, cam] is (1) max pixel value for given frame/cam
  
  - np.argwhere returns indices of the elements that satisfy condition
    in this case the indices where the max pixel value equals the
    pixel value in the max_mov array of all the pixel value; 
    basically finding all the indicies where the pixel value is at a max, 
    returning with shape (x, 2)
    
  - np.mean, axis=0 averages  up all the values down the columns 
    returning array shape (2,) for max_pt
        -since the points where the pixel values equal the max pixel 
         value are very close together, this gives an estimate of the 
         index, which is equivalent to the position of the max pixel 
         value, i.e where the LED is lighting up 
         
  - plotting max_pt (x,y) coords for each frame
"""
cam = 0

for frame in range(8**3):
    plt.cla() 
    plt.imshow(max_mov[frame,cam]) 
    
    if max_maxs[frame,cam]>=thresh: 
        max_pt = np.mean(np.argwhere(max_mov[frame,cam] == 
                                     max_maxs[frame,cam]), axis=0)
        
        plt.plot(*max_pt[::-1], 'ro')
        
    plt.pause(1) 

#%% plot scatter of all points above thresh
""" 
Confirm camera order, can tell if scatter plot is 
accurate for specific cam based on visual inspection
"""
cam = 0 

for frame in range(8**3): 
    if max_maxs[frame,cam]>=thresh:
        max_pt = np.mean(np.argwhere(max_mov[frame,cam] == max_maxs[frame,cam]), axis=0)
        plt.plot(max_pt[1], 480-max_pt[0], 'ro')    
        
plt.xlim([0,640])
plt.ylim([0,480])

#%% CREATE CSV
"""
If "EXPLORE" section was skipped, resume running cells here
"""
#%% create ground truth array
"""
Going row by row to create dimensions of the cube (8x8x8) 
for all 3 directions
"""
real = np.zeros([8**3,3])
row = 0
for z in range(8):
    for y in range(8):
        for x in range(8):
            real[row,:] = [x,y,z]
            row +=1
            
            
# convert inches to mm
real *= 25.4

#%% rotate
# # rotate 45 degrees counterclockwise and invert z axis
# rot = np.array([[2**-.5, -2**-.5, 0],
#                 [2**-.5, 2**-.5, 0],
#                 [0,0,-1]])

# trans = real@rot

# # adjust points to first quadrant
# trans[:,-1] += 7
# trans[:,1] += 4*2**0.5

# # view new coord axis
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(trans[:,0], trans[:,1], trans[:,2])

# ax.quiver(0,0,0,0,0,1,color='k')
# ax.quiver(0,0,0,0,1,0,color='k')
# ax.quiver(0,0,0,1,0,0,color='k')

#%% create array of locations where pixel > thresh for all cams
"""
Filling empty array:
- Need to double number of cameras for columns bc each frame 
  needs 2 position args  (x,y)

Looping through camera 1, frame 1, 2, 3, .... 512, camera 2.... etc. 
- max_pt: same code as before, accessing the position of max pixel value
  above the threshold for each frame in each camera

- placing x,y pos. of max pixel value (LED) into big array if this 
  value is greater than the threshold 
  - indexing: cam0: [0:2] cam1: [2:4] cam2: [4:6] cam3: [6:8] cam4 [8:10]

- Enter nan value if below thresh
"""

thresh = 250
big = np.zeros([8**3,2*num_cameras])

for cam in range(num_cameras): 
    print(cam)
    for frame in range(8**3):
        if max_maxs[frame,cam]>=thresh:
            max_pt = np.mean(np.argwhere(max_mov[frame,cam] == 
                                         max_maxs[frame,cam]), axis=0)
    
            big[frame,2*cam:2*cam+2] = max_pt[::-1]
                
        else:
            big[frame,2*cam:2*cam+2] = [np.nan, np.nan]
            
#%% save calibration  array
'''
Combines 2 arrays. 
- first 10 columns = big (x,y positions LED from each camera), 
- next 3 columns = real x,y,z positions 
- (x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x, y, z )
'''
final = np.hstack([big, real])

# np.save(f'/home/nel-lab/Desktop/Jimmy/new_cali_cube/final_{thresh}.npy', final)
# final = np.load(f'/home/nel-lab/Desktop/Jimmy/new_cali_cube/final_{thresh}.npy')

#%% plot original points in each camera
# fig = plt.figure()
# for i in range(5):
#     ax = plt.subplot(2,3,i+1)
#     ax.plot(final[:,2*i],final[:,2*i+1],'o')
    
# # plot ground truth cube
# ax = fig.add_subplot(2,3,6,projection='3d')
# ax.plot(final[:,-3],final[:,-2],final[:,-1],'.')

#%% handle nan values (drop/impute) and save as csv for BH3D code
"""
Drop row (frame) if > num_cam cameras have nan value, 
then iterate over nans
Drop if there are more nan values in the row than num_cam_missing

-final: boolean array for isnan
    -summed across row
    -if the amount of nan values is low enough, kept in final array
"""

num_cam_missing = 1

final = final[np.isnan(final).sum(axis=1) <= num_cam_missing*2]

#%% impute over remianing nan values (estimate)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

"""
IterativeImputer args: there are more that can be manipulated
    -The default method used is mean to initialize the values
    -random_state = 0, this is an instance that is generated from 
     either a seed, or np.random
     
Fit the imputer on final and return the transformed final 
"""
imp = IterativeImputer(random_state=0, max_iter=1000)

final = imp.fit_transform(final)

#%% create df with correct camera columns
"""
- Create df sized based on num_cameras
- Drop all remaining nans for final csv, keep the DataFrame 
  with valid entries in the same variable.
- Drop outlier points on edge of cube 
    - #Indexed to only include LEDs within 7**3 cubic inches
    
"""
import pandas as pd
if (num_cameras == 5):
    final_df = pd.DataFrame(final, columns = ['BOT_x', 'BOT_y', 'BR_x', 'BR_y', 'FR_x', 'FR_y', 'FL_x', 'FL_y',
       'BL_x', 'BL_y', 'X', 'Y', 'Z'])
else:
    final_df = pd.DataFrame(final, columns = ['BOT_x', 'BOT_y', 'FR_x', 'FR_y', 'FL_x', 'FL_y',
      'X', 'Y', 'Z'])

final_df.dropna(inplace = True)

final_df = final_df[final_df.X < 7*25.4]

print('final dataframe shape is', final_df.shape)

#%%save df as csv file
"""
- Save csv to computer (critical line for cali output)
    -timestamped to avoid overwriting of csv files w/ the same thresh
"""
from datetime import datetime
now = datetime.now()
dt_string=now.strftime("%d-%m_%H-%M")

path_csv= f'/home/nel/Desktop/Cali Cube/cali_{thresh}__{dt_string}.csv'
final_df.to_csv(path_csv, index=False)

#%% plot imputed points in each camera
# for i in range(5):
#     ax = plt.subplot(2,3,i+1)
#     ax.plot(final[:,2*i],final[:,2*i+1],'.')
    
#%% test using BH3D calibration
from bh3D.mapping import mapping
import pandas as pd

"""
Define path and thres
- print camera labels and order to help with model and DLCPaths below
"""
thresh=250
coordPath = path_csv

model_options = pd.read_csv(coordPath).columns
model_options = [opt[:-2] for opt in model_options[:-3][::2]]
print('Cameras labels to reference when defining model and DLCPaths variables below:\n', model_options)

#%% run calibration

"""
-The mapping model uses DLC paths to map DLC labelled coordinates to 3D
-In this step we are not doing that yet, so just use empty string

-From Github of mapping code:
    -model : list, List of camera labels as strings 
    ('cam0/1/2', 'FL/FR/BOT', etc.).
    -coordinatePath : str, Path to coordinate calibration file.
    -DLCpaths : str, Paths to DLC data for each camera
    -**kwargs : misc, Keyword arguments passed into sklearn's SVR class, 
     for example 'kernel', 'degree', and 'C' parameters.
"""

if num_cameras == 5:
    model = ['BOT','FL','FR', 'BL', 'BR']
else:
    model = ['BOT', 'FL', 'FR']

# can set dummy DLC paths as we will not use them
DLCPaths = ['']

SVR_args = {'kernel':"rbf", 'C':15000}
#super high z = high regulization

cal = mapping(model, coordPath, DLCPaths, **SVR_args)

results = cal.calibration_results()

#%% get errors and plot hist
import matplotlib.pyplot as plt
import numpy as np
"""
-array length 61 (num values after imputing)
-finding norm values (errors) for each LED of groung truth - cali results
"""
bins = 50
gt = pd.read_csv(coordPath).iloc[:,-3:]

errors = np.linalg.norm(gt-results, axis=1)
plt.hist(errors, bins)

# plt.xlim(errors.min(), errors.max())
# plt.xticks(np.linspace(errors.min(),errors.max(),10))
# plt.xticks(np.linspace(0,errors.max(),10))

if errors.max()> 1.0:
    step = round((errors.max()-errors.min())/(errors.max()), 1)
else:
    step = .05

plt.xticks(np.arange(0,errors.max()+.05, step), rotation= "45")
plt.xlabel('mm')
plt.ylabel("Frequency")
plt.title(f'Histogram of Errors, Thresh = {thresh}')

#%% PREVIOUS WORK
#%% OLD METHOD TO FIND MAX FRAMES BASED ON PIXEL INTENSITY
# from collections import Counter

# #%% helper functions
# # return array of n-long consecutive integers/indices
# def n_long(array, n):
#     new_array = []
#     for i in range(array.size):
#         if array[i]+n-1 in array and i+n-1 == int(np.where(array == array[i]+n-1)[0]):
#             new_array += list(np.arange(array[i], array[i]+n))
    
#     return np.unique(new_array)

# # return list of start/end values of consecutive integers
# def ranges(nums):
#     nums = sorted(set(nums))
#     gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
#     edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
#     return list(zip(edges, edges))

# # return integer of next number to use when adding frames that were skipped
# #   alternates between integers so tries approx. target value
# def next_num(current, target):
#     low = int(np.floor(target))
#     high = low + 1
#     if len(current)==0 or np.mean(current) >= target:
#         num = low
#     else:
#         num = high

#     current.append(num)
#     return(num)

# # get frames where all cameras are low (max intensity < 100)
# low = Counter(np.where(maxs<150)[0])
# low_arr = np.array(list(low.items()))
# low_all_cams = low_arr[low_arr[:,1]==maxs.shape[1]][:,0]

# # start = first frame where *a* camera's pixel is at max (usually 255)
# start = np.argwhere(maxs==maxs.max())[0][0]
# # end = first frame where *all* cameras are low for a 5 seconds (indicates cube has stopped)
# # end = n_long(low_all_cams, fps*5)[0]

# #%% find frames of max pixel intensity (indicates LED is on)
# # frames of max intensty for all cameras
# ids = np.argwhere(maxs1==maxs.max())
# # unique frames
# uniq = np.unique(ids[:,0])
# # ranges of unique frames
# range_uniq = ranges(uniq)
# # mean frame in each range
# max_frames = np.array([int(round(np.mean(i))) for i in range_uniq])

# #%% fix missed frames
# # see if diff between max_frames is larger than t_total + a few frames (indicates missed frame)
# diff = np.where(np.diff(max_frames) > int(np.ceil(t_total))+2)[0]
# # initialize tries for next_num function
# tries = []
# # loop until no more large diffs to fix/added all missed frames
# while len(diff):
#     # track how many frames added so frames are inserted in proper place
#     counter = 1
#     # insert missed frames
#     for i in range(len(diff)):
#         max_frames = np.insert(max_frames, diff[i]+counter, max_frames[diff[i]+i]+next_num(tries, t_total))
#         counter += 1
    
#     # update diff
#     diff = np.where(np.diff(max_frames) > int(np.ceil(t_total))+3)[0]

# # if there are not 8**3 frames, something went wrong!
# if max_frames.size != 8**3: print('something went wrong!\n\tframes:', max_frames.size)

# #%% inspect/alter max_frames
# print(max_frames)
# print(end-start)

# #max_frames = np.append(max_frames, int(max_frames[-1]+t_total))
# max_frames = np.arange(4, end-start, 10)

# assert max_frames.size == 8**3

#%% OLD METHOD TO LOCATE MAX PIXEL USING BLUR
# import cv2
# mov = np.load('Users/jimmytabet/calibration_cube.npy')

# for i in range(200):
#     # clear axis
#     plt.cla()
#     # blur current frame
#     blur = cv2.GaussianBlur(mov[i,3],(5,5),0)
#     # plot blurred frame
#     plt.imshow(blur, cmap = 'gray', vmin=0, vmax=20)
#     # get coordinates of max intensity of blurred frame
#     ind = np.unravel_index(np.argmax(blur), blur.shape)
#     # if max intensity is > threshold, plot red dot
#     if np.max(blur)>100:
#         	plt.plot(*ind[::-1], 'ro')
#     # pause for a bit before looping to next frame
#     plt.pause(0.030)
#     print(i)