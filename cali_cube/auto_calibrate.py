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

dat = np.load('/home/nel/NEL-LAB Dropbox/NEL/Datasets/Behavior3D/calicube.npz')
# data is video of the cube lighting up in form of npz file
mov = dat['movie']
# shape = (6000, 5, 480, 640) 480 and 640 are dimensions of image for each frame
#values in 480x640 array correspond to the pixel values/ intensity in each position
# 30000 2D arrays that are 480x640 
# 480 = x positions, 640 = y positions
# 6000 "groups", 5 arrays in each group 
# ex: group 1 - frame 1's array of the image taken (480x640) pixel values for each position 
#   each array represents each camera view 
maxs = np.max(mov, axis=(2,3))
# array of max pixel values for each frame and camera: shape = (6000, 5), 5 cameras, 6000 frames each 
#axis arg. removes the chosen axis by finding maxes along it, in this case 480 and 640
#%% define recording constants
fps = 50 # frames per second
d_on = 0.150  #set in arduino code (seconds on of light)
d_off = 0.050  #set in arduino code (seconds off of light)
t_on = fps*d_on # (frames on of light)
t_off = fps*d_off  # (frames off of light)
t_total = t_on+t_off # (total frames of light flash on/off for single flash)

#%% shorten movie
# start = first frame where *a* camera's pixel is at max (usually 255) - single value result
start = np.argwhere(maxs==maxs.max())[0][0] # [0][0] gives first array, could also change this but not needed

# end = frame number after lighting all LEDs (8^3 * t_total)
end = int(start+8**3*t_total)
# 8^3 LEDs times total frames for single flash + start = total frames

# visualize start/end frame
plt.plot(maxs, '.')
plt.vlines([start, end], 0, 255, color='k')
plt.title("Start and End frames")
plt.xlabel("Frames")
plt.ylabel("Camera pixel")
# vertical line at end of plot signifies where the LEDs have stopped lighting up, can see pause before 
# another cycle begins- we are not interested in
#%% shorten mov and maxs
# only include frames where pixels are lighting up (+1 frame at end)
mov1 = mov[start:end+1]
maxs1 = maxs[start:end+1]
#%% deleting full movie for storage
del mov

#%% shorten movies to just peak frames
# peak frame will be every t_total frames starting at t_total/2
# recall t_total = total frames for one "lighting up period"
max_frames = np.arange(t_total/2, t_total/2+8**3*t_total, t_total).astype(int)
# 1D array length 512
# creates array ranging from 5 frames (t_total/2) to 5125 frames incrementing in steps of 10
# (frames for a single light up)
# this isolates so we are only getting the frames when the LEDs should be lighting up 
#(halfway through t_total for each LED, respectively)
max_mov = mov1[max_frames] 
# shape (512, 5, 480, 640)
# this indexes mov1 (5121, 5, 480, 640) to be only the max pixel valued frames 
max_maxs = maxs1[max_frames] #max pixel values for each camera
# shape (512, 5)
# this is now the array of the max pixel values for each camera. there are 512 max pixel values (one for each LED)

#%% EXPLORE
#%% set cam and thresh for EDA
# thresh can be changed, this is a parameter that determines which frames will be kept during calibration
# cam is picking a specific camera
cam = 0
thresh = 250

#%% create movie to check/validate
import caiman as cm

for i in range (5):
    cm_mov = cm.movie(max_mov[:,i])
    cm_mov.fr = fps
    cm_mov.save(f'/home/nel/Desktop/Cali Cube/{i}.avi')

#%% see max values for cam
#%% see max values for cam (chosen cam)
#think about how the LED moves through the cube, there are 512 LEDs and each camera is going to pick up different visual
# Ex, top camera may not be able to see LED lighting up way at bottom, low pixel val.
plt.plot(max_maxs[:,cam])
plt.title(f'Max values for camera {cam}')
plt.ylabel('Max pixel value')
plt.xlabel('LED number')

#%% see unique max values
cam_max = np.unique(max_maxs[:,cam])
#shape (145,), sorted unique values of the max pixel values for camera 
plt.plot(cam_max[::-1])
plt.title(f'Unique max pixel values for camera {cam}')
plt.ylabel('Max pixel value')
plt.xlabel('Unique value no.')

#%% see frames where max < thresh
# this cell can give an idea of if the thresh needs to be changed
# if there are a lot of frames with bright spots, may be too low
# want thresh to be large enough so it eliminates the low pixel values, but not too high that we can not see 
# anything. can change camera value as well as thresh to make sure the best one is landed on after visual inspection
i=0
frame = 0
while i < 10:
    if max_maxs[frame,cam] <= thresh:
        plt.imshow(max_mov[frame,cam])
        # max_mov this indexes mov1 (512, 5, 480, 640) to be only the max pixel valued frames 
        #further indexed to be only frames where pixel<thresh
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

#%% see drop off in number of frames with max > thresh should see bright
# this also can help with determining a threshold value
for cam in range(5): #looping through 0, 1, 2, 3, 4, 5 (all cameras not just 1)
    lower = []
    for i in np.arange(256): # 255 element array from 0-255
        lower.append(np.sum(max_maxs[:,cam]<=i))
        #adding the sum of the number of elements of array of the max pixel values for each camera that are less than i
        #ex if in camera 0 there are 56 max pixel values that are less than 100 for i=100, append 56 to lower
        
    plt.plot(lower, np.arange(256), label = str(cam))
    #plotting elements of lower against the i value being compared to
    
    plt.xlim([8**3,0])
    
plt.hlines(thresh,0,8**3, color='k', label = 'thresh value')
plt.title(f"Drop off in number of frames with max>{thresh}")
plt.ylabel("Pixel value")
plt.xlabel(f"Number of frames with max> {thresh}")
plt.legend(title = "Cameras")

#%% plot interest point if max > thresh
# opposite of former plots where we were looking at max<thresh
cam = 0

#this time looping through all max frames (512 values)
for frame in range(8**3):
    #print(frame)
    #print(max_mov[frame,cam])
    plt.cla() #clears former axes
    plt.imshow(max_mov[frame,cam]) #shows data as image 
    #indexing max_mov with frame, cam is indexing wrt the 1st 2 array elements, giving the points
    #of the image as a result
    if max_maxs[frame,cam]>=thresh: # if this pixel value is greater than the chosen threshold
    #max_mov[frame, cam] is array of all the pixel values for given frame/cam, 
    #max_max[frame, cam] is (1) max pixel value for given frame/cam
    
        max_pt = np.mean(np.argwhere(max_mov[frame,cam] == max_maxs[frame,cam]), axis=0)
        #argwhere returns indices of the elements that satisfy condition
        #in this case the indices where the max pixel value equals the
        #pixel value in the max_mov array of all the pixel value 
        #basically finding all the indicies where the pixel value is at a max, 
        #returning with shape (x, 2)
        
        #np.mean, axis=0 averages  up all the values down the columns returning with
        #shape (2,)
        #since the points where the pixel values equal the max pixel value are
        #very close together, this give an estimate of the index, which is
        #equivalent to the position of the max pixel value, i.e where the LED
        #is lighting up 
        
        #2 value array  
        plt.plot(*max_pt[::-1], 'ro')
        # -1 reverses order, we want it in (x,y) not (y,x)
        
    plt.pause(1) 

#%% plot scatter of all points above thresh
# confirm camera order, can tell if scatter plot is accurate for specific cam based on visual inspection
cam = 0 

for frame in range(8**3): #loop through all the frames and plot the LEDs
    if max_maxs[frame,cam]>=thresh:
        max_pt = np.mean(np.argwhere(max_mov[frame,cam] == max_maxs[frame,cam]), axis=0)
        plt.plot(max_pt[1], 480-max_pt[0], 'ro')    
        
plt.xlim([0,640])
plt.ylim([0,480])

#%% CREATE CSV
#%% create ground truth array
real = np.zeros([8**3,3])
# (512, 3) number of frames rows, 3 columns empty array
row = 0
for z in range(8):
    for y in range(8):
        for x in range(8):
            real[row,:] = [x,y,z]
            row +=1
            #going row by row to create dimensions of the cube (8x8x8) for all 3 directions
            
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
#   nan value if below thresh, not interested 

thresh = 250
big = np.zeros([8**3,10])
#double number of cameras for columns bc each frame needs 2 position args  (x,y)

for cam in range(5): #camera 1, frame 1, 2, 3, .... 512, camera 2.... etc. 4 cams
    print(cam)
    for frame in range(8**3):
        if max_maxs[frame,cam]>=thresh:
            max_pt = np.mean(np.argwhere(max_mov[frame,cam] == max_maxs[frame,cam]), axis=0)
            #same code as above, accessing the position of max pixel value
            #above the threshold for each frame in each camera
            
            big[frame,2*cam:2*cam+2] = max_pt[::-1]
            #placing x,y pos. of max pixel value (LED) into big array if this 
            #value is greater than the threshold
            #recall indexing does not include second value
            #indexing: cam0: [0:2] cam1: [2:4] cam2: [4:6] cam3: [6:8] cam4 [8:10]
                
        else:
            big[frame,2*cam:2*cam+2] = [np.nan, np.nan]
            #placing nan if not greater than threshold
#%% save calibration  array
final = np.hstack([big, real])
#combines 2 arrays. 
#first 10 columns = big (x,y positions LED from each camera), 
#next 3 columns = real x,y,z positions 
# (x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x, y, z )
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
# drop row (frame) if > num_cam cameras have nan value, then iterate over nans
#drop if there are more nan values in the row than num_cam cameras

num_cam_missing = 1

final = final[np.isnan(final).sum(axis=1) <= num_cam_missing*2]
#result = boolean array for isnan
#summed across row
#if the amount of nan values is low enough, kept in final array


# impute over remianing nan values (estimate)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(random_state=0, max_iter=1000)
#args: there are more that can be manipulated
#the default method used is mean to initialize the values
#random_state = 0, this is an instance that is generated from either a seed, or np.random

final = imp.fit_transform(final)
#fit the imputer on final and return the transformed final 

# create df with correct camera columns
import pandas as pd
final_df = pd.DataFrame(final, columns = ['BOT_x', 'BOT_y', 'BR_x', 'BR_y', 'FR_x', 'FR_y', 'FL_x', 'FL_y',
       'BL_x', 'BL_y', 'X', 'Y', 'Z'])

# drop all remaining nans for final csv
final_df.dropna(inplace = True)
#Keep the DataFrame with valid entries in the same variable.

# drop outlier points on edge of cube (through visual inspection of next cell block)
final_df = final_df[final_df.X < 7*25.4]
#indexed to only include LEDs within 7**3 cubic inches

print('final dataframe shape is', final_df.shape)

#save csv to computer
#critical line for cali output
final_df.to_csv(f'/home/nel/Desktop/Cali Cube/cali_{thresh}.csv', index=False)

#%% plot imputed points in each camera
# for i in range(5):
#     ax = plt.subplot(2,3,i+1)
#     ax.plot(final[:,2*i],final[:,2*i+1],'.')
    
#%% test using BH3D calibration
from bh3D.mapping import mapping
import pandas as pd

thresh=250

#access csv
coordPath = f'/home/nel/Desktop/Cali Cube/cali_{thresh}.csv'

# print camera labels and order to help with model and DLCPaths below
model_options = pd.read_csv(coordPath).columns
model_options = [opt[:-2] for opt in model_options[:-3][::2]]
#index, [:-3][::2] cuts off laast 3 values, then prints every other value
print('Cameras labels to reference when defining model and DLCPaths variables below:\n', model_options)

#%% run calibration
# need 2 cameras at minimum, 
# model = ['BOT','FL','FR', 'BL', 'BR']
model = ['BOT', 'FL', 'FR']

# can set dummy DLC paths as we will not use them
#
DLCPaths = ['']
#the mapping model uses DLC paths to map DLC labelled coordinates to 3D
#in this step we are not doing that yet, so just use empty string

SVR_args = {'kernel':"rbf", 'C':15000}
#super high z = high regulization

cal = mapping(model, coordPath, DLCPaths, **SVR_args)
#from Github of mapping code
#model : list, List of camera labels as strings ('cam0/1/2', 'FL/FR/BOT', etc.).
#coordinatePath : str, Path to coordinate calibration file.
#DLCpaths : str, Paths to DLC data for each camera
#**kwargs : misc, Keyword arguments passed into sklearn's SVR class, for example 'kernel', 'degree', and 'C' parameters.

results = cal.calibration_results()
   #calibration_results(save=False): Calibration results.
   #what we are interested
#%% get errors and plot hist
import matplotlib.pyplot as plt
import numpy as np

bins = 50
#num equal width bins in the range
gt = pd.read_csv(coordPath).iloc[:,-3:]
#ground truth df, 61 rows 3 col

errors = np.linalg.norm(gt-results, axis=1)
#array length 61 (num values after imputing)
#finding norm values (errors) for each LED of groung truth - cali results
plt.hist(errors, bins)
# plt.xlim(errors.min(), errors.max())
# plt.xticks(np.linspace(errors.min(),errors.max(),10))
# plt.xticks(np.linspace(0,errors.max(),10))

plt.xticks(np.arange(0,errors.max()+.05,.05))
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