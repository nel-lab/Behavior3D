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
mov = dat['movie']
maxs = np.max(mov, axis=(2,3))

#%% define recording constants
fps = 50
d_on = 0.150
d_off = 0.050
t_on = fps*d_on
t_off = fps*d_off
t_total = t_on+t_off

#%% shorten movie
# start = first frame where *a* camera's pixel is at max (usually 255)
start = np.argwhere(maxs==maxs.max())[0][0]

# end = frame number after lighting all LEDs (8^3 * t_total)
end = int(start+8**3*t_total)

# visualize start/end frame
plt.plot(maxs, '.')
plt.vlines([start, end], 0, 255, color='k')

#%% shorten mov and maxs
mov1 = mov[start:end+1]
maxs1 = maxs[start:end+1]

#%% shorten movies to just peak frames
# peak frame will be every t_total frames starting at t_total/2
max_frames = np.arange(t_total/2, t_total/2+8**3*t_total, t_total).astype(int)
max_mov = mov1[max_frames]
max_maxs = maxs1[max_frames]

#%% EXPLORE
#%% set cam and thresh for EDA
cam = 0
thresh = 250

#%% create movie to check/validate
import caiman as cm

for i in range (5):
    cm_mov = cm.movie(max_mov[:,i])
    cm_mov.fr = fps
    cm_mov.save(f'/home/nel/Desktop/Cali Cube/{i}.avi')

#%% see max values for cam
plt.plot(max_maxs[:,cam])

#%% see unique max values
cam_max = np.unique(max_maxs[:,cam])
plt.plot(cam_max[::-1])

#%% see frames where max < thresh
i=0
frame = 0
while i < 10:
    if max_maxs[frame,cam] <= thresh:
        plt.imshow(max_mov[frame,cam])
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
for cam in range(5):
    lower = []
    for i in np.arange(256):
        lower.append(np.sum(max_maxs[:,cam]<=i))

    plt.plot(lower, np.arange(256), label = str(cam))
    plt.xlim([8**3,0])
    
plt.hlines(thresh,0,8**3, color='k', label = 'thresh value')
plt.legend()

#%% plot interest point if max > thresh
cam = 0

for frame in range(8**3):
    plt.cla()
    plt.imshow(max_mov[frame,cam])
    if max_maxs[frame,cam]>=thresh:
        max_pt = np.mean(np.argwhere(max_mov[frame,cam] == max_maxs[frame,cam]), axis=0)
        plt.plot(*max_pt[::-1], 'ro')
    plt.pause(.01)

#%% plot scatter of all points above thresh
cam = 0

for frame in range(8**3):
    if max_maxs[frame,cam]>=thresh:
        max_pt = np.mean(np.argwhere(max_mov[frame,cam] == max_maxs[frame,cam]), axis=0)
        plt.plot(max_pt[1], 480-max_pt[0], 'ro')    
        
plt.xlim([0,640])
plt.ylim([0,480])

#%% CREATE CSV
#%% create ground truth array
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

#%% create array of location of pixel > thresh for all cams
#   nan value if below thresh
thresh = 250
big = np.zeros([8**3,10])
for cam in range(5):
    for frame in range(8**3):
        if max_maxs[frame,cam]>=thresh:
            max_pt = np.mean(np.argwhere(max_mov[frame,cam] == max_maxs[frame,cam]), axis=0)
            big[frame,2*cam:2*cam+2] = max_pt[::-1]
        else:
            big[frame,2*cam:2*cam+2] = [np.nan, np.nan]
            
#%% save calibration  array
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
# drop row if > num_cam cameras have nan value, then iterate over nans
num_cam_missing = 1
final = final[np.isnan(final).sum(axis=1) <= num_cam_missing*2]

# impute over remianing nan values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(random_state=0, max_iter=1000)
final = imp.fit_transform(final)

# create df with correct camera columns
import pandas as pd
final_df = pd.DataFrame(final, columns = ['BOT_x', 'BOT_y', 'BR_x', 'BR_y', 'FR_x', 'FR_y', 'FL_x', 'FL_y',
       'BL_x', 'BL_y', 'X', 'Y', 'Z'])

# drop all remaining nans for final csv
final_df.dropna(inplace = True)

# drop outlier points on edge of cube (through visual inspection of next cell block)
final_df = final_df[final_df.X < 7*25.4]

print(final_df.shape)

final_df.to_csv(f'/home/nel/Desktop/Cali Cube/cali_{thresh}.csv', index=False)

#%% plot imputed points in each camera
# for i in range(5):
#     ax = plt.subplot(2,3,i+1)
#     ax.plot(final[:,2*i],final[:,2*i+1],'.')
    
#%% test using BH3D calibration
from bh3D.mapping import mapping
import pandas as pd

thresh=250

coordPath = f'/home/nel/Desktop/Cali Cube/cali_{thresh}.csv'
# print camera labels and order to help with model and DLCPaths below
model_options = pd.read_csv(coordPath).columns
model_options = [opt[:-2] for opt in model_options[:-3][::2]]
print('Cameras labels to reference when defining model and DLCPaths variables below:\n', model_options)

#%% run calibration
# model = ['BOT','FL','FR', 'BL', 'BR']
model = ['BOT', 'FL', 'FR']

# can set dummy DLC paths as we will not use them
DLCPaths = ['']

SVR_args = {'kernel':"rbf", 'C':15000}

cal = mapping(model, coordPath, DLCPaths, **SVR_args)
results = cal.calibration_results()

#%% get errors and plot hist
import matplotlib.pyplot as plt
import numpy as np

bins = 50
gt = pd.read_csv(coordPath).iloc[:,-3:]
errors = np.linalg.norm(gt-results, axis=1)
plt.hist(errors, bins)
# plt.xlim(errors.min(), errors.max())
# plt.xticks(np.linspace(errors.min(),errors.max(),10))
# plt.xticks(np.linspace(0,errors.max(),10))
plt.xticks(np.arange(0,errors.max()+.05,.05))

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