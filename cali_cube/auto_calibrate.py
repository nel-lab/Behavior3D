#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 01:13:22 2021

@author: jimmytabet
"""

#%% import modules, load data, and get max pixel values (takes time!)
import numpy as np
from collections import Counter

mov = np.load('/home/nel/Desktop/Cali Cube/calibration_cube.npy')
maxs = np.max(mov, axis=(2,3))

#%% helper functions
# return array of n-long consecutive integers/indices
def n_long(array, n):
    new_array = []
    for i in range(array.size):
        if array[i]+n-1 in array and i+n-1 == int(np.where(array == array[i]+n-1)[0]):
            new_array += list(np.arange(array[i], array[i]+n))
    
    return np.unique(new_array)

# return list of start/end values of consecutive integers
def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

# return integer of next number to use when adding frames that were skipped
#   alternates between integers so tries approx. target value
def next_num(current, target):
    low = int(np.floor(target))
    high = low + 1
    if len(current)==0 or np.mean(current) >= target:
        num = low
    else:
        num = high

    current.append(num)
    return(num)

#%% define recording constants
fps = 50
d_on = 0.050
d_off = 0.100
t_on = fps*d_on
t_off = fps*d_off
t_total = t_on+t_off

#%% shorten movie
# get frames where all cameras are low (max intensity < 100)
low = Counter(np.where(maxs<100)[0])
low_arr = np.array(list(low.items()))
low_all_cams = low_arr[low_arr[:,1]==maxs.shape[1]][:,0]

# start = first frame where *a* camera's pixel is at max (usually 255)
start = np.argwhere(maxs==maxs.max())[0][0]
# end = first frame where *all* cameras are low for a 5 seconds (indicates cube has stopped)
end = n_long(low_all_cams, fps*5)[0]

# shorten mov and maxs
mov1 = mov[start:end+1]
del mov
maxs1 = maxs[start:end+1]

#%% find frames of max pixel intensity (indicates LED is on)
# frames of max intensty for all cameras
ids = np.argwhere(maxs1==maxs.max())
# unique frames
uniq = np.unique(ids[:,0])
# ranges of unique frames
range_uniq = ranges(uniq)
# mean frame in each range
max_frames = np.array([int(round(np.mean(i))) for i in range_uniq])

#%% fix missed frames
# see if diff between max_frames is larger than t_total + a few frames (indicates missed frame)
diff = np.where(np.diff(max_frames) > int(np.ceil(t_total))+3)[0]
# initialize tries for next_num function
tries = []
# loop until no more large diffs to fix/added all missed frames
while len(diff):
    # track how many frames added so frames are inserted in proper place
    counter = 1
    # insert missed frames
    for i in range(len(diff)):
        max_frames = np.insert(max_frames, diff[i]+counter, max_frames[diff[i]+i]+next_num(tries, t_total))
        counter += 1
    
    # update diff
    diff = np.where(np.diff(max_frames) > int(np.ceil(t_total))+3)[0]

# if there are not 512 frames, something went wrong!
if max_frames.size != 512: print('something went wrong!\n\tframes:', max_frames.size)

#%% create movie to check/validate
import caiman as cm

for i in range (5):
    cam = i
    cm_mov = cm.movie(mov1[max_frames,cam])
    cm_mov.fr = fps
    cm_mov.save(f'/home/nel/Desktop/Cali Cube/{cam}.avi')

#%% EXPLORE
import matplotlib.pyplot as plt

max_mov = mov1[max_frames]
max_maxs = maxs1[max_frames]

#%% see max values for cam
cam = 0
plt.plot(max_maxs[:,cam])

#%% see unique max values
cam_max = np.unique(max_maxs[:,cam])
plt.plot(cam_max[::-1])

#%% see frames where max < thresh
i=0
frame = 0
while i < 10:
    if max_maxs[frame,cam] <= 200:
        plt.imshow(max_mov[frame,cam])
        plt.pause(0.5)
        i += 1
    frame += 1

#%% save cm movie of frames where max < thresh
temp = np.where(max_maxs[:,3] <= 190)[0]
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
    plt.xlim([512,0])
    plt.legend()
    
plt.hlines(190,0,512)

#%% plot interest point if max > thresh
cam = 0

for frame in range(10):#(512):
    plt.cla()
    plt.imshow(max_mov[frame,cam])
    if max_maxs[frame,cam]>=190:
        max_pt = np.mean(np.argwhere(max_mov[frame,cam] == max_maxs[frame,cam]), axis=0)
        plt.plot(*max_pt[::-1], 'ro')
    plt.pause(0.3)
    

# turn to animation ... WIP
# def update(frame):
#     fig.cla()
#     plt.imshow(max_mov[frame,cam])
#     if max_maxs[frame,cam]>=190:
#         max_pt = np.mean(np.argwhere(max_mov[frame,cam] == max_maxs[frame,cam]), axis=0)
#         plt.plot(*max_pt[::-1], 'ro')   
        
# fig = plt.figure()

#%% plot scatter of all points above thresh
cam = 0

for frame in range(512):
    if max_maxs[frame,cam]>=255:
        max_pt = np.mean(np.argwhere(max_mov[frame,cam] == max_maxs[frame,cam]), axis=0)
        plt.plot(max_pt[1], 480-max_pt[0], 'ro')    
        
plt.xlim([0,640])
plt.ylim([0,480])

#%% create ground truth array
real = np.zeros([512,3])
row = 0
for z in range(8):
    for y in range(8):
        for x in range(8):
            real[row,:] = [x,y,z]
            row +=1
                        
#%% create array of location of pixel > thresh for all cams
#   nan value if below thresh
big = np.zeros([512,10])
for cam in range(5):
    for frame in range(512):
        if max_maxs[frame,cam]>=255:
            max_pt = np.mean(np.argwhere(max_mov[frame,cam] == max_maxs[frame,cam]), axis=0)
            big[frame,2*cam:2*cam+2] = max_pt
        else:
            big[frame,2*cam:2*cam+2] = [np.nan, np.nan]
            
#%% save calibration  array
trial = '255'
final = np.hstack([big, real])
np.save(f'/home/nel/Desktop/Cali Cube/final_{trial}.npy', final)

#%% impute over all nan values and save as csv for BH3D code
final = np.load(f'/home/nel/Desktop/Cali Cube/final_{trial}.npy')
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(random_state=0, max_iter=500)
final = imp.fit_transform(final)

import pandas as pd
final_df = pd.DataFrame(final, columns = ['FL_x', 'FL_y', 'FR_x', 'FR_y', 'BOT_x', 'BOT_y', 'BL_x', 'BL_y',
       'BR_x', 'BR_y', 'X', 'Y', 'Z'])
# final_df.dropna(inplace = True)
final_df.to_csv(f'/home/nel/Desktop/Cali Cube/try_{trial}.csv', index=False)

#%% test using BH3D calibration
from bh3D.mapping import mapping
import pandas as pd

coordPath = '/home/nel/Desktop/Cali Cube/try_255.csv'
# print camera labels and order to help with model and DLCPaths below
model_options = pd.read_csv(coordPath).columns
model_options = [opt[:-2] for opt in model_options[:-3][::2]]
print('Cameras labels to reference when defining model and DLCPaths variables below:\n', model_options)

#%% run calibration
model = ['BOT','FL','FR', 'BL', 'BR']

# can set dummy DLC paths as we will not use them
DLCPaths = ['path/to/use_cases/mapping/DLC_bot.csv',
            'path/to/use_cases/mapping/DLC_front_left.csv',
            'path/to/use_cases/mapping/DLC_front_right.csv']

SVR_args = {'kernel':"poly", 'degree':2, 'C':1500}

cal = mapping(model, coordPath, DLCPaths, **SVR_args)
results = cal.calibration_results()

#%% get errors and plot hist
import matplotlib.pyplot as plt

bins = 50

# errors = np.linalg.norm(real-results, axis=1)
errors = np.sqrt(np.sum((real-results)**2, axis=1))
plt.hist(errors, bins)
# plt.xlim(errors.min(), errors.max())
# plt.xticks(np.linspace(errors.min(),errors.max(),10))
# plt.xticks(np.linspace(0,errors.max(),10))
plt.xticks(np.arange(0,errors.max()+.02,.02))

plt.title(f'Histogram of Errors, Thresh = {trial}, {bins} bins')

#%% OLD - testing in lab
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