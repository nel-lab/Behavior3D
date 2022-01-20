# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 21:58:28 2022

@author: sophi
"""

import numpy as np
import matplotlib.pyplot as plt
a = np.random.randint(50, size= (6,3,5,8))
#print(a)
print(a[0]) #first 3 5x8 arrays
print(a[1]) #next 3 5*8 arrays ... etc.
# first dimension = length of array 
#there are 18 2D arrays that are each 5x8
#num arrays = first 2 dimensions multipled
#6 groups, 3 arrays in each group, 5x8 arrays
    

# CALI DATA = (6000, 5, 480, 640)
#           (frames, cams, x, y)
# 50 fps, 6000 frames, 120 seconds
# 30000 2D arrays that are 480x640 
# 480 = x positions, 640 = y positions
# 6000 "groups", 5 arrays in each group 
# ex: group 1 - frame 1's array of the image taken (480x640)
#   each array represents each camera view 
#       - group 1 array 1 = top, group 1 array 2 = bottom... etc. etc. 
#     group 2 - frame 2's array of image taken... etc. etc. 
# so when the max of the movie data is found along axes 2,3 (maxs = np.max(mov, axis=(2,3)))

max_arr = np.max(a, axis = (2,3)) # shape = (6, 3) for this example
b = np.array([[1,3,6,7], [6, 5, 8,9]])
# finding the shape across the x and y axes of the image:
    # going through and finding the max value of each 480x640 array (image)
    # result is 6000x5 array that gives the max pixel value for each frame for each camera. 
fpst = 50 # frames per second
d_ont = 0.150  #set in arduino code (seconds on of light)
d_offt = 0.050  #set in arduino code (seconds off of light)
t_ont = fpst*d_ont # (frames on of light)
t_offt = fpst*d_offt  # (frames off of light)
t_totalt = t_ont+t_offt # (total frames of light flash on/off for single flash)

start_test = np.argwhere(max_arr==max_arr.max())[0][0]
#result is single value
end_test = int(start_test+8**3*t_totalt)

mov1t = a[start_test:end_test+1]
maxs1t = max_arr[start_test:end_test+1]

# peak frame will be every t_total frames starting at t_total/2
# recall t_total = total frames for one "lighting up period"
max_framest = np.arange(t_totalt/2, t_totalt/2+8**3*t_totalt, t_totalt).astype(int)
# 1D array length 512
# creates array ranging from 5 frames (t_total/2) to 5125 frames incrementing in steps of 10 (frames for a single light up)
# this isolates so we are only getting the frames when the LEDs should be lighting up (halfway through t_total for each LED, respectively)
max_movt = mov1t[max_framest] 
# shape (512, 5, 480, 640)
# this indexes mov1 (5121, 5, 480, 640) to be only the max pixel valued frames 
max_maxst = maxs1t[max_framest] #max pixel values for each camera
# shape (512, 5)
# this is now the array of the max pixel values for each camera. there are 512 max pixel values (one for each LED)

camt = 0
thresht = 250



#%% plot interest point if max > thresh
# opposite of former plots where we were looking at max<thresh

#this time looping through all max frames (512 values)
for framet in range(8**3):
    plt.cla() #clears former axes
    plt.imshow(max_movt[framet,camt]) #shows data as image 
    if max_maxst[framet,camt]>=thresht:
        max_ptt = np.mean(np.argwhere(max_movt[framet,camt] == max_maxst[framet,camt]), axis=0)
        #2 value array  
        plt.plot(*max_ptt[::-1], 'ro')
        # -1 reverses order
    plt.pause(1) 
    
# %%
c = [1, 3, 5, 8, 9, 20, 6, 7]

