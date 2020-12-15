#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 21:45:18 2020

@author: jimmytabet
"""

#%% import modules
import cv2
print('importing caiman...')
import caiman as cm
print('\ncaiman imported\n')
import numpy as np
from skimage.util import montage

#%% function to compare all behavior points/clusters at once
#   NOTE: if you'd like to rerun with different raw video, first del behavior_montage.mov
def behavior_montage(mp4_path, ids, shrink_factor=1, spread=100, montage_shape=None):

    # check for montage_shape/set it as ids.T.shape (no_points x no_clusters)
    if not montage_shape:
        montage_shape = ids.T.shape

    # check if mp4 has been processed before (if not, process)
    if not hasattr(behavior_montage, 'mov'):
        # read in mp4 frames
        cap = cv2.VideoCapture(mp4_path)  # path to raw video
        mov = []
        counter = 0
        while True:
           ret, frame = cap.read()
           if ret == False:
               break
           mov.append(frame[:,:,-1])
           if counter%1000 == 0:
               print('processing frame', counter)
           counter += 1
        
        print('mp4 processed\n')
        
        # create caiman movie from frames
        mov_caiman = cm.movie(np.array(mov))
        del mov
        print('caiman movie created from mp4\n')
        
        behavior_montage.mov = mov_caiman
    
    # create montage
    mov_caiman_shr = behavior_montage.mov.resize(1/shrink_factor,1/shrink_factor,1)
    
    # alert if edge case(s) detected
    end = behavior_montage.mov.shape[0]
    if (ids-spread<0).any() or (ids+spread>end-1).any():
        print('EDGE CASE IDENTIFIED')
        print('select id(s) within range or change spread')
        print('offending id(s) = TRUE: (array returned to variable)\n')
        bad_pts = (ids-spread<0) | (ids+spread>end-1)
        print(bad_pts)
        print()
        return bad_pts
        # alternatively, shift to fit in range of spread
        # print('edge case identified, auto-adjusting id(s)\n')
        # ids[ids-spread<0] = spread
        # ids[ids+spread>end-1] = end-1-spread
    
    ids_flat = ids.T.ravel()    # must flatten array first

    big_mov = []
    # plot (i +/- spread) frames
    for i in range(2*spread+1):
        big_frame  = montage(mov_caiman_shr[ids_flat-spread+i], grid_shape=montage_shape)
        big_mov.append(big_frame)
        
    # create caiman movie from montage frames
    big_mov = cm.movie(np.array(big_mov))
    print('caiman movie montage created')
    
    return big_mov

#%% to save/load montage most efficiently with smallest file sizes:
# test.save('test.hdf5', to32=False)
# import caiman as cm
# cm.load('test.hdf5', outtype=np.uint8)