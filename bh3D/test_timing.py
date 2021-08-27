#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:00:41 2021

@author: nel-lab
"""
from time import time
import numpy as np
#%%
a = np.random.random((1000,5,480,640)).astype(np.float16)
#%%
time_start = time()
times = []
for idx,frames in enumerate(a):
    np.save('test/img_'+str(idx),frames)
    times.append(time()-time_start)

#%%
plt.plot(np.diff(times),'.')

#%%
from tempfile import mkdtemp
import os.path as path
filename = path.join(mkdtemp(), 'newfile.dat')
#%%
fp = np.memmap(filename, dtype='float32', mode='w+', shape=(1000,5,480,640))
time_start = time()
times = []
for idx,frames in enumerate(a):
    # np.save('test/img_'+str(idx),frames)
    fp[idx] = frames
    times.append(time()-time_start)
plt.plot(np.diff(times),'.')
