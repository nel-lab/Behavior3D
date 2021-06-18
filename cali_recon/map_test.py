#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 01:57:57 2021

@author: jimmytabet
"""

#%% 
from cali_recon.scripts.map import calibration

coordPath = 'use_cases/setup_old/model_1_coordinates.csv'
DLCPaths = ['use_cases/setup_old/bot.csv','use_cases/setup_old/front_left.csv','use_cases/setup_old/front_right.csv']

SVR_args = {'kernel':"poly", 'degree':2, 'C':1500}

cal = calibration(['BOT','FL','FR'], coordPath, DLCPaths, **SVR_args)

# filtered data
data1 = cal.map_to_3D()