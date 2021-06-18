#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 01:57:57 2021

@author: jimmytabet
"""

#%% 
from cali_recon.scripts.map import calibration

coordPath = './setup_old/model_1_coordinates.csv'
DLCPaths = ['./setup_old/bot.csv','./setup_old/front_left.csv','./setup_old/front_right.csv']

cal = calibration(['BOT','FL','FR'], coordPath, DLCPaths)

# filtered data
data1 = cal.map_to_3D()