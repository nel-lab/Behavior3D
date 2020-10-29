#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:35:53 2020

@author: nel-lab
"""


from bh3D.calibration import calibration
#%%
#Customize this paths dictionary for your local paths. 
pathsDict = {
    'coordPath': '/media/nel-lab/Lexar/Calibration/My Files/My Plotting Files/coords and dlc/model_1_coordinates.csv', 
    'DLCPath': 'D:\data\Behavior data\dlc_data_201020\coords and dlc',
    'PlotPath' : 'D:\data\Behavior data\dlc_data_201020\Plotting files'
}

cal = calibration(model=['bot','fl','fr'])
cal.build_model(pathsDict['coordPath'], long=True)

positions = cal.map_to_3D()


