#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 01:57:57 2021

@author: jimmytabet

Mapping demo file. This script takes the model_coordinates csv file as would be 
generated in step 2 (labeling) and creates a 3D mapping. It then takes DeepLabCut 
labeled csv files and maps common bodyparts seen across all cameras to 3D. 
It is best run in blocks via the Spyder IDE or imported to a Jupyter Notebook.

Paths point to files in the use_cases/mapping folder of the Behavior3D repo. 
The DLC files would be generated from DeepLabCut, while the model_coordinates.csv 
file would be generated in step 2 (labeling) as a result of the camera calibration 
steps. Paths should be updated to reflect the local path of associated files in 
the use_cases/mapping folder of the Behavior3D repo.

It is imperative that the order of the DLCPaths list corresponds to the order of 
the model variable. For reference, the user-generated camera labels are printed 
in this script. Proper naming of the behavior movies as described in step 3 
(acquisition) should help this process.

For example, below the camera labels were defined as 'FL', 'FR', and 'BOT'. The 
model variable is defined in the order ['BOT','FL','FR'], so it is imperative 
that the DLCPaths are listed in the following order: 
(['DLC_bot.csv', 'DLC_front_left.csv'', 'DLC_front_right.csv'])
  
Note: the matplotlib backend may need to be changed, try using Qt5Agg 
(run '%matplotlib qt5' in IPython console) 
"""

#%% imports
from bh3D.mapping import mapping
from use_cases.mapping_demo import utils

import numpy as np
import pandas as pd

#%% setup - model options
'''
update paths to local paths in repo!
'''

coordPath = 'path/to/use_cases/mapping/demo_model_coordinates.csv'

# print camera labels and order to help with model and DLCPaths below
model_options = pd.read_csv(coordPath).columns
model_options = [opt[:-2] for opt in model_options[:-3][::2]]
print('Cameras labels to reference when defining model and DLCPaths variables below:\n', model_options)

#%% setup - define model/DLC paths
model = ['BOT','FL','FR']

'''
update paths to local paths in repo!
'''

DLCPaths = ['path/to/use_cases/mapping/DLC_bot.csv',
            'path/to/use_cases/mapping/DLC_front_left.csv',
            'path/to/use_cases/mapping/DLC_front_right.csv']

SVR_args = {'kernel':"poly", 'degree':2, 'C':1500}

if not set(model).issubset(model_options):
    raise ValueError(f'One or more model entry is not a valid camera label {tuple(model_options)}.')

#%% init mapping class (automatically builds model)
cal = mapping(model, coordPath, DLCPaths, **SVR_args)

#%% display calibration model results
cal.calibration_results()

#%% map DLC data to 3D (also filters data)
data = cal.map_to_3D()

#%% use mapping to generate figures/animations
'''
The following code blocks use the reconstructed 3D points to show a few things. 
Using functions in contained in utils.py, the wheel that the mouse was running on 
can be plotted along with a snapshot of the paw points ('scatter' function), as 
well as a 3D reconstruction of the mouse paws during the behavior video 
('animation' function). These are just examples of use cases for visualization, 
but for more in depth behavioral analysis, check out the UMouse package!

IMPORTANT:
Saving the animations proved to be tricky depending on OS, but I found that running 

"conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge"

(from https://superuser.com/questions/1420351/how-to-use-libx264-ffmpeg-in-conda-environment)

allowed ffmpeg to save the animations on Linux for me.

If you do not save the animations, the movies do not appear smooth/at the camera fps. 
These animations can be found under the 'use_cases/mapping' folder in the Behavior3D repo.

A quick google search of any errors you may encounter when saving can usually 
point you in the right direction to solve the issue.
'''
#%% define wheel endpoints and radius for plotting
lpt = data[[col for col in data.columns if col[0] == 'L' and col[-3] == 'w']]
rpt = data[[col for col in data.columns if col[0] == 'R' and col[-3] == 'w']]
lmean = lpt.mean().to_numpy()
rmean = rpt.mean().to_numpy()
mean = np.row_stack([lmean,rmean])
mean = np.mean(mean, axis=0)
R = 216/2
L = 89
mean[0] += -1.5
mean[2] += -R+1
mean[1] += 27
pt1 = mean.copy()
pt2 = mean.copy()
pt1[0] -= L/2
pt2[0] += L/2
pt1[2] += 4

#%% cumulative plot of paw points
utils.scatter(data=data, wheel_pt1=pt1, wheel_pt2=pt2, R=R, rot=False, save=False)

#%% animation
anim = utils.animate(data=data, wheel_pt1=pt1, wheel_pt2=pt2, R=R, fps=70, save=False)
