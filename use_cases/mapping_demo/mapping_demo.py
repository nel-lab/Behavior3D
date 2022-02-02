#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 01:57:57 2021

@author: jimmytabet

Mapping demo file. This script takes the model_coordinates csv file as would be 
generated in step 2 (labeling) and creates a 3D mapping. It then takes DeepLabCut 
labeled csv files and maps common bodyparts seen across all cameras to 3D. 
It is best run in blocks using the Spyder IDE, but can also be imported to a 
Jupyter Notebook or run in terminal ('python /path/to/mapping_demo.py').

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
  
Note: The matplotlib backend may need to be changed. Running 

%matplotlib auto

in the IPython console usually does the trick.
"""

#%% imports
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from bh3D.mapping import mapping
from use_cases.mapping_demo import utils

import numpy as np
import pandas as pd

#%% setup - model options
'''
DON'T FORGET TO UPDATE PATHS!
'''



coordPath = '/home/nel/Desktop/Behavior1027_cropped/cali_250.csv'



# print camera labels and order to help with model and DLCPaths below
model_options = pd.read_csv(coordPath).columns
model_options = [opt[:-2] for opt in model_options[:-3][::2]]
print('Cameras labels to reference when defining model and DLCPaths variables below:\n', model_options)

#%% setup - define model/DLC paths
model = [
    'BOT',
    'FL',
    'FR'
    ]

'''
update paths to local paths in repo!
'''



DLCPaths = [
            '/home/nel/Desktop/Behavior1027_cropped/bot.csv',
            '/home/nel/Desktop/Behavior1027_cropped/fl.csv',
            '/home/nel/Desktop/Behavior1027_cropped/fr.csv',
            # '/home/nel/Desktop/Behavior1027/bl.csv',
            # '/home/nel/Desktop/Behavior1027/br.csv'
            ]

SVR_args = {'kernel':"rbf", 'C':15000}

if not set(model).issubset(model_options):
    raise ValueError(f'One or more model entry is not a valid camera label {tuple(model_options)}.')

#%% interpret low confidence points on 2D data
# from bh3D.mapping import PreProcess_DLC_data
# from scipy import interpolate

# def interp(file):
#     # read in csv
#     df = pd.read_csv(file)
    
#     # preprocess (fill low confidence points with nan)
#     df_proc = PreProcess_DLC_data(df, thresh=0.7, filt=False)
    
#     # init interp array
#     interp = np.zeros_like(df_proc)
    
#     # for each trace, interpret over low confidence/nan values
#     for i in range(df_proc.shape[1]):
#         # copy trace
#         series = df_proc.iloc[:,i].copy()
#         if series.isna().all():
#             pass
#         else:
#             # low confidence/nan mask
#             nan_mask = series.isna()
#             # high confidence/good mask
#             good_mask = series.notnull()
#             # interpolate function
#             f = interpolate.interp1d(series[good_mask].index.values, series[good_mask],
#                                      kind='cubic', fill_value='extrapolate')
#             # interpolate over low confidence/nan values
#             series[nan_mask] = f(series[nan_mask].index.values)
#         # store in overall array
#         interp[:,i] = series
    
#     # return interpolated df
#     df_interp = pd.DataFrame(interp, columns=df_proc.columns)
#     return df_interp

# files = [
#     '/home/nel/Desktop/Behavior1027_cropped/bot.csv',
#     '/home/nel/Desktop/Behavior1027_cropped/fl.csv',
#     '/home/nel/Desktop/Behavior1027_cropped/fr.csv',
#     ]

# DLCPaths = []
# for i in files:
#     DLCPaths.append(interp(i).dropna(axis=1))

#%% init mapping class (automatically builds model)
cal = mapping(model, coordPath, DLCPaths, **SVR_args)

#%% display calibration model results
cal.calibration_results()#save='/home/nel-lab/Desktop/Jimmy/BehaviorPaper/cali_results_train_test_split.pdf')

#%% ADJUST FIG THEN SAVE
import matplotlib.pyplot as plt
plt.savefig('/home/nel-lab/Software/Behavior3D/paper_figs/cali_results_train_test_split.pdf', transparent=True, dpi=300)

#%% map DLC data to 3D (also filters data)
data = cal.map_to_3D()

#%% drop head/back/tail and back paws
# data.drop(columns = [col for col in data.columns if (col[0].islower())|(col[0] == 'B')|(col[1] == 'L')], inplace=True)
data.drop(columns = [col for col in data.columns if (col[0].islower())|(col[0] == 'B')], inplace=True)

#%% map each paw seperate
fl_all = mapping(['BOT', 'FL'], coordPath, ['/home/nel/Desktop/Behavior1027_cropped/bot.csv','/home/nel/Desktop/Behavior1027_cropped/fl.csv'], **SVR_args).map_to_3D()
fl = fl_all[[c for c in fl_all.columns if 'FL' in c]]

fr_all = mapping(['BOT', 'FR'], coordPath, ['/home/nel/Desktop/Behavior1027_cropped/bot.csv','/home/nel/Desktop/Behavior1027_cropped/fr.csv'], **SVR_args).map_to_3D()
fr = fr_all[[c for c in fr_all.columns if 'FR' in c]]

data = pd.concat([fl, fr], axis=1)


#%% save 3D reconstruction as csv
# data.to_csv('/home/nel/Desktop/Cali Cube/recon.csv', index=False)

#%% investigate head point
# import matplotlib.pyplot as plt
# head = data[['head_X', 'head_Y','head_Z']]
# print(head.max()-head.min())
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(head.iloc[:,0], head.iloc[:,1], head.iloc[:,2])

#%% let's do this the hard way... - WIP!!!
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# for i in range(10):
#     ax.set_title(f'frame number {i}')
#     plt.cla()
#     # ax.set_xlim([0,100])
#     # ax.set_ylim([0,100])
#     # ax.set_zlim([0,100])
#     # ax.scatter(data.loc[i, 'FLpaw_X'], data.loc[i, 'FLpaw_Y'], data.loc[i, 'FLpaw_Z'], color='blue')
#     ax.scatter(data.loc[i, 'FRpaw_X'], data.loc[i, 'FRpaw_Y'], data.loc[i, 'FRpaw_Z'], color='black')
#     ax.scatter(data.loc[i, 'FRdig1_X'], data.loc[i, 'FRdig1_Y'], data.loc[i, 'FRdig1_Z'], color='red')
#     ax.scatter(data.loc[i, 'FRdig2_X'], data.loc[i, 'FRdig2_Y'], data.loc[i, 'FRdig2_Z'], color='red')
#     ax.scatter(data.loc[i, 'FRdig3_X'], data.loc[i, 'FRdig3_Y'], data.loc[i, 'FRdig3_Z'], color='red')
#     ax.scatter(data.loc[i, 'FRdig4_X'], data.loc[i, 'FRdig4_Y'], data.loc[i, 'FRdig4_Z'], color='red')
#     plt.pause(0.5)
    
# plt.cla()

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
lpt = data[[col for col in data.columns if col[:2] == 'FL' and col[-3] == 'w']]
rpt = data[[col for col in data.columns if col[:2] == 'FR' and col[-3] == 'w']]
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
utils.animate(data=data, wheel_pt1=pt1, wheel_pt2=pt2, R=R, fps=70, save='/home/nel/Desktop/anim.mp4')