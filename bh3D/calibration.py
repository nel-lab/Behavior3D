#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:14:49 2020

@author: nel-lab
"""

import os
from functools import reduce    # used to get common bodyparts amoung different cameras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from IPython.display import HTML

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict

#%%
class calibration():
    def __init__(self, model, coordinatePath, regr_model=None):
        self.model = model
        self.regr_model = regr_model
        self.coordinatePath = coordinatePath
        if self.regr_model is None:
            self.build_model()
        print("This is the constructor method.")
    
    def plot_calibration_results():
        #todo
        1
        
    def build_model(self, long = False):
        """ BUILD MODEL - uses 2D coordinates (probably obtained by a calibration experiment) to estimate coordinates in 3D
        Params:
        ------
            coordinatePath: str
                path to coordinate csv file, 
                
            model:list 
                list of model IDs as strings ['fl'/'fr'/'bot'] 
            
            long: bool
                whether to show results and MSE for SVM model
                
        Return:
        -------
            regr: sklearn SVM object 
                trained SVM regressor that converts model cameras (2D) to X/Y/Z
        """
        
        All_Data = pd.read_csv(self.coordinatePath)
    
        mod_col = []
        for m in self.model:
            mod_col.append(m.upper()+'_x')
            mod_col.append(m.upper()+'_y')
        mod_col += ['X','Y','Z']
        All_Data = All_Data[mod_col]
        
        X_train_full, X_test, y_train_full, y_test = train_test_split( 
                All_Data.iloc[:,:-3], All_Data.iloc[:,-3:], random_state=42,  test_size = 0.25)
        
        svm = SVR(kernel="poly", degree=2, C=1500, epsilon=0.01, gamma="scale")
        regr = MultiOutputRegressor(svm)
        regr.fit(X_train_full, y_train_full)
    
        if long: # other function
            test_predictions = regr.predict(X_test)
            test_MSE = mean_squared_error(test_predictions, y_test)
            print('Test MSE: ' + str(test_MSE))
            
            X = All_Data.iloc[:,:-3]
            Y = All_Data.iloc[:,-3:]
            
            CV_predictions = cross_val_predict(regr, X, Y, cv=10)
            MSE = mean_squared_error(CV_predictions, Y)
            print('CV MSE: ' + str(MSE))
    
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(CV_predictions[:,0], CV_predictions[:,1], CV_predictions[:,2], color = 'blue', label = 'pred')
            ax.scatter(All_Data.iloc[:,-3], All_Data.iloc[:,-2], All_Data.iloc[:,-1], color = 'red', label = 'real')
            plt.legend()
    
        self.regr_model = regr
        
    def map_to_3D():
        '''Transforms DLC outputs ffrom multiple cameras into a single 3D representaiton
        
        Parameters:
        ----------
        
            
        
        Returns:
        -------
        
        '''
        if self.regr_model is None:
            raise Exception()#message call build_model
        

#%%
    


def PreProcess_DLC_data(df):
    # PREPROCESS DLC
    # input: DLC csv as df
    # output: processed DLC file with likelihood values and extra info removed

    for y in range (df.shape[1]):        
        df.rename(columns={df.columns[y]:df.iloc[0,y]+'_'+df.iloc[1,y]}, inplace=True)
        
    df_proc = df.iloc[2:,1:]
    
    li_columns = [col for col in df_proc.columns if 'likelihood' in col]
    df_proc = df_proc.drop(columns=li_columns)

    df_proc = df_proc.astype(float).reset_index(drop=1)
        
    return df_proc
#%%
    
# STANDARDIZE COLUMNS
# input: list of preprocessed DLC dfs
# output: list of standardized DLC dfs (only includes common bodyparts/columns), list of bodyparts/columns

def Standardize_columns(dfs):
   
    col = []
    for df in dfs:
        col.append(df.columns.tolist())

    same_col = reduce(np.intersect1d, col)
    if len(same_col) == 0:
        print('no matching body parts amongst all cameras!')
        return None, None
    
    columns = []
    for col in same_col[::2]:
        columns.append(col[:-2]+'_X')
        columns.append(col[:-2]+'_Y')
        columns.append(col[:-2]+'_Z')

    dfs_stand = []
    for df in dfs:
        dfs_stand.append(df[same_col])

    return dfs_stand, columns


#%%

def Predict_Real_Coordinates(dfs, bodyparts, numCameras, regr):
    # PREDICT X/Y/Z
    # input: list of standardized dfs, number of bodyparts, number of cameras, trained regressor
    # output: numpy array of predicted X/Y/Z (rows = frames, columns = X/Y/Z per bodypart - 3*number of bodyparts)

    df1, df2 = dfs[0], dfs[1]

    if numCameras == 3:
        df3 = dfs[2]
    elif numCameras > 3:
        print('only works with 2 or 3 cameras, can implement more later')
        return

    rows = df1.shape[0]
    columns = 2*numCameras
    
    test_data = np.zeros([rows, columns*bodyparts])
    
    for r in range(bodyparts):
        test_data[:,columns*r] = df1.iloc[:,2*r]
        test_data[:,columns*r+1] = df1.iloc[:,2*r+1]
        
        test_data[:,columns*r+2] = df2.iloc[:,2*r]
        test_data[:,columns*r+3] = df2.iloc[:,2*r+1]
        
        if numCameras == 3:
            test_data[:,columns*r+4] = df3.iloc[:,2*r]
            test_data[:,columns*r+5] = df3.iloc[:,2*r+1]
    
    Predictions = np.zeros([rows,3*bodyparts])
    
    for i in range(bodyparts):
        Predictions[:,3*i:(3*i+3)] = regr.predict(test_data[:,columns*i:(columns*i+columns)])
        
    return Predictions

#%%


# RUN ENTIRE PIPELINE
# input: path to coordinate csv, path to folder containing DLC csvs, list of model IDs as strings ['fl'/'fr'/'bot']
# output: df of predicted X/Y/Z coordinates for common bodyparts in model

def RunAll(cpath, dlcpath, model):
    
    # build model
    model_regr = Build_Model(cpath, model)

    # preproccess DLC data
    front_left = pd.read_csv(dlcpath + '/front_left.csv')
    front_right = pd.read_csv(dlcpath + '/front_right.csv') 
    bot = pd.read_csv(dlcpath + '/bot.csv')
    # back_left = pd.read_csv(dlcpath + '/back_left.csv')
    # back_right = pd.read_csv(dlcpath + '/back_right.csv')

    mod_front_left = PreProcess_DLC_data(front_left)
    mod_front_right = PreProcess_DLC_data(front_right)
    mod_bot = PreProcess_DLC_data(bot)
    # mod_back_left = PreProcess_DLC_data(back_left)
    # mod_back_right = PreProcess_DLC_data(back_right)

    # pull dfs based on model
    dfs = []
    for m in model:
        if m.lower() == 'fl':
            dfs.append(mod_front_left)
        elif m.lower() == 'fr':
            dfs.append(mod_front_right)
        elif m.lower() == 'bot':
            dfs.append(mod_bot)
        elif m.lower() == 'bl':
            dfs.append(mod_back_left)
        elif m.lower() == 'br':
            dfs.append(mod_back_right)
        else:
            print('model is not in form FL/FR/BOT/BL/BR')
            return

    # standardize dfs
    dfs_stand, columns = Standardize_columns(dfs)
    if not dfs_stand or not columns:
        return

    # predict X/Y/Z for all bodyparts
    bodyparts = dfs_stand[0].shape[1]//2
    numCameras = len(dfs_stand)
    m1 = Predict_Real_Coordinates(dfs_stand, bodyparts, numCameras, model_regr)
    m1df = pd.DataFrame(m1)
    m1df.columns = columns

    # remove tail from models with only 2 cameras since 3 camera model should be better
    if numCameras == 2 and 'tail' in [c[:4].lower() for c in columns]:
        cols = [c for c in m1df.columns if c.lower()[:4] != 'tail']
        m1df=m1df[cols]

    return m1df
