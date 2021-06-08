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

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

from scipy.signal import medfilt
from scipy import interpolate

#%%
# helper functions

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

def Standardize_columns(dfs):
    # STANDARDIZE COLUMNS
    # input: list of preprocessed DLC dfs
    # output: list of standardized DLC dfs (only includes common bodyparts/columns), list of bodyparts/columns

    col = []
    for df in dfs:
        col.append(df.columns.tolist())

    same_col = reduce(np.intersect1d, col)
    if len(same_col) == 0:
        # replace with raise Exception??
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

def Predict_Real_Coordinates(dfs, bodyparts, numCameras, regr):
    # PREDICT X/Y/Z
    # input: list of standardized dfs, number of bodyparts, number of cameras, trained regressor
    # output: numpy array of predicted X/Y/Z (rows = frames, columns = X/Y/Z per bodypart - 3*number of bodyparts)

    df1, df2 = dfs[0], dfs[1]

    if numCameras == 3:
        df3 = dfs[2]
    elif numCameras > 3:
        # replace with raise Exception??
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

def Andrea_filter(df):
    
    Predictions = df.to_numpy()

    ids = np.arange(0,Predictions.shape[-1])
    PredictionsFilt3 = medfilt(Predictions, kernel_size=(3,1))[:,ids]
    PredictionsFilt9 = medfilt(Predictions, kernel_size=(9,1))[:,ids]

    tol_pix = 6
    count = 0
    for filt3, filt9 in zip(PredictionsFilt3.T,PredictionsFilt9.T):
        PredictionOutl = np.where(np.abs(filt3-filt9)>tol_pix)[0]
        PredictionGood = np.where(np.abs(filt3-filt9)<tol_pix)[0]
        f = interpolate.interp1d(PredictionGood,filt3[PredictionGood], kind='cubic')
        filt3[PredictionOutl]=f(PredictionOutl)
        Predictions[:,count] = filt3
        count+=1

    df_filter = pd.DataFrame(Predictions, columns=df.columns)

    return df_filter

#%%
class calibration():
    def __init__(self, model, pathsDict, regr_model=None):
        self.model = model
        self.coordinatePath = pathsDict['coordPath']
        self.DLCpath = pathsDict['DLCPath']
        self.plotPath = pathsDict['PlotPath']
        self.regr_model = regr_model
        if self.regr_model is None:
            self.build_model()
        
    def build_model(self):
        """
        Uses 2D coordinates (obtained by calibration experiment) to estimate coordinates in 3D
        
        Params:
        ------
            coordinatePath: str
                path to coordinate csv file
                
            model: list 
                list of model IDs as strings ['fl'/'fr'/'bot'] 
            
            long: bool
                whether to show results and MSE for SVM model
                
        Returns:
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
        # add raise Exception if model not in coordinate csv??
        All_Data = All_Data[mod_col]
        
        X_train_full, X_test, y_train_full, y_test = train_test_split( 
                All_Data.iloc[:,:-3], All_Data.iloc[:,-3:], random_state=42,  test_size = 0.25)
        
        svm = SVR(kernel="poly", degree=2, C=1500, epsilon=0.01, gamma="scale")
        regr = MultiOutputRegressor(svm)
        regr.fit(X_train_full, y_train_full)

        self.regr_model = regr
        print('Model Built\n-----------')

    def calibration_results(self, save=False):

        All_Data = pd.read_csv(self.coordinatePath)

        mod_col = []
        for m in self.model:
            mod_col.append(m.upper()+'_x')
            mod_col.append(m.upper()+'_y')
        mod_col += ['X','Y','Z']
        All_Data = All_Data[mod_col]
        
        X_train_full, X_test, y_train_full, y_test = train_test_split( 
                All_Data.iloc[:,:-3], All_Data.iloc[:,-3:], random_state=42,  test_size = 0.25)

        test_predictions = self.regr_model.predict(X_test)
        test_MSE = mean_squared_error(test_predictions, y_test)
        print('\nCalibration Results\n-------------------')
        print('\tTest MSE: ' + str(test_MSE))
            
        X = All_Data.iloc[:,:-3]
        Y = All_Data.iloc[:,-3:]
            
        CV_predictions = cross_val_predict(self.regr_model, X, Y, cv=10)
        MSE = mean_squared_error(CV_predictions, Y)
        print('\tCV MSE: ' + str(MSE))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(CV_predictions[:,0], CV_predictions[:,1], CV_predictions[:,2], color = 'blue', label = 'pred')
        ax.scatter(All_Data.iloc[:,-3], All_Data.iloc[:,-2], All_Data.iloc[:,-1], color = 'red', label = 'real')
        ax.set_title('Calibration Results')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        plt.legend()

        if save:
            if not os.path.isdir(self.plotPath):
                os.mkdir(self.plotPath)
                print('\nPlots Folder Created')

            plt.savefig(self.plotPath+'/calibration results')
            print('\nCalibration Results Saved')
        
        plt.show()
        
        return CV_predictions

    def raw_model(self):
        '''
        Transforms DLC outputs from multiple cameras into a single 3D representation
        
        Params:
        ------
        
        Returns:
        -------
        
        '''

        # RUN ENTIRE PIPELINE
        # input: path to coordinate csv, path to folder containing DLC csvs, list of model IDs as strings ['fl'/'fr'/'bot']
        # output: df of predicted X/Y/Z coordinates for common bodyparts in model
           
        # preproccess and pull DLC data based on model
        dfs = []
        for m in self.model:
            if m.lower() == 'fl':
                df = pd.read_csv(self.DLCpath + '/front_left.csv')
            elif m.lower() == 'fr':
                df = pd.read_csv(self.DLCpath + '/front_right.csv') 
            elif m.lower() == 'bot':
                df = pd.read_csv(self.DLCpath + '/bot.csv')
            elif m.lower() == 'bl':
                df = pd.read_csv(self.DLCpath + '/back_left.csv')
            elif m.lower() == 'br':
                df = pd.read_csv(self.DLCpath + '/back_right.csv')
            else:
                # replace with raise Exception??
                print('model is not in form FL/FR/BOT/BL/BR')
                return

            mod_df = PreProcess_DLC_data(df)
            dfs.append(mod_df)

        # standardize dfs
        dfs_stand, columns = Standardize_columns(dfs)
        if not dfs_stand or not columns:
            # replace with raise Exception??
            return

        # predict X/Y/Z for all bodyparts
        bodyparts = dfs_stand[0].shape[1]//2
        numCameras = len(dfs_stand)
        m1 = Predict_Real_Coordinates(dfs_stand, bodyparts, numCameras, self.regr_model)
        m1df = pd.DataFrame(m1)
        m1df.columns = columns

        # # remove tail from models with only 2 cameras since 3 camera model should be better
        # if numCameras == 2 and 'tail' in [c[:4].lower() for c in columns]:
        #     cols = [c for c in m1df.columns if c.lower()[:4] != 'tail']
        #     m1df=m1df[cols]

        self.raw_results = m1df

        return self.raw_results

    def map_to_3D(self, filter=Andrea_filter, raw=None):

        if raw is None:
            if not hasattr(self, 'raw_results'):
                self.raw_model()
            
            raw = self.raw_results

        self.filtered_results = filter(raw)

        return self.filtered_results
