#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:14:49 2020

@author: jimmytabet

Mapping module for Behavior3D. Trains SVR model on camera calibration data and 
maps multi-camera 2D tracking coordinates from DeepLabCut (DLC) to 3D.
"""

#%% imports
from functools import reduce
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

from scipy.signal import medfilt
from scipy import interpolate

#%% functions
def PreProcess_DLC_data(df):
    '''
    Preprocesses DLC dfs.

    Parameters
    ----------
    df : pandas df
        DLC dataframe.

    Returns
    -------
    df_proc : pandas df
        Preprocessed DLC dataframe with likelihood values and extra info removed.

    '''

    # remove extra header info
    for y in range(df.shape[1]):        
        df.rename(columns={df.columns[y]:df.iloc[0,y]+'_'+df.iloc[1,y]}, inplace=True)
    
    df_proc = df.iloc[2:,1:]
    
    # remove likelihood values
    li_columns = [col for col in df_proc.columns if 'likelihood' in col]
    df_proc = df_proc.drop(columns=li_columns)

    df_proc = df_proc.astype(float).reset_index(drop=1)
        
    return df_proc

def Standardize_columns(dfs):
    '''
    Standardize preprocessed DLC dfs.

    Parameters
    ----------
    dfs : list
        List of preprocessed DLC dataframea.

    Returns
    -------
    dfs_stand : list
        List of standardized DLC dfs. These dfs only include common bodyparts/columns.
    columns : list
        List of common bodyparts/columns as 3D coordinates (X/Y/Z) for mapping.

    '''

    # get every column name in each df
    col = []
    for df in dfs:
        col.append(df.columns.tolist())

    # get common columns
    same_col = reduce(np.intersect1d, col)

    # raise error if no common columns
    if len(same_col) == 0:
        raise ValueError('No common body parts amongst cameras')
    
    # get list of common columns/bodyparts to prepare for 3D mapping
    columns = []
    for col in same_col[::2]:
        columns.append(col[:-2]+'_X')
        columns.append(col[:-2]+'_Y')
        columns.append(col[:-2]+'_Z')

    # get list of dfs only including common columns
    dfs_stand = []
    for df in dfs:
        dfs_stand.append(df[same_col])
        
    return dfs_stand, columns

def Predict_Real_Coordinates(dfs, bodyparts, numCameras, regr):
    '''
    Predict real 3D coordinates from multi-camera 2D points through SVR mapping.
    Can work with 2 or 3 cameras.

    Parameters
    ----------
    dfs : list
        List of standardized DLC dfs. These dfs only include common bodyparts/columns.
    bodyparts : int
        Number of bodyparts that are tracked.
    numCameras : int
        Number of cameras tracking bodyparts.
    regr : sklearn SVR
        Trained support vector regression that maps 2D points to 3D.

    Returns
    -------
    Predictions : numpy array
        Predicted 3D coordinates of each bodypart for each frame.
        Rows = frames, columns = 3D coordinate per bodypart (BP1X Y Z BP2X Y Z...).

    '''

    # save dfs from list
    df1, df2 = dfs[0], dfs[1]

    if numCameras == 3:
        df3 = dfs[2]
    elif numCameras > 3:
        raise ValueError('3D mapping only works with 2 or 3 cameras for now')

    # set number of rows (frames) and columns (X/Y for each camera) for DLC data
    rows = df1.shape[0]
    columns = 2*numCameras
    
    # init test_data
    test_data = np.zeros([rows, columns*bodyparts])
    
    # for each bodypart, import corresponding camera X/Y coords
    for r in range(bodyparts):
        # camera 1
        test_data[:,columns*r] = df1.iloc[:,2*r]
        test_data[:,columns*r+1] = df1.iloc[:,2*r+1]
        
        # camera 2
        test_data[:,columns*r+2] = df2.iloc[:,2*r]
        test_data[:,columns*r+3] = df2.iloc[:,2*r+1]
        
        # camera 3
        if numCameras == 3:
            test_data[:,columns*r+4] = df3.iloc[:,2*r]
            test_data[:,columns*r+5] = df3.iloc[:,2*r+1]
    
    # init predictions array
    Predictions = np.zeros([rows,3*bodyparts])
    
    # for each bodypart, map 2D coords to 3D
    for i in range(bodyparts):
        Predictions[:,3*i:(3*i+3)] = regr.predict(test_data[:,columns*i:(columns*i+columns)])
        
    return Predictions

def simple_filter(df):
    '''
    Filter 3D reconstruction to smooth points.

    Parameters
    ----------
    df : pandas df
        Dataframe to be filtered.

    Returns
    -------
    df_filter : pandas df
        Filtered dataframe.

    '''
    
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

#%% calibration class
class mapping():
    '''
    Class for calibration and 2D -> 3D mapping of DLC data.
    
    Attributes
    ----------
    model : list
        List of camera labels as strings ('cam0/1/2', 'FL/FR/BOT', etc.).
    coordinatePath : str
        Path to coordinate calibration file.
    DLCpaths : str
        Paths to DLC data for each camera.
    regr_model : regression model
        Trained regressor that converts model cameras (2D) to X/Y/Z.
    raw_results: pandas df 
        Dataframe of predicted X/Y/Z coordinates for common bodyparts in model.
    filtered_results : pandas df
        Filitered 3D mapping of DLC coordinates.
        
    Methods
    -------
    build_model():
        Uses 2D coordinates (obtained by calibration experiment) to estimate coordinates in 3D.
    calibration_results(save=False):
        Calibration results.
    raw_model():
        Transforms DLC outputs from multiple cameras into a single 3D representation.
    map_to_3D(self, filter=simple_filter, raw=None):
        Run full pipeline to build model for 2D -> 3D mapping, transform DLC output
        to 3D points, and filter data.
        
    '''
    
    def __init__(self, model, coordPath, DLCPaths, regr_model=None, **kwargs):
        '''
        Initialization.

        Parameters
        ----------
        model : list
            List of camera labels as strings ('cam0/1/2', 'FL/FR/BOT', etc.).
        coordPath : str
            Path to coordinate calibration file.
        DLCpaths : list
            List of paths to DLC data for each camera.
        regr_model : regression model, optional
            Pre-trained regression model that maps 2D points to 3D. The default 
            is None, which builds a regression model using the build_model method
            (sklearn SVR).
        **kwargs : misc
            Keyword arguments passed into sklearn's SVR class, for example 'kernel', 
            'degree', and 'C' parameters.

        Returns
        -------
        None.

        '''
        
        self.model = model
        self.coordinatePath = coordPath

        # make sure model entries are valid camera label options
        coords = pd.read_csv(self.coordinatePath)        
        model_options = [i.split('_')[0] for i in coords.columns[:-3]]
        model_options = list(set(model_options))
        if not set(model).issubset(model_options):
            raise ValueError(f'One or more model entry is not a valid camera label {tuple(model_options)}.')
        
        self.DLCpaths = DLCPaths
        self.regr_model = regr_model
        if self.regr_model is None:
            self.build_model(**kwargs)
        
    def build_model(self, **kwargs):
        """
        Uses 2D coordinates (obtained by calibration experiment) to estimate coordinates in 3D.
        
        Parameters
        ----------
        **kwargs : misc
            Keyword arguments passed into sklearn's SVR class, for example 'kernel', 
            'degree', and 'C' parameters.
                        
        Returns:
        -------
        regr: sklearn SVM object 
            trained regressor that converts model cameras (2D) to X/Y/Z
                
        """
        
        All_Data = pd.read_csv(self.coordinatePath)

        mod_col = []
        for m in self.model:
            mod_col.append(m.upper()+'_x')
            mod_col.append(m.upper()+'_y')
        mod_col += ['X','Y','Z']
        All_Data = All_Data[mod_col]
        
        X = All_Data.iloc[:,:-3]
        y = All_Data.iloc[:,-3:]
                
        svm = SVR(**kwargs)
        regr = MultiOutputRegressor(svm)
        regr.fit(X, y)


        self.regr_model = regr
        print('Model Built')

    def calibration_results(self, save=False):
        '''
        Calibration results.

        Parameters
        ----------           
        save : bool or str, optional
            Save plot of 3D calibration predictions vs actual points. If a str
            is passed, the figure will be saved with that name/location. The 
            default is False. 

        Returns
        -------
        None. Plots calibration results and prints R^2 and MSE values.

        '''

        All_Data = pd.read_csv(self.coordinatePath)
        
        mod_col = []
        for m in self.model:
            mod_col.append(m.upper()+'_x')
            mod_col.append(m.upper()+'_y')
        mod_col += ['X','Y','Z']
        All_Data = All_Data[mod_col]
            
        X = All_Data.iloc[:,:-3]
        y = All_Data.iloc[:,-3:]
            
        preds = self.regr_model.predict(X)
        
        R2 = self.regr_model.score(X, y)
        print(f'Model R^2: {R2}')
        MSE = mean_squared_error(y, preds)
        print(f'Model MSE: {MSE}')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(preds[:,0], preds[:,1], preds[:,2], color = 'blue', label = 'pred')
        ax.scatter(All_Data.iloc[:,-3], All_Data.iloc[:,-2], All_Data.iloc[:,-1], color = 'red', label = 'real')
        ax.set_title('Calibration Results')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        plt.legend()

        # save
        if isinstance(save, str):
            plt.savefig(save, dpi=300, transparent=True)
        elif save:
            plt.savefig('calibration results', dpi=300)
        else:
            pass
                
    def raw_model(self):
        '''
        Transforms DLC outputs from multiple cameras into a single 3D representation.
        
        Returns:
        -------
        raw_results: pandas df 
            Dataframe of predicted X/Y/Z coordinates for common bodyparts in model.
        
        '''
        
        # input: path to coordinate csv, path to folder containing DLC csvs, list of model IDs as strings ['fl'/'fr'/'bot']
           
        # preproccess and pull DLC data based on model
        dfs = []
        for path in self.DLCpaths:
            df = pd.read_csv(path)
            mod_df = PreProcess_DLC_data(df)
            dfs.append(mod_df)

        # standardize dfs
        dfs_stand, columns = Standardize_columns(dfs)

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

    def map_to_3D(self, filter=simple_filter, raw=None):
        '''
        Run full pipeline to build model for 2D -> 3D mapping, transform DLC output
        to 3D points, and filter data.

        Parameters
        ----------
        filter : function, optional
            Filter function to apply to data. The default is simple_filter.
        raw : df, optional
            Raw data to be filtered. The default is None.

        Returns
        -------
        filtered_results : pandas df
            Filitered 3D mapping of DLC coordinates.

        '''

        # create raw results if not created already
        if raw is None:
            if not hasattr(self, 'raw_results'):
                self.raw_model()
            
            raw = self.raw_results

        # filter raw results
        self.filtered_results = filter(raw)

        return self.filtered_results