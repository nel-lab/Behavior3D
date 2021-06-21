# Behavior3D

Package to calibrate and map multiple camera points of view into a single 3D position. Using [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) labeled data, one can then reconstruct the 3D position of tracked body parts from multiple cameras. Calibration and acquisition of behavior movies is performed through an inexpensive and reproducible setup using PS3Eye cameras.  

This package has been developed for calibration and mapping using 3 cameras, but the code base can be generalized/altered to accommodate a different number. It has been tested on Linux and is best run using the Spyder IDE or imported to a Jupyter Notebook.  

Note: The matplotlib backend may need to be changed. Running ```%matplotlib auto``` usually does the trick.

## Installation
* Install [Anaconda](https://www.anaconda.com/products/individual)
* clone Behavior3D repo
* cd into Behavior3D directory
* run ```conda env create -f environment.yml -n Behavior3D```
* run ```conda activate Behavior3D```
* clone pseyepy source code from https://github.com/bensondaled/pseyepy
  * pseyepy is used for capturing video with PS3Eye cameras
* cd into pseyepy directory
* run ```sudo path/to/env/python setup.py install```
  * it is important to specify the path to the Behavior3D environment python when using sudo. This path can be found by running ```which python``` and copying this path

## Usage
A typical workflow will follow these steps. Instructions are written here but also referenced in cell blocks of the corresponding scripts. A short example for calibration, labeling, and acquisition is provided within the scripts and point to associated output files in the ```use_cases``` folder.
### 1. ```calibration.py```
> This script enables you to capture images that you can use to calibrate your cameras. The set of images for each camera and user-defined camera labels are saved as one npz file. It is best run in blocks via the Spyder IDE or imported to a Jupyter Notebook.
> 
> **Instructions**
> 1. create realPoints.csv file (see example in ```use_cases/calibration``` folder) with planned real X,Y,Z coordinates, should be in form:  
> X, Y, Z  
> 0, 0, 0  
> 10 0, 0  
> 20 0, 0  
> ...
> 
> 2. you may need to run the following in terminal to activate usb cameras (Linux):  
> ```sudo chmod o+w /dev/bus/usb/001/*```  
> ```sudo chmod o+w /dev/bus/usb/002/*```    
> ```sudo chmod o+w /dev/bus/usb/003/*```
>         
> The script will walk you through the calibration snapshots (in 'capture calibration snapshots' cell), but plan ahead to make sure ALL real calibration coordinates can be seen in EVERY camera!

### 2. ```labeling.py```
> This script allows user to select 2D calibration point in each camera for each frame. User should click on same reference point in each frame/angle (for example, tip of micromanipulator). The reference point should be visible in all cameras in each frame.
> 
> It uses the npz file created in step 1 (calibration) and saves a model_coordinates csv file (see example in ```use_cases/labeling``` folder) containing all the info needed to map the 2D cameras to the 3D representation in step 5 (mapping). It is best run in blocks via the Spyder IDE or imported to a Jupyter Notebook.
### 3. ```aquisition.py```
> This script allows for behavior movie acquisition from multiple camera angles. After acquiring the movies, the user will be prompted to label each camera view. Movies are saved as npz files for each camera with the movie and timestamps. Movies are also saved seperately in the user-specified format (.avi, .mp4, .mov, etc.). Examples of these outputs can be found in the ```use_cases/acquisition``` folder. It is best run in blocks via the Spyder IDE or imported to a Jupyter Notebook.
>
> You may need to run the following in terminal to activate usb cameras (Linux):  
> ```sudo chmod o+w /dev/bus/usb/001/*```  
> ```sudo chmod o+w /dev/bus/usb/002/*```    
> ```sudo chmod o+w /dev/bus/usb/003/*```
### (4. Track relevant points using DeepLabCut (DLC))
### 5. ```mapping.py```
> This module creates the multi-camera 2D --> 3D mapping using a support vector regression model. This is done by preprocessing and standardizing DLC output files (see examples in ```use_cases/mapping``` folder) to ensure tracked body parts appear in all camera views. It then maps the multi-camera 2D points to 3D using the trained calibration model. Finally, a filter can be used to smooth out the recontrstructed 3D points. 
> 
> For the mapping class, it is imperative that the order of the DLCPaths list corresponds to the order of the model variable. See the ```mapping_demo.py``` file for more explanation.

## Demo
A full demo using real behavior videos can be run using the ```mapping_demo.py``` file found in the ```use_cases/mapping_demo``` folder. This demo takes a model_coordinates.csv file (as would be generated in step 2 - labeling) and DLC files to reconstruct a head-fixed mouse walking on a wheel. It includes some visualizations of the 3D reconstruction. For more behavioral analysis, check out our [UMouse repo](https://github.com/nel-lab/UMouse)!

## Developers
* Jimmy Tabet, UNC
