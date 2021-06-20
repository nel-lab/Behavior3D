# Behavior3D

Package to calibrate and map mutiple camera points of view into a single 3D position. Using [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) labeled data, one can then reconstruct the 3D position of tracked body parts from multiple cameras. Calibration and acquisition of behavior movies is performed through an inexpensive and reproducable setup using PS3Eye cameras.  

This package has been developed for calibration and mapping using 3 cameras, but the code base can be generalized/altered to accomidate a different number. It has been tested on Linux and Mac OS.

## Installation
* Install [Anaconda](https://www.anaconda.com/products/individual)
* clone Behavior3D repo
* run ```conda env create -f environment.yml -n Behavior3D```
* run ```conda activate Behavrio3D```
* clone pseyepy source code from https://github.com/bensondaled/pseyepy
  * pseyepy is used for caputuring video with PS3Eye cameras
* run ```python setup.py install```
  * although the original pseyepy repo states to run ```sudo python setup.py install```, this created issues for me which were solved by simply running ```python setup.py install``` instead. More troubleshooting can be found [here](https://github.com/nel-lab/pseye)

## Usage
A typical workflow will follow these steps. Instructions are written here but also referenced in cell blocks of the corresponding scripts.
### 1. calibration.py
> This script enables you to capture images that you can use to calibrate your cameras. The set of images for each camera and user-defined camera labels are saved > as one npz file. It is best run in blocks via Spyder or imported to a Jupyter Notebook.
> 
> 1. create realPoints.csv file with planned real X,Y,Z coordinates, should be in form:  
> > X, Y, Z  
> > 0, 0, 0  
> > 0, 0, 10  
> > 0, 0, 20  
> > ...
>
> 2. run in terminal to activate usb cameras (Linux):
> > ```sudo chmod o+w /dev/bus/usb/001/*```  
> > ```sudo chmod o+w /dev/bus/usb/002/*```    
> > ```sudo chmod o+w /dev/bus/usb/003/*```
>         
> Things to keep in mind:
> * Plan ahead when condsidering your real world coordinate system
>   * moving something from one peg to another is approximately 25.4mm. With our micromanipulator, you have approximately 40 milimeters of horizontal freedom and 30 milimeters of vertical freedom
> * Regular increments make it easier to tell if you're off or missed a picture
>   * suggestions - 5, 8, or 10mm horizontally, 5, or 10mm vertically

### 2. label_points.py
> This script allows user to select 2D calibration point in each camera for each frame. The user should click on the same reference point in each frame/angle (for example, tip of micromanipulator). The reference point should be visible in all cameras in each frame. It uses the npz file created in step 1 (calibration) and saves a csv file containing all the info needed to map the 2D cameras to the 3D representation in step 5 (map)
### 3. aquisition.py
> This script provides a method to aquire behavior recordings from multiple cameras and save as avi movies. These movies can then be processed using DeepLabCut
### (4. track relevant points using DeepLabCut (DLC))
### 5. map.py
> This module creates the multi-camera 2D --> 3D mapping using a support vector regression model. This is done by preprocessing and standardizing DLC output files to ensure tracked body parts appear in all camera views. It then maps the multi-camera 2D points to 3D using the trained calibration model. Finally, a filter can be used to smooth out the recontrstructed 3D points
