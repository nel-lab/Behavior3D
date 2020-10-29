#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:30:20 2020

@author: nel-lab
"""
import numpy as np
import os 
import pylab as plt
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import matplotlib
from scipy.fftpack import fftshift
import peakutils
from peakutils.plot import plot as pplot


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#%% 
#path = '/home/nel-lab/Desktop/WS_tests_for_DLC2/calibration2/' 
path = '/Applications/NoniCloud Desktop/GiovannucciLab/cal_images_2'
os.chdir(path)
rootDirectory = path

#%%
movDirectory = rootDirectory + '/mov_frames/'
if os.path.isdir(movDirectory) == False:
    os.mkdir(movDirectory)

figDirectory = rootDirectory + '/figDirectory/'
if os.path.isdir(figDirectory) == False:
    os.mkdir(figDirectory)

#%% exponentially weighted mean smoothing 
pred3D = pd.read_csv('m1_Predictions.csv').to_numpy()
Predictions = pred3D.reshape((6000,30))    
pred3D_df = pd.DataFrame(Predictions)
pred3D_ewm = pred3D_df.ewm(com=2).mean() 
pred3D_2 = pred3D_ewm.to_numpy()
pred3D_2 = pred3D_2.reshape((6000,10,3))
np.save('Smoothed_3D_predictions.npy', pred3D_2)

#%%
def myround(x, base):
    return x // base * base

#%% Movie of paws in 3D

pred3D_2 = np.load('Smoothed_3D_predictions.npy')
import matplotlib.style

from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=['yellow', 'greenyellow', 'green','aqua', 'dodgerblue', 'blueviolet', 'magenta', 'red', 'orangered', 'orange' ])
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fnames = []
count = 0
start = 2520

X_lower = int(np.round(np.amin(pred3D_2[:,:,0]) - 2))
X_upper = int(np.round(np.amax(pred3D_2[:,:,0]) + 2))
Y_lower = int(np.round(np.amin(pred3D_2[:,:,1]) - 3))
Y_upper = int(np.round(np.amax(pred3D_2[:,:,1]) + 3))
Z_lower = int(np.round(np.amin(pred3D_2[:,:,2]) - 1))
Z_upper = int(np.round(np.amax(pred3D_2[:,:,2]) + 1))

x_ticks = np.arange(myround(X_lower,5), myround(X_upper,5)+5, 10)
y_ticks = np.arange(myround(Y_lower,10), myround(Y_upper,10)+10, 20)
z_ticks = np.arange(myround(Z_lower,2), myround(Z_upper,2)+2, 4)

for y in range(400):
#for y in range(1):
    plt.cla()
    print(count)
    for x in range(pred3D_2.shape[1]):
        ax.scatter(*pred3D_2[start+y,x])
        ax.set_xlim([X_lower,X_upper])
        ax.set_xlabel('X')
        plt.xticks(x_ticks)
        ax.set_ylim([Y_lower,Y_upper])
        ax.set_ylabel('Y')
        plt.yticks(y_ticks)
        ax.set_zlim([Z_lower,Z_upper])
        ax.set_zlabel('Z')
        ax.set_zticks(z_ticks)
        ax.view_init(25, 250+count) # adding count changes degree of rotation by 1 each frame 
    fnames.append('mov_frames/fr_'+str(count).zfill(5)+'.png')    
    plt.savefig(fnames[-1])
    count += 1

#%%
pred3D = np.load('Filtered_Predictions.npy')
smoothed = np.load('Smoothed_3D_predictions.npy')

#%% Empirical observation of average step time
plt.figure(figsize=(30,8))
x = np.arange(0,6000,1)
pred1D = pred3D[:,5,1]
plt.plot(x,pred1D) # steps last about 50 frames, which is 1.4Hz
plt.show()

#%% Testing effects of high Pass filter
'''
We have 6000 frames, at 70FPS
steps last about 50 frames, which is 1.4Hz, we use a 1Hz Butterworth filter to be 
safe - NOTE: future analyses might require lower, or higher filters
'''

b, a = signal.butter(1, 1, 'hp', fs=70)
smoothedb = signal.filtfilt(b, a, smoothed[:,0,1], padlen=150)

x = np.arange(0,6000,1)
plt.figure(figsize=(15,8))
plt.subplot(111)
plt.plot(x,pred3D[:,0,1]) # steps last about 50 frames, which is 1.4Hz
plt.plot(x,smoothedb)
plt.show()
plt.savefig('HighpassFilter_Y.png')


#%% smoothed with exponentially weighted mean, and high pass filter
smoothed = smoothed.reshape(6000,30)

b, a = signal.butter(1, 1, 'hp', fs=70)
for i in range(smoothed.shape[1]):
    smoothed[:,i] = signal.filtfilt(b, a, smoothed[:,i], padlen=150)
    
ewm_hp_3D_predictions = smoothed.reshape(6000,10,3)

np.save('ewm_hp1_3D_predictions.npy', ewm_hp_3D_predictions)

#%% # plots an 8 second window of data for both paws and all digits
ewm_hp_3D_predictions = np.load('ewm_hp1_3D_predictions.npy')
pred3D = ewm_hp_3D_predictions[2520:2900,:,:] 

plt.figure(figsize=(15,10))    
plt.subplot(331)
[plt.plot(pred3D[:,j,0], b) for j,b in zip(range(1,5),['b','g','c','y','r'])]
plt.plot(pred3D[:,0,0], 'k--', linewidth=2)
plt.legend([ 'D1', 'D2', 'D3', 'D4', 'Palm',], fontsize=10, loc=1)
plt.ylabel('X (mm)')

plt.title('R Forelimb')
plt.subplot(334)
[plt.plot(pred3D[:,j,1], b) for j,b in zip(range(1,5),['b','g','c','y','r'])]
plt.plot(pred3D[:,0,1], 'k--', linewidth=2)
plt.ylabel('Y (mm)')
plt.subplot(337)
[plt.plot(pred3D[:,j,2], b) for j,b in zip(range(1,5),['b','g','c','y','r'])]
plt.plot(pred3D[:,0,2], 'k--', linewidth=2)
plt.ylabel('Z (mm)')
plt.xlabel('frames (70Hz)')

plt.subplot(332)
plt.title('L Forelimb')
[plt.plot(pred3D[:,j,0], b) for j,b in zip(range(6,10),['b','g','c','y','r'])]
plt.plot(pred3D[:,5,0], 'k--', linewidth=2)
plt.subplot(335)
[plt.plot(pred3D[:,j,1], b) for j,b in zip(range(6,10),['b','g','c','y','r'])]
plt.plot(pred3D[:,5,1], 'k--', linewidth=2)
plt.subplot(338)
[plt.plot(pred3D[:,j,2], b) for j,b in zip(range(6,10),['b','g','c','y','r'])]
plt.plot(pred3D[:,5,2], 'k--', linewidth=2)

plt.xlabel('frames (70Hz)')

plt.subplot(333)

plt.title('L vs R Forelimb')
plt.plot(pred3D[:,0,0], 'g') 
plt.plot(pred3D[:,5,0], 'r') 
plt.subplot(336)
plt.plot(pred3D[:,0,1], 'g') 
plt.plot(pred3D[:,5,1], 'r') 
plt.legend(['R', 'L'], fontsize=10)
plt.subplot(339)
plt.plot(pred3D[:,0,2], 'g') 
plt.plot(pred3D[:,5,2], 'r') 
plt.xlabel('frames (70Hz)')

plt.suptitle('X,Y,X movement of Paws and Digits', fontsize=14)
plt.savefig('MovementGraph.pdf')

#%% Same as above but subtracts the palm data, giving an idea of how the indvidual 
#   digits vary in relation with each other and the palm

ewm_hp_3D_predictions = np.load('ewm_hp1_3D_predictions.npy')
pred3D = ewm_hp_3D_predictions[2520:2900,:,:] 

for i in range(4):
    pred3D[:,i+1,:] = pred3D[:,i+1,:] - pred3D[:,0,:]
    pred3D[:,i+6,:] = pred3D[:,i+6,:] - pred3D[:,5,:]

pred3D[:,0,:] = pred3D[:,5,:] = 0

plt.figure(figsize=(15,10))    
plt.subplot(321)
[plt.plot(pred3D[:,j,0], b) for j,b in zip(range(1,5),['b','g','c','y','r'])]
plt.plot(pred3D[:,0,0], 'k--', linewidth=2)
plt.legend([ 'D1', 'D2', 'D3', 'D4', 'Palm',], fontsize=10, loc=1)
plt.ylabel('X (mm)')
plt.yticks([2,0,-2])
plt.ylim([-3,3.5])
plt.title('Right Forelimb')
plt.subplot(323)
[plt.plot(pred3D[:,j,1], b) for j,b in zip(range(1,5),['b','g','c','y','r'])]
plt.plot(pred3D[:,0,1], 'k--', linewidth=2)
plt.ylabel('Y (mm)')
plt.ylim([-2.5,4.5])
plt.yticks([4,2,0,-2])
plt.subplot(325)
[plt.plot(pred3D[:,j,2], b) for j,b in zip(range(1,5),['b','g','c','y','r'])]
plt.plot(pred3D[:,0,2], 'k--', linewidth=2)
plt.ylabel('Z (mm)')
plt.xlabel('frames (70Hz)')
plt.yticks([1,0,-1])
plt.ylim([-1.75,1.75])

plt.subplot(322)
plt.title('Left Forelimb')
[plt.plot(pred3D[:,j,0], b) for j,b in zip(range(6,10),['b','g','c','y','r'])]
plt.plot(pred3D[:,5,0], 'k--', linewidth=2)
plt.yticks([2,0,-2])
plt.ylim([-3,3.5])
plt.subplot(324)
[plt.plot(pred3D[:,j,1], b) for j,b in zip(range(6,10),['b','g','c','y','r'])]
plt.plot(pred3D[:,5,1], 'k--', linewidth=2)
plt.ylim([-2.5,4.5])
plt.yticks([4,2,0,-2])
plt.yticks([4,2, 0,-2])
plt.subplot(326)
[plt.plot(pred3D[:,j,2], b) for j,b in zip(range(6,10),['b','g','c','y','r'])]
plt.plot(pred3D[:,5,2], 'k--', linewidth=2)
plt.yticks([1,0,-1])
plt.ylim([-1.75,1.75])

plt.xlabel('frames (70Hz)')

plt.suptitle('X,Y,X movement of Digits with palm substracted', fontsize=14)
plt.savefig('MovementGraph_palm_subtracted.pdf')

#%% Extracting the peaks - corresponding to when the foot is furthest back on the y axis
#   of both feet
ewm_hp_3D_predictions = np.load('ewm_hp1_3D_predictions.npy')
pred3D = ewm_hp_3D_predictions

right_paw_index = 0
left_paw_index = 5
y_axis = 1

y = pred3D[:,right_paw_index,y_axis]
x = np.arange(0,pred3D.shape[0],1)
indexes = peakutils.indexes(y, thres=0.7, min_dist=30)
plt.figure(figsize=(10,6))
pplot(x, y, indexes)

y2 = pred3D[:,left_paw_index,y_axis]
x = np.arange(0,pred3D.shape[0],y_axis)
indexes2 = peakutils.indexes(y2[:], thres=0.7, min_dist=30)
pplot(x, y2, indexes2)

#%% Extracts a window 30 frames before a peak, and 20 seconds after - this roughly 
# corresponds to when the foot is at the front of the wheel, travels the the back (peak)
# and then is lefted up and brought to the front again
def extract_steps(data, indexes):
    win = 20
    indexes = indexes
    
    # a bug occurs if the index of the first step is less than the window size
    # this will delete the first step to avoid that bug
    if indexes[0] - win < 0:
        indexes = np.delete(indexes2,0,0)

    shape = indexes.shape[0]
    steps = np.zeros((shape, 2*win+10))
    for i in range(indexes.shape[0]):
        steps[i,:] = data[indexes[i]-win:indexes[i]+win+10]
        
    av_step = np.mean(steps, axis=0)
    av_var = np.var(steps, axis=0)
    
    return av_step, av_var

#%% Plots the average step trajectory in each dimension for the palm of both the 
    # right and left front paws 
x_av, x_var = extract_steps(pred3D[:,0,0], indexes)
y_av, y_var = extract_steps(pred3D[:,0,1], indexes)
z_av, z_var = extract_steps(pred3D[:,0,2], indexes)

alpha = .3
line = np.arange(0,x_av.shape[0],1)
plt.figure(figsize=(15,8))
plt.subplot(321)
plt.plot(line, x_av, label='Average')
plt.plot(line, x_av+x_var, color='lightskyblue', linestyle=':', alpha=alpha, label='+/- variance')
plt.plot(line, x_av-x_var, color='lightskyblue', alpha=alpha, linestyle=':' )
plt.yticks([2,1,0,-1])
plt.title('Right paw, steps: '+str(indexes.shape[0]))
plt.ylabel('X')
plt.legend()

plt.subplot(323)
plt.plot(line, y_av, color = 'green')
plt.plot(line, y_av+y_var, color='y', alpha=alpha, linestyle=':' )
plt.plot(line, y_av-y_var, color='y', alpha=alpha, linestyle=':' )
plt.yticks([10,0,-10])
plt.ylabel('Y')

plt.subplot(325)
plt.plot(line, z_av, color='red')
plt.plot(line, z_av+z_var, color='orange', alpha=alpha, linestyle=':' )
plt.plot(line, z_av-z_var, color='orange', alpha=alpha, linestyle=':' )
plt.yticks([2,0,-2])
plt.ylabel('Z')

x2_av, x2_var = extract_steps(pred3D[:,5,0], indexes2)
y2_av, y2_var = extract_steps(pred3D[:,5,1], indexes2)
z2_av, z2_var = extract_steps(pred3D[:,5,2], indexes2)

line = np.arange(0,x2_av.shape[0],1)
plt.subplot(322)
plt.title('Left paw, steps: '+str(indexes2.shape[0]))
plt.plot(line, x2_av, color='orange')
plt.plot(line, x2_av+x2_var, color='gold', alpha=alpha, linestyle=':' )
plt.plot(line, x2_av-x2_var, color='gold', alpha=alpha, linestyle=':' )
plt.yticks([2,1,0,-1])

plt.subplot(324)
plt.plot(line, y2_av, color='purple')
plt.plot(line, y2_av+y2_var, color='magenta', alpha=alpha, linestyle=':' )
plt.plot(line, y2_av-y2_var, color='magenta', alpha=alpha, linestyle=':' )
plt.yticks([10,0,-10])

plt.subplot(326)
plt.plot(line, z2_av, color='lightseagreen')
plt.plot(line, z2_av+z2_var, color='c', alpha=alpha, linestyle=':' )
plt.plot(line, z2_av-z2_var, color='c', alpha=alpha, linestyle=':' )
plt.yticks([2,0,-2])

plt.suptitle('Average step trajectory of palm', fontsize=14)
plt.savefig('Average_step_and_variance.pdf')

#%% used for getting average step for every digit, not just paws

def extract_steps2(data, indexes):
    win = 20
    indexes = indexes
    
    # a bug occurs if the index of the first step is less than the window size
    # this will delete the first step to avoid that bug
    if indexes[0] - win < 0:
        indexes = np.delete(indexes2,0,0)

    shape = indexes.shape[0]
    steps = np.zeros(((shape, 2*win+10,data.shape[1])))
    for j in range(data.shape[1]):
        for i in range(indexes.shape[0]):
            steps[i,:,j] = data[indexes[i]-win:indexes[i]+win+10,j]
        
    av_step = np.mean(steps, axis=0)
    av_var = np.var(steps, axis=0)
    
    return av_step, av_var

#%% Plot of average distance of each digit from paw during the average step
x_av, x_var = extract_steps2(pred3D[:,0:5,0], indexes)
y_av, y_var = extract_steps2(pred3D[:,0:5,1], indexes)
z_av, z_var = extract_steps2(pred3D[:,0:5,2], indexes)

for i in range(5):
    x_av[:,-i-1] = x_av[:,-i-1] - x_av[:,0]
    y_av[:,-i-1] = y_av[:,-i-1] - y_av[:,0]
    z_av[:,-i-1] = z_av[:,-i-1] - z_av[:,0]

alpha = .3
line = np.arange(0,x_av.shape[0],1)
plt.figure(figsize=(15,8))
plt.subplot(321)
plt.plot(line, x_av)
plt.yticks([2,1,0,-1])
plt.title('Right paw, steps: '+str(indexes.shape[0]))
plt.ylabel('X')
plt.legend(labels =['Paw', 'D1', 'D2', 'D3', 'D4'])

plt.subplot(323)
plt.plot(line, y_av)
plt.yticks([4,0,-4])
plt.ylabel('Y')

plt.subplot(325)
plt.plot(line, z_av)
plt.yticks([1.5,0,-1.5])
plt.ylabel('Z')

x2_av, x2_var = extract_steps2(pred3D[:,5:10,0], indexes2)
y2_av, y2_var = extract_steps2(pred3D[:,5:10,1], indexes2)
z2_av, z2_var = extract_steps2(pred3D[:,5:10,2], indexes2)

for i in range(5):
    x2_av[:,-i-1] = x2_av[:,-i-1] - x2_av[:,0]
    y2_av[:,-i-1] = y2_av[:,-i-1] - y2_av[:,0]
    z2_av[:,-i-1] = z2_av[:,-i-1] - z2_av[:,0]
    
line = np.arange(0,x2_av.shape[0],1)
plt.subplot(322)
plt.title('Left paw, steps: '+str(indexes2.shape[0]))
plt.plot(line, x2_av)
plt.yticks([2,1,0,-1])

plt.subplot(324)
plt.plot(line, y2_av)
plt.yticks([4,0,-4])

plt.subplot(326)
plt.plot(line, z2_av)
plt.yticks([1.5,0,-1.5])

plt.suptitle('Average trajectory of digits with paw subtracted', fontsize=14)
plt.savefig('Average_step_and_variance_paw_subtracted.pdf')


#%% Basic plotting below - first is plot of distance from palm of each digit during 
### video, second is spectrogram - these are exploratory, not important unless 
### decided to be important later 
#%%
#%%
smoothed = np.load('Smoothed_3D_predictions.npy')
#pred3D = smoothed
def preproc(X):
    # mn = np.abs(X-np.median(X, axis=1)[:,None])
    # return np.mean((mn - mn.mean(axis=0))/mn.std(axis=0), axis=1)
    mn = np.abs(X[:,1:]-X[:,0][:,None])
    return np.mean((mn - mn.min(axis=0))/(mn.max(axis=0)-mn.min(axis=0)), axis=1)

plt.figure(figsize=(15,10))  
    
idx = [0,1,2,3,4]
plt.subplot(421)
plt.plot(preproc(pred3D[:,idx,0]))
plt.subplot(423)
plt.plot(preproc(pred3D[:,idx,1]))
plt.subplot(425)
plt.plot(preproc(pred3D[:,idx,2]))
plt.subplot(427)
plt.plot(pred3D[:,idx,0])

idx = np.array([0,1,2,3,4])+5
plt.subplot(422)
plt.plot(preproc(pred3D[:,idx,0]))
plt.subplot(424)
plt.plot(preproc(pred3D[:,idx,1]))
plt.subplot(426)
plt.plot(preproc(pred3D[:,idx,2]))
plt.subplot(428)
plt.plot(pred3D[:,idx,0])

#%% Spectrogram
fs = 70
NFFT = int(fs*0.015)
pred3D = ewm_hp_3D_predictions
f, t, Sxx = signal.spectrogram(pred3D[:,0,1], fs,  nfft=200, nperseg=50)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.xlim([36,44])
plt.show()
plt.savefig('Spectragram_of_steps.pdf')





