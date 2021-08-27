#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:45:36 2021

@author: jimmytabet

This script contains functions used in the mapping demo to plot a cumulative 
scatter of all paw points, as well as animate the 3D reconstructions of tracked 
bodyparts from the Behavior3D repo.
"""

#%% imports
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%% define XYZ points of wheel (cylinder) function
def xyz_wheel(pt1, pt2, radius):
    '''
    Define XYZ points of wheel (cylinder).

    Parameters
    ----------
    pt1 : float
        Center of one endpoint of wheel.
    pt2 : flaot
        Center of other endpoint of wheel (defines length).
    radius : float
        Wheel radius.

    Returns
    -------
    X : numpy array
        Describes X coordinates of wheel.
    Y : numpy array
        Describes Y coordinates of wheel.
    Z : numpy array
        Describes Z coordinates of wheel.

    '''
    
    # vector in direction of axis
    v = pt2 - pt1
    # find magnitude of vector
    mag = np.linalg.norm(v)
    # unit vector in direction of axis
    v = v / mag
    # make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # normalize n1
    n1 /= np.linalg.norm(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    # surface ranges over t from 0 to mag of axis and 0 to 2*pi
    t = np.linspace(0, mag, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    # generate coordinates for surface
    X, Y, Z = [pt1[i] + v[i] * t + radius * np.sin(theta) * n1[i] + radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]

    return X,Y,Z

#%% scatter function
def scatter(data, wheel_pt1, wheel_pt2, R, rot=False, save=False):
    '''
    Plot a cumulative snapshot of paw points from 3D reconstruction. 

    Parameters
    ----------
    data : pandas df
        Dataframe of 3D points.
    wheel_pt1 : float
        Center of one endpoint of wheel.
    wheel_pt2 : float
        Center of other endpoint of wheel (defines length).
    R : float
        Wheel radius.
    rot : bool, optional
        Option to rotate view of scatter plot and save animation. The default 
        is False.
    save : bool or str, optional
        Save plot of cumulative snapshot. If a str is passed, the figure will 
        be saved with that name/location. The default is False. 

    Returns
    -------
    None.

    '''
    
    # find min/max for axes limits
    col_X = [col for col in data.columns if col[-1] == 'X']
    col_Y = [col for col in data.columns if col[-1] == 'Y']
    col_Z = [col for col in data.columns if col[-1] == 'Z']

    edge = 10
    Xlim = [data[col_X].min().min()-edge, data[col_X].max().max()+edge]
    Ylim = [data[col_Y].min().min()-edge, data[col_Y].max().max()+edge]
    Zlim = [data[col_Z].min().min()-edge, data[col_Z].max().max()+edge]

    # set up figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Mouse Paw Points')
    ax.set_xlim(Xlim)
    ax.set_ylim(Ylim)
    ax.set_zlim(Zlim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # SET VIEW
    ax.view_init(15,180)

    # hide axes
    ax.set_axis_off()
    
    # create legend
    patches = []
    # patch for left paw
    l_patch, = plt.plot([],[], marker="o", ls='', color='blue', label='LPaw')
    patches.append(l_patch)
    # patch for right paw
    r_patch, = plt.plot([],[], marker="o", ls='', color='red', label='RPaw')
    patches.append(r_patch)
    
    plt.legend(handles=patches, loc='lower center', ncol = len(col_X))

    # plot paws
    lpaw = data[[col for col in data.columns if col[0] == 'L' and col[-3] == 'w']]
    rpaw = data[[col for col in data.columns if col[0] == 'R' and col[-3] == 'w']]
    ax.scatter(lpaw.iloc[:,0], lpaw.iloc[:,1], lpaw.iloc[:,2], color = 'blue', alpha = 0.1)
    ax.scatter(rpaw.iloc[:,0], rpaw.iloc[:,1], rpaw.iloc[:,2], color = 'red', alpha = 0.1)

    # plot wheel
    X, Y, Z = xyz_wheel(wheel_pt1, wheel_pt2, R)
    ax.plot_surface(X, Y, Z, alpha = 0.1, color='k')

    if rot:
        # set view
        # ax.view_init(30,270-45)
        ax.view_init(30,270)

        # rotate camera for animation
        def update(num):
            # ax.view_init(30,270-45+num)
            ax.view_init(30,270+num)
            return

        # save animation
        anim = FuncAnimation(fig, update, frames=360, interval=20, repeat=0)
        anim.save('scatter_rot.mp4', progress_callback = lambda i, n: print(f'Saving frame {i+1} of {n}'))
    
    # save
    if isinstance(save, str):
        plt.savefig(save, dpi=300)
    elif save:
        plt.savefig('scatter', dpi=300)
    else:
        pass

#%% animate function
def animate(fig, data, wheel_pt1, wheel_pt2, R, fps, save=False):
    '''
    Animate all 3D reconstructed points    

    Parameters
    ----------
    data : pandas df
        Dataframe of 3D points.
    wheel_pt1 : float
        Center of one endpoint of wheel.
    wheel_pt2 : float
        Center of other endpoint of wheel (defines length).
    R : float
        Wheel radius.
    fps : int
        Framerate of animation.
    save : bool, optional
        Save animation. The default is False. 

    Returns
    -------
    anim : matplotlib FuncAnimation
        Animation of points.

    '''
    
    # find min/max for axes limits
    col_X = [col for col in data.columns if col[-1] == 'X']
    col_Y = [col for col in data.columns if col[-1] == 'Y']
    col_Z = [col for col in data.columns if col[-1] == 'Z']

    edge = 10
    Xlim = [-7.0132103916127555, 46.4446776830212]
    Ylim = [-8.612098104740042, 57.039159875762536]
    Zlim =  [-22.44768578790802, 10.927777604782086]

    # set up figure
    # fig = plt.figure()
    ax = fig.add_subplot(224, projection='3d')
    title = ax.set_title('Frame 1700 of 5999')
    ax.set_xlim(Xlim)
    ax.set_ylim(Ylim)
    ax.set_zlim(Zlim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')



    # degree vector for bouncing rotation
    factor = 5
    deg = 70*factor
    base = list(range(deg+1))+list(range(deg,-1,-1))
    base = [x/factor for x in base]
    mult, extra = 6000//(len(base)), 6000%(len(base))
    deg_vec = base*mult + base[:extra]
    
        # SET VIEW
    ax.view_init(30,270-45+deg_vec[1700])
    
    # hide axes
    ax.set_axis_off()

    # plot wheel
    X, Y, Z = xyz_wheel(wheel_pt1, wheel_pt2, R)
    ax.plot_surface(X, Y, Z, color='white')

    # plot points

    # update function for animation
    def update(num, df, graphs, col_X, deg_vec):
        
        # remove lines from previous frame
        ax.lines=[]

        # define paw centers
        lpx, lpy, lpz = df.loc[df.index.start+num,'Lpaw_X'], df.loc[df.index.start+num,'Lpaw_Y'], df.loc[df.index.start+num,'Lpaw_Z']
        rpx, rpy, rpz = df.loc[df.index.start+num,'Rpaw_X'], df.loc[df.index.start+num,'Rpaw_Y'], df.loc[df.index.start+num,'Rpaw_Z']

        for i,bpart in enumerate(col_X):
            start = 3*i
            end = 3*i+3
            temp = df[df.index==df.index.start+num].iloc[:,start:end]
            temp.columns = ['x','y','z']
            graphs[i]._offsets3d = (temp.x, temp.y, temp.z)

            # draw new lines (must use float)
            a, b, c = float(temp.x), float(temp.y), float(temp.z)
            if bpart[:4].lower() == 'lpaw':
                ax.plot3D([a, lpx], [b, lpy], [c, lpz], 'black')
            elif bpart[:4].lower() == 'rpaw':
                ax.plot3D([a, rpx], [b, rpy], [c, rpz], 'black')
        
        # SET TITLE FOR EACH FRAME
        title.set_text('Frame {} of {}'.format(df.index.start+num, df.index.stop-1))
        
        # optional, use to rotate view while animating
        ax.view_init(30,270-45+deg_vec[num])

        return

    # plot initial points

    # define paw centers
    lpx, lpy, lpz = data.loc[data.index.start,'Lpaw_X'], data.loc[data.index.start,'Lpaw_Y'], data.loc[data.index.start,'Lpaw_Z']
    rpx, rpy, rpz = data.loc[data.index.start,'Rpaw_X'], data.loc[data.index.start,'Rpaw_Y'], data.loc[data.index.start,'Rpaw_Z']

    lpaw,rpaw,lpawc,rpawc,other = (False,)*5
    graph = []
    for x,bpart in enumerate(col_X):
        if bpart[:4].lower() == 'lpaw': #and bpart[5].lower() == 'd':
            pcolor = 'blue'
            lpaw = True
        elif bpart[:4].lower() == 'rpaw': #and bpart[5].lower() == 'd':
            pcolor = 'red'
            rpaw = True
        # elif bpart[:4].lower() == 'lpaw' and bpart[5].lower() != 'd':
        #     pcolor = 'green'
        #     lpawc = True
        # elif bpart[:4].lower() == 'rpaw' and bpart[5].lower() != 'd':
        #     pcolor = 'orange'
        #     rpawc = True
        else:
            print('body part not recognized')
            pcolor = 'yellow'
            other = True

        start = 3*x
        end = 3*x+3
        temp = data[data.index==data.index.start].iloc[:,start:end]
        temp.columns = ['x','y','z']
        graph.append(ax.scatter(temp.x, temp.y, temp.z, color=pcolor))

        # draw lines (must use float)
        a, b, c = float(temp.x), float(temp.y), float(temp.z)
        if bpart[:4].lower() == 'lpaw':
            ax.plot3D([a, lpx], [b, lpy], [c, lpz], 'black')
        elif bpart[:4].lower() == 'rpaw':
            ax.plot3D([a, rpx], [b, rpy], [c, rpz], 'black')

    # create legend
    patches = []
    if lpawc == True:
        patch, = plt.plot([],[], marker="o", ls='', color='green', label='LPaw Center')
        patches.append(patch)
    if lpaw == True:
        patch, = plt.plot([],[], marker="o", ls='', color='blue', label='Left Paw')
        patches.append(patch)
    if rpawc == True:
        patch, = plt.plot([],[], marker="o", ls='', color='orange', label='RPaw Center')
        patches.append(patch)
    if rpaw == True:
        patch, = plt.plot([],[], marker="o", ls='', color='red', label='Right Paw')
        patches.append(patch)
    if other == True:
        patch, = plt.plot([],[], marker="o", ls='', color='black', label='Unknown Part')
        patches.append(patch)
    
    plt.legend(handles=patches, loc='upper center', ncol = 2)


    
    return fig