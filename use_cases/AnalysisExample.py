#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:35:53 2020

@author: jimmytabet
"""

#%% import modules
from bh3D.calibration import calibration
import numpy as np
from behavelet import wavelet_transform
import matplotlib.pyplot as plt

#%% build model
#Customize this paths dictionary for your local paths. 
pathsDict = {
    'coordPath': './setup_old/model_1_coordinates.csv', 
    'DLCPath': './setup_old/',
    'PlotPath' : './results/'
}

cal = calibration(model = ['bot','fl','fr'], pathsDict = pathsDict)

# filtered data
data = cal.map_to_3D()
means = data.mean(axis=0)
data -= means

#%% wavelet analysis
freqs, power, X_new = wavelet_transform(data.values, n_freqs=25, fsample=70, fmin=1, fmax=35)

#%% dimensionality reduction
from sklearn.decomposition import PCA, NMF, KernelPCA, FastICA
from sklearn.manifold import TSNE

comp = 2
p = PCA(n_components=comp)
n = NMF(n_components=comp, max_iter=500)
t = TSNE(n_components=comp, random_state=42)
k = KernelPCA(n_components=comp)
f = FastICA(n_components=comp, max_iter=1500)

downsample=50
pca_data = p.fit_transform(X_new[::downsample])
nmf_data = n.fit_transform(X_new[::downsample])
tsne_data = t.fit_transform(X_new[::downsample])
kpca_data = k.fit_transform(X_new[::downsample])
fica_data = f.fit_transform(X_new[::downsample])

#%% PCA then FICA
p_5 = PCA(n_components=5)
temp_data = p_5.fit_transform(X_new[::downsample])
i_p5 = FastICA(n_components=comp)
pca_fica_data = i_p5.fit_transform(temp_data)

#%% compare dim reduction methods
dim_red = [pca_data,nmf_data,tsne_data,kpca_data,fica_data,pca_fica_data]
name = ['pca','nmf','tsne','kpca','fica','pca_fica']

fig = plt.figure()
for i in range(len(dim_red)): 
    ax = fig.add_subplot(2,3,i+1)
    ax.scatter(*dim_red[i].T)
    ax.set_title(name[i]+': ds=50, mean center before wavelet')

#%% compare KMeans for dim reduction methods
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering

no_clus = 3
no_clus_tech = 2
cluster = KMeans(n_clusters = no_clus)

kmeans = []
kmeans_labels = []
kmeans_centroids = []
for i in dim_red:
    temp = cluster.fit(i)
    kmeans.append(temp)
    kmeans_labels.append(temp.labels_)
    kmeans_centroids.append(temp.cluster_centers_)
    
fig_all = plt.figure()
for i in range(len(kmeans)): 
    ax = fig_all.add_subplot(no_clus_tech,len(dim_red),i+1)
    for j in np.unique(kmeans_labels[i]):
        ax.scatter(*dim_red[i][kmeans_labels[i]==j].T)
    ax.scatter(*kmeans_centroids[i].T, c='k', marker='x')
    # ax.set_title(name[i]+' kmeans clustering')
    ax.set_title(name[i])
    if i==0:
        ax.set_ylabel('kmeans', size='large')
    
#%% compare DBSCAN for dim reduction methods - DOES NOT PRODUCE GOOD RESULTS
# eps: max distance between samples to be considered in neighborhood
# min_samples: min number of samples for point to be considered core point
# cluster = DBSCAN(eps=10, min_samples=50)

# dbscan = []
# dbscan_labels = []
# dbscan_centers = []
# for i in dim_red:
#     temp = cluster.fit(i)
#     dbscan.append(temp)
#     dbscan_labels.append(temp.labels_)
#     dbscan_centers.append(temp.components_)

# dbscan_centroids = []
# # fig = plt.figure()
# for i in range(len(dbscan)):
#     temp_centers=[]
#     ax = fig_all.add_subplot(no_clus_tech,len(dim_red),len(dim_red)+i+1)
#     for j in np.unique(dbscan_labels[i]):
#         ax.scatter(*dim_red[i][dbscan_labels[i]==j].T)
#         temp_centers.append(dim_red[i][dbscan_labels[i]==j].mean(axis=0))
#     temp_centroids = np.vstack(temp_centers)
#     dbscan_centroids.append(temp_centroids)
#     ax.scatter(*temp_centroids.T, c='k', marker='x')
#     # ax.set_title(name[i]+' dbscan clustering')
#     if i==0:
#         ax.set_ylabel('dbscan', size='large')
    
#%% compare SpectralClustering for dim reduction methods
# PCA/kPCA: "Graph is not fully connected, spectral embedding"
cluster = SpectralClustering(n_clusters=no_clus, random_state=42)

sc = []
sc_labels = []
for i in dim_red:
    temp = cluster.fit(i)
    sc.append(temp)
    sc_labels.append(temp.labels_)
    
sc_centroids = []
# fig = plt.figure()
for i in range(len(sc)):
    temp_centers=[]
    ax = fig_all.add_subplot(no_clus_tech,len(dim_red),1*len(dim_red)+i+1)
    for j in np.unique(sc_labels[i]):
        ax.scatter(*dim_red[i][sc_labels[i]==j].T)
        temp_centers.append(dim_red[i][sc_labels[i]==j].mean(axis=0))
    temp_centroids = np.vstack(temp_centers)
    sc_centroids.append(np.vstack(temp_centroids))
    ax.scatter(*temp_centroids.T, c='k', marker='x')
    # ax.set_title(name[i]+' spectral clustering')
    if i==0:
        ax.set_ylabel('spectral clustering', size='large')

#%% show comparison
fig_all.tight_layout()
fig_all.show()

#%% kclosest points to centroids function
def kclosest(k, near, dim_red_list, cluster_centroids):
    
    # get random ids of points k points from near closest points
    id_near = np.sort(np.random.choice(near, k, replace=False))
    
    # store all distances, closest ids, and k random ids from near closest
    dist = []
    closest = []
    ids = []
    for dim, centroid in zip(dim_red_list, cluster_centroids):
        temp_dist = []
        for row in dim:
            diff = row-centroid
            temp_dist.append(np.linalg.norm(diff, axis=1))
        temp_dist = np.array(temp_dist)
        dist.append(temp_dist)
        temp_closest = np.argsort(temp_dist, axis=0)
        closest.append(temp_closest)
        ids.append(temp_closest[id_near])
        
    return dist, closest, ids

#%% k random points from n_points nearest points
k = 5
n_points = 25
kmeans_dist, kmeans_closest, kmeans_ids = kclosest(k, n_points, dim_red, kmeans_centroids)
# dbscan_dist, dbscan_closest, dbscan_ids = kclosest(k, n_points, dim_red, dbscan_centroids)
sc_dist, sc_closest, sc_ids = kclosest(k, n_points, dim_red, sc_centroids)

ids_all = [kmeans_ids, sc_ids]

#%% add kclosest points to comparison plot
fig_num=0        
for ids in ids_all:
    for dim, i_dim in zip(dim_red, ids):
        fig_all.axes[fig_num].scatter(*dim[i_dim].T, c='k', marker='*')
        fig_num += 1

#%% subplot zoom function 
#   from https://stackoverflow.com/questions/44997029/matplotlib-show-single-graph-out-of-object-of-subplots
def add_subplot_zoom(figure):

    zoomed_axes = [None]
    def on_click(event):
        ax = event.inaxes

        if ax is None:
            # occurs when a region not in an axis is clicked...
            return

        # we want to allow other navigation modes as well. Only act in case
        # shift was pressed and the correct mouse button was used
        if event.key != 'shift' or event.button != 1:
            return

        if zoomed_axes[0] is None:
            # not zoomed so far. Perform zoom

            # store the original position of the axes
            zoomed_axes[0] = (ax, ax.get_position())
            ax.set_position([0.1, 0.1, 0.85, 0.85])

            # hide all the other axes...
            for axis in event.canvas.figure.axes:
                if axis is not ax:
                    axis.set_visible(False)

        else:
            # restore the original state

            zoomed_axes[0][0].set_position(zoomed_axes[0][1])
            zoomed_axes[0] = None

            # make other axes visible again
            for axis in event.canvas.figure.axes:
                axis.set_visible(True)

        # redraw to make changes visible.
        event.canvas.draw()

    figure.canvas.mpl_connect('button_press_event', on_click)

#%% show all comparisons with subplot zoom
add_subplot_zoom(fig_all)
fig_all.show()

#%% set up dictionaries
pca, nmf, tsne, kpca, fica, pca_fica = [{} for i in range(6)]
dicts = [pca, nmf, tsne, kpca, fica, pca_fica]

for dic,n,dim_data in zip(dicts, name, dim_red):
    dic['name'] = n
    dic['data'] = dim_data

for dic,clus,clus_l,clus_c,clus_d,clus_cl,clus_id in zip(dicts,kmeans,kmeans_labels,kmeans_centroids,kmeans_dist,kmeans_closest,kmeans_ids):
    dic['kmeans'] = clus
    dic['kmeans_labels'] = clus_l
    dic['kmeans_centroids'] = clus_c
    dic['kmeans_dist'] = clus_d
    dic['kmeans_closest'] = clus_cl
    dic['kmeans_ids'] = clus_id

# for dic,clus,clus_l,clus_c,clus_d,clus_cl,clus_id in zip(dicts,dbscan,dbscan_labels,dbscan_centroids,dbscan_dist,dbscan_closest,dbscan_ids):
#     dic['dbscan'] = clus
#     dic['dbscan_labels'] = clus_l
#     dic['dbscan_centroids'] = clus_c
#     dic['dbscan_dist'] = clus_d
#     dic['dbscan_closest'] = clus_cl
#     dic['dbscan_ids'] = clus_id
    
for dic,clus,clus_l,clus_c,clus_d,clus_cl,clus_id in zip(dicts,sc,sc_labels,sc_centroids,sc_dist,sc_closest,sc_ids):
    dic['sc'] = clus
    dic['sc_labels'] = clus_l
    dic['sc_centroids'] = clus_c
    dic['sc_dist'] = clus_d
    dic['sc_closest'] = clus_cl
    dic['sc_ids'] = clus_id
    
#%% set axes function
def set_axes(figure, num_clusters, num_points, show_y = False):
    subplots = figure.get_axes()
    # iterate over each cluster
    for i in range(num_clusters):
        temp_subplots = subplots[i*num_points:i*num_points+num_points]
        # initialize min/max_ylim
        min_ylim, max_ylim = np.inf, -np.inf

        # iterate over each point in each cluster
        for j in temp_subplots:
            # find min/max_ylim
            temp_min_ylim, temp_max_ylim = j.get_ylim()
            min_ylim = min(min_ylim, temp_min_ylim)
            max_ylim = max(max_ylim, temp_max_ylim)
        
        # set constant ylim for each cluster
        [ax.set_ylim([min_ylim, max_ylim]) for ax in temp_subplots]
        # "sharey" (hide yaxis for all but first point)
        [ax.yaxis.set_visible(False) for ax in temp_subplots[1:]]
        
    # set xticks (middle frame) for each point
    [ax.set_xticks([int(np.median(ax.get_xticks()))]) for ax in subplots]
    
    # set yticks (as min/max) for each point
    [ax.set_yticks(ax.get_ylim()) for ax in subplots]
    
    if not show_y:
        # remove yticks
        [ax.set_yticks([]) for ax in subplots]

#%% plot traces around kclosest points - paw center 'Y'
spread=70   # plot k closest frames (+- spread)
paws = [col for col in data.columns if col[-3:] == 'w_Y']

dicts = [pca, nmf, tsne, kpca, fica, pca_fica]
for dic in dicts:
    for c_type in ['kmeans','sc']:
        fig = plt.figure()
        fig.set_tight_layout(True)
        fig.suptitle(dic['name']+' '+c_type, size='x-large')
        num_clus = dic[c_type+'_ids'].shape[1]
        for clust in range(num_clus):
            for num, i in enumerate(dic[c_type+'_ids'][:,clust]):
                ax = fig.add_subplot(no_clus,k,k*clust+num+1)
                ax.plot(data.loc[downsample*i-spread:downsample*i+spread+1,paws])
                if num==0:
                    ax.set_ylabel('cluster '+str(clust+1), size='large')
                if clust==0:
                    ax.set_title('closest: '+str(num+1))
                if num == 0 and clust == 0:
                    ax.legend(paws)
        
        # set axes and figsize
        set_axes(fig,num_clus,k)
        DPI = fig.get_dpi()
        fig.set_size_inches(1920.0/float(DPI),1080.0/float(DPI))
        
        fig.savefig('/Users/jimmytabet/Desktop/Behavioral Classification Results/methods comp/Mean Center Before/3clusters_rand points_same ylim/traces/'+dic['name']+'/'+c_type, dpi=DPI, bbox_inches='tight')
        plt.close('all')

#%%
#%% implement behavior montage
from use_cases.behavior_montage import behavior_montage

raw_video = './vids/front_right_Dec17.mp4'

# if hasattr(behavior_montage, 'mov'): del behavior_montage.mov

mont = behavior_montage(raw_video, downsample*pca['kmeans_ids'], shrink_factor=3, spread=spread)
mont.fr = 70

#%%
mont.save('/Users/jimmytabet/Desktop/pca_kmeans.avi')

#%% PREVIOUS WORK
#%% other plotting - wavelet analysis
# plt.subplot(2,1,1)
# plt.plot(X_new[:,:12])
# plt.plot(X_new[:,12:25])
# plt.subplot(2,1,2)
# plt.plot(data.values[:,:1]/data.values[:,:1].max())

#%% other plotting - wavelet analysis
# plt.imshow(X_new.T, aspect='auto')
# plt.figure()
# plt.plot(data.values[:,1])

#%% other plotting - wavelet analysis
# plt.figure()
# plt.imshow(data.values[:,:].T, aspect='auto')

#%% other plotting - walking/stop/walking points
# from sklearn.decomposition import PCA, NMF
# nmf = NMF(n_components=2, max_iter=500)
# scores = nmf.fit_transform(X_new)

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.scatter(*scores[800::5].T, c='c', marker='o', alpha=0.1, label='other')
# ax.scatter(*scores[:400:5].T, c='r', marker='o', label='first walk')
# ax.scatter(*scores[400:600:5].T, c='b', marker='o', label='stop')
# ax.scatter(*scores[600:800:5].T, c='g', marker='o', label='second walk')
# plt.legend()
# ax.set_xlabel('component 1')
# ax.set_ylabel('component 2')
# plt.title('nmf')