#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:35:53 2020

@author: Jimmy Tabet
"""

#%% import modules
from bh3D.calibration import calibration
import pandas as pd
import numpy as np
from behavelet import wavelet_transform
import pylab as plt

#%% build model
#Customize this paths dictionary for your local paths. 
pathsDict = {
    'coordPath': './use_cases/setup_old/model_1_coordinates.csv', 
    'DLCPath': './use_cases/setup_old/',
    'PlotPath' : './use_cases/setup_old/'
}

cal = calibration(model = ['bot','fl','fr'], pathsDict = pathsDict)

# filtered data
data_filter = cal.map_to_3D()
# means = data.mean(axis=0)
# data -= means

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(data_filter)
data = pd.DataFrame(data, columns = data_filter.columns)

#%% wavelet analysis
freqs, power, X_new = wavelet_transform(data.values, n_freqs=25, fsample=70., fmin=1, fmax=35)

#%% dimensionality reduction
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE

p = PCA(n_components=2)
n = NMF(n_components=2, max_iter=500)
# t2 = TSNE(n_components=2)
# t3 = TSNE(n_components=3)

pca = p.fit_transform(X_new[::10])  # downsample
nmf = n.fit_transform(X_new[::10])  # downsample
# tsne2 = t2.fit_transform(X_new[::10])
# tsne3 = t3.fit_transform(X_new[::10])

#%% clustering
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering

cluster = KMeans(n_clusters=3)
# cluster = DBSCAN()
# cluster = SpectralClustering(n_clusters=3)

pc = cluster.fit(pca)
pca_labels = pc.labels_
pca_centroids = pc.cluster_centers_
# pca_centroids = pc.components_   # for DBSCAN centroids

nm = cluster.fit(nmf)
nmf_labels = nm.labels_
nmf_centroids = nm.cluster_centers_
# nmf_centroids = nm.components_   # for DBSCAN centroids

#%% PCA vs NMF 
plt.subplot(211)
for i in range(len(np.unique(pca_labels))):
    plt.scatter(*pca[pca_labels==i].T)
plt.scatter(*pca_centroids.T, c='k', marker='x')   # SpectralClustering does not compute centroids
plt.title('pca')

plt.subplot(212)
for i in range(len(np.unique(nmf_labels))):
    plt.scatter(*nmf[nmf_labels==i].T)
plt.scatter(*nmf_centroids.T, c='k', marker='x')   # SpectralClustering does not compute centroids
plt.title('nmf')
plt.tight_layout()

#%% k closest points to centroids - NMF
dist = nm.transform(nmf)
closest = np.argsort(dist, axis=0)
k_closest = 5
ids = closest[:k_closest]
# nmf[closest][:,:,0]

#%% plot k closest to centroid - NMF
for i in range(len(np.unique(nmf_labels))):
    plt.scatter(*nmf[nmf_labels==i].T)
plt.scatter(*nmf_centroids.T, c='k', marker='x', zorder=100)
plt.scatter(*nmf[ids].T, c='r', marker='*')
plt.title('nmf')
plt.tight_layout()

#%% plot traces around k closest points - paw center 'Z' for NMF
spread=100   # plot k closest frames (+- spread)
paws = [col for col in data.columns if col[-3:] == 'w_Z']

for lab in range(3):
    for num, i in enumerate(ids[:,lab]):
        plt.subplot(3,5,5*lab+num+1)
        plt.plot(data.loc[i-spread:i+spread,paws])
        # plt.tight_layout()

#%% PREVIOUS WORK
#%% other plotting - wavelet analysis
plt.subplot(2,1,1)
plt.plot(X_new[:,:12])
plt.plot(X_new[:,12:25])
plt.subplot(2,1,2)
plt.plot(data.values[:,:1]/data.values[:,:1].max())

#%% other plotting - wavelet analysis
plt.imshow(X_new.T, aspect='auto')
plt.figure()
plt.plot(data.values[:,1])

#%% other plotting - wavelet analysis
plt.figure()
plt.imshow(data.values[:,:].T, aspect='auto')

#%% other plotting - walking/stop/walking points
from sklearn.decomposition import PCA, NMF
nmf = NMF(n_components=2, max_iter=500)
scores = nmf.fit_transform(X_new)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(*scores[800::5].T, c='c', marker='o', alpha=0.1, label='other')
ax.scatter(*scores[:400:5].T, c='r', marker='o', label='first walk')
ax.scatter(*scores[400:600:5].T, c='b', marker='o', label='stop')
ax.scatter(*scores[600:800:5].T, c='g', marker='o', label='second walk')
plt.legend()
ax.set_xlabel('component 1')
ax.set_ylabel('component 2')
plt.title('nmf')