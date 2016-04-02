# coding: utf-
from itertools import cycle
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.datasets.samples_generator import make_blobs
from birch import birch_algo

from sklearn.cluster import Birch

# Generate centers for the blobs so that it forms a 10 X 10 grid.
xx = np.linspace(-22, 22, 10)
yy = np.linspace(-22, 22, 10)
xx, yy = np.meshgrid(xx, yy)
n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                       np.ravel(yy)[:, np.newaxis]))

# Generate blobs to do a comparison between MiniBatchKMeans and Birch.
X, y = make_blobs(n_samples=100000, centers=n_centres, random_state=0)
   

# Use all colors that matplotlib provides by default.
colors_ = cycle(colors.cnames.keys())

fig = plt.figure(figsize=(12, 4))
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)

#Compute clustering with Birch with and without the final clustering step and plot.
labels1, centroids1, n_clusters1 = birch_algo(X, clustering=None)
labels2, centroids2, n_clusters2 = birch_algo(X, clustering=100) 

labels = labels1, labels2
centroids = centroids1, centroids2
n_clusters = n_clusters1, n_clusters2


final_step = ['without global clustering', 'with global clustering']

#plot the results of birch with and without clustering.
for i in range(0, 2):
    ind = i + 1
    ax = fig.add_subplot(1, 3, ind +1)
    for this_centroids, k, col in zip(centroids[i], range(n_clusters[i]), colors_ ):
        mask = labels[i] == k
        ax.plot(X[mask, 0], X[mask, 1], 'w', markerfacecolor=col, marker='.')
        if n_clusters[i] is None:
                ax.plot(this_centroids[0], this_centroid[1], '+', markerfacecolor=col,
                        markeredgecolor='k', markersize=5)
        ax.set_ylim([-25, 25])
        ax.set_xlim([-25, 25])
        ax.set_autoscaley_on(False)
        ax.set_title('Birch %s' % final_step[i])


plt.show()
