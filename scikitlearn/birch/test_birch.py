from itertools import cycle
from time import time
import numpy as np

from sklearn.cluster import MeanShift, Birch, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import Birch
""""
xx = np.linspace(-22, 22, 10)
yy = np.linspace(-22, 22, 10)
xx, yy = np.meshgrid(xx, yy)
centres = np.hstack((np.ravel(xx)[:, np.newaxis], np.ravel(yy)[:, np.newaxis]))

# Generate blobs to
X, y = make_blobs(n_samples=100000, centers=centres, random_state=0)"""

centers = [[1, 1]]
X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=0.1)

birch1 = new Birch(X=X, threshold=1.7, clustering=100)
birch2 = new Birch(X=X, threshold=1.7, clustering=None)



labels = [birch1.labels, birch2.labels]
centroids = [birch1.centroids, birch2.centroids]
n_clusters = [birch1.n_clusters, birch2.n_clusters]

colors_ = cycle(colors.cnames.keys())

final_step = ['without global clustering', 'with global clustering']

    ax = fig.add_subplot(1, 2, ind + 1)
    
    for this_centroid, k, col in zip(centroids, range(n_clusters), colors_):
        mask = labels == k
        ax.plot(X[mask, 0], X[mask, 1], 'w',
                markerfacecolor=col, marker='.')
        if birch_model.n_clusters is None:
            ax.plot(this_centroid[0], this_centroid[1], '+', markerfacecolor=col,
                    markeredgecolor='k', markersize=5)
    ax.set_ylim([-25, 25])
    ax.set_xlim([-25, 25])
    ax.set_autoscaley_on(False)
    ax.set_title('Birch %s' % info)


birch1.print_info()
birch2.print_info()

plt.show()
