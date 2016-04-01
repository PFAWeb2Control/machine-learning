import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle

#Generates an array of vectors to cluster
centers = [[1, 1],[5,3],[-2,-1]]
X, _ = make_blobs(n_samples=10, centers=centers, cluster_std=0.5)

bandwidth = 1

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

# Giving one vector at a time to the classificator, to try the algorithm in a sequential way
for i in range(len(X)):
    ms.fit([X[i]])
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    
    # On each new graph printed, a new point is supposed to appear. But we can see that the old ones are gone.
    # That's because the function fit() reset each time it's called.
    # Meanshift can't be used for sequential processing in Scikitlearn.
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
    
