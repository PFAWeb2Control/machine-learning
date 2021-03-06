import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle

#Generates an array of vectors to cluster
centers = [[1, 1],[5,3],[-2,-1]]
X, _ = make_blobs(n_samples=1000, centers=centers, cluster_std=0.5)

# Modifying the bandwidth will affect the clustering
# A lower value will make more clusters (smaller window) 
# Estimate_bandwidth() can be used to compute a value automatically
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

# Creates the classificator and gives it the data to cluster
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

# Information returned by the algorithm
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

# Prints the computed centers for each cluster found
print("Centre(s):")
for i in range(n_clusters_):
    print i,
    print cluster_centers[i]


# Prints the graph with each cluster in a different color
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
