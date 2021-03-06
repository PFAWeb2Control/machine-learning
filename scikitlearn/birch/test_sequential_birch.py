import numpy as np
from sklearn.cluster import Birch
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle

# Generates random vectors to cluster
n_samples = 50
centers = [[0, 1], [4, -2], [-2, 2], [0, -1]]
X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.2)

# Creates the Birch classificator and gives it the vectors
brc = Birch(branching_factor=50, n_clusters=None, threshold=0.8, compute_labels=True)
brc.fit(X)

labels = brc.labels_
cluster_centers = brc.subcluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

# Prints the points generated
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.axis([-4,12,-4,12])
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

# Generates new vectors to clusters, some close to an already existant cluster ([0,1]), the others far from any clusters
n_samples_2 = 200
centers_2 = [[10,10], [0,1]]
X2, _ = make_blobs(n_samples=n_samples_2, centers=centers_2, cluster_std=0.2)

# Prints these new points
plt.plot([X2[a][0] for a in range(n_samples_2)], [X2[b][1] for b in range(n_samples_2)], '.')
plt.axis([-4,12,-4,12])
plt.show()

labels = brc.labels_
cluster_centers = brc.subcluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

# Adds the new points to the old clustering with "partial_fit"
brc.partial_fit(X2)

labels = np.concatenate([labels,brc.labels_])
cluster_centers = brc.subcluster_centers_ # All the cluster centers existing (old and new)
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

# All the points generated (old and new)
X_tot = np.concatenate([X,X2])

# Prints the different clusters computed
# We can see that the some new points were added to an old cluster ([0,1]), and the others created a new cluster
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X_tot[my_members, 0], X_tot[my_members, 1], col + '.')
    plt.axis([-4,12,-4,12])
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
