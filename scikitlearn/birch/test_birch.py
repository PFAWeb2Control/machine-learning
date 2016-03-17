import numpy as np
from sklearn.cluster import Birch
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle

n_samples = 50
centers = [[0, 1], [4, -2], [-2, 2], [0, -1]]
X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.2)

plt.plot([X[a][0] for a in range(n_samples)], [X[b][1] for b in range(n_samples)], col + '.')
plt.show()

brc = Birch(branching_factor=50, n_clusters=None, threshold=0.8, compute_labels=True)
brc.fit(X)

n_samples_2 = 200
centers_2 = [[10,10], [0,1]]
X2, _ = make_blobs(n_samples=n_samples_2, centers=centers_2, cluster_std=0.2)

plt.plot([X2[a][0] for a in range(n_samples_2)], [X2[b][1] for b in range(n_samples_2)], col + '.')
plt.show()

labels = brc.labels_
print len(labels)
cluster_centers = brc.subcluster_centers_
print len(cluster_centers)

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

brc.partial_fit(X2)

labels = np.concatenate([labels,brc.labels_])
print len(labels)

cluster_centers = brc.subcluster_centers_ #Tous les centres existants (anciens et nouveaux)
print len(cluster_centers)

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print n_clusters_

X_tot = np.concatenate([X,X2])

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X_tot[my_members, 0], X_tot[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
