import numpy as np
from sklearn.cluster import Birch
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle

n_samples = 50
centers = [[0, 1], [4, -2], [-2, 2], [0, -1]]
X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.2)

plt.plot([X[a][0] for a in range(n_samples)], [X[b][1] for b in range(n_samples)], '.')
plt.show()

#Initialiser le classificateur
def clf_init(b_factor = 50, threshold = 0.8):
    return Birch(branching_factor=b_factor, n_clusters=None, threshold=threshold, compute_labels=True)

#Ajouter des vecteurs à classer. 
#Data doit être un une liste de vecteurs. Pour ajouter un seul vecteur x -> data = [x]
#Retourne une tableau contenant le numéro de clusters de chaque data
def clf_add_data(clf, data):
    clf.partial_fit(np.asarray(data))
    return brc.labels_

#Retourne les centres de chaque cluster
def clf_cluster_centers(clf):
    return clf.subcluster_centers_

brc = init_clf()

plt.figure(1)
plt.clf()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

for i in range(len(X)):
    print i
    Y = X[:i+1]
    print Y
    print i
    labels = clf_add_data(brc, [X[i]])
    n_clusters_ = len(clf_cluster_centers(brc))

    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        print my_members
        cluster_centers = clf_cluster_centers(brc)
        cluster_center = cluster_centers[k]
        plt.plot(Y[my_members, 0], Y[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    
