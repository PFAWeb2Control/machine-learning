# coding: utf-
from itertools import cycle
from time import time
import numpy as np

from sklearn.cluster import Birch
    
def birch_algo(X, threshold=1.7, clustering=None):
        birch = Birch(threshold=threshold, n_clusters=clustering)
        birch.fit(X)
        labels = birch.labels_
        centroids = birch.subcluster_centers_
        labels_unique = np.unique(labels)
        n_clusters = labels_unique.size
        print(" The number of clusters is : %d" % n_clusters)

     #    print("Centre(s):")
     #    for i in range(n_clusters):
     #    	print i,
     #    	print centroids[i]

        return labels, centroids, n_clusters     



  


