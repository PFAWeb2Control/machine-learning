# coding: utf-
from itertools import cycle
from time import time
import numpy as np

from sklearn.cluster import Birch
"""
A function that runs the birch clustering algorithm.

Parameters:
-X is our input data (an array)
-threshold is the radius of the subcluster obtained by
merging a new sample and the closest subcluster should
be lesser than the threshold. Default value is 1.7.
-clustering is number of cluster after the final clustering
which treats the subclusters from the leaves as new sample.
By default, this final clustering step is not performed and
the subclusters are returned as they are.

This function returns the labels of the clusters, their centroids
and their number.
""" 
def birch_algo(X, threshold=1.7, clustering=None):
        birch = Birch(threshold=threshold, n_clusters=clustering)
        birch.fit(X)
        labels = birch.labels_
        centroids = birch.subcluster_centers_
        labels_unique = np.unique(labels)
        n_clusters = labels_unique.size
        print(" The number of clusters is : %d" % n_clusters)
        return labels, centroids, n_clusters     



  


