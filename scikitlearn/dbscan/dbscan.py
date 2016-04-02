import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
"""
Function that runs the DBSCAN algorithm on a set of data using 
scikit-learn library.
Parameters:
- X[array] is an array or a distance matrix
- metric[string or callable] is the metric to use when calculating 
the distance between instances.
- eps[float] the maximum distance between two sample that will be 
considered as in the same neighborhood. This parameter is optional.
- min_samples[int] is the number of sample in a neighborhood. It is
optinal.
The function returns the labels and the number of clusters.
"""
def dbscan_algo(X, metric, eps=0.3, min_samples=10):
        
	db = DBSCAN(metric=metric, eps=eps, min_samples=min_samples).fit(X)
   	labels = db.labels_
   	n_clusters = len(set(labels)) - int(-1 in labels)
	return labels, n_clusters
