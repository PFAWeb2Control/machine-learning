import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

def dbscan_algo(X, metric, eps=0.3, min_samples=10):

	db = DBSCAN(metric=metric, eps=eps, min_samples=min_samples).fit(X)
  labels = db.labels_
  n_clusters = len(set(labels)) - int(-1 in labels)
  
	return labels, n_clusters
