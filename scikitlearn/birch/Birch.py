from itertools import cycle
from time import time
import numpy as np

from sklearn.cluster import MeanShift, Birch, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs


class Birch:

    def __init__(birch, X, threshold=1.7, clustering=None):
        birch.birch_algo = Birch(threshold=threshold, n_clustters=clustering)
        birch.t = time()
        birch.fit(X)
        birch.time_ = time() - birch.t
        birch.labels = birch.birch_algo.labels_
        birch.centroids = birch.birch_algo.subcluster_centers_
        birch.n_clusters = np.unique(birch.labels).size

    def print_info(birch):
        #print("Birch with global clustering as the final step took %0.2f seconds" birch.time_)
        print(" The number of clusters is : %d" % birch.n_clusters)
        
    
        
