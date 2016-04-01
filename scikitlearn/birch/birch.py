import numpy as np
from sklearn.cluster import Birch

# Initializes the classificator
def clf_init(b_factor = 50, threshold = 0.8):
    return Birch(branching_factor=b_factor, n_clusters=None, threshold=threshold, compute_labels=True)

# Adds vectors to cluster
# Data must be a 2-dimensions array or list, even to add only one vector
# Return an array Retourne une tableau contenant le numéro de clusters de chaque data
def clf_add_data(clf, data):
    if (type(data) is list):
        data = np.asarray(data)
    clf.partial_fit(np.asarray(data))
    return brc.labels_

#Retourne les centres de chaque cluster
def clf_cluster_centers(clf):
    return clf.subcluster_centers_
