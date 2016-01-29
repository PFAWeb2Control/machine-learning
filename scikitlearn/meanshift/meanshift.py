import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

# Génération d'un vecteur temporaire correspondant à l'entrée du programme (à supprimer par la suite)
centers = [[1, 1, 1], [5, 4, 9], [-4, -8, 0]]
X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=0.6)

# Facteur à modifier pour influencer les résultats
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

# Application de l'algorithme MeanShift
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

# Récupération des étiquettes de chaque point et du centre de chaque regroupement
labels = ms.labels_
cluster_centers = ms.cluster_centers_


labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique) # Nombre de regroupements

# Affichage
print("Nombre de regroupement(s): %d" % n_clusters_)

print("Centre(s):")
for i in range(n_clusters_):
    print i,
    print cluster_centers[i]