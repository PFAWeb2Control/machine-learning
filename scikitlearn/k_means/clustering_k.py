import numpy as np
from sklearn import cluster
classificateur = cluster.KMeans(n_clusters=3)
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle

def moyenne(tableau):
    return sum(tableau, 0.0) / len(tableau)

def variance(tableau):
    m=moyenne(tableau)
    return moyenne([(x-m)**2 for x in tableau])

def distance(coord1, coord2):
    return np.sqrt((coord1[0]-coord2[0])*(coord1[0]-coord2[0])+(coord1[1]-coord2[1])*(coord1[1]-coord2[1]))

# Generation d'une matrice temporaire correspondant a l'entree du programme (a supprimer par la suite)
n = 1000
centers = [[1, 1, 1], [5, 0, 0],[-3,-5, 6],[7,-4, 3]]
X, _ = make_blobs(n_samples=n, centers=centers, cluster_std=0.6)


#Affichage pour dimension 2
plt.plot([X[a][0] for a in range(n)], [X[b][1] for b in range(n)], col + '.')
plt.show()


nb_clusters_max = 10;

#Cherche le nombre de clusters optimal dans les données data, entre 2 et nb_clusters_max
def clustering_k(data, nb_clusters_max):

    var_values = np.full((nb_clusters_max, 1),-1) #Valeur à afficher en fonction du nb de clusters
    dim = len(data[0])
    
    #Génération classificateur
    for i in range(2,nb_clusters_max):
        clusters = [[]]*i
        clf = cluster.KMeans(n_clusters=i)
        clf.fit(data)
        labels = clf.labels_
        labels_unique = np.unique(labels)
        
        for j in range(0,n):
            clusters[labels[j]] = clusters[labels[j]] + [[data[j][x] for x in range(dim)]]

        centers_mean = np.full((i,dim), -1)
        distance_mean = [[]]*i #Pour chaque cluster, distance de chaque point du cluster au centre du cluster
        var_intra = np.full((i,1),0)
        var_inter = np.full((i,1),0)
            
        for j in range(i):
            for k in range(dim):
                centers_mean[j][k] = moyenne([clusters[j][x][k] for x in range(len(clusters[j]))])

            distance_mean[j] = [np.linalg.norm(coord1 - centers_mean[j]) for coord1 in clusters[j]]
            
            var_intra[j] = [variance(distance_mean[j])]
            var_inter[j] = [moyenne(distance_mean[j])]

        
        var_values[i] = (variance(var_inter) - moyenne(var_intra))
        
    i=0
    while var_values[i] == -1:
        i = i+1
        
    plt.plot([k for k in range(0, nb_clusters_max)], var_values)
    
    plt.axis([i, nb_clusters_max, min(var_values[i:])-1, max(var_values)+1])
    plt.show()
        

clustering_k(X, nb_clusters_max);
