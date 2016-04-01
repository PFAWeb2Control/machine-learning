import numpy as np
from sklearn import cluster
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle

def mean(array):
    return sum(array, 0.0) / len(array)

def variance(array):
    m=mean(array)
    return mean([(x-m)**2 for x in array])

# Search the optimal number of clusters for the data, between 2 and nb_clusters_max
def clustering_k(data, nb_clusters_max):

    var_values = np.full((nb_clusters_max, 1),-1) #Values to compute for each number of clusters
    dim = len(data[0])
    
    #Generation of the classificator
    for i in range(2,nb_clusters_max):
        clusters = [[]]*i
        clf = cluster.KMeans(n_clusters=i)
        clf.fit(data)
        labels = clf.labels_
        labels_unique = np.unique(labels)
        
        for j in range(0,n):
            clusters[labels[j]] = clusters[labels[j]] + [[data[j][x] for x in range(dim)]]

        centers_mean = np.full((i,dim), -1)
        distance_mean = [[]]*i #For each cluster, distance between each point in the cluster and the center of the cluster
        var_intra = np.full((i,1),0)
        var_inter = np.full((i,1),0)
            
        for j in range(i):
            for k in range(dim):
                centers_mean[j][k] = mean([clusters[j][x][k] for x in range(len(clusters[j]))])

            distance_mean[j] = [np.linalg.norm(coord1 - centers_mean[j]) for coord1 in clusters[j]]
            
            var_intra[j] = [variance(distance_mean[j])]
            var_inter[j] = [mean(distance_mean[j])]

        #Compute the variance inter-clusters minus the variance intra_cluster
        var_values[i] = (variance(var_inter) - mean(var_intra))
        
    i=0
    while var_values[i] == -1:
        i = i+1
        
    #Print the graph of (variance inter - variance intra) as a function of the number of clusters k
    #Elbow method: The optimum k is the abscissa where the curve makes an angle (4 in this example)
    plt.plot([k for k in range(0, nb_clusters_max)], var_values)
    plt.axis([i, nb_clusters_max, min(var_values[i:])-1, max(var_values)+1])
    plt.show()
        

# Generate an array of vectors (2 dimensions) to cluster
n = 1000
centers = [[1, 1], [5, 0],[-3,-5],[7,-4]]
X, _ = make_blobs(n_samples=n, centers=centers, cluster_std=0.6)


#Print each point of the generated data
plt.plot([X[a][0] for a in range(n)], [X[b][1] for b in range(n)], '.')
plt.show()


nb_clusters_max = 10;
clustering_k(X, nb_clusters_max);
