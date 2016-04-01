# k_means.py
This file is an example of how can be determined the optimum number of clusters k to make.
It implements the Elbow method, where the graph of the (variance inter-clusters, - variance intra-cluster) as a function of k is computed.
The optimum k is the abscissa where the curve makes an angle.
This file creates random points in 4 clusters, prints them in a graph, then prints the computed curve showing an angle for k=4.