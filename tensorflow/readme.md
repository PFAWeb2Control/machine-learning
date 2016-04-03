Machine learning algorithms with TensorFlow
===========================================================

Each python file implements a machine learning algorithm,
by defining a clusterize() function. This function takes a list of vectors and some optional parameters
as input, and returns revelant data to compute the cluster of each vector.
The command line
`python algorithm.py`
where "algorithm" is the name of algorithm prints the output of the algorithm with random input


K_means
-------
This first algorithm is inspired from the code available [here](https://gist.github.com/dave-andersen/265e68a5e879b5540ebc)
It returns a list with two elements:
* the centroids (a.k.a. means)
* a vector where the i^th element is the index in the centroids list of the i^th data vector cluster.

basic_CEM
---------
Simple implementation of the [CEM](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#Gaussian_mixture) algorithm.
It returns a <number of vectors> * <number of clusters> matrix, where i^th * j^th element is the probability for the i^th data vector to 
belong to the j^th cluster.
It uses a pseudo random initialisation

CEM
----
A more complex implementation of CEM, where initialisation is done by running the firsts step of the algorithm with random initialisation,
and then selecting the best run as a start point.


