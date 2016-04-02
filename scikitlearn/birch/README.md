# birch.py
Implements a function to use Birch with an array of data.

# test_birch.py
Compares Birch with and without clustering on a synthetic dataset having 100000 samples.
If n_clusters is set to None, the data is reduced from 100000 samples to a set of 158 clusters.
When n_clusters is set to 100, the final clustering step reduces the previous clusters to a number of 100 clusters.

# sequential_birch.py
Implements functions to use Birch with sequential data

# test_sequential_birch.py
Tries Birch with sequential data. 
Random vectors are clusterised, then new random vectors are added to the clusters. One old cluster in completed, and one new cluster is created, showing that this algorithm works well.
