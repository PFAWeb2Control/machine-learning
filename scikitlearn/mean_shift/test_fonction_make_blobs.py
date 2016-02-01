import numpy as np
#from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
#from itertools import cycle

n = 10000 #Nombre de points à afficher
centers = [[1, 1], [-5,-5]] #Centres des nuages de points
X, _ = make_blobs(n_samples=n, centers=centers, cluster_std=0.6)

x = np.zeros(n)
y = np.zeros(n)

for i in range(n):
    x[i] = X[i][0]
    y[i] = X[i][1]

print x
print y

plt.plot(x, y, 'ro')
plt.show()
