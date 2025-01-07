import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

# Generiamo un dataset di esempio
np.random.seed(0)
n_samples = 200
n_features = 2
X = np.random.rand(n_samples, n_features)

# Creiamo due cluster non lineari
X[:100, :] += 2
X[100:, :] -= 2

# Eseguiamo il clustering spettrale
n_clusters = 2
sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
labels = sc.fit_predict(X)
