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
sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10)
labels = sc.fit_predict(X)

# Visualizziamo i risultati
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("Clustering spettrale")
plt.show()
