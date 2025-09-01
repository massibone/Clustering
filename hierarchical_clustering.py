
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Genera un dataset casuale
X, y = make_blobs(n_samples=10, centers=3, random_state=0, cluster_std=1.0)

# Esegui il clustering gerarchico
clustering = AgglomerativeClustering(n_clusters=3)
clustering.fit(X)


# Visualizza i risultati
plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, cmap='viridis', marker='o')

# Crea un dendrogramma
linked = linkage(X, 'ward')
plt.figure(figsize=(10, 5))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogramma del Clustering Gerarchico')
plt.xlabel('Campioni')
plt.ylabel('Distanza')
plt.show()
 hierarchical_clustering2.py  dentro broadcast
    ├── dbscan.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Genera un dataset casuale (formato a mezzaluna)
X, y = make_moons(n_samples=300, noise=0.05, random_state=0)

# Esegui il clustering DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
clusters = dbscan.fit_predict(X)

# Visualizza i risultati
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('Clustering DBSCAN')
plt.xlabel('Caratteristica 1')
plt.ylabel('Caratteristica 2')
plt.colorbar(label='Cluster')
plt.show()
