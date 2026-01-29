Clustering Algorithms in Python 🔍📊
Esplorazione pratica di algoritmi di clustering con implementazioni from scratch e Scikit-learn. Include K-means, Hierarchical, DBSCAN, Fuzzy, GMM e Spectral. Perfetto per FreeCodeCamp Scientific Computing e portfolio ML.
​

Clustering raggruppa dati simili: minimizza distanza intra-cluster, massimizza inter-cluster.

📁 File Principali
| Script                                            | Algoritmo                   |
| ------------------------------------------------- | --------------------------- |
| kmeans.py / kmeans2.py                            | K-means (centroidi)         |  
| hierarchical_clustering.py                        | Clustering gerarchico       |  
| dbscan2.py                                        | DBSCAN (densità-based)      |  
| fuzzy_clustering.py                               | Fuzzy C-means               |  
| gaussian_mixture.py / gaussian_mixture_models*.py | Gaussian Mixture Models     |   
| spectral*.py                                      | Spectral clustering (grafi) |  
| clustering_python.ods                             | Dati esempio                |   

🚀 Setup & Esecuzione

git clone https://github.com/massibone/Clustering
cd Clustering
pip install numpy matplotlib scikit-learn pandas openpyxl
python kmeans.py  # Genera plot cluster

Esempio K-means (adatta da kmeans.py):


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y_true = make_blobs(100, centers=3, cluster_std=0.60, random_state=0)

# K-means from scratch (semplificato)
def kmeans(X, k=3, max_iter=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return labels, centroids

labels, centroids = kmeans(X)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red')
plt.title('K-means Clustering')
plt.show()
Output: Plot con 3 cluster + centroidi.

🧪 Algoritmi Coperti
K-means: Partiziona in K cluster (elbow method per K).

Hierarchical: Dendrogramma linkage (single/complete/ward).

DBSCAN: Cluster densi, outlier automatici (eps, min_samples).

Fuzzy: Appartenenza probabilistica (non hard assignment).

GMM: Modelli probabilistici gaussiani (EM algorithm).

Spectral: Laplacian grafi per cluster non convessi.

📈 Visualizzazioni
Tutti script generano plot Matplotlib (scatter, dendrogrammi). Usa clustering_python.ods per dati reali (importa con pandas).

Esempio GMM:

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3).fit(X)
labels = gmm.predict(X)

🔗 Risorse & Repo Correlati
Probability_and_Statistics_Python
Scikit-learn docs: Clustering
MathExplainedWithPython
