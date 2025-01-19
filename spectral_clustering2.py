import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def generate_nonlinear_dataset(
    n_samples: int = 200, 
    n_features: int = 2, 
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Genera un dataset con cluster non lineari.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
X = np.random.rand(n_samples, n_features)

    
    # Crea due cluster non lineari
    X[:n_samples//2, :] += 2
    X[n_samples//2:, :] -= 2
    
    return X

def perform_spectral_clustering(
    X: np.ndarray, 
    n_clusters: int = 2, 
    affinity: str = 'nearest_neighbors',
    n_neighbors: int = 10
) -> np.ndarray:
    """
    Esegue il clustering spettrale.
    """


    sc = SpectralClustering(
        n_clusters=n_clusters, 
        affinity=affinity,
        n_neighbors=n_neighbors
    )
    
    return sc.fit_predict(X_scaled)


    # Preprocessing: standardizzazione dei dati
    X_scaled = StandardScaler().fit_transform(X)
    

def plot_clustering_results(
    X: np.ndarray, 
    labels: np.ndarray,
    title: str = "Clustering Spettrale"
):
    """
    Visualizza i risultati del clustering.
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        X[:, 0], X[:, 1], 
        c=labels, 
        cmap='viridis', 
        marker='o'
    )
    plt.title(title)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
 def evaluate_clustering(
    X: np.ndarray, 
    labels: np.ndarray
) -> Tuple[float, float]:
    """
    Valuta la qualità del clustering.
    """


    # Visualizza i risultati
    plot_clustering_results(X, labels)
    
    # Valuta la qualità del clustering
    silhouette, calinski = evaluate_clustering(X, labels)
  
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    
    return silhouette, calinski
   
def main():
    # Genera un dataset di esempio
    X = generate_nonlinear_dataset(
        n_samples=200, 
        n_features=2, 
        random_state=42
    )
    
    # Esegui il clustering spettrale
    labels = perform_spectral_clustering(
        X, 
        n_clusters=2, 
        affinity='nearest_neighbors',
        n_neighbors=10
    )
    
