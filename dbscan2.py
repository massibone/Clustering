import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

def generate_dataset(
    n_samples: int = 300, 
    noise: float = 0.05, 
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Genera un dataset a forma di mezzaluna.
    '''
    return make_moons(
        n_samples=n_samples, 
        noise=noise, 
        random_state=random_state
    )

def perform_dbscan_clustering(
    X: np.ndarray, 
    eps: float = 0.2, 
    min_samples: int = 5
) -> np.ndarray:
    """
    Esegue il clustering DBSCAN sui dati.
    """
    # Preprocessing: standardizzazione dei dati
    X_scaled = StandardScaler().fit_transform(X)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(X_scaled)

def plot_dbscan_clusters(
    X: np.ndarray, 
    clusters: np.ndarray
):
    """
    Visualizza i risultati del clustering DBSCAN.
    """
    plt.figure(figsize=(12, 8))
    
    # Distingui punti di rumore
    noise_mask = clusters == -1
    cluster_mask = ~noise_mask
    
    # Plot punti di rumore
    plt.scatter(
        X[noise_mask, 0], 
        X[noise_mask, 1], 
        c='black', 
        marker='x', 
        s=100, 
        label='Rumore'
    )
    
    # Plot cluster
    scatter = plt.scatter(
        X[cluster_mask, 0], 
        X[cluster_mask, 1], 
        c=clusters[cluster_mask], 
        cmap='viridis', 
        marker='o'
    )
    
    plt.title('Clustering DBSCAN')
    plt.xlabel('Caratteristica 1')
    plt.ylabel('Caratteristica 2')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.show()

def evaluate_clustering(
    X: np.ndarray, 
    clusters: np.ndarray
) -> Tuple[float, float]:
    """
    Valuta la qualità del clustering.
    """
    # Esclude i punti di rumore
    mask = clusters != -1
    
    silhouette = silhouette_score(
        X[mask], 
        clusters[mask]
    )
    
    calinski = calinski_harabasz_score(
        X[mask], 
        clusters[mask]
    )
    
    return silhouette, calinski

def main():
    # Genera un dataset casuale
    X, y = generate_dataset(
        n_samples=300, 
        noise=0.05, 
        random_state=42
    )
    
    # Esegui il clustering DBSCAN
    clusters = perform_dbscan_clustering(
        X, 
        eps=0.2, 
        min_samples=5
    )
    
    # Visualizza i risultati
    plot_dbscan_clusters(X, clusters)
    
    # Valuta la qualità del clustering
    silhouette, calinski = evaluate_clustering(X, clusters)
    
    print(f"Silhouette Score: {silhouette}")
    print(f"Calinski-Harabasz Score: {calinski}")

if __name__ == "__main__":
    main()
