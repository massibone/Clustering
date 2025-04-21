import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class KMeans:
    def __init__(self, n_clusters: int, max_iters: int = 100, random_state: int = None):
        """
        Inizializza l'algoritmo K-means
        
        Parameters:
        -----------
        n_clusters : int
            Numero di cluster
        max_iters : int
            Numero massimo di iterazioni
        random_state : int
            Seed per la generazione di numeri casuali
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Addestra il modello K-means sui dati
        
        Parameters:
        -----------
        X : array-like di forma (n_samples, n_features)
            Dati di training
        
        Returns:
        --------
        self : oggetto KMeans
        """
        if self.random_state:
            np.random.seed(self.random_state)
            
        # Inizializzazione casuale dei centroidi
        random_indices = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.centroids = X[random_indices]
        
        for _ in range(self.max_iters):
            # Memorizza i vecchi centroidi
            old_centroids = self.centroids.copy()
            
            # Assegna i punti ai cluster
            self.labels = self._assign_clusters(X)
            
            # Aggiorna i centroidi
            self._update_centroids(X)
            
            # Verifica la convergenza
            if np.all(old_centroids == self.centroids):
                break
                
        return self
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assegna ogni punto al cluster più vicino
        
        Parameters:
        -----------
        X : array-like di forma (n_samples, n_features)
            Dati di input
            
        Returns:
        --------
        labels : array di forma (n_samples,)
            Etichette dei cluster assegnati
        """
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X: np.ndarray) -> None:
        """
        Aggiorna la posizione dei centroidi
        
        Parameters:
        -----------
        X : array-like di forma (n_samples, n_features)
            Dati di input
        """
        for k in range(self.n_clusters):
            if np.sum(self.labels == k) > 0:  # evita divisione per zero
                self.centroids[k] = np.mean(X[self.labels == k], axis=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice il cluster di appartenenza per i dati X
        
        Parameters:
        -----------
        X : array-like di forma (n_samples, n_features)
            Nuovi dati da classificare
            
        Returns:
        --------
        labels : array di forma (n_samples,)
            Etichette dei cluster predetti
        """
        return self._assign_clusters(X)

# Esempio di utilizzo
if __name__ == "__main__":
    # Genera dati di esempio
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(0, 1, (100, 2)),
        np.random.normal(4, 1, (100, 2)),
        np.random.normal(8, 1, (100, 2))
    ])
    
    # Crea e addestra il modello
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Visualizza i risultati
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                c='red', marker='x', s=200, linewidths=3)
    plt.title('K-means Clustering')
    plt.show()
