import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
        self, 
        n_clusters: int, 
        max_iters: int = 100, 
        random_state: Optional[int] = None,
        init_method: str = 'random', 
        distance_metric: str = 'euclidean'
    ):
        """
        Inizializza l'algoritmo K-means con miglioramenti e opzioni avanzate
        
        Parameters:
        -----------
        n_clusters : int
            Numero di cluster
        max_iters : int
            Numero massimo di iterazioni
        random_state : int, optional
            Seed per la generazione di numeri casuali
        init_method : str, optional
            Metodo di inizializzazione dei centroidi ('random' o 'kmeans++')
        distance_metric : str, optional
            Metrica di distanza per calcolare la vicinanza tra punti
        """
        # Validazione dei parametri
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters deve essere un numero intero positivo")
        
        if not isinstance(max_iters, int) or max_iters <= 0:
            raise ValueError("max_iters deve essere un numero intero positivo")
        
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.init_method = init_method
        self.distance_metric = distance_metric
        
        # Inizializzazioni
        self.centroids = None
        self.labels = None
        self.inertia = None
        
        # Imposta il seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Inizializzazione dei centroidi con metodi diversi
        
        Parameters:
        -----------
        X : np.ndarray
            Dati di input
        
        Returns:
        --------
        np.ndarray
            Centroidi inizializzati
        """
        if self.init_method == 'random':
            # Selezione casuale di n_clusters punti
            random_indices = np.random.permutation(X.shape[0])[:self.n_clusters]
            return X[random_indices]
        
        elif self.init_method == 'kmeans++':
            # Implementazione dell'inizializzazione k-means++
            centroids = [X[np.random.randint(X.shape[0])]]
            
            for _ in range(1, self.n_clusters):
                # Calcola le distanze dai centroidi più vicini
                distances = cdist(X, centroids, metric=self.distance_metric)
                min_distances = distances.min(axis=1)
                
                # Sceglie il prossimo centroide con probabilità proporzionale alla distanza
                probabilities = min_distances / min_distances.sum()
                next_centroid_idx = np.random.choice(X.shape[0], p=probabilities)
                centroids.append(X[next_centroid_idx])
            
            return np.array(centroids)
        
        else:
            raise ValueError("Metodo di inizializzazione non valido")
    
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
        # Validazione input
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X deve essere una matrice 2D")
        
        # Inizializzazione centroidi
        self.centroids = self._init_centroids(X)
        
        # Parametri per la convergenza
        convergence_threshold = 1e-4
        
        for _ in range(self.max_iters):
            # Memorizza i vecchi centroidi
            old_centroids = self.centroids.copy()
            
            # Assegna i punti ai cluster
            self.labels = self._assign_clusters(X)
            
            # Aggiorna i centroidi
            self._update_centroids(X)
            
            # Calcola l'inerzia (somma delle distanze al quadrato)
            self.inertia = self._compute_inertia(X)
            
            # Verifica la convergenza
            if np.all(np.linalg.norm(self.centroids - old_centroids, axis=1) < convergence_threshold):
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
        # Calcolo efficiente delle distanze
        distances = cdist(X, self.centroids, metric=self.distance_metric)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray) -> None:
        """
        Aggiorna la posizione dei centroidi
        
        Parameters:
        -----------
        X : array-like di forma (n_samples, n_features)
            Dati di input
        """
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            
            if len(cluster_points) > 0:
                # Calcolo del nuovo centroide come media dei punti
                self.centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Gestione cluster vuoti: reinizializzazione casuale
                self.centroids[k] = X[np.random.choice(X.shape[0])]
    
    def _compute_inertia(self, X: np.ndarray) -> float:
        """
        Calcola l'inerzia (somma delle distanze al quadrato)
        
        Parameters:
        -----------
        X : array-like di forma (n_samples, n_features)
            Dati di input
        
        Returns:
        --------
        float
            Valore dell'inerzia
        """
        distances = cdist(X, self.centroids, metric=self.distance_metric)
        return np.sum(np.min(distances, axis=1) ** 2)
    
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
        X = np.asarray(X)
        if self.centroids is None:
            raise ValueError("Il modello non è stato ancora addestrato. Chiamare fit() prima.")
        
        return self._assign_clusters(X)
    
    def get_centroids(self) -> np.ndarray:
        """
        Restituisce i centroidi del clustering
        
        Returns:
        --------
        np.ndarray
            Array dei centroidi
        """
        return self.centroids

# Esempio di utilizzo con valutazione
if __name__ == "__main__":
    # Genera dati di esempio
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(0, 1, (100, 2)),
        np.random.normal(4, 1, (100, 2)),
        np.random.normal(8, 1, (100, 2))
    ])
    
    # Crea e addestra il modello con diverse configurazioni
    methods = ['random', 'kmeans++']
    metrics = ['euclidean', 'manhattan']
    
    fig, axs = plt.subplots(len(methods), len(metrics), figsize=(15, 10))
    
    for i, init_method in enumerate(methods):
        for j, distance_metric in enumerate(metrics):
            # Crea e addestra il modello
            kmeans = KMeans(
                n_clusters=3, 
                random_state=42, 
                init_method=init_method,
                distance_metric=distance_metric
            )
            kmeans.fit(X)
            
            # Visualizza i risultati
            axs[i, j].scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
            axs[i, j].scatter(
                kmeans.centroids[:, 0], 
                kmeans.centroids[:, 1], 
                c='red', 
                marker='x', 
                s=200, 
                linewidths=3
            )
            axs[i, j].set_title(f'{init_method} - {distance_metric}')
    
    plt.tight_layout()
    plt.show()
    
    # Valutazione con silhouette score
    from sklearn.metrics import silhouette_score
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels)
    print(f"Silhouette Score: {score}")