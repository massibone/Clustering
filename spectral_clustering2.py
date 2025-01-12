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
