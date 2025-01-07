import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

# Generiamo un dataset di esempio
np.random.seed(0)
n_samples = 200
n_features = 2
X = np.random.rand(n_samples, n_features)
