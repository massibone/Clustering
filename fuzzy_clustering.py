import numpy as np
import matplotlib.pyplot as plt
from skfuzzy.cluster import FuzzyCMeans


# Generiamo un dataset di esempio
np.random.seed(0)
n_samples = 200
n_features = 2
X = np.random.rand(n_samples, n_features)

# Creiamo due cluster non lineari
X[:100, :] += 2
X[100:, :] -= 2
