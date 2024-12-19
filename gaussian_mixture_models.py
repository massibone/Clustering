import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generiamo dei dati di esempio
np.random.seed(0)
n_samples = 1000
n_features = 2
n_components = 3
