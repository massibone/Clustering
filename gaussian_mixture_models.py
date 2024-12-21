import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generiamo dei dati di esempio
np.random.seed(0)
n_samples = 1000
n_features = 2
n_components = 3

# Generiamo dei dati da tre distribuzioni gaussiane diverse
mean1 = [0, 0]
cov1 = [[1, 0.5], [0.5, 1]]
data1 = np.random.multivariate_normal(mean1, cov1, n_samples // 3)

mean2 = [5, 5]
cov2 = [[2, 1], [1, 2]]
data2 = np.random.multivariate_normal(mean2, cov2, n_samples // 3)
