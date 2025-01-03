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

mean3 = [10, 10]
cov3 = [[3, 2], [2, 3]]
data3 = np.random.multivariate_normal(mean3, cov3, n_samples // 3)

# Uniamo i dati
data = np.concatenate((data1, data2, data3))
# Creiamo un modello GMM con 3 componenti
gmm = GaussianMixture(n_components=n_components)

# Addestriamo il modello sui dati
gmm.fit(data)

# Possiamo ora utilizzare il modello per fare previsioni
# Ad esempio, possiamo calcolare la probabilità di appartenenza di un punto ai vari componenti
point = np.array([2, 2])
probabilities = gmm.predict_proba([point])
print(probabilities)

# Possiamo anche visualizzare i dati e le componenti del modello
plt.scatter(data[:, 0], data[:, 1], c=gmm.predict(data))
plt.show()
