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

# Creiamo un modello GMM con 3 componenti e utilizziamo l'algoritmo EM per l'apprendimento
gmm = GaussianMixture(n_components=n_components, covariance_type='full', n_init=10)

# Addestriamo il modello sui dati
gmm.fit(data)

# Possiamo ora utilizzare il modello per fare previsioni
# Ad esempio, possiamo calcolare la probabilità di appartenenza di un punto ai vari componenti
point = np.array([2, 2])
probabilities = gmm.predict_proba([point])
print(probabilities)

# Possiamo anche visualizzare i dati e le componenti del modello
plt.scatter(data[:, 0], data[:, 1], c=gmm.predict(data), cmap='viridis')
plt.title("Dati e componenti del modello GMM")
plt.show()

# Possiamo anche visualizzare le ellissi di confidenza delle componenti
from scipy.stats import norm
from matplotlib.patches import Ellipse

# Calcoliamo le proprietà delle componenti
weights = gmm.weights_
means = gmm.means_
covs = gmm.covariances_

# Disegnamo le ellissi di confidenza
for i in range(n_components):
    eigenvalues, eigenvectors = np.linalg.eigh(covs[i])
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    vx, vy = eigenvectors[:, 0] * np.sqrt(eigenvalues[0]), eigenvectors[:, 1] * np.sqrt(eigenvalues[1])
    hull = Ellipse(xy=means[i], width=2 * vx, height=2 * vy, edgecolor='black', facecolor='none')
    plt.gca().add_patch(hull)

plt.show()
