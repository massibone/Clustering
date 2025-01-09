import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
