import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from typing import List, Tuple

class GaussianMixtureAnalyzer:
    def __init__(self, n_components: int = 3, random_state: int = 42):
        '''
        Inizializza l'analizzatore di Miscela Gaussiana
        
        Args:
            n_components (int): Numero di componenti gaussiane
            random_state (int): Seed per riproducibilità
        '''
        
        self.random_state = random_state
        self.n_components = n_components
        self.gmm = None
        self.data = None
