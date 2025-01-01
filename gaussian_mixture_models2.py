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
     def generate_multivariate_data(self, 
                                    means: List[List[float]], 
                                    covs: List[List[List[float]]], 
                                    samples_per_component: int = 333
                                    ) -> np.ndarray:
        """

        Genera dati multivariate gaussiani
        
        Args:
            means: Lista di medie per ciascuna distribuzione
            covs: Lista di matrici di covarianza
            samples_per_component: Campioni per componente

          
        Returns:
            Array numpy con dati generati
        """
        np.random.seed(self.random_state)
        
        try:
            data_components = [
                np.random.multivariate_normal(mean, cov, samples_per_component)
                for mean, cov in zip(means, covs)
            ]
            return np.concatenate(data_components)
        except Exception as e:
            print(f"Errore nella generazione dei dati: {e}")
            raise

   def fit_gmm(self, data: np.ndarray) -> None:
        """
        Addestra il modello GMM
        
        Args:
            data: Dati di addestramento
        """
        self.data = data
        
        # Ricerca griglia per numero ottimale di componenti
        param_grid = {'n_components': range(1, 10)}
        grid_search = GridSearchCV(
 GaussianMixture(random_state=self.random_state), 
            param_grid
        )
        grid_search.fit(data)
         # Selezione miglior modello
        self.gmm = grid_search.best_estimator_
        print(f"Numero ottimale di componenti: {self.gmm.n_components}")
       
