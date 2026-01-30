
from typing import Union
import numpy as np
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection
from qudet.core.base import BaseReducer

class RandomProjector(BaseReducer):
    """
    Reduces feature dimensions (columns) using Gaussian Random Projection.
    This allows high-dimensional data (e.g., 1000 cols) to fit into 
    limited qubits (e.g., 10 qubits) while preserving distances.
    
    Parameters
    ----------
    n_components : int
        The target number of dimensions (qubits) to reduce to.
    """
    
    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.model_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        self.model_ = GaussianRandomProjection(n_components=self.n_components, random_state=42)
        self.model_.fit(X)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Projector not fit yet.")
            
        return self.model_.transform(X)
