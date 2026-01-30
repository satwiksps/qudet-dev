
from typing import Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from qudet.core.base import BaseReducer

class CoresetReducer(BaseReducer):
    """
    Reduces a large dataset to a small 'Coreset' using K-Means clustering centers.
    
    Parameters
    ----------
    target_size : int
        The number of representative points to keep (e.g., 500).
    """
    
    def __init__(self, target_size: int = 100):
        self.target_size = target_size
        self.centers_ = None
        self.model_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        data = X.values if isinstance(X, pd.DataFrame) else X
        
        self.model_ = KMeans(n_clusters=self.target_size, random_state=42, n_init=10)
        self.model_.fit(data)
        
        self.centers_ = self.model_.cluster_centers_
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("Reducer has not been fit yet. Call fit() first.")
        
        return self.centers_
