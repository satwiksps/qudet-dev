
from typing import Union, List, Dict
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from qudet.core.base import BaseReducer

class StreamingHasher(BaseReducer):
    """
    Implements the 'Hashing Trick' to map high-dimensional categorical 
    streams into fixed-width vectors suitable for Quantum Encoding.
    
    This is memory-efficient and stateless (perfect for streaming).
    
    Parameters
    ----------
    n_features : int
        The target dimension (must be power of 2 for qubit mapping).
        E.g., 2**10 = 1024 dimensions maps to 10 qubits via Amplitude Encoding.
    """
    
    def __init__(self, n_features: int = 1024):
        self.n_features = n_features
        self.hasher_ = FeatureHasher(n_features=self.n_features, input_type='dict')

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, List[Dict]]) -> np.ndarray:
        """
        Converts DataFrame rows or List of Dicts into fixed-size hash vectors.
        """
        if isinstance(X, pd.DataFrame):
            data_iter = X.to_dict(orient='records')
        else:
            data_iter = X
            
        return self.hasher_.transform(data_iter).toarray()
