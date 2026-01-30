
import pandas as pd
import numpy as np
from typing import Union
from qudet.core.base import BaseReducer
from .projections import RandomProjector
from .coresets import CoresetReducer
from .sketching import StreamingHasher


class AutoReducer(BaseReducer):
    """
    Meta-Reducer that automatically selects the best reduction strategy
    based on the dataset's shape and target quantum hardware constraints.
    
    Adaptive Selection Logic:
    1. Too many columns (n_features > target_qubits)? → Random Projection
    2. Too many rows (n_samples > max_rows)? → Coreset Sampling
    3. Otherwise → No reduction needed
    
    Philosophy: Data Engineers shouldn't need to understand quantum physics
    to choose the right reduction algorithm. This class chooses for them.
    
    Example:
        >>> ar = AutoReducer(target_qubits=10, max_rows=500)
        >>> ar.fit(big_dataframe)
        >>> reduced_data = ar.transform(big_dataframe)
    """
    
    def __init__(self, target_qubits: int = 10, max_rows: int = 500):
        """
        Initialize AutoReducer with constraints.
        
        Parameters
        ----------
        target_qubits : int
            Maximum number of qubits available on target quantum hardware.
            Data dimensionality will be reduced to match this.
        max_rows : int
            Maximum number of samples to keep after reduction.
            Large datasets will be downsampled to this size.
        """
        self.target_qubits = target_qubits
        self.max_rows = max_rows
        self.pipeline_ = []
        self.reduction_strategy_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Analyze data shape and build reduction pipeline.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data to analyze
        y : array-like, optional
            Target labels (ignored, for sklearn compatibility)
            
        Returns
        -------
        self
            Fitted AutoReducer instance
        """
        self.pipeline_ = []
        
        if isinstance(X, pd.DataFrame):
            n_rows, n_cols = X.shape
        else:
            n_rows, n_cols = X.shape

        print(f"--- AutoReducer analyzing shape ({n_rows}, {n_cols}) ---")

        if n_cols > self.target_qubits:
            print(
                f"   -> Detected High Dimensionality ({n_cols} > {self.target_qubits}). "
                f"Adding RandomProjector."
            )
            proj = RandomProjector(n_components=self.target_qubits)
            proj.fit(X)
            self.pipeline_.append(("projection", proj))
            X = proj.transform(X)

        if n_rows > self.max_rows:
            print(
                f"   -> Detected Large Volume ({n_rows} > {self.max_rows}). "
                f"Adding CoresetReducer."
            )
            core = CoresetReducer(target_size=self.max_rows)
            core.fit(X)
            self.pipeline_.append(("coreset", core))
            
        if not self.pipeline_:
            print("   -> Data fits comfortably. No reduction needed.")
            self.reduction_strategy_ = "none"
        else:
            self.reduction_strategy_ = f"{len(self.pipeline_)} step(s)"
            
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Apply the chain of reducers to data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data to reduce
            
        Returns
        -------
        np.ndarray
            Reduced data
        """
        data = X
        for name, reducer in self.pipeline_:
            data = reducer.transform(data)
        return data
