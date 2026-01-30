from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np
import pandas as pd


class BaseReducer(ABC):
    """Abstract Base Class for Data Reduction."""

    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Learn the structure of the dataset."""
        pass

    @abstractmethod
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Reduce the dataset to a size compatible with Quantum Memory."""
        pass

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class BaseEncoder(ABC):
    """Abstract Base Class for Quantum Encoding."""

    @abstractmethod
    def encode(self, data: np.ndarray):
        """Returns a Quantum Circuit or State Vector."""
        pass


class BaseQuantumEstimator(ABC):
    """Abstract Base Class for Quantum Algorithms."""

    def __init__(self, backend_name: str = "aer_simulator", shots: int = 1024):
        self.backend_name = backend_name
        self.shots = shots

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform (for estimators that can transform)."""
        return self.fit(X, y).transform(X)
