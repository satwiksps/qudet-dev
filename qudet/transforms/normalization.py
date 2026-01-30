"""
Normalization methods for quantum feature spaces.

Provides various normalization and standardization techniques
optimized for quantum machine learning preprocessing.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
from qudet.core.base import BaseReducer


class QuantumNormalizer(BaseReducer):
    """
    Normalizes features to quantum-compatible scales.
    
    Quantum algorithms typically require inputs in specific ranges:
    - Unit norm normalization (L2 normalization)
    - Probability normalization (0-1, summing to 1)
    - Amplitude normalization (complex values with unit norm)
    - Quantum angle normalization (0 to 2π)
    
    Best for: Quantum algorithm preprocessing, amplitude encoding.
    """
    
    def __init__(self, method: str = "l2", scale_range: Tuple = (-1, 1)):
        """
        Initialize quantum normalizer.
        
        Args:
            method: Normalization method ('l2', 'l1', 'probability', 'amplitude', 'angle')
            scale_range: Target scale range for normalization
        """
        self.method = method.lower()
        self.scale_range = scale_range
        self.norms_ = None
        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'QuantumNormalizer':
        """
        Learn normalization parameters from data.
        
        Args:
            X: Input features
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.method == "l2":
            self.norms_ = np.linalg.norm(X, axis=1, keepdims=True)
        elif self.method == "l1":
            self.norms_ = np.sum(np.abs(X), axis=1, keepdims=True)
        elif self.method in ["probability", "amplitude", "angle"]:
            if self.method == "probability":
                self.norms_ = np.sum(X, axis=1, keepdims=True)
            else:
                self.norms_ = np.linalg.norm(X, axis=1, keepdims=True)
        
        self.fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Normalize features.
        
        Args:
            X: Input features
            
        Returns:
            Normalized features
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()
        
        if self.method in ["l2", "l1"]:
            norms = np.linalg.norm(X, axis=1, keepdims=True) if self.method == "l2" else np.sum(np.abs(X), axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = X / norms
        elif self.method == "probability":
            # Normalize to probability distribution
            normalized = np.abs(X)
            sums = np.sum(normalized, axis=1, keepdims=True)
            sums[sums == 0] = 1
            normalized = normalized / sums
        elif self.method == "amplitude":
            # Normalize to unit amplitude (complex values)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = X / norms
        elif self.method == "angle":
            # Normalize angles to [0, 2π]
            normalized = np.mod(X, 2 * np.pi)
        else:
            normalized = X
        
        # Apply scale range if needed
        if self.scale_range != (-1, 1) and self.method not in ["angle", "probability"]:
            min_val, max_val = self.scale_range
            normalized = min_val + (normalized + 1) / 2 * (max_val - min_val)
        
        return normalized

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def get_normalization_info(self) -> dict:
        """Get normalization statistics."""
        return {
            "method": self.method,
            "scale_range": self.scale_range,
            "fitted": self.fitted
        }


class RangeNormalizer(BaseReducer):
    """
    Normalizes features to specific value ranges.
    
    Supports:
    - Min-max normalization
    - Robust range normalization (using quantiles)
    - Clipping to range
    - Soft normalization (sigmoid-based)
    
    Best for: Range-based normalization, feature scaling.
    """
    
    def __init__(self, range_min: float = 0, range_max: float = 1, method: str = "minmax"):
        """
        Initialize range normalizer.
        
        Args:
            range_min: Minimum value of output range
            range_max: Maximum value of output range
            method: Normalization method ('minmax', 'robust', 'clip', 'sigmoid')
        """
        self.range_min = range_min
        self.range_max = range_max
        self.method = method.lower()
        self.min_vals_ = None
        self.max_vals_ = None
        self.q1_vals_ = None
        self.q3_vals_ = None
        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'RangeNormalizer':
        """
        Learn range parameters from data.
        
        Args:
            X: Input features
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.min_vals_ = np.min(X, axis=0)
        self.max_vals_ = np.max(X, axis=0)
        self.q1_vals_ = np.percentile(X, 25, axis=0)
        self.q3_vals_ = np.percentile(X, 75, axis=0)
        
        self.fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Normalize to specified range.
        
        Args:
            X: Input features
            
        Returns:
            Normalized features
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()
        
        if self.method == "minmax":
            normalized = self._minmax_normalize(X)
        elif self.method == "robust":
            normalized = self._robust_normalize(X)
        elif self.method == "clip":
            normalized = self._clip_normalize(X)
        elif self.method == "sigmoid":
            normalized = self._sigmoid_normalize(X)
        else:
            normalized = self._minmax_normalize(X)
        
        return normalized

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def _minmax_normalize(self, X: np.ndarray) -> np.ndarray:
        """Min-max normalization."""
        range_span = self.max_vals_ - self.min_vals_
        range_span[range_span == 0] = 1
        
        normalized = (X - self.min_vals_) / range_span
        normalized = self.range_min + normalized * (self.range_max - self.range_min)
        
        return normalized

    def _robust_normalize(self, X: np.ndarray) -> np.ndarray:
        """Robust normalization using IQR."""
        IQR = self.q3_vals_ - self.q1_vals_
        IQR[IQR == 0] = 1
        
        normalized = (X - self.q1_vals_) / IQR
        normalized = self.range_min + normalized * (self.range_max - self.range_min)
        
        return normalized

    def _clip_normalize(self, X: np.ndarray) -> np.ndarray:
        """Clip to range after normalization."""
        range_span = self.max_vals_ - self.min_vals_
        range_span[range_span == 0] = 1
        
        normalized = (X - self.min_vals_) / range_span
        normalized = np.clip(normalized, 0, 1)
        normalized = self.range_min + normalized * (self.range_max - self.range_min)
        
        return normalized

    def _sigmoid_normalize(self, X: np.ndarray) -> np.ndarray:
        """Sigmoid-based soft normalization."""
        centered = X - self.min_vals_
        range_span = self.max_vals_ - self.min_vals_
        range_span[range_span == 0] = 1
        
        # Apply sigmoid to centered data
        sigmoid_val = 1 / (1 + np.exp(-centered / range_span))
        normalized = self.range_min + sigmoid_val * (self.range_max - self.range_min)
        
        return normalized

    def get_range_info(self) -> dict:
        """Get range normalization information."""
        if not self.fitted:
            return {}
        
        return {
            "output_range": (self.range_min, self.range_max),
            "method": self.method,
            "input_min": self.min_vals_.tolist() if self.min_vals_ is not None else None,
            "input_max": self.max_vals_.tolist() if self.max_vals_ is not None else None
        }


class DecimalScaler(BaseReducer):
    """
    Scales features by moving decimal point (power-of-10 scaling).
    
    Useful for handling features with very different scales.
    Computes the decimal scaling factor for each feature.
    
    Best for: Features with extreme ranges, decimal scaling.
    """
    
    def __init__(self):
        """Initialize decimal scaler."""
        self.scale_factors_ = None
        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'DecimalScaler':
        """
        Learn decimal scale factors.
        
        Args:
            X: Input features
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.scale_factors_ = np.zeros(X.shape[1])
        
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            max_abs = np.max(np.abs(col_data))
            
            if max_abs > 0:
                # Compute power of 10 to scale to [0, 1]
                self.scale_factors_[col_idx] = np.ceil(np.log10(max_abs))
            else:
                self.scale_factors_[col_idx] = 0
        
        self.fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Apply decimal scaling.
        
        Args:
            X: Input features
            
        Returns:
            Decimal scaled features
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()
        
        scaled = np.zeros_like(X, dtype=float)
        for col_idx in range(X.shape[1]):
            scaled[:, col_idx] = X[:, col_idx] / (10 ** self.scale_factors_[col_idx])
        
        return scaled

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse decimal scaling."""
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        
        scaled = np.zeros_like(X, dtype=float)
        for col_idx in range(X.shape[1]):
            scaled[:, col_idx] = X[:, col_idx] * (10 ** self.scale_factors_[col_idx])
        
        return scaled

    def get_scale_factors(self) -> np.ndarray:
        """Get scale factors used."""
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        return self.scale_factors_.copy()


class LogTransformer(BaseReducer):
    """
    Applies logarithmic transformation to features.
    
    Useful for right-skewed distributions. Supports:
    - Natural logarithm
    - Base-10 logarithm
    - Box-Cox transformation
    - Yeo-Johnson transformation
    
    Best for: Skewed data, multiplicative relationships.
    """
    
    def __init__(self, method: str = "natural", shift: float = 1e-10):
        """
        Initialize log transformer.
        
        Args:
            method: Transformation method ('natural', 'log10', 'log2')
            shift: Small value to add before log to avoid log(0)
        """
        self.method = method.lower()
        self.shift = shift
        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'LogTransformer':
        """
        Fit log transformer (learns is possible).
        
        Args:
            X: Input features
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Check if all values are positive for log transform
        if np.any(X <= 0):
            if self.shift <= 0:
                self.shift = np.min(np.abs(X[X != 0])) / 100
        
        self.fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Apply log transformation.
        
        Args:
            X: Input features
            
        Returns:
            Log-transformed features
        """
        if not self.fitted:
            raise ValueError("Transformer not fitted")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()
        
        # Add shift to avoid log(0)
        X_shifted = X + self.shift
        
        if self.method == "natural":
            transformed = np.log(X_shifted)
        elif self.method == "log10":
            transformed = np.log10(X_shifted)
        elif self.method == "log2":
            transformed = np.log2(X_shifted)
        else:
            transformed = np.log(X_shifted)
        
        return transformed

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse log transformation."""
        if not self.fitted:
            raise ValueError("Transformer not fitted")
        
        if self.method == "natural":
            original = np.exp(X)
        elif self.method == "log10":
            original = 10 ** X
        elif self.method == "log2":
            original = 2 ** X
        else:
            original = np.exp(X)
        
        return original - self.shift

    def get_transform_info(self) -> dict:
        """Get transformation information."""
        return {
            "method": self.method,
            "shift": self.shift,
            "fitted": self.fitted
        }


class PowerTransformer(BaseReducer):
    """
    Applies power transformations to normalize distributions.
    
    Methods:
    - Box-Cox (for positive data)
    - Yeo-Johnson (for data with negative values)
    - Square root
    - Reciprocal
    
    Best for: Distribution normalization, variance stabilization.
    """
    
    def __init__(self, power: float = 0.5):
        """
        Initialize power transformer.
        
        Args:
            power: Exponent for power transformation (0.5 = sqrt, 2 = square, etc.)
        """
        self.power = power
        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'PowerTransformer':
        """
        Fit power transformer.
        
        Args:
            X: Input features
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Validate power transformation applicability
        if self.power > 0 and np.any(X <= 0):
            raise ValueError("Power transformation requires positive values for power > 0")
        
        self.fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Apply power transformation.
        
        Args:
            X: Input features
            
        Returns:
            Power transformed features
        """
        if not self.fitted:
            raise ValueError("Transformer not fitted")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()
        
        if self.power == 0:
            # Log transform when power = 0
            return np.log(np.abs(X) + 1e-10)
        else:
            # Power transform
            sign = np.sign(X)
            return sign * np.abs(X) ** self.power

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse power transformation."""
        if not self.fitted:
            raise ValueError("Transformer not fitted")
        
        if self.power == 0:
            return np.exp(X)
        else:
            sign = np.sign(X)
            if self.power != 0:
                return sign * np.abs(X) ** (1.0 / self.power)
            else:
                return np.exp(X)

    def get_power_info(self) -> dict:
        """Get power transformation information."""
        return {
            "power": self.power,
            "fitted": self.fitted
        }
