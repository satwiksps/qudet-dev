"""
Categorical feature encoding for quantum algorithms.

Converts categorical and discrete features into numerical representations
suitable for quantum machine learning.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, List
from qudet.core.base import BaseReducer


class CategoricalEncoder(BaseReducer):
    """
    Encodes categorical features to numerical values.
    
    Methods:
    - Label encoding (unique integers)
    - One-hot encoding (binary columns)
    - Ordinal encoding (custom order)
    - Binary encoding (minimal bits)
    
    Best for: Converting categorical features, preprocessing.
    """
    
    def __init__(self, method: str = "label", handle_unknown: str = "error"):
        """
        Initialize categorical encoder.
        
        Args:
            method: Encoding method ('label', 'onehot', 'ordinal', 'binary')
            handle_unknown: How to handle unknown categories ('error', 'ignore', 'use_encoded_value')
        """
        self.method = method.lower()
        self.handle_unknown = handle_unknown
        self.categories_ = {}
        self.encoding_map_ = {}
        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'CategoricalEncoder':
        """
        Learn categories from data.
        
        Args:
            X: Input features
            y: Optional target (unused)
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            features = X.columns
            X = X.values
        else:
            features = range(X.shape[1])
        
        for col_idx in range(X.shape[1]):
            unique_vals = np.unique(X[:, col_idx])
            self.categories_[col_idx] = unique_vals
            
            if self.method == "label":
                self.encoding_map_[col_idx] = {
                    val: idx for idx, val in enumerate(unique_vals)
                }
            elif self.method == "onehot":
                self.encoding_map_[col_idx] = {
                    val: idx for idx, val in enumerate(unique_vals)
                }
            elif self.method == "ordinal":
                self.encoding_map_[col_idx] = {
                    val: idx for idx, val in enumerate(unique_vals)
                }
            elif self.method == "binary":
                # Binary encoding using minimal bits
                n_categories = len(unique_vals)
                n_bits = int(np.ceil(np.log2(n_categories)))
                self.encoding_map_[col_idx] = {
                    val: idx for idx, val in enumerate(unique_vals)
                }
        
        self.fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Encode categorical features.
        
        Args:
            X: Input features
            
        Returns:
            Encoded features
        """
        if not self.fitted:
            raise ValueError("Encoder not fitted")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()
        
        if self.method == "label":
            return self._encode_label(X)
        elif self.method == "onehot":
            return self._encode_onehot(X)
        elif self.method == "binary":
            return self._encode_binary(X)
        else:
            return self._encode_label(X)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame],
                     y: Optional[Union[np.ndarray, pd.Series]] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _encode_label(self, X: np.ndarray) -> np.ndarray:
        """Label encode categorical features."""
        X_encoded = np.zeros_like(X, dtype=float)
        
        for col_idx in range(X.shape[1]):
            if col_idx in self.encoding_map_:
                for row_idx, val in enumerate(X[:, col_idx]):
                    if val in self.encoding_map_[col_idx]:
                        X_encoded[row_idx, col_idx] = self.encoding_map_[col_idx][val]
                    elif self.handle_unknown == "error":
                        raise ValueError(f"Unknown category: {val}")
        
        return X_encoded

    def _encode_onehot(self, X: np.ndarray) -> np.ndarray:
        """One-hot encode categorical features."""
        encoded_parts = []
        
        for col_idx in range(X.shape[1]):
            if col_idx in self.encoding_map_:
                n_categories = len(self.encoding_map_[col_idx])
                onehot = np.zeros((X.shape[0], n_categories))
                
                for row_idx, val in enumerate(X[:, col_idx]):
                    if val in self.encoding_map_[col_idx]:
                        cat_idx = self.encoding_map_[col_idx][val]
                        onehot[row_idx, cat_idx] = 1
                
                encoded_parts.append(onehot)
        
        return np.hstack(encoded_parts) if encoded_parts else X

    def _encode_binary(self, X: np.ndarray) -> np.ndarray:
        """Binary encode categorical features."""
        encoded_parts = []
        
        for col_idx in range(X.shape[1]):
            if col_idx in self.encoding_map_:
                mapping = self.encoding_map_[col_idx]
                n_categories = len(mapping)
                n_bits = int(np.ceil(np.log2(n_categories)))
                
                binary = np.zeros((X.shape[0], n_bits))
                
                for row_idx, val in enumerate(X[:, col_idx]):
                    if val in mapping:
                        cat_idx = mapping[val]
                        # Convert to binary
                        bits = [(cat_idx >> i) & 1 for i in range(n_bits)]
                        binary[row_idx, :] = bits
                
                encoded_parts.append(binary)
        
        return np.hstack(encoded_parts) if encoded_parts else X

    def get_n_features_out(self) -> int:
        """Get number of output features after encoding."""
        if not self.fitted:
            return 0
        
        if self.method == "label":
            return len(self.categories_)
        elif self.method == "onehot":
            return sum(len(cats) for cats in self.categories_.values())
        elif self.method == "binary":
            total = 0
            for cats in self.categories_.values():
                n_bits = int(np.ceil(np.log2(len(cats))))
                total += n_bits
            return total
        return len(self.categories_)


class TargetEncoder(BaseReducer):
    """
    Encodes features based on target variable statistics.
    
    Maps categories to mean target values, useful for regression.
    Handles:
    - Mean target encoding
    - Category frequency encoding
    - Likelihood encoding
    
    Best for: Target encoding, classification preprocessing.
    """
    
    def __init__(self, smoothing: float = 1.0, min_samples_leaf: int = 1):
        """
        Initialize target encoder.
        
        Args:
            smoothing: Regularization strength for target encoding
            min_samples_leaf: Minimum samples per category
        """
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encoding_map_ = {}
        self.global_mean_ = None
        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series]) -> 'TargetEncoder':
        """
        Learn target encoding from data.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.global_mean_ = np.mean(y)
        
        for col_idx in range(X.shape[1]):
            unique_vals = np.unique(X[:, col_idx])
            encoding = {}
            
            for val in unique_vals:
                mask = X[:, col_idx] == val
                target_vals = y[mask]
                n_samples = len(target_vals)
                
                if n_samples >= self.min_samples_leaf:
                    # Apply smoothing (Bayesian approach)
                    category_mean = np.mean(target_vals)
                    smoothed_mean = (n_samples * category_mean + 
                                   self.smoothing * self.global_mean_) / (n_samples + self.smoothing)
                    encoding[val] = smoothed_mean
                else:
                    encoding[val] = self.global_mean_
            
            self.encoding_map_[col_idx] = encoding
        
        self.fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Apply target encoding.
        
        Args:
            X: Input features
            
        Returns:
            Target encoded features
        """
        if not self.fitted:
            raise ValueError("Encoder not fitted")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()
        
        X_encoded = np.zeros_like(X, dtype=float)
        
        for col_idx in range(X.shape[1]):
            if col_idx in self.encoding_map_:
                for row_idx, val in enumerate(X[:, col_idx]):
                    if val in self.encoding_map_[col_idx]:
                        X_encoded[row_idx, col_idx] = self.encoding_map_[col_idx][val]
                    else:
                        X_encoded[row_idx, col_idx] = self.global_mean_
        
        return X_encoded

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame],
                     y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_encoding_info(self) -> dict:
        """Get encoding statistics."""
        if not self.fitted:
            return {}
        
        return {
            "global_mean": self.global_mean_,
            "smoothing": self.smoothing,
            "n_features": len(self.encoding_map_)
        }


class FrequencyEncoder(BaseReducer):
    """
    Encodes features based on category frequency.
    
    Maps categories to their relative frequencies in the training set.
    Useful for ordinal representation of categorical frequency.
    
    Best for: Frequency-based encoding, data exploration.
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize frequency encoder.
        
        Args:
            normalize: Whether to normalize frequencies
        """
        self.normalize = normalize
        self.frequency_map_ = {}
        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'FrequencyEncoder':
        """
        Learn category frequencies.
        
        Args:
            X: Input features
            y: Optional (unused)
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        for col_idx in range(X.shape[1]):
            unique_vals, counts = np.unique(X[:, col_idx], return_counts=True)
            n_total = len(X[:, col_idx])
            
            freq_map = {}
            for val, count in zip(unique_vals, counts):
                if self.normalize:
                    freq_map[val] = count / n_total
                else:
                    freq_map[val] = count
            
            self.frequency_map_[col_idx] = freq_map
        
        self.fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Apply frequency encoding.
        
        Args:
            X: Input features
            
        Returns:
            Frequency encoded features
        """
        if not self.fitted:
            raise ValueError("Encoder not fitted")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()
        
        X_encoded = np.zeros_like(X, dtype=float)
        
        for col_idx in range(X.shape[1]):
            if col_idx in self.frequency_map_:
                for row_idx, val in enumerate(X[:, col_idx]):
                    if val in self.frequency_map_[col_idx]:
                        X_encoded[row_idx, col_idx] = self.frequency_map_[col_idx][val]
        
        return X_encoded

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame],
                     y: Optional[Union[np.ndarray, pd.Series]] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_frequency_info(self) -> dict:
        """Get frequency statistics."""
        if not self.fitted:
            return {}
        
        info = {}
        for col_idx, freq_map in self.frequency_map_.items():
            info[f"feature_{col_idx}"] = {
                "n_categories": len(freq_map),
                "frequencies": freq_map
            }
        
        return info


class BinningEncoder(BaseReducer):
    """
    Bins continuous features into discrete categories.
    
    Methods:
    - Equal width binning
    - Equal frequency (quantile) binning
    - Custom boundaries
    - K-means clustering based
    
    Best for: Discretization, binning, categorical conversion.
    """
    
    def __init__(self, n_bins: int = 5, method: str = "quantile"):
        """
        Initialize binning encoder.
        
        Args:
            n_bins: Number of bins
            method: Binning method ('quantile', 'uniform', 'kmeans')
        """
        self.n_bins = n_bins
        self.method = method.lower()
        self.bin_edges_ = {}
        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'BinningEncoder':
        """
        Learn bin boundaries.
        
        Args:
            X: Input features
            y: Optional (unused)
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            
            if self.method == "quantile":
                # Equal frequency binning
                edges = np.percentile(col_data, np.linspace(0, 100, self.n_bins + 1))
            elif self.method == "uniform":
                # Equal width binning
                edges = np.linspace(col_data.min(), col_data.max(), self.n_bins + 1)
            elif self.method == "kmeans":
                # K-means clustering
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.n_bins, random_state=42)
                labels = kmeans.fit_predict(col_data.reshape(-1, 1))
                # Create edges from cluster centers
                centers = np.sort(kmeans.cluster_centers_.ravel())
                edges = np.concatenate([[col_data.min()], centers, [col_data.max()]])
            else:
                edges = np.percentile(col_data, np.linspace(0, 100, self.n_bins + 1))
            
            # Remove duplicates
            self.bin_edges_[col_idx] = np.unique(edges)
        
        self.fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Apply binning to features.
        
        Args:
            X: Input features
            
        Returns:
            Binned features (bin indices)
        """
        if not self.fitted:
            raise ValueError("Encoder not fitted")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()
        
        X_binned = np.zeros_like(X, dtype=int)
        
        for col_idx in range(X.shape[1]):
            if col_idx in self.bin_edges_:
                edges = self.bin_edges_[col_idx]
                # Use digitize to assign bin indices
                X_binned[:, col_idx] = np.digitize(X[:, col_idx], edges, right=False) - 1
        
        return X_binned

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame],
                     y: Optional[Union[np.ndarray, pd.Series]] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_bin_info(self) -> dict:
        """Get binning information."""
        if not self.fitted:
            return {}
        
        return {
            "n_bins": self.n_bins,
            "method": self.method,
            "bin_edges": {
                f"feature_{col_idx}": edges.tolist()
                for col_idx, edges in self.bin_edges_.items()
            }
        }
