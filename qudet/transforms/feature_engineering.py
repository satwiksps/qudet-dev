"""
Feature engineering and selection methods for quantum data preprocessing.

Provides feature scaling, normalization, and selection techniques
for quantum machine learning pipelines.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from qudet.core.base import BaseReducer


class FeatureScaler(BaseReducer):
    """
    Scales and normalizes features for quantum algorithms.
    
    Provides multiple scaling strategies:
    - Standard scaling (z-score normalization)
    - Min-Max scaling (0-1 range)
    - Robust scaling (resistant to outliers)
    - Quantum-aware scaling (preserves quantum structure)
    
    Best for: Feature normalization, preprocessing, standardization.
    """
    
    def __init__(self, method: str = "standard", feature_range: Tuple = (0, 1)):
        """
        Initialize feature scaler.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust', 'quantum')
            feature_range: Range for min-max scaling
        """
        self.method = method.lower()
        self.feature_range = feature_range
        self.scaler = self._create_scaler()
        self.fitted = False

    def _create_scaler(self):
        """Create appropriate scaler based on method."""
        if self.method == "standard":
            return StandardScaler()
        elif self.method == "minmax":
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.method == "robust":
            return RobustScaler()
        else:
            return StandardScaler()

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'FeatureScaler':
        """
        Fit scaler to data.
        
        Args:
            X: Input features
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Input features
            
        Returns:
            Scaled features
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        scaled = self.scaler.transform(X)
        
        # Apply quantum-aware scaling if requested
        if self.method == "quantum":
            scaled = self._apply_quantum_scaling(scaled)
        
        return scaled

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def _apply_quantum_scaling(self, X: np.ndarray) -> np.ndarray:
        """Apply quantum-aware scaling preserving structure."""
        # Normalize to unit norm per sample
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return X / norms

    def get_scaling_params(self) -> dict:
        """Get scaling parameters."""
        if not self.fitted:
            return {}
        
        params = {
            "method": self.method,
            "fitted": self.fitted
        }
        
        if hasattr(self.scaler, 'mean_'):
            params["mean"] = self.scaler.mean_
        if hasattr(self.scaler, 'scale_'):
            params["scale"] = self.scaler.scale_
        
        return params


class FeatureSelector(BaseReducer):
    """
    Selects most important features for quantum algorithms.
    
    Reduces dimensionality by selecting features with highest scores
    using statistical tests or mutual information.
    
    Best for: Feature selection, dimensionality reduction, feature importance.
    """
    
    def __init__(self, n_features: int = 10, method: str = "f_classif"):
        """
        Initialize feature selector.
        
        Args:
            n_features: Number of features to select
            method: Selection method ('f_classif', 'mutual_info')
        """
        self.n_features = n_features
        self.method = method
        self.selector = None
        self.feature_indices_ = None
        self.feature_scores_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'FeatureSelector':
        """
        Fit feature selector.
        
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
        
        if self.method == "f_classif":
            score_func = f_classif
        elif self.method == "mutual_info":
            score_func = mutual_info_classif
        else:
            score_func = f_classif
        
        self.selector = SelectKBest(score_func, k=self.n_features)
        self.selector.fit(X, y)
        
        self.feature_indices_ = self.selector.get_support(indices=True)
        self.feature_scores_ = self.selector.scores_
        
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform features to selected subset.
        
        Args:
            X: Input features
            
        Returns:
            Selected features
        """
        if self.selector is None:
            raise ValueError("Selector must be fitted before transform")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.selector.transform(X)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame],
                     y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        if self.feature_indices_ is None:
            raise ValueError("Selector not fitted")
        return self.feature_indices_

    def get_feature_scores(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.feature_scores_ is None:
            raise ValueError("Selector not fitted")
        return self.feature_scores_


class OutlierRemover(BaseReducer):
    """
    Removes outliers from datasets using multiple strategies.
    
    Methods:
    - IQR (Interquartile Range)
    - Z-score
    - Isolation Forest
    - Local Outlier Factor
    
    Best for: Data cleaning, outlier removal, robust preprocessing.
    """
    
    def __init__(self, method: str = "iqr", threshold: float = 3.0):
        """
        Initialize outlier remover.
        
        Args:
            method: Outlier detection method ('iqr', 'zscore', 'isolation', 'lof')
            threshold: Detection threshold
        """
        self.method = method.lower()
        self.threshold = threshold
        self.outlier_mask_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'OutlierRemover':
        """
        Detect outliers in data.
        
        Args:
            X: Input data
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.method == "iqr":
            self.outlier_mask_ = self._detect_iqr(X)
        elif self.method == "zscore":
            self.outlier_mask_ = self._detect_zscore(X)
        elif self.method == "isolation":
            self.outlier_mask_ = self._detect_isolation(X)
        else:
            self.outlier_mask_ = self._detect_iqr(X)
        
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Remove outliers from data.
        
        Args:
            X: Input data
            
        Returns:
            Data without outliers
        """
        if self.outlier_mask_ is None:
            raise ValueError("Remover not fitted")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return X[self.outlier_mask_]

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def _detect_iqr(self, X: np.ndarray) -> np.ndarray:
        """Detect outliers using IQR method."""
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
        return mask

    def _detect_zscore(self, X: np.ndarray) -> np.ndarray:
        """Detect outliers using z-score method."""
        z_scores = np.abs((X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10))
        mask = np.all(z_scores < self.threshold, axis=1)
        return mask

    def _detect_isolation(self, X: np.ndarray) -> np.ndarray:
        """Detect outliers using isolation approach."""
        # Simplified isolation: samples far from mean
        distances = np.linalg.norm(X - np.mean(X, axis=0), axis=1)
        threshold = np.mean(distances) + self.threshold * np.std(distances)
        mask = distances <= threshold
        return mask

    def get_outlier_ratio(self) -> float:
        """Get ratio of outliers removed."""
        if self.outlier_mask_ is None:
            return 0.0
        return 1.0 - np.mean(self.outlier_mask_)


class DataBalancer(BaseReducer):
    """
    Balances imbalanced datasets for classification.
    
    Methods:
    - Oversampling (duplicate minority class)
    - Undersampling (reduce majority class)
    - SMOTE-like (synthetic examples)
    - Stratified sampling
    
    Best for: Handling class imbalance, data balancing.
    """
    
    def __init__(self, method: str = "oversample", ratio: float = 1.0):
        """
        Initialize data balancer.
        
        Args:
            method: Balancing method ('oversample', 'undersample', 'smote', 'stratified')
            ratio: Target ratio of minority to majority class
        """
        self.method = method.lower()
        self.ratio = ratio
        self.class_counts_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series]) -> 'DataBalancer':
        """
        Analyze class distribution.
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            Self
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        unique, counts = np.unique(y, return_counts=True)
        self.class_counts_ = dict(zip(unique, counts))
        
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame],
                 y: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance dataset.
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            Tuple of (balanced_X, balanced_y)
        """
        if self.class_counts_ is None:
            raise ValueError("Balancer not fitted")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        if self.method == "oversample":
            return self._oversample(X, y)
        elif self.method == "undersample":
            return self._undersample(X, y)
        elif self.method == "smote":
            return self._smote_balance(X, y)
        else:
            return self._oversample(X, y)

    def _oversample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Oversample minority class."""
        unique_classes = np.unique(y)
        y_int = y.astype(int)
        counts = np.bincount(y_int)
        max_count = np.max(counts)
        
        X_balanced, y_balanced = [], []
        
        for cls in unique_classes:
            mask = y == cls
            X_cls = X[mask]
            y_cls = y[mask]
            
            if len(X_cls) < max_count:
                # Oversample
                indices = np.random.choice(len(X_cls), max_count, replace=True)
                X_cls = X_cls[indices]
                y_cls = y_cls[indices]
            
            X_balanced.append(X_cls)
            y_balanced.append(y_cls)
        
        return np.vstack(X_balanced), np.hstack(y_balanced)

    def _undersample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Undersample majority class."""
        unique_classes = np.unique(y)
        y_int = y.astype(int)
        counts = np.bincount(y_int)
        min_count = np.min(counts)
        
        X_balanced, y_balanced = [], []
        
        for cls in unique_classes:
            mask = y == cls
            X_cls = X[mask]
            y_cls = y[mask]
            
            if len(X_cls) > min_count:
                # Undersample
                indices = np.random.choice(len(X_cls), min_count, replace=False)
                X_cls = X_cls[indices]
                y_cls = y_cls[indices]
            
            X_balanced.append(X_cls)
            y_balanced.append(y_cls)
        
        return np.vstack(X_balanced), np.hstack(y_balanced)

    def _smote_balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE-like synthetic example generation."""
        # Simplified SMOTE: generate examples between nearest neighbors
        unique_classes = np.unique(y)
        X_balanced, y_balanced = [], []
        y_int = y.astype(int)
        counts = np.bincount(y_int)
        max_count = np.max(counts)
        
        for cls in unique_classes:
            mask = y == cls
            X_cls = X[mask]
            y_cls = y[mask]
            
            X_balanced.append(X_cls)
            y_balanced.append(y_cls)
            
            # Generate synthetic examples
            if len(X_cls) < max_count:
                n_synthetic = max_count - len(X_cls)
                for _ in range(n_synthetic):
                    # Pick random sample and random neighbor
                    idx1 = np.random.randint(len(X_cls))
                    idx2 = np.random.randint(len(X_cls))
                    # Interpolate between them
                    alpha = np.random.random()
                    synthetic = alpha * X_cls[idx1] + (1 - alpha) * X_cls[idx2]
                    X_balanced.append(synthetic.reshape(1, -1))
                    y_balanced.append(cls)
        
        return np.vstack(X_balanced), np.hstack(y_balanced)

    def get_balance_info(self) -> dict:
        """Get class balance information."""
        if self.class_counts_ is None:
            return {}
        return {
            "class_counts": self.class_counts_,
            "imbalance_ratio": max(self.class_counts_.values()) / min(self.class_counts_.values())
        }
