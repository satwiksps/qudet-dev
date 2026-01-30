"""
Quantum Principal Component Analysis (PCA) for dimensionality reduction.

This module implements PCA in a high-dimensional Quantum Hilbert Space.
Unlike classical PCA (linear), this captures non-linear structures
by diagonalizing the Quantum Kernel Matrix.
"""

import numpy as np
from sklearn.decomposition import KernelPCA
from qudet.core.base import BaseReducer
from ..analytics.anomaly import QuantumKernelAnomalyDetector


class QuantumPCA(BaseReducer):
    """
    Performs Principal Component Analysis (PCA) in a high-dimensional 
    Quantum Hilbert Space.
    
    Unlike classical PCA (linear), this captures non-linear structures
    by diagonalizing the Quantum Kernel Matrix.
    
    Attributes:
        n_components (int): Number of principal components to extract.
        n_qubits (int): Number of qubits for quantum kernel computation.
        kernel_computer: Quantum kernel computation engine.
        pca_model: Sklearn's KernelPCA fitted model.
        train_data_ (np.ndarray): Training data for kernel computation.
    """
    
    def __init__(self, n_components: int = 2, n_qubits: int = 4):
        """
        Initialize QuantumPCA.
        
        Args:
            n_components: Number of principal components to extract. Default: 2
            n_qubits: Number of qubits for quantum kernel. Default: 4
        """
        self.n_components = n_components
        self.n_qubits = n_qubits
        self.kernel_computer = QuantumKernelAnomalyDetector(n_qubits=n_qubits)
        self.pca_model = KernelPCA(n_components=n_components, kernel="precomputed")
        self.train_data_ = None

    def fit(self, X, y=None):
        """
        Learn the principal components of the Quantum Kernel.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features)
            y: Ignored. Present for scikit-learn compatibility.
            
        Returns:
            self: Fitted QuantumPCA instance
        """
        self.train_data_ = X
        
        kernel_matrix = self.kernel_computer._compute_kernel_matrix(X, X)
        
        self.pca_model.fit(kernel_matrix)
        return self

    def transform(self, X) -> np.ndarray:
        """
        Project new data onto the Quantum Principal Components.
        
        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Transformed data, shape (n_samples, n_components)
        """
        if self.train_data_ is None:
            raise ValueError("Must fit before transform")
            
        kernel_matrix_new = self.kernel_computer._compute_kernel_matrix(X, self.train_data_)
        return self.pca_model.transform(kernel_matrix_new)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features)
            y: Ignored. Present for scikit-learn compatibility.
            
        Returns:
            np.ndarray: Transformed training data
        """
        return self.fit(X, y).transform(X)
