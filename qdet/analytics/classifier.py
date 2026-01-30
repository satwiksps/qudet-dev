"""
Quantum Support Vector Classifier (QSVC) for binary classification.

Implements a Quantum Kernel Support Vector Classifier that uses high-dimensional
Hilbert space mappings to solve complex classification problems.
"""

import numpy as np
from sklearn.svm import SVC
from qdet.core.base import BaseQuantumEstimator
from ..encoders.rotation import RotationEncoder
from qiskit.quantum_info import Statevector


class QuantumSVC(BaseQuantumEstimator):
    """
    Quantum Support Vector Classifier (QSVC).
    
    Uses a Quantum Kernel to map data into a high-dimensional space where
    complex classes become linearly separable. This is particularly effective
    for non-linear classification problems where classical SVC might struggle.
    
    The quantum kernel exploits the exponentially large Hilbert space to
    achieve better separability of data points.
    
    Attributes:
        n_qubits (int): Number of qubits for quantum encoding.
        C (float): Regularization parameter for SVC.
        encoder (RotationEncoder): Quantum state encoder.
        svc_model (SVC): Underlying sklearn SVC with precomputed kernel.
        train_data_ (np.ndarray): Training data stored for kernel computation.
    """
    
    def __init__(self, n_qubits: int = 4, C: float = 1.0):
        """
        Initialize QuantumSVC.
        
        Args:
            n_qubits (int): Number of qubits for quantum encoding. Default: 4
            C (float): Regularization parameter. Smaller C = more regularization. Default: 1.0
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.C = C
        self.svc_model = SVC(kernel='precomputed', C=C)
        self.encoder = RotationEncoder(n_qubits)
        self.train_data_ = None

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Computes the Quantum Kernel Matrix K_ij = |<phi(x_i)|phi(x_j)>|^2
        
        The kernel represents the fidelity (overlap) between quantum states
        created from different data points.
        
        Args:
            X1 (np.ndarray): First set of data points, shape (n1, n_features)
            X2 (np.ndarray): Second set of data points, shape (n2, n_features)
            
        Returns:
            np.ndarray: Kernel matrix of shape (n1, n2)
        """
        n1, n2 = len(X1), len(X2)
        k_mat = np.zeros((n1, n2))
        
        states1 = []
        for x in X1:
            circuit = self.encoder.encode(x)
            if hasattr(circuit, 'parameters') and len(circuit.parameters) > 0:
                circuit = circuit.assign_parameters({p: v for p, v in zip(circuit.parameters, x[:len(circuit.parameters)])})
            states1.append(Statevector.from_instruction(circuit))
        
        if X1 is X2:
            states2 = states1
        else:
            states2 = []
            for x in X2:
                circuit = self.encoder.encode(x)
                if hasattr(circuit, 'parameters') and len(circuit.parameters) > 0:
                    circuit = circuit.assign_parameters({p: v for p, v in zip(circuit.parameters, x[:len(circuit.parameters)])})
                states2.append(Statevector.from_instruction(circuit))
            
        for i in range(n1):
            for j in range(n2):
                k_mat[i, j] = np.abs(states1[i].inner(states2[j]))**2
                
        return k_mat

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Quantum SVC on training data.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features)
            y (np.ndarray): Binary labels (0/1) of shape (n_samples,)
            
        Returns:
            self: Fitted QuantumSVC instance
        """
        if len(np.unique(y)) != 2:
            raise ValueError("QuantumSVC supports binary classification only")
            
        print(f"--- Training Quantum SVC on {len(X)} samples ---")
        self.train_data_ = X
        kernel_matrix = self._compute_kernel_matrix(X, X)
        self.svc_model.fit(kernel_matrix, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X (np.ndarray): Data to predict, shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted labels (0 or 1)
        """
        if self.train_data_ is None:
            raise ValueError("Must fit before predict")
            
        kernel_matrix = self._compute_kernel_matrix(X, self.train_data_)
        return self.svc_model.predict(kernel_matrix)
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score on test data.
        
        Args:
            X (np.ndarray): Test data, shape (n_samples, n_features)
            y (np.ndarray): True labels, shape (n_samples,)
            
        Returns:
            float: Accuracy score between 0 and 1
        """
        if self.train_data_ is None:
            raise ValueError("Must fit before score")
            
        kernel_matrix = self._compute_kernel_matrix(X, self.train_data_)
        return self.svc_model.score(kernel_matrix, y)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function of X.
        
        Args:
            X (np.ndarray): Data, shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Decision values, shape (n_samples,)
        """
        if self.train_data_ is None:
            raise ValueError("Must fit before decision_function")
            
        kernel_matrix = self._compute_kernel_matrix(X, self.train_data_)
        return self.svc_model.decision_function(kernel_matrix)
