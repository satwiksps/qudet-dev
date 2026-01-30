
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from qudet.core.base import BaseQuantumEstimator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QuantumKernelRegressor(BaseQuantumEstimator):
    """
    Performs Regression (predicting continuous values) using a Quantum Kernel.
    
    Traditional ML: Linear models assume straight-line relationships.
    Quantum Kernel: Implicitly maps to high-dimensional Hilbert space,
                    capturing non-linear patterns (e.g., time series).
    
    Kernel Method: Instead of explicitly computing the mapping, we compute
    pairwise "quantum similarities" (fidelity) between data points.
    
    Best for:
    - Non-linear regression on quantum circuits
    - Time-series prediction with feature interactions
    - Small-medium datasets (kernel computation is O(n^2) space)
    
    Example:
        >>> qreg = QuantumKernelRegressor(n_qubits=3, alpha=0.1)
        >>> qreg.fit(X_train, y_train)
        >>> predictions = qreg.predict(X_test)
    """
    
    def __init__(self, n_qubits: int, alpha: float = 1.0):
        """
        Initialize Quantum Kernel Regressor.
        
        Parameters
        ----------
        n_qubits : int
            Number of qubits for encoding circuits
        alpha : float
            Ridge regression regularization strength.
            Higher values → simpler model, lower training error.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.alpha = alpha
        self.model = KernelRidge(kernel='precomputed', alpha=alpha)
        self.train_data_ = None
        self.train_circuits_ = []
        self.is_fitted = False

    def _create_encoding_circuit(self, data: np.ndarray) -> QuantumCircuit:
        """
        Create a non-parameterized quantum encoding circuit.
        
        Parameters
        ----------
        data : np.ndarray
            Input features (1D array)
            
        Returns
        -------
        QuantumCircuit
            Quantum circuit encoding the data
        """
        qc = QuantumCircuit(self.n_qubits)
        
        normalized_data = 2 * np.pi * data[:self.n_qubits]
        
        qc.h(range(self.n_qubits))
        
        for i in range(min(len(data), self.n_qubits)):
            qc.ry(normalized_data[i], i)
            
        return qc

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute quantum kernel matrix via fidelity between encoded states.
        
        Kernel value K(x_i, x_j) = |<ψ(x_i)|ψ(x_j)>|²
        This is the "squared overlap" between quantum state representations.
        
        Parameters
        ----------
        X1 : np.ndarray
            First set of data points (n1 x n_features)
        X2 : np.ndarray
            Second set of data points (n2 x n_features)
            
        Returns
        -------
        np.ndarray
            Kernel matrix of shape (n1, n2)
        """
        n1 = len(X1)
        n2 = len(X2)
        k_mat = np.zeros((n1, n2))
        
        states1 = []
        for x in X1:
            circuit = self._create_encoding_circuit(x)
            state = Statevector(circuit)
            states1.append(state)
        
        if X1 is X2 or np.array_equal(X1, X2):
            states2 = states1
        else:
            states2 = []
            for x in X2:
                circuit = self._create_encoding_circuit(x)
                state = Statevector(circuit)
                states2.append(state)
        
        for i in range(n1):
            for j in range(n2):
                overlap = states1[i].inner(states2[j])
                k_mat[i, j] = np.abs(overlap) ** 2
                
        return k_mat

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantumKernelRegressor":
        """
        Train the quantum kernel regressor.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features (n_samples x n_features)
        y : pd.Series or np.ndarray
            Target continuous values (n_samples,)
            
        Returns
        -------
        self
            Fitted regressor instance
        """
        print(f"--- Training Quantum Regressor on {len(X)} samples ---")
        
        self.train_data_ = X.values if isinstance(X, pd.DataFrame) else X
        
        kernel_matrix = self._compute_kernel_matrix(self.train_data_, self.train_data_)
        
        y_values = y.values if isinstance(y, pd.Series) else y
        self.model.fit(kernel_matrix, y_values)
        
        self.is_fitted = True
        print(f"   Training complete. Ready for predictions.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict continuous values for new data using quantum kernel.
        
        Parameters
        ----------
        X : pd.DataFrame
            Test features (n_test x n_features)
            
        Returns
        -------
        np.ndarray
            Predicted continuous values (n_test,)
            
        Raises
        ------
        RuntimeError
            If called before fit()
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        
        kernel_matrix = self._compute_kernel_matrix(X_values, self.train_data_)
        
        return self.model.predict(kernel_matrix)

