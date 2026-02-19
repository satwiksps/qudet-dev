"""Quantum Kernel Ridge Regression.

Performs regression by computing a quantum fidelity kernel matrix and
delegating to scikit-learn ``KernelRidge(kernel='precomputed')``.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.kernel_ridge import KernelRidge

from qudet.core.base import BaseQuantumEstimator
from qudet.core.exceptions import NotFittedError, ValidationError

logger = logging.getLogger(__name__)


class QuantumKernelRegressor(BaseQuantumEstimator):
    """Regression estimator using a quantum fidelity kernel.

    Classical kernel regression assumes an explicit mapping; this estimator
    instead implicitly maps data into a high-dimensional Hilbert space by
    computing pairwise quantum fidelities ``|<ψ(x_i)|ψ(x_j)>|²``.

    Best for:
        * Non-linear regression on datasets with rich feature interactions.
        * Small-to-medium datasets (kernel computation is O(n²) in space).

    Args:
        n_qubits: Number of qubits for encoding circuits.
        alpha: Ridge regression regularisation strength.
        backend_name: Qiskit backend name (passed to base class).
        shots: Number of measurement shots (passed to base class).

    Attributes:
        model: Underlying ``KernelRidge`` with precomputed kernel.
        train_data_: Training features stored for kernel computation at
            prediction time.

    Example:
        >>> qreg = QuantumKernelRegressor(n_qubits=3, alpha=0.1)
        >>> qreg.fit(X_train, y_train)
        >>> predictions = qreg.predict(X_test)
    """

    def __init__(
        self,
        n_qubits: int,
        alpha: float = 1.0,
        backend_name: str = "aer_simulator",
        shots: int = 1024,
    ) -> None:
        if n_qubits < 1:
            raise ValidationError(f"n_qubits must be >= 1, got {n_qubits}.")
        if alpha <= 0:
            raise ValidationError(f"alpha must be > 0, got {alpha}.")
        super().__init__(backend_name=backend_name, shots=shots)
        self.n_qubits = n_qubits
        self.alpha = alpha
        self.model = KernelRidge(kernel="precomputed", alpha=alpha)
        self.train_data_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_encoding_circuit(self, data: np.ndarray) -> QuantumCircuit:
        """Create an angle-encoding circuit with Hadamard + RY layers.

        Args:
            data: 1-D feature vector.

        Returns:
            A ``QuantumCircuit`` encoding the data.
        """
        qc = QuantumCircuit(self.n_qubits)
        n_features = min(len(data), self.n_qubits)
        normalized_data = 2 * np.pi * data[:n_features]

        qc.h(range(self.n_qubits))
        for i in range(n_features):
            qc.ry(float(normalized_data[i]), i)

        return qc

    def _compute_kernel_matrix(
        self, X1: np.ndarray, X2: np.ndarray
    ) -> np.ndarray:
        """Compute quantum kernel via fidelity between encoded states.

        ``K(x_i, x_j) = |<ψ(x_i)|ψ(x_j)>|²``

        Args:
            X1: First data set, shape ``(n1, n_features)``.
            X2: Second data set, shape ``(n2, n_features)``.

        Returns:
            Kernel matrix of shape ``(n1, n2)``.
        """
        n1, n2 = len(X1), len(X2)
        k_mat = np.zeros((n1, n2))

        states1 = [
            Statevector(self._create_encoding_circuit(x)) for x in X1
        ]

        if X1 is X2 or np.array_equal(X1, X2):
            states2 = states1
        else:
            states2 = [
                Statevector(self._create_encoding_circuit(x)) for x in X2
            ]

        for i in range(n1):
            for j in range(n2):
                overlap = states1[i].inner(states2[j])
                k_mat[i, j] = np.abs(overlap) ** 2

        return k_mat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, None] = None,
    ) -> "QuantumKernelRegressor":
        """Train the quantum kernel regressor.

        Args:
            X: Training features of shape ``(n_samples, n_features)``.
                Accepts both numpy arrays and pandas DataFrames.
            y: Target continuous values of shape ``(n_samples,)``.
                Accepts both numpy arrays and pandas Series.

        Returns:
            self

        Raises:
            ValidationError: If *X* is not 2-D or *y* is ``None``.
        """
        if y is None:
            raise ValidationError("y must be provided for regression.")

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if X_arr.ndim != 2:
            raise ValidationError(f"X must be 2-D, got shape {X_arr.shape}.")

        logger.info(
            "Training QuantumKernelRegressor on %d samples.", len(X_arr)
        )

        self.train_data_ = X_arr
        kernel_matrix = self._compute_kernel_matrix(X_arr, X_arr)
        self.model.fit(kernel_matrix, y_arr)
        self._is_fitted = True

        logger.info("Training complete. Ready for predictions.")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict continuous values for new data.

        Args:
            X: Test features of shape ``(n_samples, n_features)``.
                Accepts both numpy arrays and pandas DataFrames.

        Returns:
            Predicted continuous values of shape ``(n_samples,)``.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValidationError(f"X must be 2-D, got shape {X_arr.shape}.")

        kernel_matrix = self._compute_kernel_matrix(X_arr, self.train_data_)
        return self.model.predict(kernel_matrix)
