"""Quantum kernel-based anomaly detection.

Uses a precomputed quantum fidelity kernel with a One-Class SVM to identify
outliers in data mapped to a high-dimensional Hilbert space.
"""

import logging
from typing import Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.svm import OneClassSVM

from qudet.core.base import BaseQuantumEstimator
from qudet.core.exceptions import NotFittedError, ValidationError

logger = logging.getLogger(__name__)


class QuantumKernelAnomalyDetector(BaseQuantumEstimator):
    """Anomaly detector using a quantum fidelity kernel with One-Class SVM.

    Data points are encoded into quantum states via angle encoding (RY gates).
    Pairwise similarities are computed as squared state overlaps
    ``|<ψ(x_i)|ψ(x_j)>|²``, producing a kernel matrix that is fed into a
    scikit-learn ``OneClassSVM(kernel='precomputed')``.

    Args:
        n_qubits: Number of qubits used for angle encoding.
        nu: Upper bound on the fraction of training errors and lower bound
            on the fraction of support vectors for ``OneClassSVM``.
        backend_name: Name of the Qiskit backend (unused for statevector
            computation but required by the base class contract).
        shots: Number of measurement shots (unused for statevector
            computation but required by the base class contract).

    Example:
        >>> detector = QuantumKernelAnomalyDetector(n_qubits=4, nu=0.05)
        >>> detector.fit(X_train)
        >>> labels = detector.predict(X_test)  # +1 inlier, -1 outlier
    """

    def __init__(
        self,
        n_qubits: int,
        nu: float = 0.1,
        backend_name: str = "aer_simulator",
        shots: int = 1024,
    ) -> None:
        if n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be >= 1, got {n_qubits}."
            )
        if not 0.0 < nu <= 1.0:
            raise ValidationError(
                f"nu must be in (0, 1], got {nu}."
            )
        super().__init__(backend_name=backend_name, shots=shots)
        self.n_qubits = n_qubits
        self.nu = nu
        self.svm_ = OneClassSVM(kernel="precomputed", nu=nu)
        self.train_data_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_encoding_circuit(self, x_data: np.ndarray) -> QuantumCircuit:
        """Create an angle-encoding circuit using RY rotations.

        Args:
            x_data: 1-D feature vector.

        Returns:
            A ``QuantumCircuit`` with one RY gate per feature (up to
            ``n_qubits``).
        """
        qc = QuantumCircuit(self.n_qubits)
        for i in range(min(len(x_data), self.n_qubits)):
            qc.ry(float(x_data[i]), i)
        return qc

    def _compute_kernel_matrix(
        self, X1: np.ndarray, X2: np.ndarray
    ) -> np.ndarray:
        """Compute the fidelity kernel matrix ``K(x, y) = |<ψ(x)|ψ(y)>|²``.

        Uses exact statevector simulation, which scales to approximately
        20 qubits.

        Args:
            X1: First set of data points, shape ``(n1, n_features)``.
            X2: Second set of data points, shape ``(n2, n_features)``.

        Returns:
            Kernel matrix of shape ``(n1, n2)``.
        """
        n_samples_1 = len(X1)
        n_samples_2 = len(X2)
        kernel_matrix = np.zeros((n_samples_1, n_samples_2))

        states_1 = [
            Statevector.from_instruction(self._get_encoding_circuit(x))
            for x in X1
        ]

        if X2 is X1:
            states_2 = states_1
        else:
            states_2 = [
                Statevector.from_instruction(self._get_encoding_circuit(x))
                for x in X2
            ]

        for i in range(n_samples_1):
            for j in range(n_samples_2):
                fidelity = np.abs(states_1[i].inner(states_2[j])) ** 2
                kernel_matrix[i, j] = fidelity

        return kernel_matrix

    # ------------------------------------------------------------------
    # Public API (BaseQuantumEstimator interface)
    # ------------------------------------------------------------------

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "QuantumKernelAnomalyDetector":
        """Compute the quantum kernel for *X* and fit the One-Class SVM.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            y: Ignored.  Present for API compatibility.

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValidationError(
                f"X must be 2-D, got shape {X.shape}."
            )
        logger.info(
            "Fitting QuantumKernelAnomalyDetector on %d samples.", len(X)
        )
        self.train_data_ = X
        kernel_matrix = self._compute_kernel_matrix(X, X)
        self.svm_.fit(kernel_matrix)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels for *X*.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            Array of ``+1`` (inlier) or ``-1`` (outlier) per sample.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValidationError(
                f"X must be 2-D, got shape {X.shape}."
            )
        kernel_matrix = self._compute_kernel_matrix(X, self.train_data_)
        return self.svm_.predict(kernel_matrix)
