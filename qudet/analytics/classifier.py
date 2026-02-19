"""Quantum Support Vector Classifier (QSVC) for binary classification.

Uses a quantum fidelity kernel to map data into a high-dimensional Hilbert
space where complex classes become linearly separable, then delegates to a
scikit-learn ``SVC(kernel='precomputed')``.
"""

import logging
from typing import Optional

import numpy as np
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC

from qudet.core.base import BaseQuantumEstimator
from qudet.core.exceptions import NotFittedError, ValidationError
from ..encoders.rotation import RotationEncoder

logger = logging.getLogger(__name__)


class QuantumSVC(BaseQuantumEstimator):
    """Quantum Support Vector Classifier.

    Exploits the exponentially large Hilbert space to achieve better
    separability of data points compared to classical SVC on the same
    feature set.

    Args:
        n_qubits: Number of qubits for quantum encoding.
        C: Regularization parameter.  Smaller values yield more
            regularization.
        backend_name: Qiskit backend name (passed to base class).
        shots: Number of measurement shots (passed to base class).

    Attributes:
        encoder: :class:`RotationEncoder` used to build parameterised
            circuits.
        svc_model: Underlying ``sklearn.svm.SVC`` with precomputed kernel.
        train_data_: Training data stored for kernel computation during
            prediction.

    Example:
        >>> qsvc = QuantumSVC(n_qubits=4, C=1.0)
        >>> qsvc.fit(X_train, y_train)
        >>> predictions = qsvc.predict(X_test)
    """

    def __init__(
        self,
        n_qubits: int = 4,
        C: float = 1.0,
        backend_name: str = "aer_simulator",
        shots: int = 1024,
    ) -> None:
        if n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be >= 1, got {n_qubits}."
            )
        if C <= 0:
            raise ValidationError(
                f"C must be > 0, got {C}."
            )
        super().__init__(backend_name=backend_name, shots=shots)
        self.n_qubits = n_qubits
        self.C = C
        self.svc_model = SVC(kernel="precomputed", C=C)
        self.encoder = RotationEncoder(n_qubits)
        self.train_data_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_kernel_matrix(
        self, X1: np.ndarray, X2: np.ndarray
    ) -> np.ndarray:
        """Compute quantum kernel ``K_ij = |<φ(x_i)|φ(x_j)>|²``.

        The kernel represents the fidelity (overlap) between quantum states
        created from different data points.

        Args:
            X1: First set of data points, shape ``(n1, n_features)``.
            X2: Second set of data points, shape ``(n2, n_features)``.

        Returns:
            Kernel matrix of shape ``(n1, n2)``.
        """
        n1, n2 = len(X1), len(X2)
        k_mat = np.zeros((n1, n2))

        def _to_statevector(x: np.ndarray) -> Statevector:
            circuit = self.encoder.encode(x)
            if hasattr(circuit, "parameters") and len(circuit.parameters) > 0:
                binding = {
                    p: v
                    for p, v in zip(
                        circuit.parameters, x[: len(circuit.parameters)]
                    )
                }
                circuit = circuit.assign_parameters(binding)
            return Statevector.from_instruction(circuit)

        states1 = [_to_statevector(x) for x in X1]

        if X1 is X2:
            states2 = states1
        else:
            states2 = [_to_statevector(x) for x in X2]

        for i in range(n1):
            for j in range(n2):
                k_mat[i, j] = np.abs(states1[i].inner(states2[j])) ** 2

        return k_mat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "QuantumSVC":
        """Fit the Quantum SVC on training data.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            y: Binary labels of shape ``(n_samples,)``.

        Returns:
            self

        Raises:
            ValidationError: If *y* does not contain exactly two classes.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValidationError(f"X must be 2-D, got shape {X.shape}.")
        if len(np.unique(y)) != 2:
            raise ValidationError(
                "QuantumSVC supports binary classification only. "
                f"Found {len(np.unique(y))} unique classes."
            )

        logger.info("Training QuantumSVC on %d samples.", len(X))
        self.train_data_ = X
        kernel_matrix = self._compute_kernel_matrix(X, X)
        self.svc_model.fit(kernel_matrix, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in *X*.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            Predicted labels array.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        kernel_matrix = self._compute_kernel_matrix(X, self.train_data_)
        return self.svc_model.predict(kernel_matrix)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy on test data.

        Args:
            X: Test data of shape ``(n_samples, n_features)``.
            y: True labels of shape ``(n_samples,)``.

        Returns:
            Accuracy score between 0.0 and 1.0.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        kernel_matrix = self._compute_kernel_matrix(X, self.train_data_)
        return float(self.svc_model.score(kernel_matrix, y))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the SVM decision function for *X*.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            Decision values of shape ``(n_samples,)``.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        kernel_matrix = self._compute_kernel_matrix(X, self.train_data_)
        return self.svc_model.decision_function(kernel_matrix)
