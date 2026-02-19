"""Quantum K-Means clustering.

Implements K-Means where distances between data points are computed as
quantum-state distances in Hilbert space rather than Euclidean distances.
"""

import logging
from typing import Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.base import ClusterMixin

from qudet.core.base import BaseQuantumEstimator
from qudet.core.exceptions import NotFittedError, ValidationError

logger = logging.getLogger(__name__)


class QuantumKMeans(BaseQuantumEstimator, ClusterMixin):
    """K-Means clustering using quantum-state Hilbert-space distances.

    Each data point is encoded into a quantum state via RY angle-encoding,
    and inter-point distances are defined as
    ``d(u, v) = √(2 − 2|<ψ(u)|ψ(v)>|²)``.  The rest of the algorithm
    follows standard K-Means (assign, recompute centroids, repeat).

    Args:
        n_clusters: Number of clusters.
        n_qubits: Number of qubits for angle encoding.
        max_iter: Maximum number of K-Means iterations.
        backend_name: Qiskit backend name (passed to base class).
        shots: Number of measurement shots (passed to base class).

    Attributes:
        centroids_: Cluster centroid vectors after fitting.
        labels_: Cluster assignment for each training sample after fitting.

    Example:
        >>> qkm = QuantumKMeans(n_clusters=3, n_qubits=4)
        >>> qkm.fit(X_train)
        >>> labels = qkm.predict(X_test)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_qubits: int = 4,
        max_iter: int = 10,
        backend_name: str = "aer_simulator",
        shots: int = 1024,
    ) -> None:
        if n_clusters < 1:
            raise ValidationError(
                f"n_clusters must be >= 1, got {n_clusters}."
            )
        if n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be >= 1, got {n_qubits}."
            )
        if max_iter < 1:
            raise ValidationError(
                f"max_iter must be >= 1, got {max_iter}."
            )
        super().__init__(backend_name=backend_name, shots=shots)
        self.n_clusters = n_clusters
        self.n_qubits = n_qubits
        self.max_iter = max_iter
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _quantum_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute distance in Hilbert space.

        ``d(u, v) = √(2 − 2|<u|v>|²)``

        Args:
            x1: First feature vector.
            x2: Second feature vector.

        Returns:
            Non-negative distance scalar.
        """
        qc1 = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(x1):
            if i < self.n_qubits:
                qc1.ry(float(val), i)

        qc2 = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(x2):
            if i < self.n_qubits:
                qc2.ry(float(val), i)

        sv1 = Statevector.from_instruction(qc1)
        sv2 = Statevector.from_instruction(qc2)

        overlap = np.abs(sv1.inner(sv2)) ** 2
        return float(np.sqrt(np.maximum(0.0, 2.0 - 2.0 * overlap)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "QuantumKMeans":
        """Run Quantum K-Means clustering on *X*.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
                Accepts both numpy arrays and pandas DataFrames.
            y: Ignored.  Present for API compatibility.

        Returns:
            self

        Raises:
            ValidationError: If *X* has fewer samples than ``n_clusters``.
        """
        # Normalise input to numpy early to avoid fragile DataFrame branching.
        X_arr: np.ndarray = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValidationError(f"X must be 2-D, got shape {X_arr.shape}.")
        if len(X_arr) < self.n_clusters:
            raise ValidationError(
                f"Need at least {self.n_clusters} samples for "
                f"{self.n_clusters} clusters, got {len(X_arr)}."
            )

        logger.info(
            "Fitting QuantumKMeans (k=%d) on %d samples.",
            self.n_clusters,
            len(X_arr),
        )

        # Random centroid initialisation
        indices = np.random.choice(len(X_arr), self.n_clusters, replace=False)
        self.centroids_ = X_arr[indices].copy()

        for iteration in range(self.max_iter):
            logger.info(
                "Q-Means iteration %d/%d.", iteration + 1, self.max_iter
            )

            # Assignment step
            labels = np.array(
                [
                    int(
                        np.argmin(
                            [self._quantum_distance(row, c) for c in self.centroids_]
                        )
                    )
                    for row in X_arr
                ]
            )
            self.labels_ = labels

            # Update step
            new_centroids = []
            for k in range(self.n_clusters):
                cluster_points = X_arr[self.labels_ == k]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis=0))
                else:
                    new_centroids.append(self.centroids_[k])

            new_centroids_arr = np.array(new_centroids)
            if np.allclose(self.centroids_, new_centroids_arr):
                logger.info("Converged at iteration %d.", iteration + 1)
                break

            self.centroids_ = new_centroids_arr

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign cluster labels to samples in *X*.

        Args:
            X: Data of shape ``(n_samples, n_features)``.
                Accepts both numpy arrays and pandas DataFrames.

        Returns:
            Cluster indices of shape ``(n_samples,)``.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        self._check_is_fitted()
        X_arr: np.ndarray = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValidationError(f"X must be 2-D, got shape {X_arr.shape}.")

        labels = np.array(
            [
                int(
                    np.argmin(
                        [self._quantum_distance(row, c) for c in self.centroids_]
                    )
                )
                for row in X_arr
            ]
        )
        return labels
