"""Quantum Principal Component Analysis (PCA) for dimensionality reduction.

Implements PCA in a high-dimensional quantum Hilbert space by
diagonalising the quantum kernel matrix.  Unlike classical (linear) PCA,
this captures non-linear structures in the data.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA

from qudet.core.base import BaseReducer
from qudet.core.exceptions import NotFittedError
from ..analytics.anomaly import QuantumKernelAnomalyDetector

logger = logging.getLogger(__name__)


class QuantumPCA(BaseReducer):
    """PCA in quantum Hilbert space via kernel diagonalisation.

    Computes the quantum kernel matrix between training samples using
    :class:`~qudet.analytics.anomaly.QuantumKernelAnomalyDetector`, then
    applies scikit-learn's :class:`~sklearn.decomposition.KernelPCA` with
    a precomputed kernel.

    Args:
        n_components: Number of principal components to extract.
        n_qubits: Number of qubits used for quantum kernel computation.

    Attributes:
        kernel_computer: Engine for quantum kernel evaluation.
        pca_model: Fitted :class:`KernelPCA` with ``kernel="precomputed"``.
        train_data\_: Training data retained for out-of-sample projection.

    Example:
        >>> qpca = QuantumPCA(n_components=2, n_qubits=4)
        >>> qpca.fit(X_train)
        >>> X_proj = qpca.transform(X_test)
    """

    def __init__(self, n_components: int = 2, n_qubits: int = 4) -> None:
        self.n_components = n_components
        self.n_qubits = n_qubits
        self.kernel_computer = QuantumKernelAnomalyDetector(n_qubits=n_qubits)
        self.pca_model = KernelPCA(
            n_components=n_components, kernel="precomputed"
        )
        self.train_data_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y=None,
    ) -> "QuantumPCA":
        """Learn the principal components of the quantum kernel.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.train_data_ = X

        logger.info(
            "Computing quantum kernel matrix for %d samples", X.shape[0]
        )
        kernel_matrix = self.kernel_computer._compute_kernel_matrix(X, X)

        self.pca_model.fit(kernel_matrix)
        logger.info(
            "QuantumPCA fitted with %d components", self.n_components
        )
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Project new data onto the quantum principal components.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            Projected data of shape ``(n_samples, n_components)``.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if self.train_data_ is None:
            raise NotFittedError(
                "QuantumPCA has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        kernel_matrix_new = self.kernel_computer._compute_kernel_matrix(
            X, self.train_data_
        )
        return self.pca_model.transform(kernel_matrix_new)
