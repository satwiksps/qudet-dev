"""Quantum-inspired imputation for handling missing data.

Real data frequently contains missing values (NaNs).  Classical imputation
strategies (mean / median) destroy multivariate structure.  This module
provides a quantum-kernel-aware imputer that fills gaps using cluster
centroids computed via :class:`~qudet.analytics.clustering.QuantumKMeans`.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

from qudet.core.base import BaseReducer
from qudet.core.exceptions import NotFittedError, ValidationError

logger = logging.getLogger(__name__)


class QuantumImputer(BaseReducer):
    """Fills missing values using quantum-kernel-based clustering.

    Algorithm:

    1. During ``fit``, complete (non-NaN) rows are clustered using
       :class:`~qudet.analytics.clustering.QuantumKMeans`.
    2. During ``transform``, each row that contains NaN values is assigned
       to the nearest cluster (computed on the non-missing columns), and
       the missing entries are filled with the corresponding centroid
       values.

    This preserves multivariate structure better than simple mean/median
    imputation.

    Args:
        n_clusters: Number of clusters for K-Means.

    Attributes:
        clusterer: Fitted :class:`QuantumKMeans` instance.
        n_features\_: Number of features observed during ``fit``.

    Example:
        >>> imp = QuantumImputer(n_clusters=5)
        >>> imp.fit(X_train)
        >>> X_clean = imp.transform(X_with_nans)
    """

    def __init__(self, n_clusters: int = 3) -> None:
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValidationError(
                f"n_clusters must be a positive integer, got {n_clusters!r}"
            )
        self.n_clusters = n_clusters

        # Lazy import to avoid circular dependency at module level
        from ..analytics.clustering import QuantumKMeans

        self.clusterer = QuantumKMeans(n_clusters=n_clusters)
        self.n_features_: Optional[int] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None,
    ) -> "QuantumImputer":
        """Fit the clusterer on complete (non-NaN) rows.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.  Should
                be mostly complete; rows with any NaN are dropped before
                clustering.
            y: Ignored. Present for API compatibility.

        Returns:
            self

        Raises:
            ValidationError: If no complete rows remain after dropping NaNs.
        """
        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = np.asarray(X, dtype=float)

        # Keep only complete rows for fitting
        complete_mask = ~np.isnan(data).any(axis=1)
        X_clean = data[complete_mask]

        if X_clean.shape[0] == 0:
            raise ValidationError(
                "No complete (NaN-free) rows found in X. "
                "Cannot fit QuantumImputer."
            )
        if X_clean.shape[0] < self.n_clusters:
            raise ValidationError(
                f"Only {X_clean.shape[0]} complete rows but n_clusters="
                f"{self.n_clusters}. Reduce n_clusters or provide more data."
            )

        self.n_features_ = data.shape[1]
        self.clusterer.fit(X_clean)

        logger.info(
            "QuantumImputer fitted with %d clusters on %d complete rows "
            "(out of %d total)",
            self.n_clusters,
            X_clean.shape[0],
            data.shape[0],
        )
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Fill missing values in *X* using cluster centroids.

        Args:
            X: Data of shape ``(n_samples, n_features)`` with potential
                NaN values.

        Returns:
            Array of shape ``(n_samples, n_features)`` with NaNs replaced.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
            ValidationError: If *X* has a different number of features than
                the training data.
        """
        if self.n_features_ is None:
            raise NotFittedError(
                "QuantumImputer has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            data = X.values.copy().astype(float)
        else:
            data = np.array(X, dtype=float)

        if data.shape[1] != self.n_features_:
            raise ValidationError(
                f"X has {data.shape[1]} features but the imputer was fitted "
                f"on {self.n_features_} features."
            )

        missing_mask = np.isnan(data)
        if not missing_mask.any():
            return data

        n_imputed = 0
        for idx in range(data.shape[0]):
            if not missing_mask[idx].any():
                continue

            # Temporarily fill NaN with 0 for distance computation
            temp_row = data[idx].copy()
            temp_row[missing_mask[idx]] = 0.0

            # Find nearest cluster
            distances = [
                self.clusterer._quantum_distance(temp_row, c)
                for c in self.clusterer.centroids_
            ]
            best_cluster = int(np.argmin(distances))
            centroid = self.clusterer.centroids_[best_cluster]

            # Fill only the missing values with centroid values
            for col in np.where(missing_mask[idx])[0]:
                data[idx, col] = centroid[col]

            n_imputed += 1

        logger.info("QuantumImputer imputed %d rows", n_imputed)
        return data
