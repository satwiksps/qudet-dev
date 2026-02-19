"""Quantum-inspired feature selection.

Selects an optimal subset of features using a greedy QUBO-style approach
that maximises relevance to the target while minimising inter-feature
redundancy.  Designed to interface with QAOA on real quantum hardware in
future iterations.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from qudet.core.base import BaseQuantumEstimator
from qudet.core.exceptions import NotFittedError, ValidationError

logger = logging.getLogger(__name__)


class QuantumFeatureSelector(BaseQuantumEstimator):
    """Select features via quantum-inspired greedy optimisation.

    For the current release this uses a classical greedy heuristic that
    mirrors a QUBO objective: each feature is scored by its relevance
    (absolute correlation with the target) minus its redundancy (average
    absolute correlation with already-selected features).  The top *k*
    features are retained.

    Args:
        n_features_to_select: Number of features to keep.
        backend_name: Qiskit backend name (passed to base class).
        shots: Number of measurement shots (passed to base class).

    Attributes:
        selected_features_: Column names selected after fitting.

    Example:
        >>> selector = QuantumFeatureSelector(n_features_to_select=5)
        >>> selector.fit(X_train, y_train)
        >>> X_reduced = selector.transform(X_test)
    """

    def __init__(
        self,
        n_features_to_select: int = 5,
        backend_name: str = "aer_simulator",
        shots: int = 1024,
    ) -> None:
        if n_features_to_select < 1:
            raise ValidationError(
                f"n_features_to_select must be >= 1, got {n_features_to_select}."
            )
        super().__init__(backend_name=backend_name, shots=shots)
        self.k = n_features_to_select
        self.selected_features_: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, None] = None,
    ) -> "QuantumFeatureSelector":
        """Select the best feature subset.

        The method greedily picks features that maximise
        ``relevance − redundancy`` where relevance is the absolute
        correlation with *y* and redundancy is the mean absolute
        correlation with already-selected features.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
                If a numpy array is given, synthetic column names
                (``feature_0``, ``feature_1``, …) are created.
            y: Target values of shape ``(n_samples,)``.  Required.

        Returns:
            self

        Raises:
            ValidationError: If *y* is not provided or *X* is not 2-D.
        """
        if y is None:
            raise ValidationError(
                "y must be provided for feature selection."
            )

        # Ensure we have a DataFrame for correlation-based logic.
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValidationError(
                    f"X must be 2-D, got shape {X.shape}."
                )
            X = pd.DataFrame(
                X, columns=[f"feature_{i}" for i in range(X.shape[1])]
            )
        elif not isinstance(X, pd.DataFrame):
            raise ValidationError(
                f"X must be a numpy array or pandas DataFrame, "
                f"got {type(X).__name__}."
            )

        if isinstance(y, np.ndarray):
            y = pd.Series(y, name="target")
        elif not isinstance(y, pd.Series):
            y = pd.Series(np.asarray(y), name="target")

        logger.info(
            "Selecting %d features from %d candidates on backend '%s'.",
            self.k,
            X.shape[1],
            self.backend_name,
        )

        corr_matrix = X.corr().abs()
        target_corr = X.corrwith(y).abs()

        features = X.columns.tolist()
        n = len(features)

        selected_indices: List[int] = []

        # Seed with the feature most correlated with the target.
        if not target_corr.empty:
            current_best = target_corr.idxmax()
            selected_indices.append(features.index(current_best))

        # Greedy forward selection
        for _ in range(self.k - 1):
            if len(selected_indices) >= n:
                break

            best_candidate = -1
            max_score = -np.inf

            for i in range(n):
                if i in selected_indices:
                    continue

                relevance = target_corr.iloc[i]
                redundancy = 0.0
                if selected_indices:
                    redundancy = (
                        sum(corr_matrix.iloc[i, j] for j in selected_indices)
                        / len(selected_indices)
                    )

                score = relevance - redundancy
                if score > max_score:
                    max_score = score
                    best_candidate = i

            if best_candidate != -1:
                selected_indices.append(best_candidate)

        self.selected_features_ = [features[i] for i in selected_indices]
        self._is_fitted = True
        logger.info("Selected features: %s", self.selected_features_)
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """Reduce *X* to the selected feature subset.

        Args:
            X: Data of shape ``(n_samples, n_features)``.
                If a numpy array is given, the same synthetic column names
                used during :meth:`fit` are assumed.

        Returns:
            DataFrame containing only the selected columns.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        self._check_is_fitted()

        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValidationError(
                    f"X must be 2-D, got shape {X.shape}."
                )
            X = pd.DataFrame(
                X, columns=[f"feature_{i}" for i in range(X.shape[1])]
            )

        missing = set(self.selected_features_) - set(X.columns)
        if missing:
            raise ValidationError(
                f"Columns missing from X: {missing}."
            )

        return X[self.selected_features_]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Not implemented — this is a transformer, not a predictor.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "QuantumFeatureSelector is a transformer, not a predictor. "
            "Use transform() instead."
        )
