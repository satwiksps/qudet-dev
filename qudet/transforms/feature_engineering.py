"""Feature engineering and selection methods for quantum data preprocessing.

Provides feature scaling, normalization, selection, outlier removal, and
data balancing techniques for quantum machine learning pipelines.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from qudet.core.base import BaseReducer
from qudet.core.exceptions import NotFittedError, ValidationError

logger = logging.getLogger(__name__)


class FeatureScaler(BaseReducer):
    """Scales and normalises features for quantum algorithms.

    Provides multiple scaling strategies:

    * **standard** — Z-score normalisation (zero mean, unit variance).
    * **minmax** — Linear scaling to a specified ``feature_range``.
    * **robust** — Median / IQR scaling (resistant to outliers).
    * **quantum** — Standard scaling followed by per-sample L2 normalisation.

    Args:
        method: Scaling method (``'standard'``, ``'minmax'``, ``'robust'``,
            ``'quantum'``).
        feature_range: Output range for ``'minmax'`` scaling.

    Example:
        >>> scaler = FeatureScaler(method="minmax", feature_range=(0, 1))
        >>> scaler.fit(X_train).transform(X_test)
    """

    def __init__(
        self,
        method: str = "standard",
        feature_range: Tuple[float, float] = (0, 1),
    ) -> None:
        self.method = method.lower()
        self.feature_range = feature_range
        self.scaler = self._create_scaler()
        self.fitted = False

    def _create_scaler(self):
        """Create the appropriate sklearn scaler based on *method*."""
        if self.method == "standard":
            return StandardScaler()
        elif self.method == "minmax":
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.method == "robust":
            return RobustScaler()
        elif self.method == "quantum":
            return StandardScaler()
        else:
            raise ValidationError(
                f"Unknown scaling method {self.method!r}. "
                "Choose from 'standard', 'minmax', 'robust', 'quantum'."
            )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
    ) -> "FeatureScaler":
        """Fit scaler to training data.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.scaler.fit(X)
        self.fitted = True
        logger.info("FeatureScaler (method=%s) fitted", self.method)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform features using the fitted scaler.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.

        Returns:
            Scaled features as ``np.ndarray``.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "FeatureScaler has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        scaled = self.scaler.transform(X)

        if self.method == "quantum":
            scaled = self._apply_quantum_scaling(scaled)

        return scaled

    def _apply_quantum_scaling(self, X: np.ndarray) -> np.ndarray:
        """Apply per-sample L2 normalisation (quantum-aware scaling)."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return X / norms

    def get_scaling_params(self) -> dict:
        """Return the learned scaling parameters.

        Returns:
            Dictionary containing method, fitted status, and any learned
            statistics (mean, scale).
        """
        if not self.fitted:
            return {}

        params: dict = {"method": self.method, "fitted": self.fitted}

        if hasattr(self.scaler, "mean_"):
            params["mean"] = self.scaler.mean_
        if hasattr(self.scaler, "scale_"):
            params["scale"] = self.scaler.scale_

        return params


class FeatureSelector(BaseReducer):
    """Selects the most informative features for quantum algorithms.

    Reduces dimensionality by retaining features with the highest scores
    according to a univariate statistical test.

    Supported methods:

    * **f_classif** — ANOVA F-statistic.
    * **mutual_info** — Mutual information (non-linear dependencies).

    Args:
        n_features: Number of top features to keep.
        method: Scoring method (``'f_classif'`` or ``'mutual_info'``).

    Note:
        ``fit()`` requires target labels *y* because both scoring methods
        are supervised.
    """

    def __init__(self, n_features: int = 10, method: str = "f_classif") -> None:
        if not isinstance(n_features, int) or n_features < 1:
            raise ValidationError(
                f"n_features must be a positive integer, got {n_features!r}"
            )
        self.n_features = n_features
        self.method = method
        self.selector: Optional[SelectKBest] = None
        self.feature_indices_: Optional[np.ndarray] = None
        self.feature_scores_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, None] = None,
    ) -> "FeatureSelector":
        """Fit the feature selector.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.
            y: Target values. **Required** for scoring.

        Returns:
            self

        Raises:
            ValidationError: If *y* is ``None``.
        """
        if y is None:
            raise ValidationError(
                "FeatureSelector requires target labels y for fitting."
            )
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if self.method == "mutual_info":
            score_func = mutual_info_classif
        else:
            score_func = f_classif

        self.selector = SelectKBest(score_func, k=self.n_features)
        self.selector.fit(X, y)

        self.feature_indices_ = self.selector.get_support(indices=True)
        self.feature_scores_ = self.selector.scores_

        logger.info(
            "FeatureSelector selected %d features using %s",
            self.n_features,
            self.method,
        )
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Select the fitted subset of features from *X*.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.

        Returns:
            Selected features of shape ``(n_samples, n_features_selected)``.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if self.selector is None:
            raise NotFittedError(
                "FeatureSelector has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.selector.transform(X)

    def get_selected_features(self) -> np.ndarray:
        """Return the indices of features selected during ``fit``.

        Raises:
            NotFittedError: If the selector has not been fitted.
        """
        if self.feature_indices_ is None:
            raise NotFittedError("FeatureSelector has not been fitted.")
        return self.feature_indices_

    def get_feature_scores(self) -> np.ndarray:
        """Return per-feature importance scores from ``fit``.

        Raises:
            NotFittedError: If the selector has not been fitted.
        """
        if self.feature_scores_ is None:
            raise NotFittedError("FeatureSelector has not been fitted.")
        return self.feature_scores_


class OutlierRemover(BaseReducer):
    """Removes outliers from datasets using multiple detection strategies.

    Supported methods:

    * **iqr** — Interquartile-Range fence (1.5 × IQR).
    * **zscore** — Absolute z-score exceeding ``threshold``.
    * **isolation** — Distance-from-mean exceeding
      ``mean + threshold × std``.

    Unlike the previous implementation that stored a mask from ``fit`` and
    blindly applied it during ``transform``, this version stores the
    **statistical parameters** (bounds / thresholds) during ``fit`` and
    **recomputes** the outlier mask on whatever data is passed to
    ``transform``.

    Args:
        method: Detection method (``'iqr'``, ``'zscore'``, ``'isolation'``).
        threshold: Sensitivity parameter (z-score cut-off or distance
            multiplier).
    """

    def __init__(self, method: str = "iqr", threshold: float = 3.0) -> None:
        self.method = method.lower()
        self.threshold = threshold
        # Learned parameters
        self._lower_bound: Optional[np.ndarray] = None
        self._upper_bound: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._dist_threshold: Optional[float] = None
        self.fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
    ) -> "OutlierRemover":
        """Learn outlier detection parameters from training data.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.method == "iqr":
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            iqr = q3 - q1
            self._lower_bound = q1 - 1.5 * iqr
            self._upper_bound = q3 + 1.5 * iqr
        elif self.method == "zscore":
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0) + 1e-10
        elif self.method == "isolation":
            self._mean = np.mean(X, axis=0)
            distances = np.linalg.norm(X - self._mean, axis=1)
            self._dist_threshold = float(
                np.mean(distances) + self.threshold * np.std(distances)
            )
        else:
            raise ValidationError(
                f"Unknown outlier detection method {self.method!r}. "
                "Choose from 'iqr', 'zscore', 'isolation'."
            )

        self.fitted = True
        logger.info("OutlierRemover (method=%s) fitted", self.method)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Remove outliers from *X* using parameters learned during ``fit``.

        The outlier mask is **recomputed** on the incoming data using the
        statistical boundaries / thresholds stored during ``fit``.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            Filtered data with outlier rows removed.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "OutlierRemover has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        mask = self._compute_inlier_mask(X)
        return X[mask]

    def _compute_inlier_mask(self, X: np.ndarray) -> np.ndarray:
        """Return a boolean mask where ``True`` marks inlier rows."""
        if self.method == "iqr":
            return np.all(
                (X >= self._lower_bound) & (X <= self._upper_bound), axis=1
            )
        elif self.method == "zscore":
            z_scores = np.abs((X - self._mean) / self._std)
            return np.all(z_scores < self.threshold, axis=1)
        elif self.method == "isolation":
            distances = np.linalg.norm(X - self._mean, axis=1)
            return distances <= self._dist_threshold
        # Should be unreachable after validation in fit()
        return np.ones(X.shape[0], dtype=bool)  # pragma: no cover

    def get_outlier_ratio(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        """Compute the fraction of outliers in *X*.

        Args:
            X: Data to evaluate.

        Returns:
            Ratio of outlier samples (0.0 = no outliers, 1.0 = all outliers).

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "OutlierRemover has not been fitted. Call fit() first."
            )
        if isinstance(X, pd.DataFrame):
            X = X.values
        mask = self._compute_inlier_mask(X)
        return 1.0 - float(np.mean(mask))


class DataBalancer(BaseReducer):
    """Balances imbalanced datasets for classification tasks.

    Stores the target labels *y* during ``fit`` and uses them during
    ``transform`` so that the ``transform(X)`` signature matches the
    :class:`BaseReducer` contract.

    Supported methods:

    * **oversample** — Duplicate minority-class samples.
    * **undersample** — Sub-sample majority-class samples.
    * **smote** — Generate synthetic samples via linear interpolation
      between minority-class neighbours.

    Args:
        method: Balancing method (``'oversample'``, ``'undersample'``,
            ``'smote'``).
        ratio: Target minority / majority ratio (currently unused;
            reserved for future extension).

    Note:
        Because balancing requires labels, *y* **must** be provided to
        ``fit()``.  The balanced ``(X, y)`` tuple is returned by
        ``transform()`` as a vertically stacked array where the last
        column contains the labels.
    """

    def __init__(self, method: str = "oversample", ratio: float = 1.0) -> None:
        self.method = method.lower()
        self.ratio = ratio
        self.class_counts_: Optional[dict] = None
        self._y_fit: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, None] = None,
    ) -> "DataBalancer":
        """Analyse class distribution and store labels for transform.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.
            y: Target labels. **Required**.

        Returns:
            self

        Raises:
            ValidationError: If *y* is ``None``.
        """
        if y is None:
            raise ValidationError(
                "DataBalancer requires target labels y for fitting."
            )
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        unique, counts = np.unique(y, return_counts=True)
        self.class_counts_ = dict(zip(unique, counts))
        self._y_fit = y.copy()

        logger.info(
            "DataBalancer (method=%s) fitted — class counts: %s",
            self.method,
            self.class_counts_,
        )
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Balance the dataset using labels stored during ``fit``.

        Returns the balanced feature matrix.  Use
        :meth:`transform_with_labels` if you also need the balanced *y*.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.

        Returns:
            Balanced feature array of shape ``(n_balanced, n_features)``.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
            ValidationError: If *X* length does not match stored *y*.
        """
        if self.class_counts_ is None or self._y_fit is None:
            raise NotFittedError(
                "DataBalancer has not been fitted. Call fit(X, y) first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.shape[0] != self._y_fit.shape[0]:
            raise ValidationError(
                f"X has {X.shape[0]} samples but fit() stored "
                f"{self._y_fit.shape[0]} labels. Pass the same X used in fit()."
            )

        balanced_X, _ = self._balance(X, self._y_fit)
        return balanced_X

    def transform_with_labels(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance the dataset and return both features and labels.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.

        Returns:
            ``(balanced_X, balanced_y)`` tuple.
        """
        if self.class_counts_ is None or self._y_fit is None:
            raise NotFittedError(
                "DataBalancer has not been fitted. Call fit(X, y) first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.shape[0] != self._y_fit.shape[0]:
            raise ValidationError(
                f"X has {X.shape[0]} samples but fit() stored "
                f"{self._y_fit.shape[0]} labels."
            )

        return self._balance(X, self._y_fit)

    # ------------------------------------------------------------------
    # Internal balancing strategies
    # ------------------------------------------------------------------

    def _balance(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch to the configured balancing method."""
        if self.method == "oversample":
            return self._oversample(X, y)
        elif self.method == "undersample":
            return self._undersample(X, y)
        elif self.method == "smote":
            return self._smote_balance(X, y)
        else:
            raise ValidationError(
                f"Unknown balancing method {self.method!r}. "
                "Choose from 'oversample', 'undersample', 'smote'."
            )

    def _oversample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Oversample minority classes to match the majority class count."""
        unique_classes = np.unique(y)
        max_count = max(np.sum(y == cls) for cls in unique_classes)

        X_balanced: List[np.ndarray] = []
        y_balanced: List[np.ndarray] = []

        for cls in unique_classes:
            mask = y == cls
            X_cls = X[mask]
            y_cls = y[mask]

            if len(X_cls) < max_count:
                indices = np.random.choice(len(X_cls), max_count, replace=True)
                X_cls = X_cls[indices]
                y_cls = y_cls[indices]

            X_balanced.append(X_cls)
            y_balanced.append(y_cls)

        return np.vstack(X_balanced), np.hstack(y_balanced)

    def _undersample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Undersample majority classes to match the minority class count."""
        unique_classes = np.unique(y)
        min_count = min(np.sum(y == cls) for cls in unique_classes)

        X_balanced: List[np.ndarray] = []
        y_balanced: List[np.ndarray] = []

        for cls in unique_classes:
            mask = y == cls
            X_cls = X[mask]
            y_cls = y[mask]

            if len(X_cls) > min_count:
                indices = np.random.choice(len(X_cls), min_count, replace=False)
                X_cls = X_cls[indices]
                y_cls = y_cls[indices]

            X_balanced.append(X_cls)
            y_balanced.append(y_cls)

        return np.vstack(X_balanced), np.hstack(y_balanced)

    def _smote_balance(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE-like synthetic sample generation for minority classes."""
        unique_classes = np.unique(y)
        max_count = max(np.sum(y == cls) for cls in unique_classes)

        X_balanced: List[np.ndarray] = []
        y_balanced: List[np.ndarray] = []

        for cls in unique_classes:
            mask = y == cls
            X_cls = X[mask]
            y_cls = y[mask]

            X_balanced.append(X_cls)
            y_balanced.append(y_cls)

            if len(X_cls) < max_count:
                n_synthetic = max_count - len(X_cls)
                for _ in range(n_synthetic):
                    idx1 = np.random.randint(len(X_cls))
                    idx2 = np.random.randint(len(X_cls))
                    alpha = np.random.random()
                    synthetic = alpha * X_cls[idx1] + (1 - alpha) * X_cls[idx2]
                    X_balanced.append(synthetic.reshape(1, -1))
                    y_balanced.append(np.array([cls]))

        return np.vstack(X_balanced), np.hstack(y_balanced)

    def get_balance_info(self) -> dict:
        """Return class-balance statistics from the training data.

        Returns:
            Dictionary with ``class_counts`` and ``imbalance_ratio``.
        """
        if self.class_counts_ is None:
            return {}
        return {
            "class_counts": self.class_counts_,
            "imbalance_ratio": (
                max(self.class_counts_.values())
                / min(self.class_counts_.values())
            ),
        }
