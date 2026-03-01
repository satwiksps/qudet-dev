"""Categorical feature encoding for quantum algorithms.

Converts categorical and discrete features into numerical representations
suitable for quantum machine learning.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from qudet.core.base import BaseReducer
from qudet.core.exceptions import NotFittedError, ValidationError

logger = logging.getLogger(__name__)


class CategoricalEncoder(BaseReducer):
    """Encodes categorical features to numerical values.

    Supported methods:

    * **label** — Map each unique category to a unique integer.
    * **onehot** — Create binary indicator columns per category.
    * **ordinal** — Equivalent to label encoding (natural ordering).
    * **binary** — Encode category indices in binary (minimal bits).

    Args:
        method: Encoding method.
        handle_unknown: How to handle categories not seen during ``fit``
            (``'error'`` raises, ``'ignore'`` maps to zero).

    Example:
        >>> enc = CategoricalEncoder(method="onehot")
        >>> enc.fit(X_train).transform(X_test)
    """

    def __init__(
        self, method: str = "label", handle_unknown: str = "error"
    ) -> None:
        self.method = method.lower()
        self.handle_unknown = handle_unknown
        self.categories_: Dict[int, np.ndarray] = {}
        self.encoding_map_: Dict[int, Dict] = {}
        self.fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "CategoricalEncoder":
        """Learn categories from data.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        for col_idx in range(X.shape[1]):
            unique_vals = np.unique(X[:, col_idx])
            self.categories_[col_idx] = unique_vals
            self.encoding_map_[col_idx] = {
                val: idx for idx, val in enumerate(unique_vals)
            }

        self.fitted = True
        logger.info(
            "CategoricalEncoder (method=%s) fitted on %d features",
            self.method,
            X.shape[1],
        )
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Encode categorical features.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.

        Returns:
            Encoded features as ``np.ndarray``.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "CategoricalEncoder has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()

        if self.method == "label":
            return self._encode_label(X)
        elif self.method == "onehot":
            return self._encode_onehot(X)
        elif self.method == "binary":
            return self._encode_binary(X)
        else:
            return self._encode_label(X)

    def _encode_label(self, X: np.ndarray) -> np.ndarray:
        """Label-encode categorical features."""
        X_encoded = np.zeros_like(X, dtype=float)

        for col_idx in range(X.shape[1]):
            if col_idx in self.encoding_map_:
                for row_idx, val in enumerate(X[:, col_idx]):
                    if val in self.encoding_map_[col_idx]:
                        X_encoded[row_idx, col_idx] = self.encoding_map_[col_idx][val]
                    elif self.handle_unknown == "error":
                        raise ValidationError(f"Unknown category: {val}")

        return X_encoded

    def _encode_onehot(self, X: np.ndarray) -> np.ndarray:
        """One-hot-encode categorical features."""
        encoded_parts: List[np.ndarray] = []

        for col_idx in range(X.shape[1]):
            if col_idx in self.encoding_map_:
                n_categories = len(self.encoding_map_[col_idx])
                onehot = np.zeros((X.shape[0], n_categories))

                for row_idx, val in enumerate(X[:, col_idx]):
                    if val in self.encoding_map_[col_idx]:
                        cat_idx = self.encoding_map_[col_idx][val]
                        onehot[row_idx, cat_idx] = 1

                encoded_parts.append(onehot)

        return np.hstack(encoded_parts) if encoded_parts else X.astype(float)

    def _encode_binary(self, X: np.ndarray) -> np.ndarray:
        """Binary-encode categorical features (minimal bits)."""
        encoded_parts: List[np.ndarray] = []

        for col_idx in range(X.shape[1]):
            if col_idx in self.encoding_map_:
                mapping = self.encoding_map_[col_idx]
                n_categories = len(mapping)
                n_bits = max(1, int(np.ceil(np.log2(n_categories))))

                binary = np.zeros((X.shape[0], n_bits))

                for row_idx, val in enumerate(X[:, col_idx]):
                    if val in mapping:
                        cat_idx = mapping[val]
                        bits = [(cat_idx >> i) & 1 for i in range(n_bits)]
                        binary[row_idx, :] = bits

                encoded_parts.append(binary)

        return np.hstack(encoded_parts) if encoded_parts else X.astype(float)

    def get_n_features_out(self) -> int:
        """Return the number of output features after encoding.

        Returns:
            Number of columns produced by ``transform``.
        """
        if not self.fitted:
            return 0

        if self.method == "label":
            return len(self.categories_)
        elif self.method == "onehot":
            return sum(len(cats) for cats in self.categories_.values())
        elif self.method == "binary":
            total = 0
            for cats in self.categories_.values():
                n_bits = max(1, int(np.ceil(np.log2(len(cats)))))
                total += n_bits
            return total
        return len(self.categories_)


class TargetEncoder(BaseReducer):
    """Encodes features based on target-variable statistics.

    Maps each category to the smoothed mean of the target variable for
    samples in that category (Bayesian target encoding).

    Args:
        smoothing: Regularisation strength.  Higher values bias category
            means toward the global mean.
        min_samples_leaf: Minimum samples per category; categories below
            this threshold receive the global mean.

    Note:
        ``fit()`` requires *y* because encoding is supervised.

    Example:
        >>> te = TargetEncoder(smoothing=1.0)
        >>> te.fit(X_train, y_train).transform(X_test)
    """

    def __init__(
        self, smoothing: float = 1.0, min_samples_leaf: int = 1
    ) -> None:
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encoding_map_: Dict[int, Dict] = {}
        self.global_mean_: Optional[float] = None
        self.fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, None] = None,
    ) -> "TargetEncoder":
        """Learn target encoding from data.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.
            y: Target values. **Required**.

        Returns:
            self

        Raises:
            ValidationError: If *y* is ``None``.
        """
        if y is None:
            raise ValidationError(
                "TargetEncoder requires target values y for fitting."
            )
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.global_mean_ = float(np.mean(y))

        for col_idx in range(X.shape[1]):
            unique_vals = np.unique(X[:, col_idx])
            encoding: Dict = {}

            for val in unique_vals:
                mask = X[:, col_idx] == val
                target_vals = y[mask]
                n_samples = len(target_vals)

                if n_samples >= self.min_samples_leaf:
                    category_mean = np.mean(target_vals)
                    smoothed_mean = (
                        n_samples * category_mean
                        + self.smoothing * self.global_mean_
                    ) / (n_samples + self.smoothing)
                    encoding[val] = smoothed_mean
                else:
                    encoding[val] = self.global_mean_

            self.encoding_map_[col_idx] = encoding

        self.fitted = True
        logger.info("TargetEncoder fitted on %d features", X.shape[1])
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply target encoding.

        Unknown categories (not seen during ``fit``) are mapped to the
        global mean.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.

        Returns:
            Target-encoded features as ``np.ndarray``.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "TargetEncoder has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()

        X_encoded = np.zeros_like(X, dtype=float)

        for col_idx in range(X.shape[1]):
            if col_idx in self.encoding_map_:
                for row_idx, val in enumerate(X[:, col_idx]):
                    if val in self.encoding_map_[col_idx]:
                        X_encoded[row_idx, col_idx] = self.encoding_map_[col_idx][val]
                    else:
                        X_encoded[row_idx, col_idx] = self.global_mean_

        return X_encoded

    def get_encoding_info(self) -> dict:
        """Return encoding statistics.

        Returns:
            Dictionary with global mean, smoothing, and feature count.
        """
        if not self.fitted:
            return {}
        return {
            "global_mean": self.global_mean_,
            "smoothing": self.smoothing,
            "n_features": len(self.encoding_map_),
        }


class FrequencyEncoder(BaseReducer):
    """Encodes features based on category frequency.

    Maps each category to its relative (or absolute) frequency in the
    training data.

    Args:
        normalize: If ``True``, use relative frequencies (0–1);
            otherwise absolute counts.

    Example:
        >>> fe = FrequencyEncoder(normalize=True)
        >>> fe.fit(X_train).transform(X_test)
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize
        self.frequency_map_: Dict[int, Dict] = {}
        self.fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "FrequencyEncoder":
        """Learn category frequencies.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        for col_idx in range(X.shape[1]):
            unique_vals, counts = np.unique(X[:, col_idx], return_counts=True)
            n_total = len(X[:, col_idx])

            freq_map: Dict = {}
            for val, count in zip(unique_vals, counts):
                freq_map[val] = count / n_total if self.normalize else count

            self.frequency_map_[col_idx] = freq_map

        self.fitted = True
        logger.info("FrequencyEncoder fitted on %d features", X.shape[1])
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply frequency encoding.

        Unknown categories (not seen during ``fit``) are mapped to zero.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.

        Returns:
            Frequency-encoded features as ``np.ndarray``.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "FrequencyEncoder has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()

        X_encoded = np.zeros_like(X, dtype=float)

        for col_idx in range(X.shape[1]):
            if col_idx in self.frequency_map_:
                for row_idx, val in enumerate(X[:, col_idx]):
                    if val in self.frequency_map_[col_idx]:
                        X_encoded[row_idx, col_idx] = self.frequency_map_[col_idx][val]

        return X_encoded

    def get_frequency_info(self) -> dict:
        """Return per-feature frequency statistics.

        Returns:
            Dictionary keyed by ``feature_<idx>`` with category counts
            and frequency maps.
        """
        if not self.fitted:
            return {}

        info: Dict = {}
        for col_idx, freq_map in self.frequency_map_.items():
            info[f"feature_{col_idx}"] = {
                "n_categories": len(freq_map),
                "frequencies": freq_map,
            }

        return info


class BinningEncoder(BaseReducer):
    """Bins continuous features into discrete categories.

    Supported methods:

    * **quantile** — Equal-frequency (quantile-based) bin edges.
    * **uniform** — Equal-width bins.
    * **kmeans** — Bin edges derived from K-Means cluster centres.

    Args:
        n_bins: Number of bins per feature.
        method: Binning strategy.

    Example:
        >>> be = BinningEncoder(n_bins=10, method="quantile")
        >>> be.fit(X_train).transform(X_test)
    """

    def __init__(self, n_bins: int = 5, method: str = "quantile") -> None:
        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValidationError(
                f"n_bins must be an integer ≥ 2, got {n_bins!r}"
            )
        self.n_bins = n_bins
        self.method = method.lower()
        self.bin_edges_: Dict[int, np.ndarray] = {}
        self.fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "BinningEncoder":
        """Learn bin boundaries from data.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]

            if self.method == "quantile":
                edges = np.percentile(
                    col_data, np.linspace(0, 100, self.n_bins + 1)
                )
            elif self.method == "uniform":
                edges = np.linspace(
                    col_data.min(), col_data.max(), self.n_bins + 1
                )
            elif self.method == "kmeans":
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=self.n_bins, random_state=42, n_init=10)
                kmeans.fit_predict(col_data.reshape(-1, 1))
                centers = np.sort(kmeans.cluster_centers_.ravel())
                edges = np.concatenate(
                    [[col_data.min()], centers, [col_data.max()]]
                )
            else:
                edges = np.percentile(
                    col_data, np.linspace(0, 100, self.n_bins + 1)
                )

            self.bin_edges_[col_idx] = np.unique(edges)

        self.fitted = True
        logger.info(
            "BinningEncoder (method=%s, n_bins=%d) fitted on %d features",
            self.method,
            self.n_bins,
            X.shape[1],
        )
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply binning to features.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.

        Returns:
            Bin indices as an integer ``np.ndarray``.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "BinningEncoder has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()

        X_binned = np.zeros_like(X, dtype=int)

        for col_idx in range(X.shape[1]):
            if col_idx in self.bin_edges_:
                edges = self.bin_edges_[col_idx]
                X_binned[:, col_idx] = (
                    np.digitize(X[:, col_idx], edges, right=False) - 1
                )

        return X_binned

    def get_bin_info(self) -> dict:
        """Return binning metadata.

        Returns:
            Dictionary with n_bins, method, and per-feature bin edges.
        """
        if not self.fitted:
            return {}
        return {
            "n_bins": self.n_bins,
            "method": self.method,
            "bin_edges": {
                f"feature_{col_idx}": edges.tolist()
                for col_idx, edges in self.bin_edges_.items()
            },
        }
