"""Normalization methods for quantum feature spaces.

Provides various normalization and standardization techniques optimised
for quantum machine learning preprocessing, including L2, probability,
amplitude, angle, range, decimal, logarithmic, and power normalisations.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from qudet.core.base import BaseReducer
from qudet.core.exceptions import NotFittedError, ValidationError

logger = logging.getLogger(__name__)


class QuantumNormalizer(BaseReducer):
    """Normalises features to quantum-compatible scales.

    Quantum algorithms typically require inputs in specific ranges.
    This normaliser supports several strategies:

    * **l2** — Per-sample L2 (Euclidean) normalisation.
    * **l1** — Per-sample L1 (Manhattan) normalisation.
    * **probability** — Non-negative values summing to 1 per sample.
    * **amplitude** — Same as L2 (unit-norm complex amplitudes).
    * **angle** — Wrap values into the ``[0, 2π)`` range.

    Args:
        method: Normalisation method.
        scale_range: Target output range applied after normalisation
            (ignored for ``'angle'`` and ``'probability'``).

    Example:
        >>> norm = QuantumNormalizer(method="l2")
        >>> norm.fit(X_train).transform(X_test)
    """

    def __init__(
        self,
        method: str = "l2",
        scale_range: Tuple[float, float] = (-1, 1),
    ) -> None:
        self.method = method.lower()
        self.scale_range = scale_range
        self.norms_: Optional[np.ndarray] = None
        self.fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
    ) -> "QuantumNormalizer":
        """Learn normalisation parameters from data.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.method == "l2":
            self.norms_ = np.linalg.norm(X, axis=1, keepdims=True)
        elif self.method == "l1":
            self.norms_ = np.sum(np.abs(X), axis=1, keepdims=True)
        elif self.method in ("probability", "amplitude", "angle"):
            if self.method == "probability":
                self.norms_ = np.sum(X, axis=1, keepdims=True)
            else:
                self.norms_ = np.linalg.norm(X, axis=1, keepdims=True)
        else:
            raise ValidationError(
                f"Unknown normalisation method {self.method!r}. "
                "Choose from 'l2', 'l1', 'probability', 'amplitude', 'angle'."
            )

        self.fitted = True
        logger.info("QuantumNormalizer (method=%s) fitted", self.method)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Normalise features.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.

        Returns:
            Normalised features as ``np.ndarray``.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "QuantumNormalizer has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()

        if self.method in ("l2", "l1"):
            norms = (
                np.linalg.norm(X, axis=1, keepdims=True)
                if self.method == "l2"
                else np.sum(np.abs(X), axis=1, keepdims=True)
            )
            norms[norms == 0] = 1
            normalized = X / norms
        elif self.method == "probability":
            normalized = np.abs(X)
            sums = np.sum(normalized, axis=1, keepdims=True)
            sums[sums == 0] = 1
            normalized = normalized / sums
        elif self.method == "amplitude":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = X / norms
        elif self.method == "angle":
            normalized = np.mod(X, 2 * np.pi)
        else:
            normalized = X

        # Apply scale range when relevant
        if self.scale_range != (-1, 1) and self.method not in (
            "angle",
            "probability",
        ):
            min_val, max_val = self.scale_range
            normalized = min_val + (normalized + 1) / 2 * (max_val - min_val)

        return normalized

    def get_normalization_info(self) -> dict:
        """Return normalisation statistics.

        Returns:
            Dictionary with method, scale_range, and fitted status.
        """
        return {
            "method": self.method,
            "scale_range": self.scale_range,
            "fitted": self.fitted,
        }


class RangeNormalizer(BaseReducer):
    """Normalises features to a specific output range.

    Supported methods:

    * **minmax** — Classic min-max linear scaling.
    * **robust** — IQR-based scaling (resistant to outliers).
    * **clip** — Min-max scaling followed by hard clipping to ``[0, 1]``.
    * **sigmoid** — Soft (sigmoid) normalisation.

    Args:
        range_min: Minimum value of the output range.
        range_max: Maximum value of the output range.
        method: Normalisation method.

    Example:
        >>> rn = RangeNormalizer(range_min=-1, range_max=1, method="minmax")
        >>> rn.fit(X_train).transform(X_test)
    """

    def __init__(
        self,
        range_min: float = 0,
        range_max: float = 1,
        method: str = "minmax",
    ) -> None:
        self.range_min = range_min
        self.range_max = range_max
        self.method = method.lower()
        self.min_vals_: Optional[np.ndarray] = None
        self.max_vals_: Optional[np.ndarray] = None
        self.q1_vals_: Optional[np.ndarray] = None
        self.q3_vals_: Optional[np.ndarray] = None
        self.fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
    ) -> "RangeNormalizer":
        """Learn range parameters from training data.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.min_vals_ = np.min(X, axis=0)
        self.max_vals_ = np.max(X, axis=0)
        self.q1_vals_ = np.percentile(X, 25, axis=0)
        self.q3_vals_ = np.percentile(X, 75, axis=0)

        self.fitted = True
        logger.info("RangeNormalizer (method=%s) fitted", self.method)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Normalise features to the configured output range.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.

        Returns:
            Normalised features as ``np.ndarray``.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "RangeNormalizer has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()

        if self.method == "minmax":
            normalized = self._minmax_normalize(X)
        elif self.method == "robust":
            normalized = self._robust_normalize(X)
        elif self.method == "clip":
            normalized = self._clip_normalize(X)
        elif self.method == "sigmoid":
            normalized = self._sigmoid_normalize(X)
        else:
            normalized = self._minmax_normalize(X)

        return normalized

    def _minmax_normalize(self, X: np.ndarray) -> np.ndarray:
        """Min-max normalisation."""
        range_span = self.max_vals_ - self.min_vals_
        range_span[range_span == 0] = 1

        normalized = (X - self.min_vals_) / range_span
        return self.range_min + normalized * (self.range_max - self.range_min)

    def _robust_normalize(self, X: np.ndarray) -> np.ndarray:
        """Robust normalisation using IQR."""
        iqr = self.q3_vals_ - self.q1_vals_
        iqr[iqr == 0] = 1

        normalized = (X - self.q1_vals_) / iqr
        return self.range_min + normalized * (self.range_max - self.range_min)

    def _clip_normalize(self, X: np.ndarray) -> np.ndarray:
        """Min-max normalisation with hard clipping."""
        range_span = self.max_vals_ - self.min_vals_
        range_span[range_span == 0] = 1

        normalized = (X - self.min_vals_) / range_span
        normalized = np.clip(normalized, 0, 1)
        return self.range_min + normalized * (self.range_max - self.range_min)

    def _sigmoid_normalize(self, X: np.ndarray) -> np.ndarray:
        """Sigmoid-based soft normalisation."""
        centered = X - self.min_vals_
        range_span = self.max_vals_ - self.min_vals_
        range_span[range_span == 0] = 1

        sigmoid_val = 1 / (1 + np.exp(-centered / range_span))
        return self.range_min + sigmoid_val * (self.range_max - self.range_min)

    def get_range_info(self) -> dict:
        """Return range normalisation statistics.

        Returns:
            Dictionary with output range, method, and learned input bounds.
        """
        if not self.fitted:
            return {}
        return {
            "output_range": (self.range_min, self.range_max),
            "method": self.method,
            "input_min": (
                self.min_vals_.tolist() if self.min_vals_ is not None else None
            ),
            "input_max": (
                self.max_vals_.tolist() if self.max_vals_ is not None else None
            ),
        }


class DecimalScaler(BaseReducer):
    """Scales features by moving the decimal point (power-of-10 scaling).

    For each feature, the maximum absolute value is used to compute a
    decimal scaling factor so that all values fall within ``[-1, 1]``.

    Example:
        >>> ds = DecimalScaler()
        >>> ds.fit(X_train).transform(X_test)
    """

    def __init__(self) -> None:
        self.scale_factors_: Optional[np.ndarray] = None
        self.fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
    ) -> "DecimalScaler":
        """Learn decimal scale factors.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.scale_factors_ = np.zeros(X.shape[1])

        for col_idx in range(X.shape[1]):
            max_abs = np.max(np.abs(X[:, col_idx]))
            if max_abs > 0:
                self.scale_factors_[col_idx] = np.ceil(np.log10(max_abs))
            else:
                self.scale_factors_[col_idx] = 0

        self.fitted = True
        logger.info("DecimalScaler fitted with %d features", X.shape[1])
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply decimal scaling.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.

        Returns:
            Decimal-scaled features as ``np.ndarray``.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "DecimalScaler has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()

        scaled = np.zeros_like(X, dtype=float)
        for col_idx in range(X.shape[1]):
            scaled[:, col_idx] = X[:, col_idx] / (
                10 ** self.scale_factors_[col_idx]
            )

        return scaled

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the decimal scaling.

        Args:
            X: Scaled features.

        Returns:
            Original-scale features.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "DecimalScaler has not been fitted. Call fit() first."
            )

        scaled = np.zeros_like(X, dtype=float)
        for col_idx in range(X.shape[1]):
            scaled[:, col_idx] = X[:, col_idx] * (
                10 ** self.scale_factors_[col_idx]
            )

        return scaled

    def get_scale_factors(self) -> np.ndarray:
        """Return the per-feature scale factors.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "DecimalScaler has not been fitted. Call fit() first."
            )
        return self.scale_factors_.copy()


class LogTransformer(BaseReducer):
    """Applies logarithmic transformations to features.

    Useful for right-skewed distributions where a log transform
    can approximate normality.

    Supported methods:

    * **natural** — ``ln(x + shift)``
    * **log10** — ``log₁₀(x + shift)``
    * **log2** — ``log₂(x + shift)``

    Args:
        method: Log base to use.
        shift: Small constant added before taking the log to handle
            zero or negative values.

    Example:
        >>> lt = LogTransformer(method="natural")
        >>> lt.fit(X_train).transform(X_test)
    """

    def __init__(self, method: str = "natural", shift: float = 1e-10) -> None:
        self.method = method.lower()
        self.shift = shift
        self.fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
    ) -> "LogTransformer":
        """Fit the log transformer (learn adaptive shift if needed).

        Args:
            X: Input features of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if np.any(X <= 0):
            if self.shift <= 0:
                nonzero = X[X != 0]
                self.shift = (
                    float(np.min(np.abs(nonzero)) / 100) if len(nonzero) > 0 else 1e-10
                )

        self.fitted = True
        logger.info("LogTransformer (method=%s) fitted", self.method)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply the log transformation.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.

        Returns:
            Log-transformed features.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "LogTransformer has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()

        X_shifted = X + self.shift

        if self.method == "natural":
            transformed = np.log(X_shifted)
        elif self.method == "log10":
            transformed = np.log10(X_shifted)
        elif self.method == "log2":
            transformed = np.log2(X_shifted)
        else:
            transformed = np.log(X_shifted)

        return transformed

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the log transformation.

        Args:
            X: Log-transformed features.

        Returns:
            Original-scale features.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "LogTransformer has not been fitted. Call fit() first."
            )

        if self.method == "natural":
            original = np.exp(X)
        elif self.method == "log10":
            original = 10 ** X
        elif self.method == "log2":
            original = 2 ** X
        else:
            original = np.exp(X)

        return original - self.shift

    def get_transform_info(self) -> dict:
        """Return transformation metadata.

        Returns:
            Dictionary with method, shift, and fitted status.
        """
        return {
            "method": self.method,
            "shift": self.shift,
            "fitted": self.fitted,
        }


class PowerTransformer(BaseReducer):
    """Applies power transformations to normalise distributions.

    Raises each element to the configured ``power`` exponent, preserving
    sign (``sign(x) × |x|^power``).  A power of 0 falls back to a log
    transformation.

    Args:
        power: Exponent for the power transformation (0.5 = sqrt, 2 = square,
            etc.).

    Example:
        >>> pt = PowerTransformer(power=0.5)
        >>> pt.fit(X_train).transform(X_test)
    """

    def __init__(self, power: float = 0.5) -> None:
        self.power = power
        self.fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
    ) -> "PowerTransformer":
        """Fit the power transformer (validate input data).

        Args:
            X: Input features of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self

        Raises:
            ValidationError: If data contains non-positive values with
                ``power > 0``.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.power > 0 and np.any(X <= 0):
            raise ValidationError(
                "Power transformation requires positive values for power > 0. "
                f"Found non-positive values with power={self.power}."
            )

        self.fitted = True
        logger.info("PowerTransformer (power=%.3f) fitted", self.power)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply the power transformation.

        Args:
            X: Input features of shape ``(n_samples, n_features)``.

        Returns:
            Power-transformed features.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "PowerTransformer has not been fitted. Call fit() first."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = X.copy()

        if self.power == 0:
            return np.log(np.abs(X) + 1e-10)
        else:
            sign = np.sign(X)
            return sign * np.abs(X) ** self.power

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the power transformation.

        Args:
            X: Power-transformed features.

        Returns:
            Original-scale features.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.fitted:
            raise NotFittedError(
                "PowerTransformer has not been fitted. Call fit() first."
            )

        if self.power == 0:
            return np.exp(X)
        else:
            sign = np.sign(X)
            return sign * np.abs(X) ** (1.0 / self.power)

    def get_power_info(self) -> dict:
        """Return power transformation metadata.

        Returns:
            Dictionary with power exponent and fitted status.
        """
        return {"power": self.power, "fitted": self.fitted}
