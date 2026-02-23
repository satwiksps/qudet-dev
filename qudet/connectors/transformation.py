"""
Data transformation, metadata tracking, quality checking, and profiling.

Provides reusable data-processing primitives that sit between raw data
ingestion and quantum encoding.
"""

import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


class DataTransformer:
    """Transform data through normalization, min-max scaling, or z-score standardization.

    Supported ``transformation_type`` values:

    * ``"normalize"`` — min-max normalization to [0, 1].
    * ``"scale"`` — min-max scaling (same formula as normalize; alias).
    * ``"standardize"`` — z-score standardization (zero mean, unit variance).

    Args:
        transformation_type: Transformation strategy.

    Raises:
        ValueError: If *transformation_type* is not one of the supported values.
    """

    _VALID_TYPES = {"normalize", "scale", "standardize"}

    def __init__(self, transformation_type: str = "normalize") -> None:
        if transformation_type not in self._VALID_TYPES:
            raise ValueError(
                f"Unknown transformation_type {transformation_type!r}. "
                f"Choose from {sorted(self._VALID_TYPES)}."
            )
        self.transformation_type = transformation_type
        self.fit_params: Dict[str, np.ndarray] = {}

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> "DataTransformer":
        """Fit the transformer on *data*.

        Args:
            data: Training data (2-D array or ``DataFrame``).

        Returns:
            ``self`` for method chaining.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        if self.transformation_type in ("normalize", "scale"):
            self.fit_params["min"] = np.min(data, axis=0)
            self.fit_params["max"] = np.max(data, axis=0)
        elif self.transformation_type == "standardize":
            self.fit_params["mean"] = np.mean(data, axis=0)
            self.fit_params["std"] = np.std(data, axis=0)

        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply the fitted transformation.

        Args:
            data: Data to transform.

        Returns:
            Transformed data as a ``numpy.ndarray``.

        Raises:
            ValueError: If the transformer has not been fitted.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        if not self.fit_params:
            raise ValueError("Transformer not fitted. Call fit() first.")

        if self.transformation_type in ("normalize", "scale"):
            min_val = self.fit_params["min"].copy()
            max_val = self.fit_params["max"].copy()
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0
            return (data - min_val) / range_val

        elif self.transformation_type == "standardize":
            mean_val = self.fit_params["mean"].copy()
            std_val = self.fit_params["std"].copy()
            std_val[std_val == 0] = 1.0
            return (data - mean_val) / std_val

        return data  # pragma: no cover

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


class DataMetadataTracker:
    """Track metadata for data provenance and versioning.

    Args:
        source_name: Name of the data source.
    """

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name
        self.metadata: Dict = {
            "source": source_name,
            "created": datetime.now().isoformat(),
            "versions": [],
            "checksums": {},
            "transformations": [],
        }

    def record_load(self, file_path: str, n_records: int, n_features: int) -> None:
        """Record a data-loading event.

        Args:
            file_path: Path of the loaded file.
            n_records: Number of records loaded.
            n_features: Number of features per record.
        """
        self.metadata["versions"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "file": file_path,
                "n_records": n_records,
                "n_features": n_features,
                "operation": "load",
            }
        )

    def record_transformation(self, transform_type: str, params: Dict) -> None:
        """Record a transformation event.

        Args:
            transform_type: Kind of transformation applied.
            params: Parameters used.
        """
        self.metadata["transformations"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "type": transform_type,
                "params": params,
            }
        )

    def compute_checksum(self, data: Union[np.ndarray, pd.DataFrame]) -> str:
        """Compute a SHA-256 checksum of *data*.

        Args:
            data: Data to checksum.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        data_bytes = data.tobytes()
        checksum = hashlib.sha256(data_bytes).hexdigest()

        self.metadata["checksums"][datetime.now().isoformat()] = checksum
        return checksum

    def get_metadata(self) -> Dict:
        """Return all recorded metadata."""
        return self.metadata

    def get_lineage(self) -> List[Dict]:
        """Return the data-loading lineage (version history)."""
        return self.metadata["versions"]


class DataQualityChecker:
    """Check data quality metrics (completeness, validity, outliers).

    Args:
        min_completeness: Minimum fraction of non-null values required.
        max_outlier_ratio: Maximum fraction of outliers allowed.
    """

    def __init__(
        self,
        min_completeness: float = 0.95,
        max_outlier_ratio: float = 0.05,
    ) -> None:
        self.min_completeness = min_completeness
        self.max_outlier_ratio = max_outlier_ratio
        self.quality_report: Dict = {}

    def check_quality(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Run all quality checks.

        Args:
            data: Data to check.

        Returns:
            ``True`` if all checks pass.
        """
        self.quality_report = {}

        is_complete = self._check_completeness(data)
        is_valid = self._check_validity(data)
        is_distribution_ok = self._check_distribution(data)

        self.quality_report["complete"] = is_complete
        self.quality_report["valid"] = is_valid
        self.quality_report["distribution_ok"] = is_distribution_ok

        return is_complete and is_valid and is_distribution_ok

    def _check_completeness(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Check data completeness (fraction of non-null values)."""
        if isinstance(data, pd.DataFrame):
            completeness = 1.0 - data.isna().sum().sum() / (
                len(data) * len(data.columns)
            )
        else:
            completeness = 1.0 - np.isnan(data).sum() / data.size

        self.quality_report["completeness"] = completeness
        return bool(completeness >= self.min_completeness)

    def _check_validity(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Check that all values are finite."""
        if isinstance(data, pd.DataFrame):
            data = data.values

        is_finite = np.isfinite(data).all()
        self.quality_report["is_finite"] = is_finite
        return bool(is_finite)

    def _check_distribution(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Check data distribution for outliers (IQR method)."""
        if isinstance(data, pd.DataFrame):
            data = data.values

        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_ratio = np.sum((data < lower_bound) | (data > upper_bound)) / data.size

        self.quality_report["outlier_ratio"] = outlier_ratio
        return bool(outlier_ratio <= self.max_outlier_ratio)

    def get_report(self) -> Dict:
        """Return the most recent quality report."""
        return self.quality_report


class DataProfiler:
    """Generate statistical profiles of data."""

    def __init__(self) -> None:
        self.profile: Dict = {}

    def profile_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict:
        """Generate a full statistical profile.

        Args:
            data: Data to profile.

        Returns:
            Dictionary containing row/column counts, per-column statistics
            (mean, std, min, max, median, quartiles, skewness, kurtosis).
        """
        if isinstance(data, pd.DataFrame):
            columns = data.columns.tolist()
            data = data.values
        else:
            columns = [f"col_{i}" for i in range(data.shape[1])]

        self.profile = {
            "n_rows": len(data),
            "n_cols": data.shape[1] if len(data.shape) > 1 else 1,
            "columns": columns,
            "dtypes": [str(data.dtype)],
            "memory_usage": data.nbytes,
            "statistics": {},
        }

        for i, col in enumerate(columns):
            col_data = data[:, i] if len(data.shape) > 1 else data
            self.profile["statistics"][col] = {
                "mean": float(np.mean(col_data)),
                "std": float(np.std(col_data)),
                "min": float(np.min(col_data)),
                "max": float(np.max(col_data)),
                "median": float(np.median(col_data)),
                "q25": float(np.percentile(col_data, 25)),
                "q75": float(np.percentile(col_data, 75)),
                "skewness": float(self._compute_skewness(col_data)),
                "kurtosis": float(self._compute_kurtosis(col_data)),
            }

        return self.profile

    @staticmethod
    def _compute_skewness(data: np.ndarray) -> float:
        """Compute sample skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    @staticmethod
    def _compute_kurtosis(data: np.ndarray) -> float:
        """Compute excess kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)

    def get_profile(self) -> Dict:
        """Return the most recently computed profile."""
        return self.profile
