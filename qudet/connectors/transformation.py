import numpy as np
import pandas as pd
import hashlib
import json
from typing import Optional, Dict, List, Union
from datetime import datetime


class DataTransformer:
    """Transform data through normalization, scaling, and encoding."""

    def __init__(self, transformation_type: str = "normalize"):
        """
        Args:
            transformation_type: Type of transformation (normalize, scale, standardize)
        """
        self.transformation_type = transformation_type
        self.fit_params = {}

    def fit(self, data: Union[np.ndarray, pd.DataFrame]):
        """
        Fit transformer on data.
        
        Args:
            data: Training data
            
        Returns:
            self
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if self.transformation_type == "normalize":
            self.fit_params['min'] = np.min(data, axis=0)
            self.fit_params['max'] = np.max(data, axis=0)
        elif self.transformation_type == "scale":
            self.fit_params['mean'] = np.mean(data, axis=0)
            self.fit_params['std'] = np.std(data, axis=0)
        elif self.transformation_type == "standardize":
            self.fit_params['mean'] = np.mean(data, axis=0)
            self.fit_params['std'] = np.std(data, axis=0)
        
        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Apply transformation.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if not self.fit_params:
            raise ValueError("Transformer not fitted")
        
        if self.transformation_type == "normalize":
            min_val = self.fit_params['min']
            max_val = self.fit_params['max']
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0
            return (data - min_val) / range_val
        
        elif self.transformation_type == "scale":
            mean_val = self.fit_params['mean']
            std_val = self.fit_params['std']
            std_val[std_val == 0] = 1.0
            return (data - mean_val) / std_val
        
        elif self.transformation_type == "standardize":
            mean_val = self.fit_params['mean']
            std_val = self.fit_params['std']
            std_val[std_val == 0] = 1.0
            return (data - mean_val) / std_val
        
        return data

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


class DataMetadataTracker:
    """Track metadata for data provenance and versioning."""

    def __init__(self, source_name: str):
        """
        Args:
            source_name: Name of data source
        """
        self.source_name = source_name
        self.metadata = {
            'source': source_name,
            'created': datetime.now().isoformat(),
            'versions': [],
            'checksums': {},
            'transformations': []
        }

    def record_load(self, file_path: str, n_records: int, n_features: int):
        """Record data loading."""
        self.metadata['versions'].append({
            'timestamp': datetime.now().isoformat(),
            'file': file_path,
            'n_records': n_records,
            'n_features': n_features,
            'operation': 'load'
        })

    def record_transformation(self, transform_type: str, params: Dict):
        """Record transformation."""
        self.metadata['transformations'].append({
            'timestamp': datetime.now().isoformat(),
            'type': transform_type,
            'params': params
        })

    def compute_checksum(self, data: Union[np.ndarray, pd.DataFrame]) -> str:
        """
        Compute checksum for data integrity.
        
        Args:
            data: Data to checksum
            
        Returns:
            Hex checksum
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        data_bytes = data.tobytes()
        checksum = hashlib.sha256(data_bytes).hexdigest()
        
        self.metadata['checksums'][datetime.now().isoformat()] = checksum
        return checksum

    def get_metadata(self) -> Dict:
        """Get all metadata."""
        return self.metadata

    def get_lineage(self) -> List[Dict]:
        """Get data lineage."""
        return self.metadata['versions']


class DataQualityChecker:
    """Check data quality metrics."""

    def __init__(self, min_completeness: float = 0.95,
                 max_outlier_ratio: float = 0.05):
        """
        Args:
            min_completeness: Minimum fraction of non-null values
            max_outlier_ratio: Maximum fraction of outliers allowed
        """
        self.min_completeness = min_completeness
        self.max_outlier_ratio = max_outlier_ratio
        self.quality_report = {}

    def check_quality(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """
        Check overall data quality.
        
        Args:
            data: Data to check
            
        Returns:
            True if quality is acceptable
        """
        self.quality_report = {}
        
        is_complete = self._check_completeness(data)
        is_valid = self._check_validity(data)
        is_distribution_ok = self._check_distribution(data)
        
        self.quality_report['complete'] = is_complete
        self.quality_report['valid'] = is_valid
        self.quality_report['distribution_ok'] = is_distribution_ok
        
        return is_complete and is_valid and is_distribution_ok

    def _check_completeness(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Check data completeness."""
        if isinstance(data, pd.DataFrame):
            completeness = 1.0 - data.isna().sum().sum() / (len(data) * len(data.columns))
        else:
            completeness = 1.0 - np.isnan(data).sum() / data.size
        
        self.quality_report['completeness'] = completeness
        return bool(completeness >= self.min_completeness)

    def _check_validity(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Check data validity."""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        is_finite = np.isfinite(data).all()
        self.quality_report['is_finite'] = is_finite
        return bool(is_finite)

    def _check_distribution(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Check data distribution for outliers."""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_ratio = np.sum((data < lower_bound) | (data > upper_bound)) / data.size
        
        self.quality_report['outlier_ratio'] = outlier_ratio
        return bool(outlier_ratio <= self.max_outlier_ratio)

    def get_report(self) -> Dict:
        """Get quality report."""
        return self.quality_report


class DataProfiler:
    """Generate statistical profiles of data."""

    def __init__(self):
        """Initialize data profiler."""
        self.profile = {}

    def profile_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict:
        """
        Generate data profile.
        
        Args:
            data: Data to profile
            
        Returns:
            Statistical profile
        """
        if isinstance(data, pd.DataFrame):
            columns = data.columns.tolist()
            data = data.values
        else:
            columns = [f"col_{i}" for i in range(data.shape[1])]
        
        self.profile = {
            'n_rows': len(data),
            'n_cols': data.shape[1] if len(data.shape) > 1 else 1,
            'columns': columns,
            'dtypes': [str(data.dtype)],
            'memory_usage': data.nbytes,
            'statistics': {}
        }
        
        for i, col in enumerate(columns):
            col_data = data[:, i] if len(data.shape) > 1 else data
            self.profile['statistics'][col] = {
                'mean': float(np.mean(col_data)),
                'std': float(np.std(col_data)),
                'min': float(np.min(col_data)),
                'max': float(np.max(col_data)),
                'median': float(np.median(col_data)),
                'q25': float(np.percentile(col_data, 25)),
                'q75': float(np.percentile(col_data, 75)),
                'skewness': float(self._compute_skewness(col_data)),
                'kurtosis': float(self._compute_kurtosis(col_data))
            }
        
        return self.profile

    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def get_profile(self) -> Dict:
        """Get current profile."""
        return self.profile
