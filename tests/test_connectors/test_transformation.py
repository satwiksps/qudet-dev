import pytest
import numpy as np
import pandas as pd
from qudet.connectors.transformation import (
    DataTransformer,
    DataMetadataTracker,
    DataQualityChecker,
    DataProfiler
)


class TestDataTransformer:
    """Test suite for Data Transformer."""

    def test_initialization(self):
        """Test transformer initialization."""
        transformer = DataTransformer(transformation_type="normalize")
        assert transformer.transformation_type == "normalize"

    def test_fit_normalize(self):
        """Test fitting normalization."""
        transformer = DataTransformer(transformation_type="normalize")
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        transformer.fit(data)
        
        assert 'min' in transformer.fit_params
        assert 'max' in transformer.fit_params

    def test_normalize_transform(self):
        """Test normalization transformation."""
        transformer = DataTransformer(transformation_type="normalize")
        data = np.array([[0, 0], [10, 10]])
        
        transformer.fit(data)
        result = transformer.transform(data)
        
        assert np.allclose(result[0], [0, 0])
        assert np.allclose(result[1], [1, 1])

    def test_standardize_transform(self):
        """Test standardization transformation."""
        transformer = DataTransformer(transformation_type="standardize")
        data = np.array([[1, 2], [2, 3], [3, 4]])
        
        transformer.fit(data)
        result = transformer.transform(data)
        
        assert np.isclose(np.mean(result), 0, atol=1e-6)

    def test_fit_transform(self):
        """Test fit_transform method."""
        transformer = DataTransformer(transformation_type="normalize")
        data = np.array([[0, 10], [5, 5], [10, 0]])
        
        result = transformer.fit_transform(data)
        
        assert result.shape == data.shape

    def test_transform_without_fit_raises_error(self):
        """Test transform without fit raises error."""
        transformer = DataTransformer()
        data = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError):
            transformer.transform(data)

    def test_dataframe_support(self):
        """Test transformer works with DataFrames."""
        transformer = DataTransformer(transformation_type="normalize")
        df = pd.DataFrame({'a': [0, 10], 'b': [0, 10]})
        
        result = transformer.fit_transform(df)
        
        assert result.shape == (2, 2)


class TestDataMetadataTracker:
    """Test suite for Data Metadata Tracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = DataMetadataTracker(source_name="test_source")
        assert tracker.source_name == "test_source"

    def test_record_load(self):
        """Test recording data load."""
        tracker = DataMetadataTracker("source")
        tracker.record_load("file.csv", 100, 5)
        
        assert len(tracker.metadata['versions']) == 1
        assert tracker.metadata['versions'][0]['n_records'] == 100

    def test_record_transformation(self):
        """Test recording transformation."""
        tracker = DataMetadataTracker("source")
        tracker.record_transformation("normalize", {"scale": 1.0})
        
        assert len(tracker.metadata['transformations']) == 1

    def test_compute_checksum(self):
        """Test checksum computation."""
        tracker = DataMetadataTracker("source")
        data = np.array([[1, 2], [3, 4]])
        
        checksum = tracker.compute_checksum(data)
        
        assert isinstance(checksum, str)
        assert len(checksum) > 0

    def test_get_metadata(self):
        """Test getting metadata."""
        tracker = DataMetadataTracker("source")
        tracker.record_load("file.csv", 50, 3)
        
        metadata = tracker.get_metadata()
        
        assert metadata['source'] == "source"
        assert len(metadata['versions']) > 0

    def test_get_lineage(self):
        """Test getting data lineage."""
        tracker = DataMetadataTracker("source")
        tracker.record_load("file1.csv", 50, 3)
        tracker.record_load("file2.csv", 100, 3)
        
        lineage = tracker.get_lineage()
        
        assert len(lineage) == 2


class TestDataQualityChecker:
    """Test suite for Data Quality Checker."""

    def test_initialization(self):
        """Test quality checker initialization."""
        checker = DataQualityChecker(min_completeness=0.9)
        assert checker.min_completeness == 0.9

    def test_check_quality_clean_data(self):
        """Test quality check on clean data."""
        checker = DataQualityChecker()
        data = np.random.rand(100, 5)
        
        is_quality = checker.check_quality(data)
        
        assert is_quality == True

    def test_completeness_check(self):
        """Test completeness checking."""
        checker = DataQualityChecker(min_completeness=0.9)
        df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [5, 6, 7, 8]})
        
        checker._check_completeness(df)
        
        assert 'completeness' in checker.quality_report

    def test_validity_check(self):
        """Test validity checking."""
        checker = DataQualityChecker()
        data = np.array([1, 2, np.inf, 4])
        
        is_valid = checker._check_validity(data)
        
        assert is_valid == False

    def test_distribution_check(self):
        """Test distribution checking."""
        checker = DataQualityChecker(max_outlier_ratio=0.3)
        np.random.seed(42)
        data = np.random.randn(100, 2)
        
        is_ok = checker._check_distribution(data)
        
        assert isinstance(is_ok, bool)

    def test_get_report(self):
        """Test getting quality report."""
        checker = DataQualityChecker()
        data = np.random.rand(50, 3)
        
        checker.check_quality(data)
        report = checker.get_report()
        
        assert 'complete' in report
        assert 'valid' in report


class TestDataProfiler:
    """Test suite for Data Profiler."""

    def test_initialization(self):
        """Test profiler initialization."""
        profiler = DataProfiler()
        assert profiler.profile == {}

    def test_profile_numpy_array(self):
        """Test profiling numpy array."""
        profiler = DataProfiler()
        data = np.random.rand(100, 5)
        
        profile = profiler.profile_data(data)
        
        assert profile['n_rows'] == 100
        assert profile['n_cols'] == 5

    def test_profile_dataframe(self):
        """Test profiling DataFrame."""
        profiler = DataProfiler()
        df = pd.DataFrame({'a': np.random.rand(50), 'b': np.random.rand(50)})
        
        profile = profiler.profile_data(df)
        
        assert profile['n_rows'] == 50
        assert 'a' in profile['statistics']
        assert 'b' in profile['statistics']

    def test_profile_statistics(self):
        """Test profile statistics."""
        profiler = DataProfiler()
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        profile = profiler.profile_data(data)
        
        assert 'mean' in profile['statistics']['col_0']
        assert 'std' in profile['statistics']['col_0']
        assert 'min' in profile['statistics']['col_0']
        assert 'max' in profile['statistics']['col_0']

    def test_compute_skewness(self):
        """Test skewness computation."""
        profiler = DataProfiler()
        data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
        
        skewness = profiler._compute_skewness(data)
        
        assert isinstance(skewness, float)

    def test_compute_kurtosis(self):
        """Test kurtosis computation."""
        profiler = DataProfiler()
        data = np.random.randn(100)
        
        kurtosis = profiler._compute_kurtosis(data)
        
        assert isinstance(kurtosis, float)

    def test_get_profile(self):
        """Test getting profile."""
        profiler = DataProfiler()
        data = np.random.rand(30, 3)
        profiler.profile_data(data)
        
        profile = profiler.get_profile()
        
        assert len(profile) > 0
        assert 'n_rows' in profile
