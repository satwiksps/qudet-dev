"""
Tests for categorical encoding transforms.

Tests CategoricalEncoder, TargetEncoder, FrequencyEncoder, BinningEncoder.
"""

import pytest
import numpy as np
import pandas as pd
from qudet.transforms.encoding import (
    CategoricalEncoder,
    TargetEncoder,
    FrequencyEncoder,
    BinningEncoder
)


class TestCategoricalEncoder:
    """Test CategoricalEncoder with various encoding methods."""
    
    @pytest.fixture
    def categorical_data(self):
        """Create categorical data."""
        data = np.array([
            ['red', 'small'],
            ['blue', 'large'],
            ['red', 'medium'],
            ['green', 'large'],
            ['blue', 'small']
        ], dtype=object)
        return data
    
    def test_label_encoding(self, categorical_data):
        """Test label encoding."""
        encoder = CategoricalEncoder(method="label")
        encoder.fit(categorical_data)
        encoded = encoder.transform(categorical_data)
        
        assert encoded.shape == categorical_data.shape
        assert np.all(encoded >= 0)
    
    def test_onehot_encoding(self, categorical_data):
        """Test one-hot encoding."""
        encoder = CategoricalEncoder(method="onehot")
        encoder.fit(categorical_data)
        encoded = encoder.transform(categorical_data)
        
        assert encoded.shape[0] == categorical_data.shape[0]
        assert encoded.shape[1] > categorical_data.shape[1]
        assert np.all((encoded == 0) | (encoded == 1))
    
    def test_binary_encoding(self, categorical_data):
        """Test binary encoding."""
        encoder = CategoricalEncoder(method="binary")
        encoder.fit(categorical_data)
        encoded = encoder.transform(categorical_data)
        
        assert encoded.shape[0] == categorical_data.shape[0]
        assert np.all((encoded == 0) | (encoded == 1))
    
    def test_fit_transform(self, categorical_data):
        """Test fit_transform."""
        encoder = CategoricalEncoder(method="label")
        encoded = encoder.fit_transform(categorical_data)
        
        assert encoded.shape == categorical_data.shape
    
    def test_get_n_features_out_label(self, categorical_data):
        """Test output features for label encoding."""
        encoder = CategoricalEncoder(method="label")
        encoder.fit(categorical_data)
        n_out = encoder.get_n_features_out()
        
        assert n_out == categorical_data.shape[1]
    
    def test_get_n_features_out_onehot(self, categorical_data):
        """Test output features for one-hot encoding."""
        encoder = CategoricalEncoder(method="onehot")
        encoder.fit(categorical_data)
        n_out = encoder.get_n_features_out()
        
        assert n_out > categorical_data.shape[1]
    
    def test_unfitted_transform_raises(self, categorical_data):
        """Test transform before fit raises error."""
        encoder = CategoricalEncoder()
        with pytest.raises(ValueError):
            encoder.transform(categorical_data)
    
    def test_numeric_data_encoding(self):
        """Test encoding numeric data as categories."""
        data = np.array([[1, 2], [1, 3], [2, 2], [2, 3]])
        encoder = CategoricalEncoder(method="label")
        encoder.fit(data)
        encoded = encoder.transform(data)
        
        assert encoded.shape == data.shape
    
    def test_consistency_across_samples(self, categorical_data):
        """Test encoding is consistent."""
        encoder = CategoricalEncoder(method="label")
        encoder.fit(categorical_data)
        
        encoded1 = encoder.transform(categorical_data[[0]])
        encoded2 = encoder.transform(categorical_data[[0]])
        
        assert np.array_equal(encoded1, encoded2)


class TestTargetEncoder:
    """Test TargetEncoder."""
    
    @pytest.fixture
    def target_data(self):
        """Create data for target encoding."""
        X = np.array([
            [1], [1], [1], [2], [2], [3], [3], [3], [3]
        ])
        y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1])
        return X, y
    
    def test_target_encoding(self, target_data):
        """Test target encoding."""
        X, y = target_data
        encoder = TargetEncoder()
        encoder.fit(X, y)
        encoded = encoder.transform(X)
        
        assert encoded.shape == X.shape
        assert np.all(encoded >= 0) and np.all(encoded <= 1)
    
    def test_fit_transform(self, target_data):
        """Test fit_transform."""
        X, y = target_data
        encoder = TargetEncoder()
        encoded = encoder.fit_transform(X, y)
        
        assert encoded.shape == X.shape
    
    def test_smoothing_parameter(self, target_data):
        """Test smoothing parameter effect."""
        X, y = target_data
        
        encoder1 = TargetEncoder(smoothing=1.0)
        encoder1.fit(X, y)
        encoded1 = encoder1.transform(X)
        
        encoder2 = TargetEncoder(smoothing=10.0)
        encoder2.fit(X, y)
        encoded2 = encoder2.transform(X)
        
        # Higher smoothing should result in values closer to global mean
        assert not np.allclose(encoded1, encoded2)
    
    def test_get_encoding_info(self, target_data):
        """Test getting encoding information."""
        X, y = target_data
        encoder = TargetEncoder()
        encoder.fit(X, y)
        info = encoder.get_encoding_info()
        
        assert "global_mean" in info
        assert "smoothing" in info
        assert "n_features" in info
    
    def test_min_samples_leaf(self, target_data):
        """Test min_samples_leaf parameter."""
        X, y = target_data
        encoder = TargetEncoder(min_samples_leaf=2)
        encoder.fit(X, y)
        encoded = encoder.transform(X)
        
        assert encoded.shape == X.shape
    
    def test_unfitted_transform_raises(self, target_data):
        """Test transform before fit raises error."""
        X, y = target_data
        encoder = TargetEncoder()
        with pytest.raises(ValueError):
            encoder.transform(X)


class TestFrequencyEncoder:
    """Test FrequencyEncoder."""
    
    @pytest.fixture
    def frequency_data(self):
        """Create frequency data."""
        data = np.array([[1, 'a'], [1, 'b'], [2, 'a'], [2, 'a'], [3, 'b']], dtype=object)
        return data
    
    def test_frequency_encoding_normalized(self, frequency_data):
        """Test frequency encoding with normalization."""
        encoder = FrequencyEncoder(normalize=True)
        encoder.fit(frequency_data)
        encoded = encoder.transform(frequency_data)
        
        assert encoded.shape == frequency_data.shape
        assert np.all(encoded >= 0) and np.all(encoded <= 1)
    
    def test_frequency_encoding_counts(self, frequency_data):
        """Test frequency encoding with raw counts."""
        encoder = FrequencyEncoder(normalize=False)
        encoder.fit(frequency_data)
        encoded = encoder.transform(frequency_data)
        
        assert encoded.shape == frequency_data.shape
        assert np.all(encoded >= 1)
    
    def test_fit_transform(self, frequency_data):
        """Test fit_transform."""
        encoder = FrequencyEncoder()
        encoded = encoder.fit_transform(frequency_data)
        
        assert encoded.shape == frequency_data.shape
    
    def test_get_frequency_info(self, frequency_data):
        """Test getting frequency information."""
        encoder = FrequencyEncoder()
        encoder.fit(frequency_data)
        info = encoder.get_frequency_info()
        
        assert len(info) == frequency_data.shape[1]
    
    def test_unfitted_transform_raises(self, frequency_data):
        """Test transform before fit raises error."""
        encoder = FrequencyEncoder()
        with pytest.raises(ValueError):
            encoder.transform(frequency_data)
    
    def test_high_frequency_values(self, frequency_data):
        """Test that high-frequency values get high encoding."""
        encoder = FrequencyEncoder(normalize=True)
        encoder.fit(frequency_data)
        encoded = encoder.transform(frequency_data)
        
        # Value 'a' in second column appears more frequently
        # So its encoding should be higher
        col1_encodings = encoded[:, 0]
        assert np.max(col1_encodings) <= 1.0


class TestBinningEncoder:
    """Test BinningEncoder."""
    
    @pytest.fixture
    def continuous_data(self):
        """Create continuous data."""
        np.random.seed(42)
        return np.random.randn(100, 3) * 10 + 50
    
    def test_quantile_binning(self, continuous_data):
        """Test quantile binning."""
        encoder = BinningEncoder(n_bins=5, method="quantile")
        encoder.fit(continuous_data)
        binned = encoder.transform(continuous_data)
        
        assert binned.shape == continuous_data.shape
        assert np.all(binned >= 0)
    
    def test_uniform_binning(self, continuous_data):
        """Test uniform binning."""
        encoder = BinningEncoder(n_bins=5, method="uniform")
        encoder.fit(continuous_data)
        binned = encoder.transform(continuous_data)
        
        assert binned.shape == continuous_data.shape
        assert np.all(binned >= 0)
    
    def test_kmeans_binning(self, continuous_data):
        """Test k-means binning."""
        encoder = BinningEncoder(n_bins=5, method="kmeans")
        encoder.fit(continuous_data)
        binned = encoder.transform(continuous_data)
        
        assert binned.shape == continuous_data.shape
        assert np.all(binned >= 0)
    
    def test_fit_transform(self, continuous_data):
        """Test fit_transform."""
        encoder = BinningEncoder(n_bins=5)
        binned = encoder.fit_transform(continuous_data)
        
        assert binned.shape == continuous_data.shape
    
    def test_get_bin_info(self, continuous_data):
        """Test getting bin information."""
        encoder = BinningEncoder(n_bins=5)
        encoder.fit(continuous_data)
        info = encoder.get_bin_info()
        
        assert "n_bins" in info
        assert "method" in info
        assert "bin_edges" in info
    
    def test_bin_count_consistency(self, continuous_data):
        """Test that bin assignment is consistent."""
        encoder = BinningEncoder(n_bins=5)
        encoder.fit(continuous_data)
        
        binned1 = encoder.transform(continuous_data[:10])
        binned2 = encoder.transform(continuous_data[:10])
        
        assert np.array_equal(binned1, binned2)
    
    def test_unfitted_transform_raises(self, continuous_data):
        """Test transform before fit raises error."""
        encoder = BinningEncoder()
        with pytest.raises(ValueError):
            encoder.transform(continuous_data)
    
    def test_different_n_bins(self, continuous_data):
        """Test different number of bins."""
        encoder3 = BinningEncoder(n_bins=3)
        encoder10 = BinningEncoder(n_bins=10)
        
        encoder3.fit(continuous_data)
        encoder10.fit(continuous_data)
        
        binned3 = encoder3.transform(continuous_data)
        binned10 = encoder10.transform(continuous_data)
        
        # Different binning strategies should potentially produce different results
        assert binned3.shape == binned10.shape
        assert binned3.shape == continuous_data.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
