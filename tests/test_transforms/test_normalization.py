"""
Tests for normalization transforms.

Tests QuantumNormalizer, RangeNormalizer, DecimalScaler, LogTransformer, PowerTransformer.
"""

import pytest
import numpy as np
import pandas as pd
from qudet.transforms.normalization import (
    QuantumNormalizer,
    RangeNormalizer,
    DecimalScaler,
    LogTransformer,
    PowerTransformer
)


class TestQuantumNormalizer:
    """Test QuantumNormalizer for quantum-compatible normalization."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return np.random.randn(20, 5) * 10
    
    def test_l2_normalization(self, sample_data):
        """Test L2 normalization."""
        normalizer = QuantumNormalizer(method="l2")
        normalizer.fit(sample_data)
        normalized = normalizer.transform(sample_data)
        
        # Check L2 norm per sample is 1
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1, atol=1e-10)
    
    def test_l1_normalization(self, sample_data):
        """Test L1 normalization."""
        normalizer = QuantumNormalizer(method="l1")
        normalizer.fit(sample_data)
        normalized = normalizer.transform(sample_data)
        
        # Check L1 norm per sample is 1
        l1_norms = np.sum(np.abs(normalized), axis=1)
        assert np.allclose(l1_norms, 1, atol=1e-10)
    
    def test_probability_normalization(self, sample_data):
        """Test probability normalization."""
        # Use positive data for probability
        pos_data = np.abs(sample_data)
        normalizer = QuantumNormalizer(method="probability")
        normalizer.fit(pos_data)
        normalized = normalizer.transform(pos_data)
        
        # Check sums to 1 per sample
        sums = np.sum(normalized, axis=1)
        assert np.allclose(sums, 1, atol=1e-10)
    
    def test_amplitude_normalization(self, sample_data):
        """Test amplitude normalization."""
        normalizer = QuantumNormalizer(method="amplitude")
        normalizer.fit(sample_data)
        normalized = normalizer.transform(sample_data)
        
        # Check norm per sample is 1
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1, atol=1e-10)
    
    def test_angle_normalization(self, sample_data):
        """Test angle normalization."""
        normalizer = QuantumNormalizer(method="angle")
        normalizer.fit(sample_data)
        normalized = normalizer.transform(sample_data)
        
        # Check angles are in [0, 2π]
        assert np.all(normalized >= 0)
        assert np.all(normalized < 2 * np.pi)
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform."""
        normalizer = QuantumNormalizer(method="l2")
        normalized = normalizer.fit_transform(sample_data)
        
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1, atol=1e-10)
    
    def test_get_normalization_info(self, sample_data):
        """Test getting normalization info."""
        normalizer = QuantumNormalizer(method="l2")
        normalizer.fit(sample_data)
        info = normalizer.get_normalization_info()
        
        assert "method" in info
        assert "scale_range" in info
        assert "fitted" in info
    
    def test_unfitted_transform_raises(self, sample_data):
        """Test transform before fit raises error."""
        normalizer = QuantumNormalizer()
        with pytest.raises(ValueError):
            normalizer.transform(sample_data)


class TestRangeNormalizer:
    """Test RangeNormalizer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return np.random.randn(20, 5) * 100 + 500
    
    def test_minmax_normalization(self, sample_data):
        """Test min-max normalization."""
        normalizer = RangeNormalizer(range_min=0, range_max=1, method="minmax")
        normalizer.fit(sample_data)
        normalized = normalizer.transform(sample_data)
        
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
    
    def test_robust_normalization(self, sample_data):
        """Test robust normalization."""
        normalizer = RangeNormalizer(method="robust")
        normalizer.fit(sample_data)
        normalized = normalizer.transform(sample_data)
        
        assert normalized.shape == sample_data.shape
    
    def test_clip_normalization(self, sample_data):
        """Test clip normalization."""
        normalizer = RangeNormalizer(method="clip")
        normalizer.fit(sample_data)
        normalized = normalizer.transform(sample_data)
        
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
    
    def test_sigmoid_normalization(self, sample_data):
        """Test sigmoid normalization."""
        normalizer = RangeNormalizer(method="sigmoid")
        normalizer.fit(sample_data)
        normalized = normalizer.transform(sample_data)
        
        assert normalized.shape == sample_data.shape
    
    def test_custom_range(self, sample_data):
        """Test custom range."""
        normalizer = RangeNormalizer(range_min=-1, range_max=1)
        normalizer.fit(sample_data)
        normalized = normalizer.transform(sample_data)
        
        assert np.all(normalized >= -1)
        assert np.all(normalized <= 1)
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform."""
        normalizer = RangeNormalizer()
        normalized = normalizer.fit_transform(sample_data)
        
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
    
    def test_get_range_info(self, sample_data):
        """Test getting range info."""
        normalizer = RangeNormalizer()
        normalizer.fit(sample_data)
        info = normalizer.get_range_info()
        
        assert "output_range" in info
        assert "method" in info
        assert "input_min" in info
        assert "input_max" in info
    
    def test_unfitted_transform_raises(self, sample_data):
        """Test transform before fit raises error."""
        normalizer = RangeNormalizer()
        with pytest.raises(ValueError):
            normalizer.transform(sample_data)


class TestDecimalScaler:
    """Test DecimalScaler."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return np.array([
            [123.45, 0.0045, 50000],
            [234.56, 0.0078, 60000],
            [345.67, 0.0012, 70000]
        ])
    
    def test_decimal_scaling(self, sample_data):
        """Test decimal scaling."""
        scaler = DecimalScaler()
        scaler.fit(sample_data)
        scaled = scaler.transform(sample_data)
        
        assert scaled.shape == sample_data.shape
        assert np.all(scaled <= 1)
    
    def test_inverse_transform(self, sample_data):
        """Test inverse transformation."""
        scaler = DecimalScaler()
        scaler.fit(sample_data)
        scaled = scaler.transform(sample_data)
        original = scaler.inverse_transform(scaled)
        
        assert np.allclose(original, sample_data, rtol=1e-10)
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform."""
        scaler = DecimalScaler()
        scaled = scaler.fit_transform(sample_data)
        
        assert scaled.shape == sample_data.shape
    
    def test_get_scale_factors(self, sample_data):
        """Test getting scale factors."""
        scaler = DecimalScaler()
        scaler.fit(sample_data)
        factors = scaler.get_scale_factors()
        
        assert len(factors) == sample_data.shape[1]
    
    def test_unfitted_transform_raises(self, sample_data):
        """Test transform before fit raises error."""
        scaler = DecimalScaler()
        with pytest.raises(ValueError):
            scaler.transform(sample_data)
    
    def test_unfitted_inverse_transform_raises(self, sample_data):
        """Test inverse_transform before fit raises error."""
        scaler = DecimalScaler()
        with pytest.raises(ValueError):
            scaler.inverse_transform(sample_data)


class TestLogTransformer:
    """Test LogTransformer."""
    
    @pytest.fixture
    def positive_data(self):
        """Create positive data."""
        np.random.seed(42)
        return np.random.exponential(2, size=(20, 3)) + 1
    
    def test_natural_log_transform(self, positive_data):
        """Test natural log transformation."""
        transformer = LogTransformer(method="natural")
        transformer.fit(positive_data)
        transformed = transformer.transform(positive_data)
        
        assert transformed.shape == positive_data.shape
        assert np.all(np.isfinite(transformed))
    
    def test_log10_transform(self, positive_data):
        """Test base-10 log transformation."""
        transformer = LogTransformer(method="log10")
        transformer.fit(positive_data)
        transformed = transformer.transform(positive_data)
        
        assert transformed.shape == positive_data.shape
        assert np.all(np.isfinite(transformed))
    
    def test_log2_transform(self, positive_data):
        """Test base-2 log transformation."""
        transformer = LogTransformer(method="log2")
        transformer.fit(positive_data)
        transformed = transformer.transform(positive_data)
        
        assert transformed.shape == positive_data.shape
        assert np.all(np.isfinite(transformed))
    
    def test_inverse_transform(self, positive_data):
        """Test inverse transformation."""
        transformer = LogTransformer()
        transformer.fit(positive_data)
        transformed = transformer.transform(positive_data)
        original = transformer.inverse_transform(transformed)
        
        assert np.allclose(original, positive_data, rtol=1e-10)
    
    def test_fit_transform(self, positive_data):
        """Test fit_transform."""
        transformer = LogTransformer()
        transformed = transformer.fit_transform(positive_data)
        
        assert transformed.shape == positive_data.shape
    
    def test_get_transform_info(self, positive_data):
        """Test getting transform info."""
        transformer = LogTransformer()
        transformer.fit(positive_data)
        info = transformer.get_transform_info()
        
        assert "method" in info
        assert "shift" in info
        assert "fitted" in info
    
    def test_unfitted_transform_raises(self, positive_data):
        """Test transform before fit raises error."""
        transformer = LogTransformer()
        with pytest.raises(ValueError):
            transformer.transform(positive_data)
    
    def test_negative_data_handling(self):
        """Test handling of negative data."""
        data = np.array([[0.5, 1, 2], [2, 3, 4]], dtype=float)
        transformer = LogTransformer()
        transformer.fit(data)
        transformed = transformer.transform(data)
        
        assert transformed.shape == data.shape
        assert np.all(np.isfinite(transformed))


class TestPowerTransformer:
    """Test PowerTransformer."""
    
    @pytest.fixture
    def positive_data(self):
        """Create positive data."""
        np.random.seed(42)
        return np.random.exponential(2, size=(20, 3)) + 1
    
    def test_sqrt_transform(self, positive_data):
        """Test square root transformation."""
        transformer = PowerTransformer(power=0.5)
        transformer.fit(positive_data)
        transformed = transformer.transform(positive_data)
        
        assert transformed.shape == positive_data.shape
        assert np.allclose(transformed ** 2, positive_data, atol=1e-10)
    
    def test_square_transform(self, positive_data):
        """Test square transformation."""
        transformer = PowerTransformer(power=2)
        transformer.fit(positive_data)
        transformed = transformer.transform(positive_data)
        
        assert transformed.shape == positive_data.shape
        assert np.allclose(transformed ** 0.5, positive_data, atol=1e-10)
    
    def test_log_via_power_zero(self, positive_data):
        """Test log transform via power=0."""
        transformer = PowerTransformer(power=0)
        transformer.fit(positive_data)
        transformed = transformer.transform(positive_data)
        
        assert transformed.shape == positive_data.shape
    
    def test_inverse_transform(self, positive_data):
        """Test inverse transformation."""
        transformer = PowerTransformer(power=0.5)
        transformer.fit(positive_data)
        transformed = transformer.transform(positive_data)
        original = transformer.inverse_transform(transformed)
        
        assert np.allclose(original, positive_data, rtol=1e-10)
    
    def test_fit_transform(self, positive_data):
        """Test fit_transform."""
        transformer = PowerTransformer(power=0.5)
        transformed = transformer.fit_transform(positive_data)
        
        assert transformed.shape == positive_data.shape
    
    def test_get_power_info(self, positive_data):
        """Test getting power info."""
        transformer = PowerTransformer(power=0.5)
        transformer.fit(positive_data)
        info = transformer.get_power_info()
        
        assert "power" in info
        assert "fitted" in info
    
    def test_unfitted_transform_raises(self, positive_data):
        """Test transform before fit raises error."""
        transformer = PowerTransformer()
        with pytest.raises(ValueError):
            transformer.transform(positive_data)
    
    def test_unfitted_inverse_transform_raises(self, positive_data):
        """Test inverse_transform before fit raises error."""
        transformer = PowerTransformer()
        with pytest.raises(ValueError):
            transformer.inverse_transform(positive_data)
    
    def test_invalid_power_raises(self):
        """Test that invalid power for negative data raises error."""
        data = np.array([[-1, 0, 1], [2, 3, 4]], dtype=float)
        transformer = PowerTransformer(power=2)
        
        with pytest.raises(ValueError):
            transformer.fit(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
