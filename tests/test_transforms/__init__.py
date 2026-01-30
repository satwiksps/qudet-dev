"""
Tests for Quantum Transformation Module (PCA)
"""

import numpy as np
import pytest
from qudet.transforms.pca import QuantumPCA


class TestQuantumPCA:
    """Test suite for QuantumPCA class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        # Create simple 2D data with clusters
        data = np.vstack([
            np.random.randn(10, 4) + [0, 0, 0, 0],
            np.random.randn(10, 4) + [3, 3, 3, 3],
            np.random.randn(10, 4) + [-3, -3, -3, -3]
        ])
        return data
    
    def test_initialization(self):
        """Test QuantumPCA initialization."""
        pca = QuantumPCA(n_components=2, n_qubits=4)
        assert pca.n_components == 2
        assert pca.n_qubits == 4
        assert pca.train_data_ is None
    
    def test_fit(self, sample_data):
        """Test fitting QuantumPCA."""
        pca = QuantumPCA(n_components=2, n_qubits=4)
        result = pca.fit(sample_data)
        
        # Should return self
        assert result is pca
        # Should store training data
        assert pca.train_data_ is not None
        assert pca.train_data_.shape == sample_data.shape
    
    def test_transform(self, sample_data):
        """Test transforming data with QuantumPCA."""
        pca = QuantumPCA(n_components=2, n_qubits=4)
        pca.fit(sample_data)
        
        transformed = pca.transform(sample_data)
        
        # Check output shape
        assert transformed.shape == (len(sample_data), 2)
        assert isinstance(transformed, np.ndarray)
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        pca = QuantumPCA(n_components=2, n_qubits=4)
        transformed = pca.fit_transform(sample_data)
        
        assert transformed.shape == (len(sample_data), 2)
        assert pca.train_data_ is not None
    
    def test_transform_without_fit_raises_error(self, sample_data):
        """Test that transform without fit raises error."""
        pca = QuantumPCA(n_components=2, n_qubits=4)
        
        with pytest.raises(ValueError, match="Must fit before transform"):
            pca.transform(sample_data)
    
    def test_different_n_components(self, sample_data):
        """Test different numbers of components."""
        for n_comp in [1, 2, 3]:
            pca = QuantumPCA(n_components=n_comp, n_qubits=4)
            transformed = pca.fit_transform(sample_data)
            assert transformed.shape[1] == n_comp
    
    def test_output_is_float(self, sample_data):
        """Test that output is float type."""
        pca = QuantumPCA(n_components=2, n_qubits=4)
        transformed = pca.fit_transform(sample_data)
        assert transformed.dtype in [np.float32, np.float64]
