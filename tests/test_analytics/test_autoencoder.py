"""
Test suite for Quantum Autoencoder (qudet.analytics.autoencoder)
"""

import pytest
import numpy as np
from qudet.analytics.autoencoder import QuantumAutoencoder


class TestQuantumAutoencoder:
    """Test cases for QuantumAutoencoder class."""
    
    @pytest.fixture
    def qae(self):
        """Create a test autoencoder instance."""
        return QuantumAutoencoder(n_input_qubits=8, n_latent_qubits=4, n_layers=2)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        return np.random.randn(10, 8)
    
    def test_initialization(self, qae):
        """Test autoencoder initialization."""
        assert qae.n_input == 8
        assert qae.n_latent == 4
        assert qae.n_trash == 4
        assert qae.n_layers == 2
        assert qae.params.shape == (2 * 8 * 2,)
    
    def test_invalid_latent_qubits(self):
        """Test error when latent qubits >= input qubits."""
        with pytest.raises(ValueError, match="must be <"):
            QuantumAutoencoder(n_input_qubits=4, n_latent_qubits=4)
    
    def test_fit(self, qae, sample_data):
        """Test fit method."""
        result = qae.fit(sample_data)
        assert result is qae  # Returns self for chaining
        assert hasattr(qae, "_is_trained")
    
    def test_fit_wrong_features(self, qae, sample_data):
        """Test fit with wrong number of features."""
        wrong_data = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="Expected 8"):
            qae.fit(wrong_data)
    
    def test_compress(self, qae, sample_data):
        """Test data compression."""
        qae.fit(sample_data)
        compressed = qae.compress(sample_data)
        
        # Check shape
        assert compressed.shape == (10, 4)
        # Compressed data should be less than original
        assert compressed.shape[1] < sample_data.shape[1]
    
    def test_compress_without_fit(self, qae, sample_data):
        """Test error when compressing without training."""
        with pytest.raises(RuntimeError, match="not trained"):
            qae.compress(sample_data)
    
    def test_compress_wrong_features(self, qae, sample_data):
        """Test error when compressing with wrong features."""
        qae.fit(sample_data)
        wrong_data = np.random.randn(5, 5)
        with pytest.raises(ValueError, match="Expected 8"):
            qae.compress(wrong_data)
    
    def test_decompress(self, qae, sample_data):
        """Test data decompression."""
        qae.fit(sample_data)
        compressed = qae.compress(sample_data)
        decompressed = qae.decompress(compressed)
        
        # Check shape
        assert decompressed.shape == (10, 8)
    
    def test_decompress_wrong_features(self, qae, sample_data):
        """Test error when decompressing with wrong features."""
        qae.fit(sample_data)
        wrong_data = np.random.randn(5, 3)
        with pytest.raises(ValueError, match="Expected 4"):
            qae.decompress(wrong_data)
    
    def test_get_compression_ratio(self, qae):
        """Test compression ratio calculation."""
        ratio = qae.get_compression_ratio()
        assert ratio == pytest.approx(0.5)  # 4/8 = 0.5
    
    def test_build_ansatz(self, qae):
        """Test ansatz circuit construction."""
        circuit = qae._build_ansatz(qae.params)
        assert circuit.num_qubits == 8
        # Should have multiple gates (cx, ry, rz)
        assert circuit.size() > 0
    
    def test_different_configurations(self):
        """Test autoencoder with different configurations."""
        configs = [
            (4, 2, 1),
            (8, 4, 2),
            (16, 8, 3),
        ]
        
        for n_input, n_latent, n_layers in configs:
            qae = QuantumAutoencoder(n_input, n_latent, n_layers)
            assert qae.n_trash == n_input - n_latent
            assert qae.get_compression_ratio() == pytest.approx(n_latent / n_input)
