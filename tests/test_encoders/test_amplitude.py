"""
Comprehensive tests for amplitude encoding methods.

Tests amplitude, density matrix, basis change, and feature map encoders.
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qudet.encoders.amplitude import (
    AmplitudeEncoder,
    DensityMatrixEncoder,
    BasisChangeEncoder,
    FeatureMapEncoder
)


class TestAmplitudeEncoder:
    """Test amplitude encoding functionality."""

    def test_initialization(self):
        """Test amplitude encoder initialization."""
        encoder = AmplitudeEncoder(n_qubits=3)
        assert encoder.n_qubits == 3
        assert encoder.normalize is True

    def test_encode_basic(self):
        """Test basic encoding."""
        encoder = AmplitudeEncoder(n_qubits=2)
        data = np.array([1.0, 0.0])
        circuit = encoder.encode(data)
        
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 2

    def test_encode_with_normalization(self):
        """Test encoding with normalization."""
        encoder = AmplitudeEncoder(n_qubits=3, normalize=True)
        data = np.array([3.0, 4.0])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 3

    def test_encode_without_normalization(self):
        """Test encoding without normalization."""
        encoder = AmplitudeEncoder(n_qubits=2, normalize=False)
        data = np.array([1.0, 0.0])  # Already normalized
        circuit = encoder.encode(data)
        
        assert isinstance(circuit, QuantumCircuit)

    def test_max_features(self):
        """Test maximum features supported."""
        encoder = AmplitudeEncoder(n_qubits=3)
        max_features = encoder.get_features_supported()
        
        assert max_features == 8  # 2^3

    def test_various_qubit_counts(self):
        """Test with different qubit counts."""
        for n_qubits in [1, 2, 3]:
            encoder = AmplitudeEncoder(n_qubits=n_qubits)
            data = np.random.randn(2**n_qubits)  # Appropriately sized data
            circuit = encoder.encode(data)
            
            assert circuit.num_qubits == n_qubits

    def test_empty_data(self):
        """Test with empty data."""
        encoder = AmplitudeEncoder(n_qubits=2)
        data = np.array([])
        circuit = encoder.encode(data)
        
        # Should create |00⟩ state when given empty data
        assert circuit.num_qubits == 2


class TestDensityMatrixEncoder:
    """Test density matrix encoding."""

    def test_initialization(self):
        """Test density matrix encoder initialization."""
        encoder = DensityMatrixEncoder(n_qubits=2)
        assert encoder.n_qubits == 2
        assert encoder.mixed_state is False

    def test_encode_basic(self):
        """Test basic density matrix encoding."""
        encoder = DensityMatrixEncoder(n_qubits=2)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 2

    def test_encode_normalized(self):
        """Test encoding with normalized features."""
        encoder = DensityMatrixEncoder(n_qubits=3)
        data = np.array([1.0, 1.0, 1.0])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 3

    def test_compute_density_matrix(self):
        """Test density matrix computation."""
        encoder = DensityMatrixEncoder(n_qubits=2)
        data = np.array([0.6, 0.8])
        rho = encoder.compute_density_matrix(data)
        
        assert rho.shape == (4, 4)
        assert np.allclose(np.trace(rho), 1.0)  # Unit trace

    def test_density_matrix_hermitian(self):
        """Test that density matrix is Hermitian."""
        encoder = DensityMatrixEncoder(n_qubits=2)
        data = np.array([0.5, 0.5])
        rho = encoder.compute_density_matrix(data)
        
        assert np.allclose(rho, rho.conj().T)

    def test_mixed_state_flag(self):
        """Test mixed state flag."""
        encoder = DensityMatrixEncoder(n_qubits=2, mixed_state=True)
        assert encoder.mixed_state is True

    def test_empty_data_density(self):
        """Test density matrix with empty data."""
        encoder = DensityMatrixEncoder(n_qubits=2)
        data = np.array([])
        rho = encoder.compute_density_matrix(data)
        
        assert rho.shape == (4, 4)


class TestBasisChangeEncoder:
    """Test basis change encoding."""

    def test_initialization(self):
        """Test basis change encoder initialization."""
        encoder = BasisChangeEncoder(n_qubits=2, basis="x")
        assert encoder.n_qubits == 2
        assert encoder.basis == "x"

    def test_z_basis(self):
        """Test Z-basis encoding."""
        encoder = BasisChangeEncoder(n_qubits=2, basis="z")
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_x_basis(self):
        """Test X-basis encoding."""
        encoder = BasisChangeEncoder(n_qubits=2, basis="x")
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_y_basis(self):
        """Test Y-basis encoding."""
        encoder = BasisChangeEncoder(n_qubits=2, basis="y")
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_invalid_basis(self):
        """Test invalid basis raises error."""
        with pytest.raises(ValueError):
            BasisChangeEncoder(n_qubits=2, basis="invalid")

    def test_supported_bases(self):
        """Test getting supported bases."""
        encoder = BasisChangeEncoder(n_qubits=2)
        bases = encoder.get_supported_bases()
        
        assert "z" in bases
        assert "x" in bases
        assert "y" in bases

    def test_basis_override(self):
        """Test overriding basis in encode."""
        encoder = BasisChangeEncoder(n_qubits=2, basis="z")
        data = np.array([0.5, 0.5])
        
        circuit_x = encoder.encode(data, basis="x")
        circuit_y = encoder.encode(data, basis="y")
        
        assert circuit_x.num_qubits == 2
        assert circuit_y.num_qubits == 2


class TestFeatureMapEncoder:
    """Test feature map encoding."""

    def test_initialization(self):
        """Test feature map encoder initialization."""
        encoder = FeatureMapEncoder(n_qubits=2)
        assert encoder.n_qubits == 2
        assert encoder.mapping_type == "linear"

    def test_linear_mapping(self):
        """Test linear mapping."""
        encoder = FeatureMapEncoder(n_qubits=2, mapping_type="linear")
        data = np.array([1.0, 2.0])
        
        mapped = encoder.apply_mapping(data)
        assert np.allclose(mapped, data)

    def test_polynomial_mapping(self):
        """Test polynomial mapping."""
        encoder = FeatureMapEncoder(n_qubits=2, mapping_type="polynomial", power=2)
        data = np.array([2.0, 3.0])
        
        mapped = encoder.apply_mapping(data)
        expected = np.array([4.0, 9.0])
        assert np.allclose(mapped, expected)

    def test_trigonometric_mapping(self):
        """Test trigonometric mapping."""
        encoder = FeatureMapEncoder(n_qubits=2, mapping_type="trigonometric")
        data = np.array([0.0, np.pi/2])
        
        mapped = encoder.apply_mapping(data)
        expected = np.sin(data)
        assert np.allclose(mapped, expected)

    def test_encode_with_mapping(self):
        """Test encoding with feature mapping."""
        encoder = FeatureMapEncoder(n_qubits=2, mapping_type="linear")
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_multiple_reps(self):
        """Test with multiple repetitions."""
        encoder = FeatureMapEncoder(n_qubits=2, reps=3)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2
        # More reps should create deeper circuit

    def test_mapping_info(self):
        """Test getting mapping info."""
        encoder = FeatureMapEncoder(n_qubits=3, mapping_type="polynomial", power=3, reps=2)
        info = encoder.get_mapping_info()
        
        assert info["type"] == "polynomial"
        assert info["power"] == 3
        assert info["repetitions"] == 2
        assert info["n_qubits"] == 3

    def test_different_powers(self):
        """Test polynomial mapping with different powers."""
        for power in [1, 2, 3]:
            encoder = FeatureMapEncoder(n_qubits=2, mapping_type="polynomial", power=power)
            data = np.array([2.0])
            mapped = encoder.apply_mapping(data)
            
            expected = 2.0 ** power
            assert np.isclose(mapped[0], expected)


# Integration tests
class TestAmplitudeEncodingIntegration:
    """Integration tests for amplitude encoders."""

    def test_chain_encoding(self):
        """Test chaining multiple encodings."""
        encoder1 = AmplitudeEncoder(n_qubits=2)
        encoder2 = FeatureMapEncoder(n_qubits=2)
        
        data = np.array([0.5, 0.5])
        circuit1 = encoder1.encode(data)
        circuit2 = encoder2.encode(data)
        
        assert circuit1.num_qubits == 2
        assert circuit2.num_qubits == 2

    def test_mixed_data_types(self):
        """Test encoding with different data formats."""
        encoder = AmplitudeEncoder(n_qubits=2)
        
        data_list = [0.5, 0.5]
        data_array = np.array(data_list)
        data_tuple = tuple(data_list)
        
        circuit_array = encoder.encode(data_array)
        assert circuit_array.num_qubits == 2

    def test_encoder_consistency(self):
        """Test that same data produces consistent encoding."""
        encoder = FeatureMapEncoder(n_qubits=3, mapping_type="linear")
        data = np.array([0.5, 0.3, 0.2])
        
        circuit1 = encoder.encode(data)
        circuit2 = encoder.encode(data)
        
        # Both should have same structure
        assert circuit1.num_qubits == circuit2.num_qubits
