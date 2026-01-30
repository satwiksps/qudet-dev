"""
Comprehensive tests for angle and phase encoding methods.

Tests angle, phase, hybrid, multi-axis, and parametric encoders.
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qudet.encoders.angle_phase import (
    AngleEncoder,
    PhaseEncoder,
    HybridAnglePhaseEncoder,
    MultiAxisRotationEncoder,
    ParametricAngleEncoder
)


class TestAngleEncoder:
    """Test angle encoding functionality."""

    def test_initialization(self):
        """Test angle encoder initialization."""
        encoder = AngleEncoder(n_qubits=2, angle_type="rx")
        assert encoder.n_qubits == 2
        assert encoder.angle_type == "rx"

    def test_rx_encoding(self):
        """Test RX rotation encoding."""
        encoder = AngleEncoder(n_qubits=2, angle_type="rx")
        data = np.array([np.pi / 4, np.pi / 6])
        circuit = encoder.encode(data)
        
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 2

    def test_ry_encoding(self):
        """Test RY rotation encoding."""
        encoder = AngleEncoder(n_qubits=2, angle_type="ry")
        data = np.array([np.pi / 4, np.pi / 6])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_rz_encoding(self):
        """Test RZ rotation encoding."""
        encoder = AngleEncoder(n_qubits=2, angle_type="rz")
        data = np.array([np.pi / 4, np.pi / 6])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_auto_angle_type(self):
        """Test auto angle type selection."""
        encoder = AngleEncoder(n_qubits=3, angle_type="auto")
        data = np.array([0.5, 0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 3

    def test_invalid_angle_type(self):
        """Test invalid angle type raises error."""
        with pytest.raises(ValueError):
            AngleEncoder(n_qubits=2, angle_type="invalid")

    def test_scaled_encoding(self):
        """Test scaled angle encoding."""
        encoder = AngleEncoder(n_qubits=2)
        data = np.array([1.0, 1.0])
        
        circuit = encoder.encode_scaled(data, scale_factor=np.pi)
        assert circuit.num_qubits == 2

    def test_multiple_reps(self):
        """Test encoding with multiple repetitions."""
        encoder = AngleEncoder(n_qubits=2, reps=3)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_angle_override(self):
        """Test overriding angle type."""
        encoder = AngleEncoder(n_qubits=2, angle_type="rx")
        data = np.array([0.5, 0.5])
        
        circuit_ry = encoder.encode(data, angle_type="ry")
        assert circuit_ry.num_qubits == 2


class TestPhaseEncoder:
    """Test phase encoding functionality."""

    def test_initialization(self):
        """Test phase encoder initialization."""
        encoder = PhaseEncoder(n_qubits=2)
        assert encoder.n_qubits == 2
        assert encoder.global_phase is False

    def test_encode_basic(self):
        """Test basic phase encoding."""
        encoder = PhaseEncoder(n_qubits=2)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 2

    def test_encode_normalized(self):
        """Test encoding with normalization."""
        encoder = PhaseEncoder(n_qubits=2)
        data = np.array([1.0, 2.0])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_global_phase_flag(self):
        """Test global phase flag."""
        encoder = PhaseEncoder(n_qubits=2, global_phase=True)
        assert encoder.global_phase is True

    def test_apply_global_phase(self):
        """Test applying global phase."""
        encoder = PhaseEncoder(n_qubits=2)
        circuit = QuantumCircuit(2)
        
        modified = encoder.apply_global_phase(circuit, np.pi / 4)
        assert modified.global_phase == np.pi / 4

    def test_phase_with_zero_norm(self):
        """Test phase encoding with zero norm data."""
        encoder = PhaseEncoder(n_qubits=2)
        data = np.array([0.0, 0.0])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2


class TestHybridAnglePhaseEncoder:
    """Test hybrid angle-phase encoding."""

    def test_initialization(self):
        """Test hybrid encoder initialization."""
        encoder = HybridAnglePhaseEncoder(n_qubits=2)
        assert encoder.n_qubits == 2

    def test_encode_balanced(self):
        """Test balanced angle-phase encoding."""
        encoder = HybridAnglePhaseEncoder(n_qubits=2, angle_weight=0.5, phase_weight=0.5)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_angle_dominant(self):
        """Test angle-dominant encoding."""
        encoder = HybridAnglePhaseEncoder(n_qubits=2, angle_weight=0.8, phase_weight=0.2)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_phase_dominant(self):
        """Test phase-dominant encoding."""
        encoder = HybridAnglePhaseEncoder(n_qubits=2, angle_weight=0.2, phase_weight=0.8)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_get_weights(self):
        """Test getting encoding weights."""
        encoder = HybridAnglePhaseEncoder(n_qubits=2, angle_weight=0.6, phase_weight=0.4)
        weights = encoder.get_encoding_weights()
        
        assert "angle_weight" in weights
        assert "phase_weight" in weights
        assert np.isclose(weights["angle_weight"] + weights["phase_weight"], 1.0)


class TestMultiAxisRotationEncoder:
    """Test multi-axis rotation encoding."""

    def test_initialization(self):
        """Test multi-axis encoder initialization."""
        encoder = MultiAxisRotationEncoder(n_qubits=2)
        assert encoder.n_qubits == 2

    def test_encode_default_axes(self):
        """Test encoding with default axes."""
        encoder = MultiAxisRotationEncoder(n_qubits=3)
        data = np.array([0.5, 0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 3

    def test_custom_axes(self):
        """Test encoding with custom axes."""
        encoder = MultiAxisRotationEncoder(n_qubits=3, axes=['x', 'y', 'z'])
        data = np.array([0.5, 0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 3

    def test_single_axis(self):
        """Test with single axis."""
        encoder = MultiAxisRotationEncoder(n_qubits=2, axes=['x'])
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_invalid_axis(self):
        """Test invalid axis raises error."""
        with pytest.raises(ValueError):
            MultiAxisRotationEncoder(n_qubits=2, axes=['invalid'])

    def test_get_available_axes(self):
        """Test getting available axes."""
        encoder = MultiAxisRotationEncoder(n_qubits=2)
        axes = encoder.get_available_axes()
        
        assert 'x' in axes
        assert 'y' in axes
        assert 'z' in axes


class TestParametricAngleEncoder:
    """Test parametric angle encoding."""

    def test_initialization(self):
        """Test parametric encoder initialization."""
        encoder = ParametricAngleEncoder(n_qubits=2)
        assert encoder.n_qubits == 2
        assert encoder.param_sharing is False

    def test_encode_basic(self):
        """Test basic parametric encoding."""
        encoder = ParametricAngleEncoder(n_qubits=2)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_custom_parameters(self):
        """Test encoding with custom parameters."""
        encoder = ParametricAngleEncoder(n_qubits=2, n_params=3)
        data = np.array([0.5, 0.5])
        params = np.array([0.1, 0.2, 0.3])
        
        circuit = encoder.encode(data, parameters=params)
        assert circuit.num_qubits == 2

    def test_param_sharing(self):
        """Test parameter sharing."""
        encoder = ParametricAngleEncoder(n_qubits=4, n_params=2, param_sharing=True)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 4

    def test_update_parameters(self):
        """Test updating parameters."""
        encoder = ParametricAngleEncoder(n_qubits=2, n_params=2)
        new_params = np.array([0.1, 0.2])
        
        encoder.update_parameters(new_params)
        current = encoder.get_parameters()
        
        assert np.allclose(current, new_params)

    def test_get_parameters(self):
        """Test getting parameters."""
        encoder = ParametricAngleEncoder(n_qubits=2, n_params=2)
        params = encoder.get_parameters()
        
        assert len(params) == 2

    def test_invalid_parameter_count(self):
        """Test updating with invalid parameter count raises error."""
        encoder = ParametricAngleEncoder(n_qubits=2, n_params=2)
        
        with pytest.raises(ValueError):
            encoder.update_parameters(np.array([0.1]))

    def test_parameter_reproducibility(self):
        """Test parameter consistency across calls."""
        encoder = ParametricAngleEncoder(n_qubits=2, n_params=2)
        params1 = encoder.get_parameters()
        params2 = encoder.get_parameters()
        
        assert np.allclose(params1, params2)


# Integration tests
class TestAnglePhaseIntegration:
    """Integration tests for angle and phase encoders."""

    def test_angle_phase_combination(self):
        """Test combining angle and phase encoders."""
        angle_enc = AngleEncoder(n_qubits=2)
        phase_enc = PhaseEncoder(n_qubits=2)
        
        data = np.array([0.5, 0.5])
        circuit_angle = angle_enc.encode(data)
        circuit_phase = phase_enc.encode(data)
        
        assert circuit_angle.num_qubits == 2
        assert circuit_phase.num_qubits == 2

    def test_parametric_with_data(self):
        """Test parametric encoder with various data."""
        encoder = ParametricAngleEncoder(n_qubits=3, n_params=3)
        
        for _ in range(5):
            data = np.random.randn(3)
            circuit = encoder.encode(data)
            assert circuit.num_qubits == 3

    def test_multi_axis_coverage(self):
        """Test multi-axis encoder covers all axes."""
        encoder = MultiAxisRotationEncoder(n_qubits=3, axes=['x', 'y', 'z'])
        data = np.array([0.1, 0.2, 0.3])
        
        circuit = encoder.encode(data)
        assert circuit.num_qubits == 3
