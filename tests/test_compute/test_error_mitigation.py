import pytest
import numpy as np
from qudet.compute.error_mitigation import (
    QuantumErrorMitigation,
    QuantumNoiseModel,
    QuantumCalibrationalAnalyzer
)


class TestQuantumErrorMitigation:
    """Test suite for Quantum Error Mitigation."""

    def test_initialization(self):
        """Test error mitigation initialization."""
        mitigator = QuantumErrorMitigation(mitigation_method="zero_noise_extrapolation")
        assert mitigator.mitigation_method == "zero_noise_extrapolation"

    def test_calibrate(self):
        """Test calibration of error mitigation."""
        mitigator = QuantumErrorMitigation()
        test_circuits = [{'gates': [{'type': 'h'}]} for _ in range(5)]
        results = np.array([0.9, 0.95, 0.92, 0.94, 0.93])
        
        mitigator.calibrate(test_circuits, results)
        
        assert mitigator.calibration_data is not None
        assert 'error_rate' in mitigator.calibration_data

    def test_mitigate_without_calibration_raises_error(self):
        """Test that mitigation without calibration raises error."""
        mitigator = QuantumErrorMitigation()
        
        with pytest.raises(ValueError):
            mitigator.mitigate(np.array([0.5, 0.6, 0.7]))

    def test_mitigate_after_calibration(self):
        """Test mitigation after calibration."""
        mitigator = QuantumErrorMitigation()
        test_circuits = [{'gates': []} for _ in range(3)]
        results = np.array([0.95, 0.96, 0.94])
        
        mitigator.calibrate(test_circuits, results)
        noisy_results = np.array([0.85, 0.87, 0.83])
        
        mitigated = mitigator.mitigate(noisy_results)
        
        assert len(mitigated) == len(noisy_results)
        assert all(0 <= m <= 1 for m in mitigated)

    def test_zne_mitigation(self):
        """Test zero-noise extrapolation mitigation."""
        mitigator = QuantumErrorMitigation()
        test_circuits = [{}]
        mitigator.calibrate(test_circuits, np.array([0.9]))
        
        results = np.array([0.5])
        mitigated = mitigator._zne_mitigation(results)
        
        assert mitigated[0] > results[0]


class TestQuantumNoiseModel:
    """Test suite for Quantum Noise Model."""

    def test_initialization(self):
        """Test noise model initialization."""
        noise_model = QuantumNoiseModel(noise_type="depolarizing", error_rate=0.01)
        assert noise_model.noise_type == "depolarizing"
        assert noise_model.error_rate == 0.01

    def test_apply_depolarizing_noise(self):
        """Test application of depolarizing noise."""
        noise_model = QuantumNoiseModel(noise_type="depolarizing", error_rate=0.1)
        state = np.array([1.0, 0.0])
        
        noisy_state = noise_model.apply_noise(state)
        
        assert len(noisy_state) == len(state)
        assert np.isclose(np.linalg.norm(noisy_state), 1.0)

    def test_apply_amplitude_damping_noise(self):
        """Test application of amplitude damping noise."""
        noise_model = QuantumNoiseModel(noise_type="amplitude_damping", error_rate=0.05)
        state = np.array([1.0, 0.0])
        
        noisy_state = noise_model.apply_noise(state)
        
        assert len(noisy_state) == len(state)
        assert np.isclose(np.linalg.norm(noisy_state), 1.0)

    def test_get_noise_parameters(self):
        """Test retrieval of noise parameters."""
        noise_model = QuantumNoiseModel(noise_type="amplitude_damping", error_rate=0.02)
        params = noise_model.get_noise_parameters()
        
        assert params['type'] == "amplitude_damping"
        assert params['error_rate'] == 0.02

    def test_noise_normalization(self):
        """Test that noisy states remain normalized."""
        noise_model = QuantumNoiseModel(error_rate=0.2)
        state = np.array([0.707, 0.707])
        
        for _ in range(10):
            state = noise_model.apply_noise(state)
            assert np.isclose(np.linalg.norm(state), 1.0)


class TestQuantumCalibrationalAnalyzer:
    """Test suite for Quantum Calibration Analyzer."""

    def test_initialization(self):
        """Test calibration analyzer initialization."""
        analyzer = QuantumCalibrationalAnalyzer()
        assert analyzer.calibration_data == {}

    def test_store_calibration(self):
        """Test storing calibration data."""
        analyzer = QuantumCalibrationalAnalyzer()
        cal_data = {
            'single_qubit_error': 1e-3,
            'two_qubit_error': 5e-3,
            'readout_error': 1e-2
        }
        
        analyzer.store_calibration("ibm_brisbane", cal_data)
        
        assert "ibm_brisbane" in analyzer.calibration_data

    def test_extract_gate_errors(self):
        """Test extraction of gate errors."""
        analyzer = QuantumCalibrationalAnalyzer()
        cal_data = {
            'single_qubit_error': 1e-3,
            'two_qubit_error': 5e-3
        }
        
        analyzer.store_calibration("sim", cal_data)
        
        assert "sim" in analyzer.gate_errors
        assert 'single_qubit' in analyzer.gate_errors["sim"]
        assert 'two_qubit' in analyzer.gate_errors["sim"]

    def test_extract_readout_errors(self):
        """Test extraction of readout errors."""
        analyzer = QuantumCalibrationalAnalyzer()
        cal_data = {'readout_error': 0.02}
        
        analyzer.store_calibration("backend", cal_data)
        
        assert "backend" in analyzer.readout_errors

    def test_get_gate_error(self):
        """Test retrieval of gate error rates."""
        analyzer = QuantumCalibrationalAnalyzer()
        cal_data = {
            'single_qubit_error': 1e-3,
            'two_qubit_error': 5e-3
        }
        
        analyzer.store_calibration("hw", cal_data)
        
        assert analyzer.get_gate_error("hw", "single_qubit") == 1e-3
        assert analyzer.get_gate_error("hw", "two_qubit") == 5e-3

    def test_get_readout_error(self):
        """Test retrieval of readout error."""
        analyzer = QuantumCalibrationalAnalyzer()
        cal_data = {'readout_error': 0.015}
        
        analyzer.store_calibration("backend", cal_data)
        
        assert analyzer.get_readout_error("backend") == 0.015

    def test_estimate_circuit_fidelity(self):
        """Test circuit fidelity estimation."""
        analyzer = QuantumCalibrationalAnalyzer()
        cal_data = {
            'single_qubit_error': 1e-3,
            'two_qubit_error': 5e-3,
            'readout_error': 1e-2
        }
        
        analyzer.store_calibration("backend", cal_data)
        fidelity = analyzer.estimate_circuit_fidelity("backend", n_gates=100)
        
        assert 0 <= fidelity <= 1
        assert fidelity < 1.0

    def test_fidelity_decreases_with_gates(self):
        """Test that fidelity decreases with more gates."""
        analyzer = QuantumCalibrationalAnalyzer()
        cal_data = {
            'single_qubit_error': 1e-3,
            'two_qubit_error': 5e-3,
            'readout_error': 1e-2
        }
        
        analyzer.store_calibration("backend", cal_data)
        fidelity_10 = analyzer.estimate_circuit_fidelity("backend", n_gates=10)
        fidelity_100 = analyzer.estimate_circuit_fidelity("backend", n_gates=100)
        
        assert fidelity_10 > fidelity_100
