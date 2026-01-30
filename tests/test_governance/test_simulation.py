"""Tests for Noise Profiler"""

import pytest
from qudet.governance.simulation import NoiseSimulator


class TestNoiseSimulator:
    """Test suite for NoiseSimulator class."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("qiskit_aer", minversion=None),
        reason="qiskit-aer not installed"
    )
    def test_get_noiseless_backend(self):
        """Test getting a noiseless backend."""
        try:
            backend = NoiseSimulator.get_noiseless_backend()
            assert backend is not None
            assert hasattr(backend, 'run')
        except ImportError:
            pytest.skip("qiskit-aer not installed")
    
    @pytest.mark.skipif(
        not pytest.importorskip("qiskit_aer", minversion=None),
        reason="qiskit-aer not installed"
    )
    def test_get_noisy_backend(self):
        """Test getting a noisy backend."""
        try:
            backend = NoiseSimulator.get_noisy_backend(error_prob=0.01)
            assert backend is not None
            assert hasattr(backend, 'run')
        except ImportError:
            pytest.skip("qiskit-aer not installed")
    
    @pytest.mark.skipif(
        not pytest.importorskip("qiskit_aer", minversion=None),
        reason="qiskit-aer not installed"
    )
    def test_get_thermal_backend(self):
        """Test getting a thermal backend."""
        try:
            backend = NoiseSimulator.get_thermal_backend(
                t1=50e-6,
                t2=70e-6,
                gate_time=100e-9
            )
            assert backend is not None
        except ImportError:
            pytest.skip("qiskit-aer not installed")
    
    @pytest.mark.skipif(
        not pytest.importorskip("qiskit_aer", minversion=None),
        reason="qiskit-aer not installed"
    )
    def test_get_ibm_like_backend(self):
        """Test IBM-like backend."""
        try:
            backend = NoiseSimulator.get_ibm_like_backend(error_prob=0.005)
            assert backend is not None
        except ImportError:
            pytest.skip("qiskit-aer not installed")
    
    @pytest.mark.skipif(
        not pytest.importorskip("qiskit_aer", minversion=None),
        reason="qiskit-aer not installed"
    )
    def test_get_high_noise_backend(self):
        """Test high noise backend."""
        try:
            backend = NoiseSimulator.get_high_noise_backend(error_prob=0.05)
            assert backend is not None
        except ImportError:
            pytest.skip("qiskit-aer not installed")
    
    def test_estimate_accuracy_degradation(self):
        """Test accuracy degradation estimation."""
        # This doesn't require qiskit-aer
        baseline = 0.95
        estimated = NoiseSimulator.estimate_accuracy_degradation(baseline, error_prob=0.01)
        
        assert 0 <= estimated <= 1
        assert estimated < baseline  # Should be degraded
    
    def test_accuracy_degradation_with_high_error(self):
        """Test degradation with high error rate."""
        baseline = 0.95
        
        # Low error
        low_err = NoiseSimulator.estimate_accuracy_degradation(baseline, error_prob=0.001)
        
        # High error
        high_err = NoiseSimulator.estimate_accuracy_degradation(baseline, error_prob=0.1)
        
        # High error should degrade more
        assert low_err > high_err
    
    def test_estimate_accuracy_degradation_non_negative(self):
        """Test that degradation never goes negative."""
        baseline = 0.5
        high_error = 0.5
        
        estimated = NoiseSimulator.estimate_accuracy_degradation(baseline, error_prob=high_error)
        
        assert estimated >= 0
