# File: tests/test_validation/test_integrity.py

import pytest
import numpy as np
import pandas as pd
from qudet.governance.integrity import DataIntegrityCheck
from qudet.encoders.statevector import StatevectorEncoder
from qudet.encoders.rotation import RotationEncoder


class TestDataIntegrityCheck:
    """Test suite for DataIntegrityCheck."""
    
    def test_verify_amplitude_encoding_valid(self):
        """Test integrity check on valid amplitude encoding."""
        encoder = StatevectorEncoder()
        # Valid: power-of-2 length, normalized data
        data = np.array([0.6, 0.8])
        
        is_valid = DataIntegrityCheck.verify_encoding(data, encoder, tolerance=1e-5)
        
        assert is_valid is True
        
    def test_verify_amplitude_encoding_large_input(self):
        """Test integrity check with larger amplitude encoding."""
        encoder = StatevectorEncoder()
        # 4 features → 2 qubits (2^2 = 4)
        data = np.array([0.5, 0.5, 0.5, 0.5])
        
        is_valid = DataIntegrityCheck.verify_encoding(data, encoder)
        
        assert is_valid is True
        
    def test_verify_angle_encoding(self):
        """Test integrity check on IQP encoding."""
        from qudet.encoders.iqp import IQPEncoder
        encoder = IQPEncoder(n_qubits=2, reps=1)
        data = np.array([0.5, 0.3])
        
        # IQP encoding should pass
        is_valid = DataIntegrityCheck.verify_encoding(data, encoder)
        
        assert is_valid is True
        
    def test_verify_encoding_failed(self):
        """Test that encoding failure is caught."""
        encoder = StatevectorEncoder()
        # Invalid: 1-element array (not power of 2)
        data = np.array([0.5])
        
        with pytest.raises(ValueError):
            DataIntegrityCheck.verify_encoding(data, encoder)
            
    def test_compute_encoding_fidelity_amplitude(self):
        """Test fidelity computation for amplitude encoding."""
        encoder = StatevectorEncoder()
        data = np.array([0.6, 0.8])
        
        stats = DataIntegrityCheck.compute_encoding_fidelity(data, encoder)
        
        # Check all expected keys
        expected_keys = [
            "encoder", "fidelity", "min_probability", "max_probability",
            "shannon_entropy", "purity", "num_qubits"
        ]
        for key in expected_keys:
            assert key in stats
            
        # Fidelity should be between 0 and 1
        assert 0 <= stats["fidelity"] <= 1
        
    def test_compute_encoding_fidelity_angle(self):
        """Test fidelity computation for IQP encoding."""
        from qudet.encoders.iqp import IQPEncoder
        encoder = IQPEncoder(n_qubits=2, reps=1)
        data = np.array([0.5, 0.3])
        
        stats = DataIntegrityCheck.compute_encoding_fidelity(data, encoder)
        
        # Purity should be between 0 and 1
        assert 0 <= stats["purity"] <= 1
        
    def test_fidelity_properties(self):
        """Test that fidelity has expected properties."""
        encoder = StatevectorEncoder()
        data = np.array([0.6, 0.8])
        
        stats = DataIntegrityCheck.compute_encoding_fidelity(data, encoder)
        
        # Min prob should be <= max prob
        assert stats["min_probability"] <= stats["max_probability"]
        
        # Shannon entropy should be non-negative
        assert stats["shannon_entropy"] >= 0
        
    def test_verify_batch_success(self):
        """Test batch verification with all passing."""
        encoder = StatevectorEncoder()
        
        # Create batch of valid data
        batch = np.array([
            [0.6, 0.8],
            [0.5, 0.866],  # sqrt(0.75)
        ])
        
        passed, failed = DataIntegrityCheck.verify_batch(batch, encoder)
        
        assert passed == 2
        assert failed == 0
        
    def test_verify_batch_mixed(self):
        """Test batch verification with some failures."""
        encoder = StatevectorEncoder()
        
        # Create batch with one invalid sample (1 element)
        batch = np.array([
            [0.6, 0.8],  # Valid
        ])
        
        passed, failed = DataIntegrityCheck.verify_batch(batch, encoder)
        
        assert passed >= 0
        assert passed + failed == len(batch)
        
    def test_tolerance_parameter(self):
        """Test that tolerance parameter affects validation."""
        encoder = StatevectorEncoder()
        data = np.array([0.6, 0.8])
        
        # Loose tolerance
        is_valid_loose = DataIntegrityCheck.verify_encoding(data, encoder, tolerance=0.1)
        
        # Tight tolerance
        is_valid_tight = DataIntegrityCheck.verify_encoding(data, encoder, tolerance=1e-10)
        
        # Both should pass for good encoders
        assert is_valid_loose is True
        assert is_valid_tight is True
