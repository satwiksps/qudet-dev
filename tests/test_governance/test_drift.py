"""
Test suite for Quantum Drift Detector (qudet.validation.drift)
"""

import pytest
import numpy as np
from qudet.governance.drift import QuantumDriftDetector


class MockQuantumKernelComputer:
    """Mock kernel computer for testing."""
    
    def _compute_kernel_matrix(self, X1, X2):
        """Return mock kernel matrix (dot product)."""
        # Simple kernel: dot product matrix
        return np.dot(X1, X2.T)


class TestQuantumDriftDetector:
    """Test cases for QuantumDriftDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return QuantumDriftDetector(n_qubits=4, threshold=0.1)
    
    @pytest.fixture
    def reference_data(self):
        """Create reference (training) data."""
        np.random.seed(42)
        return np.random.randn(20, 5)
    
    @pytest.fixture
    def stable_data(self, reference_data):
        """Create data similar to reference (no drift)."""
        # Add small noise to reference
        return reference_data + np.random.randn(*reference_data.shape) * 0.01
    
    @pytest.fixture
    def drifted_data(self):
        """Create data with different distribution (drift)."""
        np.random.seed(123)
        return np.random.randn(20, 5) * 5 + 5  # Shifted distribution
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.n_qubits == 4
        assert detector.threshold == 0.1
        assert detector.reference_data_ is None
    
    def test_fit_reference(self, detector, reference_data):
        """Test fitting reference data."""
        result = detector.fit_reference(reference_data)
        assert result is detector  # Returns self for chaining
        assert detector.reference_data_ is not None
        assert detector.reference_data_.shape == reference_data.shape
    
    def test_fit_reference_wrong_shape(self, detector):
        """Test error on 1D data."""
        with pytest.raises(ValueError, match="Expected 2D"):
            detector.fit_reference(np.array([1, 2, 3]))
    
    def test_detect_drift_without_reference(self, detector, stable_data):
        """Test error when reference not set."""
        with pytest.raises(ValueError, match="Reference data not set"):
            detector.detect_drift(stable_data)
    
    def test_detect_drift_result_structure(self, detector, reference_data, stable_data):
        """Test drift detection result structure."""
        detector.fit_reference(reference_data)
        result = detector.detect_drift(stable_data)
        
        # Check result keys
        assert "drift_detected" in result
        assert "mmd_score" in result
        assert "threshold" in result
        assert "status" in result
        
        # Check types
        assert isinstance(result["drift_detected"], bool)
        assert isinstance(result["mmd_score"], float)
        assert isinstance(result["threshold"], float)
        assert isinstance(result["status"], str)
    
    def test_detect_drift_no_drift(self, detector, reference_data, stable_data):
        """Test detection with stable data (no drift)."""
        # Mock the kernel computer for predictable results
        detector.fit_reference(reference_data)
        detector._kernel_computer = MockQuantumKernelComputer()
        
        result = detector.detect_drift(stable_data)
        
        # Low MMD score, no drift detected
        assert result["mmd_score"] >= 0  # MMD is non-negative
        assert "STABLE" in result["status"]
    
    def test_detect_drift_wrong_features(self, detector, reference_data):
        """Test error on feature mismatch."""
        detector.fit_reference(reference_data)
        wrong_data = np.random.randn(10, 3)  # Wrong number of features
        
        with pytest.raises(ValueError, match="Feature mismatch"):
            detector.detect_drift(wrong_data)
    
    def test_detect_drift_wrong_shape(self, detector, reference_data):
        """Test error on 1D data."""
        detector.fit_reference(reference_data)
        with pytest.raises(ValueError, match="Expected 2D"):
            detector.detect_drift(np.array([1, 2, 3, 4, 5]))
    
    def test_set_threshold(self, detector):
        """Test threshold update."""
        result = detector.set_threshold(0.5)
        assert result is detector  # Returns self
        assert detector.threshold == 0.5
    
    def test_set_threshold_negative(self, detector):
        """Test error on negative threshold."""
        with pytest.raises(ValueError, match="non-negative"):
            detector.set_threshold(-0.1)
    
    def test_get_config(self, detector, reference_data):
        """Test configuration retrieval."""
        detector.fit_reference(reference_data)
        config = detector.get_config()
        
        assert config["n_qubits"] == 4
        assert config["threshold"] == 0.1
        assert config["reference_shape"] == reference_data.shape
    
    def test_different_thresholds(self, detector, reference_data, drifted_data):
        """Test detection with different thresholds."""
        detector.fit_reference(reference_data)
        detector._kernel_computer = MockQuantumKernelComputer()
        
        # Low threshold - more likely to detect drift
        detector.set_threshold(0.01)
        result_low = detector.detect_drift(drifted_data)
        
        # High threshold - less likely to detect drift
        detector.set_threshold(1.0)
        result_high = detector.detect_drift(drifted_data)
        
        # Both should be valid results
        assert "status" in result_low
        assert "status" in result_high
    
    def test_mmd_score_non_negative(self, detector, reference_data, stable_data):
        """Test that MMD score is always non-negative."""
        detector.fit_reference(reference_data)
        detector._kernel_computer = MockQuantumKernelComputer()
        
        result = detector.detect_drift(stable_data)
        assert result["mmd_score"] >= 0
