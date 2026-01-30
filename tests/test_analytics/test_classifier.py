"""Tests for Quantum Support Vector Classifier"""

import numpy as np
import pytest
from qudet.analytics.classifier import QuantumSVC


class TestQuantumSVC:
    """Test suite for QuantumSVC class."""
    
    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        # Class 0
        X0 = np.random.randn(15, 4) + [0, 0, 0, 0]
        # Class 1
        X1 = np.random.randn(15, 4) + [2, 2, 2, 2]
        
        X = np.vstack([X0, X1])
        y = np.array([0]*15 + [1]*15)
        
        return X, y
    
    def test_initialization(self):
        """Test QuantumSVC initialization."""
        svc = QuantumSVC(n_qubits=4, C=1.0)
        assert svc.n_qubits == 4
        assert svc.C == 1.0
        assert svc.train_data_ is None
    
    def test_fit(self, binary_data):
        """Test fitting QuantumSVC."""
        X, y = binary_data
        svc = QuantumSVC(n_qubits=4, C=1.0)
        result = svc.fit(X, y)
        
        # Should return self
        assert result is svc
        # Should store training data
        assert svc.train_data_ is not None
    
    def test_predict(self, binary_data):
        """Test predictions."""
        X, y = binary_data
        svc = QuantumSVC(n_qubits=4, C=1.0)
        svc.fit(X, y)
        
        predictions = svc.predict(X)
        
        assert predictions.shape == y.shape
        assert set(predictions).issubset({0, 1})
    
    def test_score(self, binary_data):
        """Test scoring."""
        X, y = binary_data
        svc = QuantumSVC(n_qubits=4, C=1.0)
        svc.fit(X, y)
        
        score = svc.score(X, y)
        
        assert 0 <= score <= 1
    
    def test_fit_with_non_binary_labels_raises_error(self):
        """Test that non-binary labels raise error."""
        X = np.random.randn(30, 4)
        y = np.array([0]*10 + [1]*10 + [2]*10)  # 3 classes
        
        svc = QuantumSVC(n_qubits=4)
        
        with pytest.raises(ValueError, match="binary classification"):
            svc.fit(X, y)
    
    def test_predict_without_fit_raises_error(self):
        """Test that predict without fit raises error."""
        X = np.random.randn(10, 4)
        svc = QuantumSVC(n_qubits=4)
        
        with pytest.raises(ValueError, match="Must fit"):
            svc.predict(X)
    
    def test_decision_function(self, binary_data):
        """Test decision function."""
        X, y = binary_data
        svc = QuantumSVC(n_qubits=4, C=1.0)
        svc.fit(X, y)
        
        decisions = svc.decision_function(X)
        
        assert decisions.shape == (len(X),)
        assert all(isinstance(d, (int, float, np.number)) for d in decisions)
    
    def test_different_regularization(self, binary_data):
        """Test different C values."""
        X, y = binary_data
        
        for C in [0.1, 1.0, 10.0]:
            svc = QuantumSVC(n_qubits=4, C=C)
            svc.fit(X, y)
            score = svc.score(X, y)
            assert 0 <= score <= 1
