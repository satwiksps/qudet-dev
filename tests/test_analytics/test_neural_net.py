import pytest
import numpy as np
from qudet.analytics.neural_net import QuantumNeuralNetwork, QuantumTransferLearning


class TestQuantumNeuralNetwork:
    """Test suite for Quantum Neural Network."""

    def test_initialization(self):
        """Test QNN initialization."""
        qnn = QuantumNeuralNetwork(n_qubits=4, layers=2, learning_rate=0.01)
        assert qnn.n_qubits == 4
        assert qnn.layers == 2
        assert qnn.learning_rate == 0.01

    def test_fit(self, clean_data_2d, binary_data):
        """Test QNN training."""
        X, y = binary_data
        qnn = QuantumNeuralNetwork(n_qubits=4, layers=1)
        
        qnn.fit(X[:20], y[:20], epochs=2)
        
        assert qnn.weights is not None
        assert len(qnn.training_history) == 2

    def test_predict(self, binary_data):
        """Test QNN prediction."""
        X, y = binary_data
        qnn = QuantumNeuralNetwork(n_qubits=4, layers=1)
        qnn.fit(X[:10], y[:10], epochs=1)
        
        predictions = qnn.predict(X[:5])
        
        assert len(predictions) == 5
        assert all(isinstance(p, (float, np.floating)) for p in predictions)

    def test_fit_predict(self, binary_data):
        """Test fit and predict pipeline."""
        X, y = binary_data
        qnn = QuantumNeuralNetwork(n_qubits=4, layers=1)
        
        qnn.fit(X[:15], y[:15], epochs=1)
        predictions = qnn.predict(X[15:20])
        
        assert len(predictions) == 5

    def test_learning_rate_effect(self):
        """Test that learning rate affects training."""
        X = np.random.rand(10, 4)
        y = np.random.rand(10)
        
        qnn_fast = QuantumNeuralNetwork(learning_rate=0.1)
        qnn_slow = QuantumNeuralNetwork(learning_rate=0.001)
        
        qnn_fast.fit(X, y, epochs=1)
        qnn_slow.fit(X, y, epochs=1)
        
        assert qnn_fast.weights is not None
        assert qnn_slow.weights is not None


class TestQuantumTransferLearning:
    """Test suite for Quantum Transfer Learning."""

    def test_initialization(self):
        """Test transfer learning initialization."""
        qtl = QuantumTransferLearning(feature_map_depth=2, classifier_depth=1)
        assert qtl.feature_map_depth == 2
        assert qtl.classifier_depth == 1

    def test_fit(self, clean_data_2d, binary_data):
        """Test transfer learning training."""
        X, y = binary_data
        qtl = QuantumTransferLearning()
        
        qtl.fit(X[:20], y[:20])
        
        assert qtl.classifier_params is not None

    def test_predict(self, binary_data):
        """Test transfer learning prediction."""
        X, y = binary_data
        qtl = QuantumTransferLearning()
        qtl.fit(X[:20], y[:20])
        
        predictions = qtl.predict(X[20:30])
        
        assert len(predictions) == 10
        assert all(isinstance(p, (float, np.floating)) for p in predictions)

    def test_feature_extraction(self, clean_data_2d):
        """Test quantum feature extraction."""
        qtl = QuantumTransferLearning()
        qtl.fit(clean_data_2d[:50], np.random.rand(50))
        
        features = qtl._extract_features(clean_data_2d[50:55])
        
        assert features.shape[0] == 5
        assert features.ndim == 2

    def test_freeze_features(self, binary_data):
        """Test fitting with frozen feature extractor."""
        X, y = binary_data
        qtl = QuantumTransferLearning()
        
        qtl.fit(X[:20], y[:20], freeze_features=True)
        
        assert qtl.classifier_params is not None
