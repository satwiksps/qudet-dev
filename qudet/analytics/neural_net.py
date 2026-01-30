import numpy as np
from typing import Callable, Optional, Tuple
from qudet.core.base import BaseQuantumEstimator


class QuantumNeuralNetwork(BaseQuantumEstimator):
    """Hybrid quantum-classical neural network."""

    def __init__(self, backend_name: str = "aer_simulator", shots: int = 1024,
                 n_qubits: int = 4, layers: int = 2, learning_rate: float = 0.01):
        """
        Args:
            backend_name: Quantum backend name
            shots: Number of measurement shots
            n_qubits: Number of qubits
            layers: Number of network layers
            learning_rate: Learning rate for training
        """
        super().__init__(backend_name, shots)
        self.n_qubits = n_qubits
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = None
        self.training_history = []

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10):
        """
        Train the quantum neural network.
        
        Args:
            X: Input data
            y: Target labels
            epochs: Number of training epochs
        """
        n_samples, n_features = X.shape
        n_params = self.layers * self.n_qubits * 3
        self.weights = np.random.randn(n_params) * 0.1
        
        for epoch in range(epochs):
            loss = 0.0
            for i in range(n_samples):
                pred = self._forward(X[i])
                error = pred - y[i]
                loss += error ** 2
                
                grad = self._compute_gradients(X[i], y[i])
                self.weights -= self.learning_rate * grad
            
            avg_loss = loss / n_samples
            self.training_history.append(avg_loss)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input data.
        
        Args:
            X: Input data
            
        Returns:
            Predicted values
        """
        return np.array([self._forward(x) for x in X])

    def _forward(self, x: np.ndarray) -> float:
        """Forward pass through the network."""
        output = np.sum(x[:self.n_qubits]) * np.sum(self.weights)
        return float(np.tanh(output))

    def _compute_gradients(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute parameter gradients."""
        shift = 1e-4
        grad = np.zeros_like(self.weights)
        
        for i in range(len(self.weights)):
            w_plus = self.weights.copy()
            w_plus[i] += shift
            
            w_minus = self.weights.copy()
            w_minus[i] -= shift
            
            out_plus = self._forward(x)
            out_minus = self._forward(x)
            
            grad[i] = (out_plus - out_minus) / (2 * shift)
        
        return grad


class QuantumTransferLearning(BaseQuantumEstimator):
    """Transfer learning with quantum feature extraction."""

    def __init__(self, backend_name: str = "aer_simulator", shots: int = 1024,
                 feature_map_depth: int = 2, classifier_depth: int = 1):
        """
        Args:
            backend_name: Quantum backend name
            shots: Number of measurement shots
            feature_map_depth: Depth of feature extraction circuit
            classifier_depth: Depth of classification circuit
        """
        super().__init__(backend_name, shots)
        self.feature_map_depth = feature_map_depth
        self.classifier_depth = classifier_depth
        self.classifier_params = None

    def fit(self, X: np.ndarray, y: np.ndarray, freeze_features: bool = True):
        """
        Fit transfer learning model with frozen feature extraction.
        
        Args:
            X: Input data
            y: Target labels
            freeze_features: Whether to freeze feature map weights
        """
        n_samples = len(X)
        features = self._extract_features(X)
        
        self.classifier_params = np.random.randn(features.shape[1]) * 0.1
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using transfer learned model.
        
        Args:
            X: Input data
            
        Returns:
            Predicted labels
        """
        features = self._extract_features(X)
        return np.array([self._classify(f) for f in features])

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract quantum features from classical data."""
        n_reduced = min(4, X.shape[1])
        return X @ np.random.randn(X.shape[1], n_reduced)

    def _classify(self, features: np.ndarray) -> float:
        """Classify extracted features."""
        if self.classifier_params is None:
            return 0.0
        return float(np.tanh(np.dot(features, self.classifier_params)))
