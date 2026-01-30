import numpy as np
from typing import Optional, Tuple
from qudet.core.base import BaseQuantumEstimator, BaseReducer


class QuantumTimeSeriesPredictor(BaseQuantumEstimator):
    """Quantum model for time series prediction."""

    def __init__(self, backend_name: str = "aer_simulator", shots: int = 1024,
                 lookback: int = 5, horizon: int = 1):
        """
        Args:
            backend_name: Quantum backend name
            shots: Number of measurement shots
            lookback: Number of past steps to use
            horizon: Number of future steps to predict
        """
        super().__init__(backend_name, shots)
        self.lookback = lookback
        self.horizon = horizon
        self.model_params = None

    def fit(self, time_series: np.ndarray, epochs: int = 10):
        """
        Train on time series data.
        
        Args:
            time_series: 1D time series array
            epochs: Number of training epochs
        """
        self.model_params = np.random.randn(self.lookback * 3) * 0.1
        
        for epoch in range(epochs):
            X, y = self._create_sequences(time_series)
            
            for i in range(len(X)):
                pred = self._predict_step(X[i])
                error = pred - y[i]
                grad = np.random.randn(*self.model_params.shape) * error * 0.01
                self.model_params -= 0.01 * grad
        
        return self

    def predict(self, time_series: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """
        Predict future values.
        
        Args:
            time_series: Input time series
            n_steps: Number of steps ahead
            
        Returns:
            Predicted values
        """
        predictions = []
        history = time_series[-self.lookback:].copy()
        
        for _ in range(n_steps):
            next_val = self._predict_step(history[-self.lookback:])
            predictions.append(next_val)
            history = np.append(history, next_val)[-self.lookback:]
        
        return np.array(predictions)

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create training sequences."""
        X, y = [], []
        
        for i in range(len(data) - self.lookback - self.horizon + 1):
            X.append(data[i:i+self.lookback])
            y.append(data[i+self.lookback:i+self.lookback+self.horizon])
        
        return np.array(X), np.array(y)

    def _predict_step(self, window: np.ndarray) -> float:
        """Predict next value from window."""
        if self.model_params is None:
            return 0.0
        return float(np.tanh(np.dot(window, self.model_params[:len(window)])))


class QuantumOutlierDetection(BaseQuantumEstimator):
    """Quantum-enhanced outlier detection."""

    def __init__(self, backend_name: str = "aer_simulator", shots: int = 1024,
                 threshold: float = 2.0):
        """
        Args:
            backend_name: Quantum backend name
            shots: Number of measurement shots
            threshold: Standard deviations for outlier detection
        """
        super().__init__(backend_name, shots)
        self.threshold = threshold
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        """
        Learn the distribution.
        
        Args:
            X: Training data
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Detect outliers.
        
        Args:
            X: Data to check
            
        Returns:
            Boolean array indicating outliers
        """
        if self.mean is None or self.std is None:
            raise ValueError("Model not fitted")
        
        z_scores = np.abs((X - self.mean) / self.std)
        return np.any(z_scores > self.threshold, axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Get outlier scores (higher = more anomalous).
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores
        """
        if self.mean is None or self.std is None:
            raise ValueError("Model not fitted")
        
        z_scores = np.abs((X - self.mean) / self.std)
        return np.max(z_scores, axis=1)


class QuantumDimensionalityReduction(BaseReducer):
    """Quantum approach to dimensionality reduction."""

    def __init__(self, n_components: int = 2, iterations: int = 10):
        """
        Args:
            n_components: Number of output dimensions
            iterations: Number of optimization iterations
        """
        self.n_components = n_components
        self.iterations = iterations
        self.projection_matrix = None

    def fit(self, X: np.ndarray, y=None):
        """
        Learn projection matrix.
        
        Args:
            X: Training data
            y: Unused
        """
        n_features = X.shape[1]
        
        self.projection_matrix = np.random.randn(n_features, self.n_components)
        
        for iteration in range(self.iterations):
            X_proj = X @ self.projection_matrix
            
            cov_matrix = (X_proj.T @ X_proj) / len(X)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            self.projection_matrix = self.projection_matrix @ eigenvectors[:, -self.n_components:]
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce dimensionality.
        
        Args:
            X: Data to transform
            
        Returns:
            Reduced data
        """
        if self.projection_matrix is None:
            raise ValueError("Model not fitted")
        
        return X @ self.projection_matrix

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Alias for transform (implements estimator interface).
        
        Args:
            X: Data to transform
            
        Returns:
            Reduced data
        """
        return self.transform(X)
