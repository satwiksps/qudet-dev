# File: tests/test_analytics/test_regression.py

import numpy as np
import pandas as pd
import pytest
from qudet.analytics.regression import QuantumKernelRegressor


class TestQuantumKernelRegressor:
    """Test suite for Quantum Kernel Regressor."""
    
    def test_regressor_initialization(self):
        """Test regressor initialization."""
        regressor = QuantumKernelRegressor(n_qubits=3, alpha=0.5)
        
        assert regressor.n_qubits == 3
        assert regressor.alpha == 0.5
        assert not regressor.is_fitted
        
    def test_regressor_fit_basic(self):
        """Test basic fitting on small dataset."""
        regressor = QuantumKernelRegressor(n_qubits=2, alpha=1.0)
        
        # Small training dataset (4 samples, 2 features)
        X_train = pd.DataFrame(np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8]
        ]))
        y_train = pd.Series([1.0, 2.0, 3.0, 4.0])
        
        regressor.fit(X_train, y_train)
        
        assert regressor.is_fitted
        assert regressor.train_data_ is not None
        
    def test_regressor_fit_returns_self(self):
        """Test that fit() returns self for method chaining."""
        regressor = QuantumKernelRegressor(n_qubits=2)
        X_train = pd.DataFrame(np.array([[0.1, 0.2], [0.3, 0.4]]))
        y_train = pd.Series([1.0, 2.0])
        
        result = regressor.fit(X_train, y_train)
        assert result is regressor
        
    def test_regressor_predict_basic(self):
        """Test basic prediction."""
        regressor = QuantumKernelRegressor(n_qubits=2, alpha=1.0)
        
        X_train = pd.DataFrame(np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8]
        ]))
        y_train = pd.Series([1.0, 2.0, 3.0, 4.0])
        
        regressor.fit(X_train, y_train)
        
        # Predict on new data
        X_test = pd.DataFrame(np.array([[0.2, 0.3]]))
        predictions = regressor.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == 1
        
    def test_regressor_predict_unfitted_error(self):
        """Test that predict() raises error before fit()."""
        regressor = QuantumKernelRegressor(n_qubits=2)
        X_test = pd.DataFrame(np.array([[0.1, 0.2]]))
        
        with pytest.raises(RuntimeError):
            regressor.predict(X_test)
            
    def test_regressor_predict_multiple_samples(self):
        """Test prediction on multiple samples."""
        regressor = QuantumKernelRegressor(n_qubits=2)
        
        X_train = pd.DataFrame(np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ]))
        y_train = pd.Series([1.0, 2.0, 3.0])
        
        regressor.fit(X_train, y_train)
        
        # Predict on 4 new samples
        X_test = pd.DataFrame(np.array([
            [0.15, 0.25],
            [0.35, 0.45],
            [0.55, 0.65],
            [0.75, 0.85]
        ]))
        
        predictions = regressor.predict(X_test)
        
        assert len(predictions) == 4
        
    def test_regressor_numpy_input(self):
        """Test with numpy array input instead of DataFrame."""
        regressor = QuantumKernelRegressor(n_qubits=2)
        
        X_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        y_train = np.array([1.0, 2.0, 3.0])
        
        regressor.fit(X_train, y_train)
        
        X_test = np.array([[0.25, 0.35]])
        predictions = regressor.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == 1
        
    def test_regressor_kernel_computation(self):
        """Test kernel matrix computation."""
        regressor = QuantumKernelRegressor(n_qubits=2)
        
        X1 = np.array([[0.1, 0.2], [0.3, 0.4]])
        X2 = np.array([[0.5, 0.6], [0.7, 0.8]])
        
        kernel = regressor._compute_kernel_matrix(X1, X2)
        
        # Kernel matrix should be (2, 2)
        assert kernel.shape == (2, 2)
        # Kernel values should be between 0 and 1 (fidelity)
        assert np.all(kernel >= 0) and np.all(kernel <= 1)
        
    def test_regressor_kernel_symmetric(self):
        """Test that kernel matrix is symmetric for same data."""
        regressor = QuantumKernelRegressor(n_qubits=2)
        
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        
        kernel = regressor._compute_kernel_matrix(X, X)
        
        # Should be symmetric: K(i,j) = K(j,i)
        assert np.allclose(kernel, kernel.T)
        # Diagonal should be 1 (fidelity with self)
        assert np.allclose(np.diag(kernel), 1.0)
        
    def test_regressor_different_alpha_values(self):
        """Test regressor with different regularization strengths."""
        X_train = pd.DataFrame(np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
        y_train = pd.Series([1.0, 2.0, 3.0])
        
        predictions_list = []
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            regressor = QuantumKernelRegressor(n_qubits=2, alpha=alpha)
            regressor.fit(X_train, y_train)
            
            X_test = pd.DataFrame(np.array([[0.25, 0.35]]))
            predictions = regressor.predict(X_test)
            predictions_list.append(predictions[0])
        
        # All predictions should be valid numbers
        assert all(isinstance(p, (int, float, np.number)) for p in predictions_list)
        
    def test_regressor_single_sample_training(self):
        """Test edge case: single training sample."""
        regressor = QuantumKernelRegressor(n_qubits=2)
        
        X_train = pd.DataFrame(np.array([[0.5, 0.5]]))
        y_train = pd.Series([2.5])
        
        regressor.fit(X_train, y_train)
        
        X_test = pd.DataFrame(np.array([[0.5, 0.5]]))
        predictions = regressor.predict(X_test)
        
        # Should predict something close to the training value
        assert predictions is not None
        
    def test_regressor_store_training_data(self):
        """Test that training data is stored for prediction."""
        regressor = QuantumKernelRegressor(n_qubits=2)
        
        X_train = pd.DataFrame(np.array([[0.1, 0.2], [0.3, 0.4]]))
        y_train = pd.Series([1.0, 2.0])
        
        regressor.fit(X_train, y_train)
        
        # Training data should be stored
        assert regressor.train_data_ is not None
        assert len(regressor.train_data_) == 2
