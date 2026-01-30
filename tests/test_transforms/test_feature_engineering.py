"""
Tests for feature engineering and selection transforms.

Tests FeatureScaler, FeatureSelector, OutlierRemover, DataBalancer.
"""

import pytest
import numpy as np
import pandas as pd
from qudet.transforms.feature_engineering import (
    FeatureScaler,
    FeatureSelector,
    OutlierRemover,
    DataBalancer
)


class TestFeatureScaler:
    """Test FeatureScaler with various scaling methods."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return np.random.randn(20, 5) * 100 + 500
    
    def test_standard_scaling(self, sample_data):
        """Test standard scaling."""
        scaler = FeatureScaler(method="standard")
        scaler.fit(sample_data)
        scaled = scaler.transform(sample_data)
        
        assert scaled.shape == sample_data.shape
        assert np.allclose(np.mean(scaled, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(scaled, axis=0), 1, atol=1e-10)
    
    def test_minmax_scaling(self, sample_data):
        """Test min-max scaling."""
        scaler = FeatureScaler(method="minmax", feature_range=(0, 1))
        scaler.fit(sample_data)
        scaled = scaler.transform(sample_data)
        
        assert scaled.shape == sample_data.shape
        assert np.all(scaled >= 0) and np.all(scaled <= 1)
    
    def test_robust_scaling(self, sample_data):
        """Test robust scaling."""
        scaler = FeatureScaler(method="robust")
        scaler.fit(sample_data)
        scaled = scaler.transform(sample_data)
        
        assert scaled.shape == sample_data.shape
    
    def test_quantum_scaling(self, sample_data):
        """Test quantum-aware scaling."""
        scaler = FeatureScaler(method="quantum")
        scaler.fit(sample_data)
        scaled = scaler.transform(sample_data)
        
        # Check unit norm per sample
        norms = np.linalg.norm(scaled, axis=1)
        assert np.allclose(norms, 1, atol=1e-10)
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform."""
        scaler = FeatureScaler(method="standard")
        scaled = scaler.fit_transform(sample_data)
        
        assert scaled.shape == sample_data.shape
    
    def test_dataframe_input(self, sample_data):
        """Test with DataFrame input."""
        df = pd.DataFrame(sample_data)
        scaler = FeatureScaler(method="standard")
        scaler.fit(df)
        scaled = scaler.transform(df)
        
        assert scaled.shape == sample_data.shape
    
    def test_unfitted_transform_raises(self, sample_data):
        """Test transform before fit raises error."""
        scaler = FeatureScaler()
        with pytest.raises(ValueError):
            scaler.transform(sample_data)
    
    def test_get_scaling_params(self, sample_data):
        """Test getting scaling parameters."""
        scaler = FeatureScaler(method="standard")
        scaler.fit(sample_data)
        params = scaler.get_scaling_params()
        
        assert "method" in params
        assert "fitted" in params
        assert params["fitted"] is True


class TestFeatureSelector:
    """Test FeatureSelector."""
    
    @pytest.fixture
    def regression_data(self):
        """Create regression data."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = X[:, 0] + 2 * X[:, 1] + np.random.randn(50) * 0.1
        return X, y
    
    def test_f_classif_selection(self):
        """Test f_classif feature selection."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        
        selector = FeatureSelector(n_features=5, method="f_classif")
        selector.fit(X, y)
        selected = selector.transform(X)
        
        assert selected.shape == (100, 5)
    
    def test_mutual_info_selection(self):
        """Test mutual information feature selection."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        
        selector = FeatureSelector(n_features=5, method="mutual_info")
        selector.fit(X, y)
        selected = selector.transform(X)
        
        assert selected.shape == (100, 5)
    
    def test_fit_transform(self):
        """Test fit_transform."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        
        selector = FeatureSelector(n_features=5)
        selected = selector.fit_transform(X, y)
        
        assert selected.shape == (100, 5)
    
    def test_get_selected_features(self):
        """Test getting selected feature indices."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        
        selector = FeatureSelector(n_features=5)
        selector.fit(X, y)
        indices = selector.get_selected_features()
        
        assert len(indices) == 5
        assert all(0 <= idx < 20 for idx in indices)
    
    def test_get_feature_scores(self):
        """Test getting feature scores."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        
        selector = FeatureSelector(n_features=5)
        selector.fit(X, y)
        scores = selector.get_feature_scores()
        
        assert len(scores) == 20
    
    def test_unfitted_transform_raises(self):
        """Test transform before fit raises error."""
        X = np.random.randn(100, 20)
        
        selector = FeatureSelector(n_features=5)
        with pytest.raises(ValueError):
            selector.transform(X)


class TestOutlierRemover:
    """Test OutlierRemover."""
    
    @pytest.fixture
    def data_with_outliers(self):
        """Create data with outliers."""
        np.random.seed(42)
        X = np.random.randn(100, 5) * 10 + 50
        # Add some outliers
        X[-5:] = np.random.randn(5, 5) * 100 + 200
        return X
    
    def test_iqr_detection(self, data_with_outliers):
        """Test IQR outlier detection."""
        remover = OutlierRemover(method="iqr")
        remover.fit(data_with_outliers)
        clean = remover.transform(data_with_outliers)
        
        assert clean.shape[0] <= data_with_outliers.shape[0]
        assert clean.shape[1] == data_with_outliers.shape[1]
    
    def test_zscore_detection(self, data_with_outliers):
        """Test z-score outlier detection."""
        remover = OutlierRemover(method="zscore", threshold=3)
        remover.fit(data_with_outliers)
        clean = remover.transform(data_with_outliers)
        
        assert clean.shape[0] <= data_with_outliers.shape[0]
    
    def test_isolation_detection(self, data_with_outliers):
        """Test isolation outlier detection."""
        remover = OutlierRemover(method="isolation", threshold=2)
        remover.fit(data_with_outliers)
        clean = remover.transform(data_with_outliers)
        
        assert clean.shape[0] <= data_with_outliers.shape[0]
    
    def test_fit_transform(self, data_with_outliers):
        """Test fit_transform."""
        remover = OutlierRemover(method="iqr")
        clean = remover.fit_transform(data_with_outliers)
        
        assert clean.shape[0] <= data_with_outliers.shape[0]
    
    def test_outlier_ratio(self, data_with_outliers):
        """Test getting outlier ratio."""
        remover = OutlierRemover(method="iqr")
        remover.fit(data_with_outliers)
        ratio = remover.get_outlier_ratio()
        
        assert 0 <= ratio <= 1
        assert ratio > 0  # Should have some outliers
    
    def test_unfitted_transform_raises(self, data_with_outliers):
        """Test transform before fit raises error."""
        remover = OutlierRemover()
        with pytest.raises(ValueError):
            remover.transform(data_with_outliers)


class TestDataBalancer:
    """Test DataBalancer."""
    
    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced dataset."""
        np.random.seed(42)
        X1 = np.random.randn(100, 5) + 1
        X2 = np.random.randn(20, 5) - 1
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(100), np.ones(20)])
        return X, y
    
    def test_oversample_balance(self, imbalanced_data):
        """Test oversampling for balancing."""
        X, y = imbalanced_data
        balancer = DataBalancer(method="oversample")
        balancer.fit(X, y)
        X_bal, y_bal = balancer.transform(X, y)
        
        assert len(X_bal) >= len(X)
        # Check class counts are equal
        unique, counts = np.unique(y_bal, return_counts=True)
        assert len(set(counts)) == 1
    
    def test_undersample_balance(self, imbalanced_data):
        """Test undersampling for balancing."""
        X, y = imbalanced_data
        balancer = DataBalancer(method="undersample")
        balancer.fit(X, y)
        X_bal, y_bal = balancer.transform(X, y)
        
        assert len(X_bal) <= len(X)
        # Check class counts are equal
        unique, counts = np.unique(y_bal, return_counts=True)
        assert len(set(counts)) == 1
    
    def test_smote_balance(self, imbalanced_data):
        """Test SMOTE-like balancing."""
        X, y = imbalanced_data
        balancer = DataBalancer(method="smote")
        balancer.fit(X, y)
        X_bal, y_bal = balancer.transform(X, y)
        
        assert len(X_bal) >= len(X)
        # Check class counts
        unique, counts = np.unique(y_bal, return_counts=True)
        assert len(counts) == 2
    
    def test_balance_info(self, imbalanced_data):
        """Test getting balance information."""
        X, y = imbalanced_data
        balancer = DataBalancer()
        balancer.fit(X, y)
        info = balancer.get_balance_info()
        
        assert "class_counts" in info
        assert "imbalance_ratio" in info
        assert info["imbalance_ratio"] > 1
    
    def test_unfitted_transform_raises(self, imbalanced_data):
        """Test transform before fit raises error."""
        X, y = imbalanced_data
        balancer = DataBalancer()
        with pytest.raises(ValueError):
            balancer.transform(X, y)
    
    def test_balanced_preserved_after_balance(self, imbalanced_data):
        """Test that balanced data stays balanced."""
        X, y = imbalanced_data
        balancer = DataBalancer(method="oversample")
        balancer.fit(X, y)
        X_bal, y_bal = balancer.transform(X, y)
        
        assert len(X_bal) == len(y_bal)
        assert X_bal.shape[1] == X.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
