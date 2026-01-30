"""Tests for Quantum Cleaning Module (Imputation)"""

import numpy as np
import pandas as pd
import pytest
from qudet.transforms.imputation import QuantumImputer


class TestQuantumImputer:
    """Test suite for QuantumImputer class."""
    
    @pytest.fixture
    def clean_data(self):
        """Generate clean data for training."""
        np.random.seed(42)
        data = pd.DataFrame({
            'A': np.random.randn(20),
            'B': np.random.randn(20),
            'C': np.random.randn(20),
            'D': np.random.randn(20)
        })
        return data
    
    @pytest.fixture
    def dirty_data(self, clean_data):
        """Generate data with missing values."""
        dirty = clean_data.copy()
        # Add some NaNs
        dirty.loc[0, 'A'] = np.nan
        dirty.loc[5, 'B'] = np.nan
        dirty.loc[10, 'C'] = np.nan
        dirty.loc[15, ['A', 'D']] = np.nan
        return dirty
    
    def test_initialization(self):
        """Test QuantumImputer initialization."""
        imputer = QuantumImputer(n_clusters=3)
        assert imputer.n_clusters == 3
    
    def test_fit(self, clean_data):
        """Test fitting QuantumImputer."""
        imputer = QuantumImputer(n_clusters=3)
        result = imputer.fit(clean_data)
        
        # Should return self
        assert result is imputer
    
    def test_transform_with_missing_values(self, clean_data, dirty_data):
        """Test transforming data with missing values."""
        imputer = QuantumImputer(n_clusters=3)
        imputer.fit(clean_data)
        
        # Count NaNs before imputation
        nan_before = dirty_data.isna().sum().sum()
        assert nan_before > 0
        
        # Impute
        repaired = imputer.transform(dirty_data)
        
        # Check no NaNs remain
        assert repaired.isna().sum().sum() == 0
        
        # Check non-missing values unchanged
        for col in dirty_data.columns:
            mask = dirty_data[col].notna()
            np.testing.assert_array_almost_equal(
                repaired.loc[mask, col].values,
                dirty_data.loc[mask, col].values
            )
    
    def test_transform_without_missing_values(self, clean_data):
        """Test transform on data without missing values."""
        imputer = QuantumImputer(n_clusters=3)
        imputer.fit(clean_data)
        
        # Transform clean data
        result = imputer.transform(clean_data)
        
        # Should be identical
        pd.testing.assert_frame_equal(result, clean_data)
    
    def test_fit_transform(self, clean_data, dirty_data):
        """Test fit_transform method."""
        imputer = QuantumImputer(n_clusters=3)
        
        # Train on clean, impute dirty
        repaired = imputer.fit_transform(dirty_data)
        
        # Check no NaNs
        assert repaired.isna().sum().sum() == 0
    
    def test_different_cluster_counts(self, clean_data, dirty_data):
        """Test with different numbers of clusters."""
        for n_clusters in [2, 3, 5]:
            imputer = QuantumImputer(n_clusters=n_clusters)
            imputer.fit(clean_data)
            
            repaired = imputer.transform(dirty_data)
            assert repaired.isna().sum().sum() == 0
    
    def test_imputed_values_in_data_range(self, clean_data, dirty_data):
        """Test that imputed values are reasonable."""
        imputer = QuantumImputer(n_clusters=3)
        imputer.fit(clean_data)
        
        repaired = imputer.transform(dirty_data)
        
        # Imputed values should be roughly in the range of the training data
        for col in clean_data.columns:
            train_min = clean_data[col].min()
            train_max = clean_data[col].max()
            train_std = clean_data[col].std()
            
            repaired_col = repaired[col]
            
            # Allow some tolerance
            tolerance = 3 * train_std
            assert repaired_col.min() >= train_min - tolerance
            assert repaired_col.max() <= train_max + tolerance
    
    def test_output_preserves_columns(self, clean_data, dirty_data):
        """Test that columns are preserved."""
        imputer = QuantumImputer(n_clusters=3)
        imputer.fit(clean_data)
        
        repaired = imputer.transform(dirty_data)
        
        assert list(repaired.columns) == list(dirty_data.columns)
        assert len(repaired) == len(dirty_data)
