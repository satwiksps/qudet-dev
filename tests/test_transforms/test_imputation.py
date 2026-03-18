"""Tests for Quantum Imputation Module."""

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
        return pd.DataFrame(
            {
                "A": np.random.randn(20),
                "B": np.random.randn(20),
                "C": np.random.randn(20),
                "D": np.random.randn(20),
            }
        )

    @pytest.fixture
    def dirty_data(self, clean_data):
        """Generate data with missing values."""
        dirty = clean_data.copy()
        dirty.loc[0, "A"] = np.nan
        dirty.loc[5, "B"] = np.nan
        dirty.loc[10, "C"] = np.nan
        dirty.loc[15, ["A", "D"]] = np.nan
        return dirty

    def test_initialization(self):
        """Test QuantumImputer initialization."""
        imputer = QuantumImputer(n_clusters=3)
        assert imputer.n_clusters == 3

    def test_fit(self, clean_data):
        """Test fitting QuantumImputer."""
        imputer = QuantumImputer(n_clusters=3)
        result = imputer.fit(clean_data)
        assert result is imputer

    def test_transform_with_missing_values(self, clean_data, dirty_data):
        """Test transforming data with missing values."""
        imputer = QuantumImputer(n_clusters=3)
        imputer.fit(clean_data)

        repaired = imputer.transform(dirty_data)

        # Result should be a numpy array with no NaNs
        assert isinstance(repaired, np.ndarray)
        assert not np.isnan(repaired).any()
        assert repaired.shape == dirty_data.shape

    def test_transform_without_missing_values(self, clean_data):
        """Test transform on data without missing values."""
        imputer = QuantumImputer(n_clusters=3)
        imputer.fit(clean_data)

        result = imputer.transform(clean_data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, clean_data.values)

    def test_fit_transform(self, dirty_data):
        """Test fit_transform method."""
        imputer = QuantumImputer(n_clusters=3)
        repaired = imputer.fit_transform(dirty_data)

        assert isinstance(repaired, np.ndarray)
        assert not np.isnan(repaired).any()

    def test_different_cluster_counts(self, clean_data, dirty_data):
        """Test with different numbers of clusters."""
        for n_clusters in [2, 3, 5]:
            imputer = QuantumImputer(n_clusters=n_clusters)
            imputer.fit(clean_data)
            repaired = imputer.transform(dirty_data)
            assert not np.isnan(repaired).any()

    def test_imputed_values_in_data_range(self, clean_data, dirty_data):
        """Test that imputed values are reasonable."""
        imputer = QuantumImputer(n_clusters=3)
        imputer.fit(clean_data)
        repaired = imputer.transform(dirty_data)

        clean_arr = clean_data.values
        for col_idx in range(clean_arr.shape[1]):
            train_min = clean_arr[:, col_idx].min()
            train_max = clean_arr[:, col_idx].max()
            train_std = clean_arr[:, col_idx].std()
            tolerance = 3 * train_std
            assert repaired[:, col_idx].min() >= train_min - tolerance
            assert repaired[:, col_idx].max() <= train_max + tolerance

    def test_output_preserves_shape(self, clean_data, dirty_data):
        """Test that output shape is preserved."""
        imputer = QuantumImputer(n_clusters=3)
        imputer.fit(clean_data)
        repaired = imputer.transform(dirty_data)
        assert repaired.shape == (len(dirty_data), clean_data.shape[1])
