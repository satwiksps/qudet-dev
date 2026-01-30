# File: tests/test_reduction/test_auto.py

import numpy as np
import pandas as pd
import pytest
from qudet.transforms.auto import AutoReducer


class TestAutoReducer:
    """Test suite for AutoReducer meta-reducer."""
    
    def test_auto_reducer_no_reduction_needed(self):
        """Test case where data is small enough (no reduction)."""
        ar = AutoReducer(target_qubits=10, max_rows=100)
        
        # Create small dataset (5 features, 20 samples)
        X = pd.DataFrame(np.random.rand(20, 5))
        ar.fit(X)
        
        # No reduction needed
        assert ar.reduction_strategy_ == "none"
        assert len(ar.pipeline_) == 0
        
    def test_auto_reducer_projection_only(self):
        """Test case where only dimensionality reduction is needed."""
        ar = AutoReducer(target_qubits=5, max_rows=1000)
        
        # Create dataset with many features (20 > 5 qubits)
        X = pd.DataFrame(np.random.rand(50, 20))
        ar.fit(X)
        
        # Should add projection reducer
        assert len(ar.pipeline_) > 0
        assert "projection" in [name for name, _ in ar.pipeline_]
        
    def test_auto_reducer_coreset_only(self):
        """Test case where only numerosity reduction is needed."""
        ar = AutoReducer(target_qubits=100, max_rows=100)
        
        # Create dataset with many rows (600 > 100)
        X = pd.DataFrame(np.random.rand(600, 5))
        ar.fit(X)
        
        # Should add coreset reducer
        assert len(ar.pipeline_) > 0
        assert "coreset" in [name for name, _ in ar.pipeline_]
        
    def test_auto_reducer_both_reductions(self):
        """Test case where both dimensionality and numerosity reduction needed."""
        ar = AutoReducer(target_qubits=5, max_rows=100)
        
        # Create large, high-dimensional dataset
        X = pd.DataFrame(np.random.rand(500, 20))
        ar.fit(X)
        
        # Should add both projection and coreset
        assert len(ar.pipeline_) == 2
        names = [name for name, _ in ar.pipeline_]
        assert "projection" in names
        assert "coreset" in names
        
    def test_auto_reducer_transform_no_reduction(self):
        """Test transform when no reduction is applied."""
        ar = AutoReducer(target_qubits=10, max_rows=1000)
        
        X = pd.DataFrame(np.random.rand(20, 5))
        ar.fit(X)
        X_reduced = ar.transform(X)
        
        # Shape should match original (no reduction)
        assert X_reduced.shape == (20, 5)
        
    def test_auto_reducer_transform_with_numpy(self):
        """Test with numpy array input."""
        ar = AutoReducer(target_qubits=10, max_rows=1000)
        
        X = np.random.rand(20, 5)
        ar.fit(X)
        X_reduced = ar.transform(X)
        
        # Should return numpy array
        assert isinstance(X_reduced, np.ndarray)
        assert X_reduced.shape[0] <= 20
        
    def test_auto_reducer_returns_self(self):
        """Test that fit() returns self for method chaining."""
        ar = AutoReducer(target_qubits=10, max_rows=100)
        X = pd.DataFrame(np.random.rand(50, 5))
        
        result = ar.fit(X)
        assert result is ar
        
    def test_auto_reducer_projection_then_coreset(self):
        """Test that reductions are applied in correct order."""
        ar = AutoReducer(target_qubits=3, max_rows=50)
        
        # Large, high-dimensional data
        X = pd.DataFrame(np.random.rand(100, 10))
        ar.fit(X)
        
        # Transform should apply projection first, then coreset
        X_reduced = ar.transform(X)
        
        # Result should have at most 3 features and at most 50 samples
        assert X_reduced.shape[1] <= 3
        assert X_reduced.shape[0] <= 50
        
    def test_auto_reducer_boundary_exactly_at_limits(self):
        """Test boundary case: data exactly at limits."""
        ar = AutoReducer(target_qubits=10, max_rows=100)
        
        # Data exactly at limits
        X = pd.DataFrame(np.random.rand(100, 10))
        ar.fit(X)
        
        # Should not apply reduction
        assert len(ar.pipeline_) == 0
        
    def test_auto_reducer_boundary_one_over_limit(self):
        """Test boundary case: data one over limits."""
        ar = AutoReducer(target_qubits=10, max_rows=100)
        
        # Data one row over and one feature over
        X = pd.DataFrame(np.random.rand(101, 11))
        ar.fit(X)
        
        # Should apply both reductions
        assert len(ar.pipeline_) == 2
        
    def test_auto_reducer_single_sample(self):
        """Test edge case: single sample."""
        ar = AutoReducer(target_qubits=5, max_rows=100)
        
        X = pd.DataFrame(np.random.rand(1, 20))
        ar.fit(X)
        
        # Should only apply projection
        assert "projection" in [name for name, _ in ar.pipeline_]
