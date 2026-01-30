# File: tests/test_analytics/test_algorithms.py

import pytest
import pandas as pd
import numpy as np
from qudet.analytics import QuantumFeatureSelector, QuantumKMeans


def test_feature_selector_output_shape():
    """Test if the selector respects the k limit (n_features_to_select)."""
    # Create dataset with 5 features
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [1, 0, 1, 0, 1],
        'D': [2, 3, 4, 5, 6],
        'E': [0, 1, 0, 1, 0]
    })
    y = pd.Series([1, 2, 3, 4, 5])  # Target
    
    # Select top 2 features
    selector = QuantumFeatureSelector(n_features_to_select=2)
    selector.fit(df, y)
    selected = selector.transform(df)
    
    # Must have exactly 2 columns
    assert selected.shape[1] == 2, f"Expected 2 features, got {selected.shape[1]}"
    # Must have same number of rows
    assert selected.shape[0] == 5


def test_feature_selector_with_target():
    """Test feature selection with explicit target correlation."""
    # Create dataset where Col A is identical to Target Y
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],  # Inverse correlated
        'C': [1, 0, 1, 0, 1]   # Noise
    })
    y = df['A']
    
    # We want top 1 feature. It SHOULD be 'A' (perfect correlation).
    selector = QuantumFeatureSelector(n_features_to_select=1)
    selector.fit(df, y)
    selected = selector.transform(df)
    
    assert selected.shape[1] == 1
    # The selected feature should be 'A' (most correlated with y)
    assert 'A' in selected.columns or selected.columns[0] == 0  # Index-based or name-based


def test_kmeans_convergence():
    """
    Test if Q-Means runs without crashing and assigns labels.
    """
    X = np.array([
        [0.1, 0.1], [0.2, 0.2],  # Cluster 1
        [0.9, 0.9], [0.8, 0.8]   # Cluster 2
    ])
    X_df = pd.DataFrame(X)
    
    kmeans = QuantumKMeans(n_clusters=2, n_qubits=2, max_iter=3)
    kmeans.fit(X_df)
    
    # Should have 2 centroids
    assert len(kmeans.centroids_) == 2
    # Should have 4 labels (one for each point)
    assert len(kmeans.labels_) == 4
    # Points 0 and 1 should likely have the same label
    assert kmeans.labels_[0] == kmeans.labels_[1]


def test_kmeans_predictions():
    """Test that K-Means can make predictions on new data."""
    X_train = np.array([
        [0.1, 0.1], [0.15, 0.15],
        [0.9, 0.9], [0.95, 0.95]
    ])
    X_test = np.array([[0.12, 0.12], [0.88, 0.88]])
    
    kmeans = QuantumKMeans(n_clusters=2, n_qubits=2, max_iter=2)
    kmeans.fit(pd.DataFrame(X_train))
    preds = kmeans.predict(pd.DataFrame(X_test))
    
    # Should return 2 predictions
    assert len(preds) == 2
    # Predictions should be cluster indices (0 or 1)
    assert all(p in [0, 1] for p in preds)


def test_kmeans_unfitted_error():
    """Test that predict raises error if not fitted."""
    kmeans = QuantumKMeans(n_clusters=2, n_qubits=2, max_iter=1)
    X_test = np.array([[0.1, 0.1], [0.9, 0.9]])
    
    with pytest.raises(RuntimeError):
        kmeans.predict(pd.DataFrame(X_test))
