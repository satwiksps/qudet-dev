# File: tests/test_transforms/test_coresets.py

import numpy as np
import pandas as pd
import pytest
from qudet.transforms.coresets import CoresetReducer


def test_coreset_reduction_shape():
    """Test that transform maps each point to its nearest center."""
    X_large = np.random.rand(1000, 5)
    reducer = CoresetReducer(target_size=10)
    reducer.fit(X_large)
    X_reduced = reducer.transform(X_large)

    # Transform maps each input to nearest center - output shape matches input
    assert X_reduced.shape == (1000, 5)
    assert isinstance(X_reduced, np.ndarray)
    # Each row should be one of the 10 cluster centers
    assert len(np.unique(X_reduced, axis=0)) <= 10


def test_pandas_compatibility():
    """Test that it works with Pandas DataFrames."""
    df = pd.DataFrame(np.random.rand(100, 3), columns=["a", "b", "c"])
    reducer = CoresetReducer(target_size=5)
    reducer.fit(df)
    X_reduced = reducer.transform(df)

    assert X_reduced.shape == (100, 3)
    assert len(np.unique(X_reduced, axis=0)) <= 5


def test_fit_transform():
    """Test fit_transform convenience method."""
    X = np.random.rand(50, 4)
    reducer = CoresetReducer(target_size=5)
    result = reducer.fit_transform(X)
    assert result.shape == (50, 4)


def test_unfitted_raises():
    """Test that transform before fit raises error."""
    reducer = CoresetReducer(target_size=5)
    with pytest.raises(Exception):
        reducer.transform(np.random.rand(10, 3))
