# File: tests/test_reduction/test_coresets.py

import pytest
import numpy as np
import pandas as pd
from qudet.transforms.coresets import CoresetReducer

def test_coreset_reduction_shape():
    """Test if the reducer actually reduces the data size."""
    # 1. Create fake big data (1000 rows, 5 columns)
    X_large = np.random.rand(1000, 5)
    
    # 2. Initialize reducer to keep only 10 points
    reducer = CoresetReducer(target_size=10)
    
    # 3. Run the pipeline
    reducer.fit(X_large)
    X_reduced = reducer.transform(X_large)
    
    # 4. Assertions
    assert X_reduced.shape == (10, 5)
    assert isinstance(X_reduced, np.ndarray)

def test_pandas_compatibility():
    """Test if it works with Pandas DataFrames."""
    df = pd.DataFrame(np.random.rand(100, 3), columns=['a', 'b', 'c'])
    reducer = CoresetReducer(target_size=5)
    
    reducer.fit(df)
    X_reduced = reducer.transform(df)
    
    assert X_reduced.shape == (5, 3)
