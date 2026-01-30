# File: tests/test_flow.py

import pytest
import numpy as np
import pandas as pd
from qudet.transforms import CoresetReducer
from qudet.encoders import RotationEncoder
from qudet.analytics import QuantumFeatureSelector

def test_full_pipeline_execution():
    """
    Simulates a Data Engineering Pipeline:
    Raw Data -> Feature Selection -> Reduction -> Encoding
    """
    # 1. Create Synthetic Financial Data
    # 100 rows, 10 columns
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(100, 10), columns=[f"col_{i}" for i in range(10)])
    # Target variable (linear combo of col_0 and col_1)
    y = df['col_0'] * 2 + df['col_1'] + np.random.normal(0, 0.1, 100)

    # 2. STEP 1: Feature Selection (Select top 3 features)
    selector = QuantumFeatureSelector(n_features_to_select=3)
    selector.fit(df, y)
    df_selected = selector.transform(df)
    
    assert df_selected.shape[1] == 3
    assert 'col_0' in df_selected.columns # Should find the most important one

    # 3. STEP 2: Data Reduction (Reduce 100 rows -> 5 quantum-ready points)
    reducer = CoresetReducer(target_size=5)
    reducer.fit(df_selected)
    data_reduced = reducer.transform(df_selected)
    
    assert data_reduced.shape == (5, 3)

    # 4. STEP 3: Quantum Encoding
    # Map the 3 selected features to 3 Qubits
    encoder = RotationEncoder(n_qubits=3)
    # Encode just the first point to check circuit generation
    circuit = encoder.encode(data_reduced[0])
    
    # Check if we got a Qiskit object back
    from qiskit import QuantumCircuit
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 3
