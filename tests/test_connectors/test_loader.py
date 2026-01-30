# File: tests/test_connectors/test_loader.py

import pytest
import pandas as pd
import numpy as np
from qiskit import QuantumCircuit
from qudet.connectors import QuantumDataLoader


def test_loader_batching():
    """
    Test if a dataset of 100 rows with batch_size=32 
    yields correct number of batches (3 full + 1 partial).
    """
    # Create dummy dataframe (100 rows, 4 cols)
    df = pd.DataFrame(np.random.rand(100, 4))
    
    loader = QuantumDataLoader(df, batch_size=32, encoder_type='angle')
    
    # Check length (ceil(100/32) = 4)
    assert len(loader) == 4
    
    batches = list(loader)
    
    # Check first batch size
    assert len(batches[0][0]) == 32
    # Check last batch size (100 % 32 = 4)
    assert len(batches[-1][0]) == 4


def test_loader_circuit_generation():
    """
    Test if the loader actually yields Quantum Circuits.
    """
    df = pd.DataFrame(np.random.rand(10, 2))
    loader = QuantumDataLoader(df, batch_size=5, encoder_type='angle')
    
    data, circuits = next(iter(loader))
    
    assert isinstance(circuits[0], QuantumCircuit)
    assert len(circuits) == 5
    assert data.shape == (5, 2)


def test_loader_batch_size_one():
    """Test edge case: batch_size=1."""
    df = pd.DataFrame(np.random.rand(5, 3))
    loader = QuantumDataLoader(df, batch_size=1, encoder_type='angle')
    
    assert len(loader) == 5
    
    for data, circuits in loader:
        assert len(data) == 1
        assert len(circuits) == 1


def test_loader_batch_size_larger_than_data():
    """Test edge case: batch_size > dataset size."""
    df = pd.DataFrame(np.random.rand(5, 3))
    loader = QuantumDataLoader(df, batch_size=100, encoder_type='angle')
    
    # Should only have 1 batch with 5 samples
    assert len(loader) == 1
    
    data, circuits = next(iter(loader))
    assert len(data) == 5
    assert len(circuits) == 5


def test_loader_amplitude_encoding():
    """Test data loader with amplitude encoder."""
    df = pd.DataFrame(np.random.rand(20, 8))
    loader = QuantumDataLoader(df, batch_size=10, encoder_type='amplitude')
    
    assert len(loader) == 2
    
    data, circuits = next(iter(loader))
    assert len(circuits) == 10
    assert all(isinstance(c, QuantumCircuit) for c in circuits)


def test_loader_invalid_encoder():
    """Test that invalid encoder type raises error."""
    df = pd.DataFrame(np.random.rand(10, 2))
    
    with pytest.raises(ValueError):
        loader = QuantumDataLoader(df, batch_size=5, encoder_type='invalid')
