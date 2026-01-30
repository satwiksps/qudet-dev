# File: tests/test_orchestration/test_distributed.py

import pytest
import numpy as np
import pandas as pd
from qudet.compute.distributed import DistributedQuantumProcessor
from qudet.encoders.statevector import StatevectorEncoder


class TestDistributedQuantumProcessor:
    """Test suite for DistributedQuantumProcessor."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        encoder = StatevectorEncoder()
        processor = DistributedQuantumProcessor(encoder=encoder, n_workers=2)
        
        assert processor.encoder is encoder
        assert processor.n_workers == 2
        
        # Cleanup
        processor.shutdown()
        
    def test_process_small_dataset(self):
        """Test processing a small dataset (serial fallback)."""
        encoder = StatevectorEncoder()
        processor = DistributedQuantumProcessor(encoder=encoder, n_workers=2)
        
        # Create small dataset
        data = pd.DataFrame(np.array([
            [0.6, 0.8],
            [0.5, 0.866],
        ]))
        
        circuits = processor.process_large_dataset(data)
        
        # Should return list of circuits
        assert isinstance(circuits, list)
        assert len(circuits) == 2
        
        # Cleanup
        processor.shutdown()
        
    def test_process_numpy_array(self):
        """Test processing numpy array input."""
        from qudet.encoders.iqp import IQPEncoder
        encoder = IQPEncoder(n_qubits=2, reps=1)
        processor = DistributedQuantumProcessor(encoder=encoder, n_workers=2)
        
        # Create numpy array data
        data = np.array([
            [0.6, 0.8],
            [0.5, 0.3],
        ])
        
        circuits = processor.process_large_dataset(data)
        
        assert len(circuits) == 2
        
        # Cleanup
        processor.shutdown()
        
    def test_shutdown(self):
        """Test graceful shutdown."""
        encoder = StatevectorEncoder()
        processor = DistributedQuantumProcessor(encoder=encoder, n_workers=1)
        
        # Should not raise error
        processor.shutdown()
        
    def test_different_worker_counts(self):
        """Test with different worker configurations."""
        encoder = StatevectorEncoder()
        
        for n_workers in [1, 2, 4]:
            processor = DistributedQuantumProcessor(encoder=encoder, n_workers=n_workers)
            
            data = pd.DataFrame(np.array([
                [0.6, 0.8],
                [0.5, 0.866],
            ]))
            
            circuits = processor.process_large_dataset(data)
            assert len(circuits) == 2
            
            processor.shutdown()
