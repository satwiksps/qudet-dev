"""
Test suite for Quantum Parquet Loader (qudet.connectors.parquet)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# Check if PyArrow is available
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


@pytest.mark.skipif(not HAS_PARQUET, reason="PyArrow not installed")
class TestQuantumParquetLoader:
    """Test cases for QuantumParquetLoader class."""
    
    @pytest.fixture
    def sample_parquet_file(self):
        """Create a temporary Parquet file for testing."""
        # Create sample data
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'label': np.random.randint(0, 2, 100)
        })
        
        # Write to temporary Parquet file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            df.to_parquet(tmp.name)
            tmp_path = tmp.name
        
        yield tmp_path
        
        # Cleanup
        Path(tmp_path).unlink()
    
    def test_initialization(self, sample_parquet_file):
        """Test loader initialization."""
        from qudet.connectors.parquet import QuantumParquetLoader
        
        loader = QuantumParquetLoader(sample_parquet_file, batch_size=25)
        
        assert loader.filepath == sample_parquet_file
        assert loader.batch_size == 25
        assert loader.encoder_type == 'angle'
    
    def test_initialization_invalid_file(self):
        """Test error on invalid file path."""
        from qudet.connectors.parquet import QuantumParquetLoader
        
        with pytest.raises(ValueError, match="Could not open"):
            QuantumParquetLoader('/nonexistent/file.parquet')
    
    def test_get_metadata(self, sample_parquet_file):
        """Test metadata retrieval."""
        from qudet.connectors.parquet import QuantumParquetLoader
        
        loader = QuantumParquetLoader(sample_parquet_file)
        metadata = loader.get_metadata()
        
        assert "filepath" in metadata
        assert "num_rows" in metadata
        assert "num_columns" in metadata
        assert "num_row_groups" in metadata
        assert "column_names" in metadata
        
        assert metadata["num_rows"] == 100
        assert metadata["num_columns"] == 4
    
    def test_get_schema(self, sample_parquet_file):
        """Test schema retrieval."""
        from qudet.connectors.parquet import QuantumParquetLoader
        
        loader = QuantumParquetLoader(sample_parquet_file)
        schema = loader.get_schema()
        
        assert "feature1" in schema
        assert "feature2" in schema
        assert "feature3" in schema
        assert "label" in schema
    
    def test_read_sample(self, sample_parquet_file):
        """Test reading sample rows."""
        from qudet.connectors.parquet import QuantumParquetLoader
        
        loader = QuantumParquetLoader(sample_parquet_file)
        sample = loader.read_sample(n_rows=5)
        
        assert isinstance(sample, pd.DataFrame)
        assert len(sample) == 5
        assert list(sample.columns) == ['feature1', 'feature2', 'feature3', 'label']
    
    def test_iteration(self, sample_parquet_file):
        """Test iteration through batches."""
        from qudet.connectors.parquet import QuantumParquetLoader
        
        loader = QuantumParquetLoader(sample_parquet_file, batch_size=25)
        
        total_rows = 0
        batch_count = 0
        
        for batch_data, batch_circuits in loader:
            # batch_data is numpy array from QuantumDataLoader
            assert isinstance(batch_data, np.ndarray)
            assert isinstance(batch_circuits, list)
            
            total_rows += len(batch_data)
            batch_count += 1
        
        assert total_rows == 100
        assert batch_count > 0
    
    def test_context_manager(self, sample_parquet_file):
        """Test context manager usage."""
        from qudet.connectors.parquet import QuantumParquetLoader
        
        with QuantumParquetLoader(sample_parquet_file) as loader:
            assert loader is not None
            assert loader.filepath == sample_parquet_file
    
    def test_encoder_type_parameter(self, sample_parquet_file):
        """Test different encoder types."""
        from qudet.connectors.parquet import QuantumParquetLoader
        
        for encoder_type in ['angle', 'amplitude', 'iqp']:
            loader = QuantumParquetLoader(
                sample_parquet_file,
                encoder_type=encoder_type
            )
            assert loader.encoder_type == encoder_type


class TestQuantumParquetLoaderWithoutDependency:
    """Test cases when PyArrow is not available."""
    
    @pytest.mark.skipif(HAS_PARQUET, reason="PyArrow is installed")
    def test_initialization_without_parquet(self):
        """Test error when PyArrow not available."""
        from qudet.connectors.parquet import QuantumParquetLoader
        
        with pytest.raises(ImportError, match="PyArrow not installed"):
            QuantumParquetLoader('/some/file.parquet')
