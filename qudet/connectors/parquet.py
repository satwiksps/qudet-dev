"""
High-performance Parquet file loader for quantum data pipelines.

Parquet is the columnar storage format used by Apache Spark, AWS Athena,
Dask, and modern big data platforms. This loader reads row groups efficiently.
"""

import pandas as pd
from typing import Iterator, Tuple, Optional
from .loader import QuantumDataLoader

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


class QuantumParquetLoader:
    """
    High-performance loader for Apache Parquet files.
    
    Reads Parquet files 'Row Group' by 'Row Group' to handle datasets
    larger than available RAM. Each row group is processed into batches
    and converted to quantum circuits.
    
    Parquet is the gold standard for big data because:
    - Columnar storage (only reads needed columns)
    - Compression (reduces disk I/O)
    - Row groups (enables chunk-based processing)
    - Part of Spark, Dask, DuckDB ecosystems
    
    Parameters
    ----------
    filepath : str
        Path to Parquet file.
    batch_size : int, optional
        Number of samples per batch. Default: 1000
    encoder_type : str, optional
        Encoding for quantum circuits: 'angle', 'amplitude', or 'iqp'.
        Default: 'angle'
        
    Attributes
    ----------
    parquet_file : pyarrow.parquet.ParquetFile
        Parquet file handle.
    batch_size : int
        Batch size for iteration.
    encoder_type : str
        Type of quantum encoding.
        
    Examples
    --------
    >>> loader = QuantumParquetLoader('data/large_dataset.parquet', batch_size=500)
    >>> for batch_data, batch_circuits in loader:
    ...     predictions = model.predict(batch_circuits)
    ...     results.extend(predictions)
    
    Notes
    -----
    Requires PyArrow: pip install pyarrow
    """
    
    def __init__(
        self,
        filepath: str,
        batch_size: int = 1000,
        encoder_type: str = 'angle'
    ):
        """Initialize Quantum Parquet Loader."""
        if not HAS_PARQUET:
            raise ImportError(
                "PyArrow not installed. "
                "Install with: pip install pyarrow"
            )
        
        try:
            self.parquet_file = pq.ParquetFile(filepath)
        except Exception as e:
            raise ValueError(f"Could not open Parquet file '{filepath}': {e}")
        
        self.filepath = filepath
        self.batch_size = batch_size
        self.encoder_type = encoder_type
        
        print(f"Parquet file loaded: {filepath}")
        print(f"   • Row groups: {self.parquet_file.num_row_groups}")
        print(f"   • Total rows: {self.parquet_file.metadata.num_rows}")
        print(f"   • Columns: {self.parquet_file.metadata.num_columns}")
        print(f"   • Column names: {self.parquet_file.schema.names}")

    def __iter__(self) -> Iterator[Tuple[pd.DataFrame, list]]:
        """
        Iterate through Parquet file in batches.
        
        Yields chunks from row groups, further split by batch_size.
        Each batch is converted to quantum circuits.
        
        Yields
        ------
        tuple
            (batch_data: pd.DataFrame, batch_circuits: list)
            - batch_data: DataFrame with batch_size rows
            - batch_circuits: List of QuantumCircuit objects
        """
        total_batches = 0
        total_rows = 0
        
        for row_group_idx in range(self.parquet_file.num_row_groups):
            print(f"\nProcessing row group {row_group_idx + 1}/{self.parquet_file.num_row_groups}...")
            
            table_chunk = self.parquet_file.read_row_group(row_group_idx)
            df_chunk = table_chunk.to_pandas()
            
            print(f"   • Rows in group: {len(df_chunk)}")
            
            mini_loader = QuantumDataLoader(
                df_chunk,
                batch_size=self.batch_size,
                encoder_type=self.encoder_type
            )
            
            for batch_data, batch_circuits in mini_loader:
                total_rows += len(batch_data)
                total_batches += 1
                
                yield batch_data, batch_circuits
        
        print(f"\nParquet loading complete!")
        print(f"   • Total batches: {total_batches}")
        print(f"   • Total rows processed: {total_rows}")

    def get_metadata(self) -> dict:
        """
        Return metadata about the Parquet file.
        
        Returns
        -------
        dict
            Metadata including row count, column count, row group info.
        """
        return {
            "filepath": self.filepath,
            "num_rows": self.parquet_file.metadata.num_rows,
            "num_columns": self.parquet_file.metadata.num_columns,
            "num_row_groups": self.parquet_file.num_row_groups,
            "column_names": self.parquet_file.schema.names,
            "batch_size": self.batch_size,
            "encoder_type": self.encoder_type
        }

    def get_schema(self) -> dict:
        """
        Return schema of the Parquet file.
        
        Returns
        -------
        dict
            Column names mapped to PyArrow data types.
        """
        schema = {}
        pa_schema = self.parquet_file.schema
        for i, name in enumerate(pa_schema.names):
            schema[name] = str(pa_schema[i])
        return schema

    def read_sample(self, n_rows: int = 5) -> pd.DataFrame:
        """
        Read first n_rows from the Parquet file (preview).
        
        Parameters
        ----------
        n_rows : int, optional
            Number of rows to read. Default: 5
            
        Returns
        -------
        pd.DataFrame
            First n_rows of the Parquet file.
        """
        if self.parquet_file.num_row_groups == 0:
            raise ValueError("Parquet file is empty")
        
        table = self.parquet_file.read_row_group(0)
        df = table.to_pandas()
        
        return df.head(n_rows)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
