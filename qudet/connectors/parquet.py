"""
High-performance Parquet file loader for quantum data pipelines.

Parquet is the columnar storage format used by Apache Spark, AWS Athena,
Dask, and modern big data platforms. This loader reads row groups efficiently.
"""

import logging
from typing import Iterator, Tuple

import pandas as pd

from .loader import QuantumDataLoader

logger = logging.getLogger(__name__)

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


class QuantumParquetLoader:
    """High-performance loader for Apache Parquet files.

    Reads Parquet files row-group by row-group to handle datasets larger
    than available RAM.  Each row group is processed into batches and
    converted to quantum circuits via :class:`QuantumDataLoader`.

    Parquet is the gold standard for big data because:

    * Columnar storage (only reads needed columns).
    * Compression (reduces disk I/O).
    * Row groups (enables chunk-based processing).
    * Part of Spark, Dask, DuckDB ecosystems.

    Args:
        filepath: Path to Parquet file.
        batch_size: Number of samples per batch.
        encoder_type: Encoding for quantum circuits — ``'angle'`` or
            ``'amplitude'``.

    Attributes:
        parquet_file (pyarrow.parquet.ParquetFile): Parquet file handle.
        batch_size (int): Batch size for iteration.
        encoder_type (str): Type of quantum encoding.

    Raises:
        ImportError: If PyArrow is not installed.
        ValueError: If *filepath* cannot be opened.

    Example:
        >>> loader = QuantumParquetLoader('data/large.parquet', batch_size=500)
        >>> for batch_data, batch_circuits in loader:
        ...     predictions = model.predict(batch_circuits)
        ...     results.extend(predictions)

    Note:
        Requires PyArrow: ``pip install pyarrow``
    """

    def __init__(
        self,
        filepath: str,
        batch_size: int = 1000,
        encoder_type: str = "angle",
    ) -> None:
        if not HAS_PARQUET:
            raise ImportError(
                "PyArrow not installed. Install with: pip install pyarrow"
            )

        try:
            self.parquet_file = pq.ParquetFile(filepath)
        except Exception as exc:
            raise ValueError(
                f"Could not open Parquet file '{filepath}': {exc}"
            ) from exc

        self.filepath = filepath
        self.batch_size = batch_size
        self.encoder_type = encoder_type

        logger.info("Parquet file loaded: %s", filepath)
        logger.info("   • Row groups: %d", self.parquet_file.num_row_groups)
        logger.info("   • Total rows: %d", self.parquet_file.metadata.num_rows)
        logger.info("   • Columns: %d", self.parquet_file.metadata.num_columns)
        logger.info("   • Column names: %s", self.parquet_file.schema.names)

    def __iter__(self) -> Iterator[Tuple[pd.DataFrame, list]]:
        """Iterate through Parquet file in batches.

        Yields chunks from row groups, further split by ``batch_size``.
        Each batch is converted to quantum circuits.

        Yields:
            Tuple of ``(batch_data, batch_circuits)`` where *batch_data*
            is a ``DataFrame`` and *batch_circuits* is a list of
            ``QuantumCircuit`` objects.
        """
        total_batches = 0
        total_rows = 0

        for row_group_idx in range(self.parquet_file.num_row_groups):
            logger.info(
                "Processing row group %d/%d...",
                row_group_idx + 1,
                self.parquet_file.num_row_groups,
            )

            table_chunk = self.parquet_file.read_row_group(row_group_idx)
            df_chunk = table_chunk.to_pandas()

            logger.debug("   • Rows in group: %d", len(df_chunk))

            mini_loader = QuantumDataLoader(
                df_chunk,
                batch_size=self.batch_size,
                encoder_type=self.encoder_type,
            )

            for batch_data, batch_circuits in mini_loader:
                total_rows += len(batch_data)
                total_batches += 1
                yield batch_data, batch_circuits

        logger.info("Parquet loading complete!")
        logger.info("   • Total batches: %d", total_batches)
        logger.info("   • Total rows processed: %d", total_rows)

    def get_metadata(self) -> dict:
        """Return metadata about the Parquet file.

        Returns:
            Dictionary with row count, column count, row-group info, and
            current loader settings.
        """
        return {
            "filepath": self.filepath,
            "num_rows": self.parquet_file.metadata.num_rows,
            "num_columns": self.parquet_file.metadata.num_columns,
            "num_row_groups": self.parquet_file.num_row_groups,
            "column_names": self.parquet_file.schema.names,
            "batch_size": self.batch_size,
            "encoder_type": self.encoder_type,
        }

    def get_schema(self) -> dict:
        """Return schema of the Parquet file.

        Returns:
            Column names mapped to their PyArrow data-type strings.
        """
        schema: dict = {}
        pa_schema = self.parquet_file.schema
        for i, name in enumerate(pa_schema.names):
            schema[name] = str(pa_schema[i])
        return schema

    def read_sample(self, n_rows: int = 5) -> pd.DataFrame:
        """Read first *n_rows* from the Parquet file (preview).

        Args:
            n_rows: Number of rows to read.

        Returns:
            A ``DataFrame`` containing the first *n_rows* rows.

        Raises:
            ValueError: If the Parquet file contains no row groups.
        """
        if self.parquet_file.num_row_groups == 0:
            raise ValueError("Parquet file is empty")

        table = self.parquet_file.read_row_group(0)
        df = table.to_pandas()
        return df.head(n_rows)

    # -- context-manager support ------------------------------------------

    def __enter__(self) -> "QuantumParquetLoader":
        """Enter the runtime context and return self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the runtime context, closing the underlying Parquet file."""
        if hasattr(self, "parquet_file") and self.parquet_file is not None:
            # ParquetFile wraps a file-handle; close it if available.
            reader = getattr(self.parquet_file, "reader", None)
            if reader is not None and hasattr(reader, "close"):
                reader.close()
            self.parquet_file = None
            logger.debug("Parquet file handle closed.")
