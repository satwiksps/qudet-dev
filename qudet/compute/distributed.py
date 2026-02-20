"""
Distributed quantum circuit encoding using Dask.

Splits large datasets into partitions and encodes them in parallel across
CPU workers before batch submission to a quantum backend.
"""

import logging
from typing import List, Union

import numpy as np
import pandas as pd

from qudet.core.base import BaseEncoder

logger = logging.getLogger(__name__)

HAS_DASK = False
try:
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster
    HAS_DASK = True
except ImportError:
    HAS_DASK = False
except Exception:
    HAS_DASK = False


class DistributedQuantumProcessor:
    """Manage parallel quantum encoding of large datasets using Dask.

    Splits massive datasets into chunks, encodes them on parallel CPU workers,
    and prepares them for batch submission to the QPU. Falls back to serial
    processing when Dask is not available.

    Example:
        >>> processor = DistributedQuantumProcessor(
        ...     encoder=StatevectorEncoder(4), n_workers=4
        ... )
        >>> circuits = processor.process_large_dataset(large_df)
        >>> processor.shutdown()
    """

    def __init__(self, encoder: BaseEncoder, n_workers: int = 4) -> None:
        """Initialize the distributed quantum processor.

        Args:
            encoder: Quantum encoder to apply to each data row.
            n_workers: Number of parallel Dask worker processes.
        """
        self.encoder = encoder
        self.n_workers = n_workers
        self.client = None
        self.cluster = None

        if HAS_DASK:
            self.cluster = LocalCluster(n_workers=n_workers, silence_logs=False)
            self.client = Client(self.cluster)
            logger.info("Dask cluster started: %s", self.client.dashboard_link)
        else:
            logger.warning("Dask not installed. Running in serial mode.")

    def process_large_dataset(self, data: Union[pd.DataFrame, "dd.DataFrame"]) -> List:
        """Encode a large dataset in parallel across the Dask cluster.

        Args:
            data: Input data to encode. Rows are treated as individual samples.
                Accepts a pandas or Dask DataFrame.

        Returns:
            List of encoded quantum circuits (one per row).
        """
        if not HAS_DASK:
            logger.info("Dask unavailable — encoding serially")
            if isinstance(data, pd.DataFrame):
                data_values = data.values
            else:
                data_values = data
            return [self.encoder.encode(row) for row in data_values]

        logger.info("Distributing %d rows to %d workers", len(data), self.n_workers)

        if isinstance(data, pd.DataFrame):
            dask_df = dd.from_pandas(data, npartitions=self.n_workers)
        else:
            dask_df = data

        def encode_partition(df_partition):
            """Encode a single partition of data (runs on a worker)."""
            circuits = []
            for row in df_partition.values:
                circuits.append(self.encoder.encode(row))
            return circuits

        results = dask_df.map_partitions(
            encode_partition,
            meta=('circuits', 'object')
        ).compute()

        flat_list = [item for sublist in results if sublist for item in sublist]

        logger.info("Distributed encoding complete: %d circuits", len(flat_list))
        return flat_list

    def shutdown(self) -> None:
        """Gracefully shut down the Dask cluster.

        Should be called when processing is complete to free resources.
        """
        if self.client:
            self.client.close()
            self.cluster.close()
            logger.info("Dask cluster shut down")
