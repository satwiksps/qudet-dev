
import numpy as np
import pandas as pd
from typing import List, Union
from qudet.core.base import BaseEncoder

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
    """
    Manages parallel execution of Quantum Encoding using Dask.
    
    Splits massive datasets into chunks, encodes them on parallel CPUs,
    and prepares them for batch submission to the QPU.
    
    Philosophy:
    "Large Scale" means you cannot process 1 million rows in a simple loop.
    This class distributes encoding across multiple CPUs before queuing
    for the Quantum Processing Unit.
    
    Example:
        >>> processor = DistributedQuantumProcessor(encoder=StatevectorEncoder(4), n_workers=4)
        >>> circuits = processor.process_large_dataset(large_df)
        >>> processor.shutdown()
    """
    
    def __init__(self, encoder: BaseEncoder, n_workers: int = 4):
        """
        Initialize Distributed Quantum Processor.
        
        Parameters
        ----------
        encoder : BaseEncoder
            Quantum encoder to apply to each row
        n_workers : int
            Number of parallel worker processes (CPU cores to use)
        """
        self.encoder = encoder
        self.n_workers = n_workers
        self.client = None
        self.cluster = None
        
        if HAS_DASK:
            self.cluster = LocalCluster(n_workers=n_workers, silence_logs=False)
            self.client = Client(self.cluster)
            print(f"--- Dask Cluster Started: {self.client.dashboard_link} ---")
        else:
            print("--- Dask not installed. Running in serial mode. ---")

    def process_large_dataset(self, data: Union[pd.DataFrame, "dd.DataFrame"]) -> List:
        """
        Parallelizes the encoding step across the cluster.
        
        Parameters
        ----------
        data : pd.DataFrame or dd.DataFrame
            Input data to encode (rows are samples)
            
        Returns
        -------
        List
            List of encoded quantum circuits (one per row)
        """
        if not HAS_DASK:
            print("--- Dask unavailable. Encoding serially ---")
            if isinstance(data, pd.DataFrame):
                data_values = data.values
            else:
                data_values = data
            return [self.encoder.encode(row) for row in data_values]

        print(f"--- Distributing {len(data)} rows to {self.n_workers} workers ---")
        
        if isinstance(data, pd.DataFrame):
            dask_df = dd.from_pandas(data, npartitions=self.n_workers)
        else:
            dask_df = data

        def encode_partition(df_partition):
            """
            Encodes a partition of data (called on each worker).
            
            Parameters
            ----------
            df_partition : pd.DataFrame
                A chunk of the data to process
                
            Returns
            -------
            List
                List of encoded quantum circuits
            """
            circuits = []
            for row in df_partition.values:
                circuits.append(self.encoder.encode(row))
            return circuits

        results = dask_df.map_partitions(
            encode_partition, 
            meta=('circuits', 'object')
        ).compute()
        
        flat_list = [item for sublist in results if sublist for item in sublist]
        
        print(f"--- Distributed Encoding Complete: {len(flat_list)} circuits ---")
        return flat_list

    def shutdown(self):
        """
        Gracefully shutdown the Dask cluster.
        Should be called when done processing.
        """
        if self.client:
            self.client.close()
            self.cluster.close()
            print("--- Dask Cluster Shutdown ---")
