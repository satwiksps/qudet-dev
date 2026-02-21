"""
Connectors module for QuDET.

Provides data loading, streaming, transformation, serialization, and utility
classes for bridging classical data sources with quantum computing pipelines.

Submodules:
    loader: Core batch data loader with quantum encoding.
    sql: SQL database connector with chunked iteration.
    parquet: Apache Parquet file connector with row-group streaming.
    serialization: QASM / JSON / pickle persistence for circuits and models.
    streaming: Streaming buffers, iterators, validators, caches, and aggregators.
    transformation: Data transformations, metadata tracking, quality checks, profiling.
    utilities: Factory, batch processing, format conversion, splitting, sampling.
"""

from . import loader
from . import sql
from . import parquet
from . import serialization
from . import streaming
from . import transformation
from . import utilities
from .loader import QuantumDataLoader
from .sql import QuantumSQLLoader
from .parquet import QuantumParquetLoader
from .serialization import QuantumSerializer
from .streaming import (
    StreamingDataBuffer,
    DataStreamIterator,
    DataValidator,
    DataCacher,
    BatchAggregator
)
from .transformation import (
    DataTransformer,
    DataMetadataTracker,
    DataQualityChecker,
    DataProfiler
)
from .utilities import (
    DataConnectorFactory,
    DataBatchProcessor,
    DataFormatConverter,
    DataSplitter,
    DataSampler
)

__all__ = [
    "loader",
    "sql",
    "parquet",
    "serialization",
    "streaming",
    "transformation",
    "utilities",
    "QuantumDataLoader",
    "QuantumSQLLoader",
    "QuantumParquetLoader",
    "QuantumSerializer",
    "StreamingDataBuffer",
    "DataStreamIterator",
    "DataValidator",
    "DataCacher",
    "BatchAggregator",
    "DataTransformer",
    "DataMetadataTracker",
    "DataQualityChecker",
    "DataProfiler",
    "DataConnectorFactory",
    "DataBatchProcessor",
    "DataFormatConverter",
    "DataSplitter",
    "DataSampler",
]
