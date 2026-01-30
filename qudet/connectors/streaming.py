import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union, Iterator, Tuple
from collections import deque


class StreamingDataBuffer:
    """Buffer for streaming data with sliding window capabilities."""

    def __init__(self, buffer_size: int = 1000, window_size: int = 100):
        """
        Args:
            buffer_size: Maximum buffer capacity
            window_size: Size of sliding window for aggregation
        """
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.buffer = deque(maxlen=buffer_size)
        self.metadata = {}

    def add_batch(self, data: np.ndarray, batch_id: str = None):
        """
        Add batch to buffer.
        
        Args:
            data: Batch data to add
            batch_id: Optional batch identifier
        """
        for row in data:
            self.buffer.append(row)
        
        if batch_id:
            self.metadata[batch_id] = {
                'size': len(data),
                'timestamp': pd.Timestamp.now()
            }

    def get_sliding_window(self) -> np.ndarray:
        """
        Get sliding window of recent data.
        
        Returns:
            Window data as numpy array
        """
        window_size = min(self.window_size, len(self.buffer))
        if window_size == 0:
            return np.array([])
        
        recent_data = list(self.buffer)[-window_size:]
        return np.array(recent_data)

    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {'size': 0, 'mean': None, 'std': None}
        
        data = np.array(list(self.buffer))
        return {
            'size': len(self.buffer),
            'capacity': self.buffer_size,
            'utilization': len(self.buffer) / self.buffer_size,
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data),
            'max': np.max(data)
        }

    def flush(self) -> np.ndarray:
        """Flush buffer and return all data."""
        data = np.array(list(self.buffer))
        self.buffer.clear()
        self.metadata.clear()
        return data


class DataStreamIterator:
    """Iterator for streaming data from various sources."""

    def __init__(self, data_source: Union[np.ndarray, pd.DataFrame], 
                 batch_size: int = 32, shuffle: bool = False):
        """
        Args:
            data_source: Data array or DataFrame
            batch_size: Batch size for streaming
            shuffle: Whether to shuffle batches
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if isinstance(data_source, pd.DataFrame):
            self.data = data_source.values
        else:
            self.data = np.array(data_source)
        
        self.n_samples = len(self.data)
        self.indices = np.arange(self.n_samples)
        
        if shuffle:
            np.random.shuffle(self.indices)
        
        self.current_batch = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over batches."""
        self.current_batch = 0
        return self

    def __next__(self) -> np.ndarray:
        """Get next batch."""
        start = self.current_batch * self.batch_size
        
        if start >= self.n_samples:
            raise StopIteration
        
        end = min(start + self.batch_size, self.n_samples)
        batch_indices = self.indices[start:end]
        batch = self.data[batch_indices]
        
        self.current_batch += 1
        return batch

    def get_batch_count(self) -> int:
        """Get total number of batches."""
        return int(np.ceil(self.n_samples / self.batch_size))


class DataValidator:
    """Validate data integrity and schema compliance."""

    def __init__(self, expected_shape: Tuple = None, 
                 expected_dtypes: Dict = None, 
                 allow_nan: bool = False):
        """
        Args:
            expected_shape: Expected data shape
            expected_dtypes: Expected column data types
            allow_nan: Whether to allow NaN values
        """
        self.expected_shape = expected_shape
        self.expected_dtypes = expected_dtypes
        self.allow_nan = allow_nan
        self.validation_errors = []

    def validate(self, data: Union[np.ndarray, pd.DataFrame]) -> bool:
        """
        Validate data.
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid, False otherwise
        """
        self.validation_errors = []
        
        if isinstance(data, pd.DataFrame):
            return self._validate_dataframe(data)
        else:
            return self._validate_array(data)

    def _validate_array(self, data: np.ndarray) -> bool:
        """Validate numpy array."""
        if self.expected_shape:
            if len(data.shape) != len(self.expected_shape):
                self.validation_errors.append(
                    f"Shape dimension mismatch: {data.shape} vs {self.expected_shape}"
                )
                return False
        
        if not self.allow_nan and np.isnan(data).any():
            self.validation_errors.append("Array contains NaN values")
            return False
        
        if not np.isfinite(data).all():
            self.validation_errors.append("Array contains non-finite values")
            return False
        
        return len(self.validation_errors) == 0

    def _validate_dataframe(self, data: pd.DataFrame) -> bool:
        """Validate DataFrame."""
        if self.expected_dtypes:
            for col, dtype in self.expected_dtypes.items():
                if col not in data.columns:
                    self.validation_errors.append(f"Missing column: {col}")
                elif not pd.api.types.is_dtype_equal(data[col].dtype, dtype):
                    self.validation_errors.append(
                        f"Column {col} dtype mismatch: {data[col].dtype} vs {dtype}"
                    )
        
        if not self.allow_nan and data.isna().any().any():
            self.validation_errors.append("DataFrame contains NaN values")
            return False
        
        return len(self.validation_errors) == 0

    def get_errors(self) -> List[str]:
        """Get validation errors."""
        return self.validation_errors


class DataCacher:
    """Cache data with LRU eviction policy."""

    def __init__(self, max_cache_size: int = 100):
        """
        Args:
            max_cache_size: Maximum number of cached items
        """
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_count = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found
        """
        if key in self.cache:
            self.cache_hits += 1
            self.access_count[key] += 1
            return self.cache[key]
        
        self.cache_misses += 1
        return None

    def put(self, key: str, data: np.ndarray):
        """
        Put item in cache with LRU eviction.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        if key in self.cache:
            self.cache[key] = data
            self.access_count[key] += 1
        else:
            if len(self.cache) >= self.max_cache_size:
                lru_key = min(self.access_count, key=self.access_count.get)
                del self.cache[lru_key]
                del self.access_count[lru_key]
            
            self.cache[key] = data
            self.access_count[key] = 1

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cached_items': len(self.cache),
            'max_size': self.max_cache_size
        }

    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()


class BatchAggregator:
    """Aggregate batches with statistics tracking."""

    def __init__(self, aggregation_method: str = "mean"):
        """
        Args:
            aggregation_method: Aggregation method (mean, max, min, sum)
        """
        self.aggregation_method = aggregation_method
        self.batches = []
        self.stats = {}

    def add_batch(self, batch: np.ndarray):
        """Add batch to aggregator."""
        self.batches.append(batch)

    def aggregate(self) -> np.ndarray:
        """
        Aggregate all batches.
        
        Returns:
            Aggregated result
        """
        if not self.batches:
            return np.array([])
        
        combined = np.vstack(self.batches)
        
        if self.aggregation_method == "mean":
            result = np.mean(combined, axis=0)
        elif self.aggregation_method == "max":
            result = np.max(combined, axis=0)
        elif self.aggregation_method == "min":
            result = np.min(combined, axis=0)
        elif self.aggregation_method == "sum":
            result = np.sum(combined, axis=0)
        else:
            result = np.mean(combined, axis=0)
        
        self._compute_stats(combined)
        return result

    def _compute_stats(self, data: np.ndarray):
        """Compute statistics on aggregated data."""
        self.stats = {
            'n_batches': len(self.batches),
            'n_samples': len(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)
        }

    def get_stats(self) -> Dict:
        """Get aggregation statistics."""
        return self.stats

    def clear(self):
        """Clear batches."""
        self.batches.clear()
        self.stats.clear()
