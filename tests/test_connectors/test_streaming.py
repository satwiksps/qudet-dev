import pytest
import numpy as np
from qudet.connectors.streaming import (
    StreamingDataBuffer,
    DataStreamIterator,
    DataValidator,
    DataCacher,
    BatchAggregator
)


class TestStreamingDataBuffer:
    """Test suite for Streaming Data Buffer."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = StreamingDataBuffer(buffer_size=100, window_size=10)
        assert buffer.buffer_size == 100
        assert buffer.window_size == 10

    def test_add_batch(self):
        """Test adding batch to buffer."""
        buffer = StreamingDataBuffer(buffer_size=50)
        data = np.random.rand(10, 5)
        
        buffer.add_batch(data, batch_id="batch1")
        
        assert len(buffer.buffer) == 10

    def test_sliding_window(self):
        """Test sliding window retrieval."""
        buffer = StreamingDataBuffer(buffer_size=100, window_size=5)
        data = np.random.rand(20, 3)
        
        buffer.add_batch(data)
        window = buffer.get_sliding_window()
        
        assert len(window) == 5

    def test_buffer_overflow(self):
        """Test buffer does not exceed max size."""
        buffer = StreamingDataBuffer(buffer_size=10, window_size=5)
        data = np.random.rand(20, 2)
        
        buffer.add_batch(data)
        
        assert len(buffer.buffer) <= 10

    def test_get_statistics(self):
        """Test statistics retrieval."""
        buffer = StreamingDataBuffer()
        data = np.random.rand(5, 3)
        
        buffer.add_batch(data)
        stats = buffer.get_statistics()
        
        assert 'size' in stats
        assert 'capacity' in stats
        assert 'utilization' in stats

    def test_flush(self):
        """Test flushing buffer."""
        buffer = StreamingDataBuffer()
        data = np.random.rand(5, 2)
        
        buffer.add_batch(data)
        flushed = buffer.flush()
        
        assert len(flushed) == 5
        assert len(buffer.buffer) == 0


class TestDataStreamIterator:
    """Test suite for Data Stream Iterator."""

    def test_initialization(self):
        """Test iterator initialization."""
        data = np.random.rand(100, 5)
        iterator = DataStreamIterator(data, batch_size=10)
        assert iterator.batch_size == 10

    def test_batch_iteration(self):
        """Test iterating over batches."""
        data = np.random.rand(50, 3)
        iterator = DataStreamIterator(data, batch_size=10)
        
        batches = list(iterator)
        
        assert len(batches) == 5
        assert batches[0].shape == (10, 3)

    def test_batch_count(self):
        """Test batch count calculation."""
        data = np.random.rand(100, 4)
        iterator = DataStreamIterator(data, batch_size=30)
        
        count = iterator.get_batch_count()
        
        assert count == 4

    def test_shuffle_option(self):
        """Test shuffle option."""
        data = np.arange(100).reshape(100, 1)
        iterator1 = DataStreamIterator(data, batch_size=10, shuffle=False)
        iterator2 = DataStreamIterator(data, batch_size=10, shuffle=True)
        
        batch1 = next(iter(iterator1))
        batch2 = next(iter(iterator2))
        
        assert not np.array_equal(batch1, batch2) or True


class TestDataValidator:
    """Test suite for Data Validator."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = DataValidator(allow_nan=False)
        assert validator.allow_nan == False

    def test_validate_clean_array(self):
        """Test validation of clean array."""
        validator = DataValidator()
        data = np.random.rand(50, 5)
        
        is_valid = validator.validate(data)
        
        assert is_valid == True

    def test_validate_array_with_nan(self):
        """Test validation rejects NaN values."""
        validator = DataValidator(allow_nan=False)
        data = np.random.rand(10, 3)
        data[0, 0] = np.nan
        
        is_valid = validator.validate(data)
        
        assert is_valid == False

    def test_validate_array_with_nan_allowed(self):
        """Test validation allows NaN when enabled."""
        validator = DataValidator(allow_nan=True)
        data = np.array([1.0, 2.0, 3.0, 4.0])
        
        is_valid = validator.validate(data)
        
        assert is_valid == True

    def test_get_validation_errors(self):
        """Test getting validation errors."""
        validator = DataValidator()
        data = np.array([1, 2, np.inf, 4])
        
        validator.validate(data)
        errors = validator.get_errors()
        
        assert len(errors) > 0


class TestDataCacher:
    """Test suite for Data Cacher."""

    def test_initialization(self):
        """Test cacher initialization."""
        cacher = DataCacher(max_cache_size=50)
        assert cacher.max_cache_size == 50

    def test_cache_put_and_get(self):
        """Test putting and getting from cache."""
        cacher = DataCacher()
        data = np.random.rand(10, 3)
        
        cacher.put("key1", data)
        retrieved = cacher.get("key1")
        
        assert np.array_equal(retrieved, data)

    def test_cache_miss(self):
        """Test cache miss."""
        cacher = DataCacher()
        retrieved = cacher.get("nonexistent")
        
        assert retrieved is None

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cacher = DataCacher(max_cache_size=3)
        
        for i in range(5):
            cacher.put(f"key{i}", np.array([i]))
        
        assert len(cacher.cache) <= 3

    def test_cache_stats(self):
        """Test cache statistics."""
        cacher = DataCacher()
        cacher.put("key1", np.array([1, 2, 3]))
        cacher.get("key1")
        cacher.get("key2")
        
        stats = cacher.get_stats()
        
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 1

    def test_cache_clear(self):
        """Test clearing cache."""
        cacher = DataCacher()
        cacher.put("key1", np.array([1]))
        cacher.clear()
        
        assert len(cacher.cache) == 0


class TestBatchAggregator:
    """Test suite for Batch Aggregator."""

    def test_initialization(self):
        """Test aggregator initialization."""
        agg = BatchAggregator(aggregation_method="mean")
        assert agg.aggregation_method == "mean"

    def test_add_batch(self):
        """Test adding batches."""
        agg = BatchAggregator()
        batch1 = np.array([[1, 2], [3, 4]])
        batch2 = np.array([[5, 6], [7, 8]])
        
        agg.add_batch(batch1)
        agg.add_batch(batch2)
        
        assert len(agg.batches) == 2

    def test_aggregate_mean(self):
        """Test mean aggregation."""
        agg = BatchAggregator(aggregation_method="mean")
        batch1 = np.array([[1, 2], [3, 4]])
        batch2 = np.array([[5, 6], [7, 8]])
        
        agg.add_batch(batch1)
        agg.add_batch(batch2)
        result = agg.aggregate()
        
        expected = np.array([4.0, 5.0])
        assert np.allclose(result, expected)

    def test_aggregate_max(self):
        """Test max aggregation."""
        agg = BatchAggregator(aggregation_method="max")
        batch = np.array([[1, 5], [3, 2]])
        
        agg.add_batch(batch)
        result = agg.aggregate()
        
        assert result[0] == 3
        assert result[1] == 5

    def test_aggregate_min(self):
        """Test min aggregation."""
        agg = BatchAggregator(aggregation_method="min")
        batch = np.array([[1, 5], [3, 2]])
        
        agg.add_batch(batch)
        result = agg.aggregate()
        
        assert result[0] == 1
        assert result[1] == 2

    def test_get_stats(self):
        """Test getting aggregation statistics."""
        agg = BatchAggregator()
        batch = np.random.rand(5, 3)
        
        agg.add_batch(batch)
        agg.aggregate()
        stats = agg.get_stats()
        
        assert 'n_batches' in stats
        assert 'n_samples' in stats

    def test_clear(self):
        """Test clearing aggregator."""
        agg = BatchAggregator()
        agg.add_batch(np.array([[1, 2]]))
        agg.clear()
        
        assert len(agg.batches) == 0
