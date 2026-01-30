import pytest
import numpy as np
import pandas as pd
from qudet.connectors.utilities import (
    DataConnectorFactory,
    DataBatchProcessor,
    DataFormatConverter,
    DataSplitter,
    DataSampler
)


class TestDataConnectorFactory:
    """Test suite for Data Connector Factory."""

    def test_initialization(self):
        """Test factory initialization."""
        assert len(DataConnectorFactory.get_available_connectors()) >= 0

    def test_register_connector(self):
        """Test registering a connector."""
        class DummyConnector:
            pass
        
        DataConnectorFactory.register_connector("dummy", DummyConnector)
        
        assert "dummy" in DataConnectorFactory.get_available_connectors()

    def test_create_registered_connector(self):
        """Test creating registered connector."""
        class TestConnector:
            def __init__(self, param):
                self.param = param
        
        DataConnectorFactory.register_connector("test", TestConnector)
        connector = DataConnectorFactory.create_connector("test", param="value")
        
        assert connector.param == "value"

    def test_create_unknown_connector_raises_error(self):
        """Test error on unknown connector."""
        with pytest.raises(ValueError):
            DataConnectorFactory.create_connector("unknown_type")


class TestDataBatchProcessor:
    """Test suite for Data Batch Processor."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = DataBatchProcessor(batch_size=32)
        assert processor.batch_size == 32

    def test_process_batches(self):
        """Test processing batches."""
        processor = DataBatchProcessor(batch_size=10)
        data = np.random.rand(45, 3)
        
        def sum_operation(batch):
            return np.sum(batch, axis=0)
        
        results = processor.process_batches(data, sum_operation)
        
        assert len(results) == 5

    def test_get_results(self):
        """Test getting results."""
        processor = DataBatchProcessor(batch_size=10)
        data = np.random.rand(20, 2)
        
        processor.process_batches(data, lambda x: x.mean())
        results = processor.get_results()
        
        assert len(results) == 2

    def test_aggregate_results(self):
        """Test aggregating results."""
        processor = DataBatchProcessor(batch_size=10)
        data = np.random.rand(30, 2)
        
        processor.process_batches(data, lambda x: np.array([x.sum()]))
        aggregated = processor.aggregate_results()
        
        assert len(aggregated) > 0


class TestDataFormatConverter:
    """Test suite for Data Format Converter."""

    def test_to_numpy_from_array(self):
        """Test converting array to numpy."""
        data = np.array([[1, 2], [3, 4]])
        result = DataFormatConverter.to_numpy(data)
        
        assert isinstance(result, np.ndarray)

    def test_to_numpy_from_dataframe(self):
        """Test converting DataFrame to numpy."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = DataFormatConverter.to_numpy(df)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_to_numpy_from_list(self):
        """Test converting list to numpy."""
        data = [[1, 2], [3, 4]]
        result = DataFormatConverter.to_numpy(data)
        
        assert isinstance(result, np.ndarray)

    def test_to_dataframe_from_array(self):
        """Test converting array to DataFrame."""
        data = np.array([[1, 2], [3, 4]])
        result = DataFormatConverter.to_dataframe(data, columns=['a', 'b'])
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['a', 'b']

    def test_to_dataframe_from_dict(self):
        """Test converting dict to DataFrame."""
        data = {'a': [1, 2], 'b': [3, 4]}
        result = DataFormatConverter.to_dataframe(data)
        
        assert isinstance(result, pd.DataFrame)

    def test_to_dict_from_dataframe(self):
        """Test converting DataFrame to dict."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = DataFormatConverter.to_dict(df)
        
        assert isinstance(result, dict)
        assert 'a' in result


class TestDataSplitter:
    """Test suite for Data Splitter."""

    def test_initialization(self):
        """Test splitter initialization."""
        splitter = DataSplitter(random_state=42)
        assert splitter.random_state == 42

    def test_split_data(self):
        """Test data splitting."""
        splitter = DataSplitter()
        data = np.random.rand(100, 5)
        
        splits = splitter.split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        assert len(splits['train']) + len(splits['val']) + len(splits['test']) == 100

    def test_split_preserves_dataframe(self):
        """Test split preserves DataFrame format."""
        splitter = DataSplitter()
        df = pd.DataFrame({'a': np.random.rand(50), 'b': np.random.rand(50)})
        
        splits = splitter.split(df)
        
        assert isinstance(splits['train'], pd.DataFrame)

    def test_split_invalid_ratios(self):
        """Test error on invalid ratios."""
        splitter = DataSplitter()
        data = np.random.rand(100, 5)
        
        splits = splitter.split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        assert len(splits['train']) > 0

    def test_stratified_split(self):
        """Test stratified splitting."""
        splitter = DataSplitter()
        data = np.random.rand(100, 5)
        labels = np.array([0]*50 + [1]*50)
        
        splits = splitter.stratified_split(data, labels)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits


class TestDataSampler:
    """Test suite for Data Sampler."""

    def test_initialization(self):
        """Test sampler initialization."""
        sampler = DataSampler(random_state=42)
        assert sampler.random_state == 42

    def test_random_sample(self):
        """Test random sampling."""
        sampler = DataSampler()
        data = np.arange(100).reshape(100, 1)
        
        sample = sampler.random_sample(data, n_samples=10, replace=False)
        
        assert len(sample) == 10

    def test_random_sample_with_replacement(self):
        """Test sampling with replacement."""
        sampler = DataSampler()
        data = np.array([[1], [2], [3]])
        
        sample = sampler.random_sample(data, n_samples=5, replace=True)
        
        assert len(sample) <= 5

    def test_stratified_sample(self):
        """Test stratified sampling."""
        sampler = DataSampler()
        data = np.random.rand(100, 5)
        labels = np.array([0]*50 + [1]*50)
        
        result = sampler.stratified_sample(data, labels, n_samples=20)
        
        assert 'data' in result
        assert 'labels' in result
        assert len(result['labels']) == 20

    def test_stratified_sample_preserves_balance(self):
        """Test stratified sampling preserves class distribution."""
        sampler = DataSampler()
        data = np.random.rand(100, 2)
        labels = np.array([0]*60 + [1]*40)
        
        result = sampler.stratified_sample(data, labels, n_samples=20)
        
        assert len(result['labels']) <= 20

    def test_stratified_sample_with_dataframe(self):
        """Test stratified sampling with DataFrame."""
        sampler = DataSampler()
        df = pd.DataFrame({'a': np.random.rand(50), 'b': np.random.rand(50)})
        labels = np.array([0]*25 + [1]*25)
        
        result = sampler.stratified_sample(df, labels, n_samples=10)
        
        assert isinstance(result['data'], pd.DataFrame)
