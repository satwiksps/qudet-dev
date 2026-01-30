import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union


class DataConnectorFactory:
    """Factory for creating data connectors with various backends."""

    _connectors = {}

    @classmethod
    def register_connector(cls, name: str, connector_class):
        """Register a connector type."""
        cls._connectors[name] = connector_class

    @classmethod
    def create_connector(cls, connector_type: str, **kwargs):
        """
        Create a connector instance.
        
        Args:
            connector_type: Type of connector
            **kwargs: Arguments for connector
            
        Returns:
            Connector instance
        """
        if connector_type not in cls._connectors:
            raise ValueError(f"Unknown connector type: {connector_type}")
        
        return cls._connectors[connector_type](**kwargs)

    @classmethod
    def get_available_connectors(cls) -> List[str]:
        """Get list of available connectors."""
        return list(cls._connectors.keys())


class DataBatchProcessor:
    """Process data in batches with custom operations."""

    def __init__(self, batch_size: int = 32):
        """
        Args:
            batch_size: Size of batches for processing
        """
        self.batch_size = batch_size
        self.processing_results = []

    def process_batches(self, data: Union[np.ndarray, pd.DataFrame], 
                       operation: callable) -> List:
        """
        Process data in batches using custom operation.
        
        Args:
            data: Data to process
            operation: Function to apply to each batch
            
        Returns:
            List of results from operation
        """
        self.processing_results = []
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        n_batches = int(np.ceil(len(data) / self.batch_size))
        
        for i in range(n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(data))
            batch = data[start:end]
            
            result = operation(batch)
            self.processing_results.append(result)
        
        return self.processing_results

    def get_results(self) -> List:
        """Get processing results."""
        return self.processing_results

    def aggregate_results(self, aggregation_fn: callable = None):
        """
        Aggregate processing results.
        
        Args:
            aggregation_fn: Function to aggregate results
            
        Returns:
            Aggregated result
        """
        if not self.processing_results:
            return None
        
        if aggregation_fn:
            return aggregation_fn(self.processing_results)
        else:
            return np.concatenate(self.processing_results)


class DataFormatConverter:
    """Convert between different data formats."""

    @staticmethod
    def to_numpy(data: Union[np.ndarray, pd.DataFrame, list]) -> np.ndarray:
        """Convert to numpy array."""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, list):
            return np.array(data)
        else:
            return np.array(data)

    @staticmethod
    def to_dataframe(data: Union[np.ndarray, pd.DataFrame, dict], 
                     columns: List[str] = None) -> pd.DataFrame:
        """Convert to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data, columns=columns)
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(data, columns=columns)

    @staticmethod
    def to_dict(data: Union[np.ndarray, pd.DataFrame, dict], 
                orient: str = 'list') -> dict:
        """Convert to dictionary."""
        if isinstance(data, dict):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.to_dict(orient=orient)
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 2:
                return {f'col_{i}': data[:, i].tolist() for i in range(data.shape[1])}
            else:
                return {'data': data.tolist()}
        else:
            return {'data': data}


class DataSplitter:
    """Split data into train/validation/test sets."""

    def __init__(self, random_state: Optional[int] = None):
        """
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def split(self, data: Union[np.ndarray, pd.DataFrame],
              train_ratio: float = 0.7,
              val_ratio: float = 0.15,
              test_ratio: float = 0.15) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Split data into train/val/test sets.
        
        Args:
            data: Data to split
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            data_array = data.values
        else:
            data_array = data
        
        n_samples = len(data_array)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        if is_dataframe:
            return {
                'train': data.iloc[train_idx],
                'val': data.iloc[val_idx],
                'test': data.iloc[test_idx]
            }
        else:
            return {
                'train': data_array[train_idx],
                'val': data_array[val_idx],
                'test': data_array[test_idx]
            }

    def stratified_split(self, data: Union[np.ndarray, pd.DataFrame],
                        labels: np.ndarray,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15) -> Dict:
        """
        Split data while preserving label distribution.
        
        Args:
            data: Data to split
            labels: Class labels
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            
        Returns:
            Dictionary with stratified splits
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            data_array = data.values
        else:
            data_array = data
        
        unique_labels = np.unique(labels)
        train_idx = []
        val_idx = []
        test_idx = []
        
        for label in unique_labels:
            mask = labels == label
            label_indices = np.where(mask)[0]
            np.random.shuffle(label_indices)
            
            n = len(label_indices)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_idx.extend(label_indices[:train_end])
            val_idx.extend(label_indices[train_end:val_end])
            test_idx.extend(label_indices[val_end:])
        
        if is_dataframe:
            return {
                'train': data.iloc[train_idx],
                'val': data.iloc[val_idx],
                'test': data.iloc[test_idx]
            }
        else:
            return {
                'train': data_array[train_idx],
                'val': data_array[val_idx],
                'test': data_array[test_idx]
            }


class DataSampler:
    """Sample data with various strategies."""

    def __init__(self, random_state: Optional[int] = None):
        """
        Args:
            random_state: Random seed
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def random_sample(self, data: Union[np.ndarray, pd.DataFrame],
                     n_samples: int, replace: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Random sampling.
        
        Args:
            data: Data to sample from
            n_samples: Number of samples
            replace: Whether to sample with replacement
            
        Returns:
            Sampled data
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        n = len(data)
        
        indices = np.random.choice(n, size=min(n_samples, n), replace=replace)
        
        if is_dataframe:
            return data.iloc[indices]
        else:
            return data[indices]

    def stratified_sample(self, data: Union[np.ndarray, pd.DataFrame],
                        labels: np.ndarray,
                        n_samples: int) -> Dict:
        """
        Stratified sampling by label.
        
        Args:
            data: Data to sample from
            labels: Labels for stratification
            n_samples: Total number of samples
            
        Returns:
            Dictionary with sampled data and labels
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            data_array = data.values
        else:
            data_array = data
        
        unique_labels = np.unique(labels)
        samples_per_class = n_samples // len(unique_labels)
        
        sampled_data = []
        sampled_labels = []
        
        for label in unique_labels:
            mask = labels == label
            class_indices = np.where(mask)[0]
            
            selected = np.random.choice(
                class_indices,
                size=min(samples_per_class, len(class_indices)),
                replace=False
            )
            
            sampled_data.append(data_array[selected])
            sampled_labels.extend([label] * len(selected))
        
        sampled_data = np.vstack(sampled_data)
        
        if is_dataframe:
            return {
                'data': pd.DataFrame(sampled_data),
                'labels': np.array(sampled_labels)
            }
        else:
            return {
                'data': sampled_data,
                'labels': np.array(sampled_labels)
            }
