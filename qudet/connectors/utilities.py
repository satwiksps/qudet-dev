"""
Data connector utilities.

Factory, batch processing, format conversion, train/test splitting, and
sampling helpers for quantum data pipelines.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union


class DataConnectorFactory:
    """Factory for creating data connectors with various backends.

    Register connector classes by name then instantiate them via
    :meth:`create_connector`.
    """

    _connectors: Dict[str, type] = {}

    @classmethod
    def register_connector(cls, name: str, connector_class: type) -> None:
        """Register a connector type.

        Args:
            name: Short name to identify the connector.
            connector_class: Class to instantiate when *name* is requested.
        """
        cls._connectors[name] = connector_class

    @classmethod
    def create_connector(cls, connector_type: str, **kwargs):
        """Create a connector instance.

        Args:
            connector_type: Registered connector name.
            **kwargs: Arguments forwarded to the connector constructor.

        Returns:
            Connector instance.

        Raises:
            ValueError: If *connector_type* is not registered.
        """
        if connector_type not in cls._connectors:
            raise ValueError(
                f"Unknown connector type: {connector_type!r}. "
                f"Available: {sorted(cls._connectors)}"
            )

        return cls._connectors[connector_type](**kwargs)

    @classmethod
    def get_available_connectors(cls) -> List[str]:
        """Return list of registered connector names."""
        return list(cls._connectors.keys())


class DataBatchProcessor:
    """Process data in batches with a user-supplied operation.

    Args:
        batch_size: Size of batches for processing.
    """

    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self.processing_results: List = []

    def process_batches(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        operation: callable,
    ) -> List:
        """Apply *operation* to each batch.

        Args:
            data: Data to process.
            operation: Callable applied to each batch array.

        Returns:
            List of per-batch results.
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
        """Return processing results from the last run."""
        return self.processing_results

    def aggregate_results(self, aggregation_fn: callable = None):
        """Aggregate processing results.

        Args:
            aggregation_fn: Optional function to aggregate the list of
                results.  Defaults to ``np.concatenate``.

        Returns:
            Aggregated result, or ``None`` if no results are available.
        """
        if not self.processing_results:
            return None

        if aggregation_fn:
            return aggregation_fn(self.processing_results)
        else:
            return np.concatenate(self.processing_results)


class DataFormatConverter:
    """Convert between numpy arrays, DataFrames, and dictionaries."""

    @staticmethod
    def to_numpy(data: Union[np.ndarray, pd.DataFrame, list]) -> np.ndarray:
        """Convert *data* to a ``numpy.ndarray``."""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.values
        else:
            return np.array(data)

    @staticmethod
    def to_dataframe(
        data: Union[np.ndarray, pd.DataFrame, dict],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Convert *data* to a ``pandas.DataFrame``."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data, columns=columns)
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(data, columns=columns)

    @staticmethod
    def to_dict(
        data: Union[np.ndarray, pd.DataFrame, dict],
        orient: str = "list",
    ) -> dict:
        """Convert *data* to a dictionary."""
        if isinstance(data, dict):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.to_dict(orient=orient)
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 2:
                return {
                    f"col_{i}": data[:, i].tolist() for i in range(data.shape[1])
                }
            else:
                return {"data": data.tolist()}
        else:
            return {"data": data}


class DataSplitter:
    """Split data into train / validation / test sets.

    Uses ``numpy.random.Generator`` (via :func:`numpy.random.default_rng`)
    to avoid polluting global random state.

    Args:
        random_state: Seed for the random number generator.
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def split(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """Split *data* into train / val / test sets.

        Args:
            data: Data to split.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.

        Returns:
            Dictionary with ``'train'``, ``'val'``, ``'test'`` keys.

        Raises:
            ValueError: If the ratios do not sum to 1.0.
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")

        is_dataframe = isinstance(data, pd.DataFrame)
        data_array = data.values if is_dataframe else data

        n_samples = len(data_array)
        indices = np.arange(n_samples)
        self._rng.shuffle(indices)

        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        if is_dataframe:
            return {
                "train": data.iloc[train_idx],
                "val": data.iloc[val_idx],
                "test": data.iloc[test_idx],
            }
        else:
            return {
                "train": data_array[train_idx],
                "val": data_array[val_idx],
                "test": data_array[test_idx],
            }

    def stratified_split(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict:
        """Split *data* while preserving label distribution.

        Args:
            data: Data to split.
            labels: Class labels (1-D array).
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.

        Returns:
            Dictionary with stratified ``'train'``, ``'val'``, ``'test'``
            splits.
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        data_array = data.values if is_dataframe else data

        unique_labels = np.unique(labels)
        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []

        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            self._rng.shuffle(label_indices)

            n = len(label_indices)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_idx.extend(label_indices[:train_end])
            val_idx.extend(label_indices[train_end:val_end])
            test_idx.extend(label_indices[val_end:])

        if is_dataframe:
            return {
                "train": data.iloc[train_idx],
                "val": data.iloc[val_idx],
                "test": data.iloc[test_idx],
            }
        else:
            return {
                "train": data_array[train_idx],
                "val": data_array[val_idx],
                "test": data_array[test_idx],
            }


class DataSampler:
    """Sample data with various strategies.

    Uses ``numpy.random.Generator`` (via :func:`numpy.random.default_rng`)
    to avoid polluting global random state.

    Args:
        random_state: Seed for the random number generator.
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def random_sample(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        n_samples: int,
        replace: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Draw a random sample.

        Args:
            data: Data to sample from.
            n_samples: Number of samples to draw.
            replace: Whether to sample with replacement.

        Returns:
            Sampled subset of *data*.
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        n = len(data)

        indices = self._rng.choice(n, size=min(n_samples, n), replace=replace)

        if is_dataframe:
            return data.iloc[indices]
        else:
            return data[indices]

    def stratified_sample(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: np.ndarray,
        n_samples: int,
    ) -> Dict:
        """Draw a stratified sample by label.

        Args:
            data: Data to sample from.
            labels: Labels for stratification.
            n_samples: Total number of samples.

        Returns:
            Dictionary with ``'data'`` and ``'labels'`` keys.
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        data_array = data.values if is_dataframe else data

        unique_labels = np.unique(labels)
        samples_per_class = n_samples // len(unique_labels)

        sampled_data = []
        sampled_labels = []

        for label in unique_labels:
            class_indices = np.where(labels == label)[0]

            selected = self._rng.choice(
                class_indices,
                size=min(samples_per_class, len(class_indices)),
                replace=False,
            )

            sampled_data.append(data_array[selected])
            sampled_labels.extend([label] * len(selected))

        sampled_data = np.vstack(sampled_data)

        if is_dataframe:
            return {
                "data": pd.DataFrame(sampled_data),
                "labels": np.array(sampled_labels),
            }
        else:
            return {
                "data": sampled_data,
                "labels": np.array(sampled_labels),
            }
