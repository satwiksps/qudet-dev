"""
Core quantum data loader.

Provides a batch iterator that loads classical data, encodes it into quantum
circuits, and yields ``(batch_data, circuits)`` tuples — analogous to
PyTorch's ``DataLoader`` but targeting quantum backends.
"""

import logging
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from ..encoders.rotation import RotationEncoder
from ..encoders.statevector import StatevectorEncoder

logger = logging.getLogger(__name__)

_VALID_ENCODER_TYPES = {"angle", "amplitude"}


class QuantumDataLoader:
    """Batch iterator that converts classical data to quantum circuits.

    Acts like PyTorch's ``DataLoader`` but for quantum backends.  Each
    iteration yields a tuple of the raw batch array and a list of
    ``QuantumCircuit`` objects produced by the chosen encoder.

    Args:
        data: Classical feature data.  Must be a 2-D ``DataFrame``.
        batch_size: Number of samples per batch.  Must be a positive integer.
        encoder_type: Quantum encoding strategy — ``'angle'`` (rotation
            encoding) or ``'amplitude'`` (statevector encoding).

    Raises:
        TypeError: If *data* is not a ``pandas.DataFrame``.
        ValueError: If *batch_size* is not a positive integer or
            *encoder_type* is not recognised.

    Example:
        >>> loader = QuantumDataLoader(df, batch_size=64, encoder_type='angle')
        >>> for batch_data, circuits in loader:
        ...     results = backend.run(circuits)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        batch_size: int = 32,
        encoder_type: str = "angle",
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"'data' must be a pandas DataFrame, got {type(data).__name__}."
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"'batch_size' must be a positive integer, got {batch_size!r}."
            )
        if encoder_type not in _VALID_ENCODER_TYPES:
            raise ValueError(
                f"Unknown encoder_type {encoder_type!r}. "
                f"Choose from {sorted(_VALID_ENCODER_TYPES)}."
            )

        self.data = data
        self.batch_size = batch_size
        self.encoder_type = encoder_type
        self.n_features = data.shape[1]

        if encoder_type == "angle":
            self.encoder = RotationEncoder(n_qubits=self.n_features)
        else:  # amplitude
            self.encoder = StatevectorEncoder()

        logger.info(
            "QuantumDataLoader initialised: %d samples, batch_size=%d, encoder=%s",
            len(data),
            batch_size,
            encoder_type,
        )

    def __iter__(self) -> Iterator[Tuple[np.ndarray, list]]:
        """Iterate over batches of data and their quantum-circuit encodings.

        Yields:
            Tuple of ``(batch_data, circuits)`` where *batch_data* is a
            2-D ``numpy.ndarray`` and *circuits* is a list of
            ``QuantumCircuit`` objects.
        """
        n_samples = len(self.data)
        indices = np.arange(n_samples)

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_values = self.data.iloc[batch_indices].values

            circuits = [self.encoder.encode(row) for row in batch_values]

            yield batch_values, circuits

    def __len__(self) -> int:
        """Return the number of batches in a full pass over the data."""
        return int(np.ceil(len(self.data) / self.batch_size))
