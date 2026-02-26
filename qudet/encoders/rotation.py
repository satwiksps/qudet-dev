"""Rotation encoding for quantum feature embedding.

Encodes classical features directly into qubit rotation angles,
providing a simple and efficient quantum data loading strategy.
"""

import logging
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit

from qudet.core.base import BaseEncoder
from qudet.core.exceptions import EncodingError, ValidationError

logger = logging.getLogger(__name__)


class RotationEncoder(BaseEncoder):
    """Encodes classical features into qubit rotation angles.

    Each feature ``x_i`` is mapped to an ``R_y(x_i)`` rotation on qubit *i*.
    Features beyond ``n_qubits`` are silently truncated; if fewer features
    than qubits are provided, the remaining qubits stay in |0⟩.

    Best for: Simple feature encoding, fast data loading, shallow circuits.

    Example::

        encoder = RotationEncoder(n_qubits=4)
        qc = encoder.encode(np.array([0.1, 0.5, 1.2, 0.8]))
    """

    def __init__(self, n_qubits: int) -> None:
        """Initialize the rotation encoder.

        Args:
            n_qubits: Number of qubits in the encoding circuit.  Must be ≥ 1.

        Raises:
            ValidationError: If ``n_qubits`` is not a positive integer.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        self.n_qubits: int = int(n_qubits)

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode a classical feature vector as R_y rotations.

        Args:
            data: 1-D array of feature values.  Length may differ from
                ``n_qubits``; extra features are truncated and missing
                qubits are left in |0⟩.

        Returns:
            A ``QuantumCircuit`` with one ``R_y`` gate per encoded feature.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        qc = QuantumCircuit(self.n_qubits)

        n_features = min(len(data), self.n_qubits)
        if len(data) > self.n_qubits:
            logger.warning(
                "Data has %d features but encoder has only %d qubits; "
                "extra features are truncated.",
                len(data),
                self.n_qubits,
            )

        for i in range(n_features):
            qc.ry(data[i], i)

        return qc
