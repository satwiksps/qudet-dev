"""IQP (Instantaneous Quantum Polynomial) encoding.

Creates entanglement between features via single-qubit Z-rotations and
two-qubit ZZ-interaction gates, making it well suited for quantum kernel
methods such as QSVM.
"""

import logging
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit

from qudet.core.base import BaseEncoder
from qudet.core.exceptions import EncodingError, ValidationError

logger = logging.getLogger(__name__)


class IQPEncoder(BaseEncoder):
    """Implements IQP (Instantaneous Quantum Polynomial) Encoding.

    Unlike simple angle encoding, IQP creates **entanglement** between
    features:

    1. Hadamard layer — prepares superposition.
    2. Single-qubit ``R_z(x_i)`` rotations.
    3. Two-qubit ``R_zz(x_i · x_j)`` entangling gates.

    Multiple repetition layers allow deeper feature mixing.

    Best for: Quantum Support Vector Machines (QSVM), complex feature
    interactions, quantum kernel estimation.

    Example::

        encoder = IQPEncoder(n_qubits=4, reps=2)
        qc = encoder.encode(np.array([0.1, 0.5, 1.2, 0.8]))
    """

    def __init__(self, n_qubits: int, reps: int = 2) -> None:
        """Initialize the IQP encoder.

        Args:
            n_qubits: Number of qubits in the encoding circuit.  Must be ≥ 1.
            reps: Number of repetition layers (circuit depth).  Must be ≥ 1.

        Raises:
            ValidationError: If ``n_qubits`` or ``reps`` is not a positive
                integer.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        if not isinstance(reps, (int, np.integer)) or reps < 1:
            raise ValidationError(
                f"reps must be a positive integer, got {reps!r}."
            )
        self.n_qubits: int = int(n_qubits)
        self.reps: int = int(reps)

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode classical data using IQP gates.

        Args:
            data: 1-D array of input features.  Features beyond
                ``n_qubits`` are truncated.

        Returns:
            A ``QuantumCircuit`` encoding the data with IQP structure.

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

        for _ in range(self.reps):
            # Hadamard layer
            qc.h(range(self.n_qubits))

            # Single-qubit Z-rotations
            for i in range(n_features):
                qc.rz(data[i], i)

            # Two-qubit ZZ-interactions
            for i in range(n_features - 1):
                interaction_strength = data[i] * data[i + 1]
                qc.rzz(interaction_strength, i, i + 1)

        return qc
