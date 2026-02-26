"""Statevector (amplitude) encoding for quantum feature embedding.

Encodes a classical data vector directly into the amplitudes of a quantum
state, achieving logarithmic compression: *N* features are encoded into
``ceil(log2(N))`` qubits.
"""

import logging
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit

from qudet.core.base import BaseEncoder
from qudet.core.exceptions import EncodingError, ValidationError

logger = logging.getLogger(__name__)


class StatevectorEncoder(BaseEncoder):
    """Encodes a data vector into quantum state amplitudes.

    The input vector is zero-padded to the next power of two, normalised to
    unit L2 norm, and then loaded as the initial statevector of the circuit.

    If ``n_qubits`` is supplied at construction time the circuit will always
    use that many qubits (padding or truncating data as needed).  If left as
    ``None`` the qubit count is inferred from the data length.

    Best for: Dense feature loading, quantum state preparation.

    Example::

        encoder = StatevectorEncoder(n_qubits=3)
        qc = encoder.encode(np.array([1.0, 0.5, 0.3, 0.1]))
    """

    def __init__(self, n_qubits: Optional[int] = None) -> None:
        """Initialize the statevector encoder.

        Args:
            n_qubits: Fixed qubit count.  When ``None`` the number is
                inferred from data at encode time.  Must be ≥ 1 if given.

        Raises:
            ValidationError: If ``n_qubits`` is provided but is not a
                positive integer.
        """
        if n_qubits is not None:
            if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
                raise ValidationError(
                    f"n_qubits must be a positive integer or None, "
                    f"got {n_qubits!r}."
                )
            n_qubits = int(n_qubits)
        self.n_qubits: Optional[int] = n_qubits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """Return the L2-normalised version of *v*.

        Args:
            v: 1-D numeric array.

        Returns:
            Normalised copy of *v*, or *v* unchanged when the norm is zero.
        """
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode a feature vector as a quantum statevector.

        Args:
            data: 1-D array of feature values.

        Returns:
            A ``QuantumCircuit`` initialised to the state ``|x⟩``.

        Raises:
            EncodingError: If *data* is empty, not 1-D, or results in a
                degenerate (0-qubit) circuit.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )
        if len(data) == 0:
            raise EncodingError("data must not be empty.")

        # Determine qubit count -----------------------------------------
        if self.n_qubits is not None:
            n_qubits = self.n_qubits
            required_len = 2 ** n_qubits
        else:
            # Infer from data length, ensuring at least 1 qubit
            n_qubits = max(1, int(np.ceil(np.log2(max(len(data), 2)))))
            required_len = 2 ** n_qubits

        # Pad / truncate ------------------------------------------------
        if len(data) < required_len:
            padded_data = np.pad(data, (0, required_len - len(data)))
        else:
            padded_data = data[:required_len]
            if len(data) > required_len:
                logger.warning(
                    "Data has %d features but circuit supports %d "
                    "(2^%d); extra features are truncated.",
                    len(data),
                    required_len,
                    n_qubits,
                )

        state_vector = self._normalize(padded_data)

        # Guard: all-zero vector → default to |0…0⟩
        if np.linalg.norm(state_vector) == 0:
            state_vector = np.zeros(required_len)
            state_vector[0] = 1.0
            logger.warning("All-zero data vector; defaulting to |0…0⟩.")

        qc = QuantumCircuit(n_qubits)
        qc.initialize(state_vector, range(n_qubits))

        return qc
