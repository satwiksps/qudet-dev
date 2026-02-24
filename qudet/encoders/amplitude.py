"""Amplitude encoding implementations for quantum feature space representation.

Amplitude encoding directly maps classical data into quantum state amplitudes,
providing efficient data loading for quantum algorithms.
"""

import logging
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit

from qudet.core.base import BaseEncoder
from qudet.core.exceptions import EncodingError, ValidationError

logger = logging.getLogger(__name__)


class AmplitudeEncoder(BaseEncoder):
    """Encodes normalised data into quantum state amplitudes.

    Maps a classical data vector directly into amplitudes of a quantum state:

    .. math::

        |\\psi\\rangle = \\sum_i \\alpha_i |i\\rangle

    Requires data to be normalised to unit norm (handled automatically when
    ``normalize=True``).  Uses logarithmic qubit count: ``2^n_qubits``
    amplitudes for ``n_qubits`` qubits.

    Best for: Quantum state preparation, quantum machine learning, data
    loading.

    Example::

        encoder = AmplitudeEncoder(n_qubits=3, normalize=True)
        qc = encoder.encode(np.array([1.0, 0.5, 0.3, 0.1]))
    """

    def __init__(self, n_qubits: int, normalize: bool = True) -> None:
        """Initialize amplitude encoder.

        Args:
            n_qubits: Number of qubits for encoding.  Must be ≥ 1.
            normalize: Whether to normalise input data to unit norm.

        Raises:
            ValidationError: If ``n_qubits`` is not a positive integer.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        self.n_qubits: int = int(n_qubits)
        self.normalize: bool = normalize
        self.max_features: int = 2 ** n_qubits

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode classical data as quantum state amplitudes.

        Args:
            data: Feature vector of size ≤ ``2^n_qubits``.

        Returns:
            A ``QuantumCircuit`` with amplitude encoding.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        # Pad to exact size needed
        padded_data = np.zeros(self.max_features)
        if len(data) > 0:
            n_copy = min(len(data), self.max_features)
            padded_data[:n_copy] = data[:n_copy]

        if len(data) > self.max_features:
            logger.warning(
                "Data has %d features but encoder supports %d "
                "(2^%d); extra features are truncated.",
                len(data),
                self.max_features,
                self.n_qubits,
            )

        # Normalise if requested (required for valid state initialisation)
        norm = np.linalg.norm(padded_data)
        if self.normalize or norm == 0:
            if norm > 0:
                padded_data = padded_data / norm
            else:
                # Handle all-zero data → default to |0…0⟩
                padded_data[0] = 1.0
                logger.warning(
                    "All-zero data vector; defaulting to |0…0⟩."
                )

        qc = QuantumCircuit(self.n_qubits)
        qc.initialize(padded_data, qc.qubits)

        return qc

    def get_features_supported(self) -> int:
        """Get the maximum number of features that can be encoded.

        Returns:
            ``2^n_qubits``.
        """
        return self.max_features


class DensityMatrixEncoder(BaseEncoder):
    """Encodes data using quantum density-matrix representation.

    Captures both classical information and quantum correlations.
    Useful for encoding mixed quantum states and working with kernel
    methods.

    Best for: Quantum kernel methods, metric learning, correlation
    matrices.

    Example::

        encoder = DensityMatrixEncoder(n_qubits=3)
        qc = encoder.encode(data)
        rho = encoder.compute_density_matrix(data)
    """

    def __init__(self, n_qubits: int) -> None:
        """Initialize density matrix encoder.

        Args:
            n_qubits: Number of qubits.  Must be ≥ 1.

        Raises:
            ValidationError: If ``n_qubits`` is not a positive integer.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        self.n_qubits: int = int(n_qubits)

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode data into a density-matrix-inspired circuit structure.

        Args:
            data: Input feature vector (1-D array).

        Returns:
            A ``QuantumCircuit`` with rotation + entanglement encoding.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        qc = QuantumCircuit(self.n_qubits)

        # Normalise features
        norm = np.linalg.norm(data)
        normalized = data / norm if norm > 0 else data

        # Encode as parametrised rotations creating coherence
        for i in range(min(len(normalized), self.n_qubits)):
            angle = np.pi * normalized[i]
            qc.ry(angle, i)

        # Add entanglement for correlation capture
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)

        return qc

    def compute_density_matrix(self, data: np.ndarray) -> np.ndarray:
        """Compute the density matrix from the data vector.

        Creates an outer product ``|ψ⟩⟨ψ|`` of the normalised,
        zero-padded data.

        Args:
            data: Input feature vector.

        Returns:
            Complex density matrix of shape ``(2^n_qubits, 2^n_qubits)``.
        """
        data = np.asarray(data, dtype=float)
        dim = 2 ** self.n_qubits
        rho = np.zeros((dim, dim), dtype=complex)

        norm = np.linalg.norm(data)
        if norm > 0:
            psi = data / norm
            padded = np.zeros(dim)
            padded[: len(psi)] = psi
            rho = np.outer(padded, padded.conj())

        return rho


class BasisChangeEncoder(BaseEncoder):
    """Encodes data by applying basis transformations to quantum states.

    Supports encoding in different computational bases:

    * ``'z'`` — computational basis (no extra transform).
    * ``'x'`` — Hadamard basis.
    * ``'y'`` — Y-basis (S† then Hadamard).

    Best for: Basis-adaptive encoding, flexibility in feature
    representation.

    Example::

        encoder = BasisChangeEncoder(n_qubits=4, basis="x")
        qc = encoder.encode(data)
    """

    _VALID_BASES = {"z", "x", "y"}

    def __init__(self, n_qubits: int, basis: str = "z") -> None:
        """Initialize basis change encoder.

        Args:
            n_qubits: Number of qubits.  Must be ≥ 1.
            basis: Measurement basis (``'z'``, ``'x'``, or ``'y'``).

        Raises:
            ValidationError: On invalid arguments.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        basis = basis.lower()
        if basis not in self._VALID_BASES:
            raise ValidationError(
                f"basis must be one of {self._VALID_BASES}, got {basis!r}."
            )
        self.n_qubits: int = int(n_qubits)
        self.basis: str = basis

    def encode(
        self,
        data: np.ndarray,
        basis: Optional[str] = None,
    ) -> QuantumCircuit:
        """Encode data with basis transformation.

        Args:
            data: Input features (1-D array).
            basis: Override the default basis for this encoding.

        Returns:
            A ``QuantumCircuit`` with basis-transformed encoding.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        use_basis = basis or self.basis
        qc = QuantumCircuit(self.n_qubits)

        # Prepare initial state with rotation angles
        for i in range(min(len(data), self.n_qubits)):
            angle = np.pi * data[i] / (np.max(np.abs(data)) + 1e-10)
            qc.ry(angle, i)

        # Apply basis transformation
        if use_basis == "x":
            qc.h(range(self.n_qubits))
        elif use_basis == "y":
            qc.sdg(range(self.n_qubits))
            qc.h(range(self.n_qubits))
        # z-basis needs no transformation (computational basis)

        return qc

    def get_supported_bases(self) -> list:
        """Get list of supported measurement bases.

        Returns:
            ``['z', 'x', 'y']``.
        """
        return ["z", "x", "y"]


class FeatureMapEncoder(BaseEncoder):
    """General-purpose feature map encoder for flexible data encoding.

    Combines Hadamard layers, feature-dependent rotations, and
    entanglement in a repeatable block structure.  Supports multiple
    mapping types:

    * ``'linear'`` — identity mapping.
    * ``'polynomial'`` — raise features to a configurable power.
    * ``'trigonometric'`` — apply ``sin()`` to features.

    Best for: Adaptable encoding for various data types and distributions.

    Example::

        encoder = FeatureMapEncoder(n_qubits=4, mapping_type="polynomial", power=2)
        qc = encoder.encode(data)
    """

    _VALID_MAPPINGS = {"linear", "polynomial", "trigonometric"}

    def __init__(
        self,
        n_qubits: int,
        mapping_type: str = "linear",
        power: int = 1,
        reps: int = 1,
    ) -> None:
        """Initialize feature map encoder.

        Args:
            n_qubits: Number of qubits.  Must be ≥ 1.
            mapping_type: Type of mapping (``'linear'``, ``'polynomial'``,
                or ``'trigonometric'``).
            power: Exponent for polynomial mapping.  Must be ≥ 1.
            reps: Number of repetitions.  Must be ≥ 1.

        Raises:
            ValidationError: On invalid arguments.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        if mapping_type not in self._VALID_MAPPINGS:
            raise ValidationError(
                f"mapping_type must be one of {self._VALID_MAPPINGS}, "
                f"got {mapping_type!r}."
            )
        self.n_qubits: int = int(n_qubits)
        self.mapping_type: str = mapping_type
        self.power: int = int(power)
        self.reps: int = int(reps)

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode data using the selected feature map.

        Args:
            data: Input features (1-D array).

        Returns:
            A ``QuantumCircuit`` with feature-map encoding.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        qc = QuantumCircuit(self.n_qubits)

        for _rep in range(self.reps):
            # Hadamard layer
            qc.h(range(self.n_qubits))

            # Feature-dependent rotations
            for i in range(min(len(data), self.n_qubits)):
                if self.mapping_type == "linear":
                    angle = data[i]
                elif self.mapping_type == "polynomial":
                    angle = data[i] ** self.power
                elif self.mapping_type == "trigonometric":
                    angle = np.sin(data[i])
                else:
                    angle = data[i]

                qc.rz(angle, i)

            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)

        return qc

    def apply_mapping(self, data: np.ndarray) -> np.ndarray:
        """Apply the feature-map transformation to classical data.

        Args:
            data: Input features.

        Returns:
            Transformed feature array.
        """
        data = np.asarray(data, dtype=float)
        if self.mapping_type == "linear":
            return data
        elif self.mapping_type == "polynomial":
            return np.power(data, self.power)
        elif self.mapping_type == "trigonometric":
            return np.sin(data)
        else:
            return data

    def get_mapping_info(self) -> dict:
        """Get information about the current mapping configuration.

        Returns:
            Dictionary with type, power, repetitions, and n_qubits.
        """
        return {
            "type": self.mapping_type,
            "power": self.power,
            "repetitions": self.reps,
            "n_qubits": self.n_qubits,
        }
