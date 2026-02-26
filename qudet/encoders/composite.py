"""Composite and hybrid encoding methods combining multiple encoding strategies.

Provides flexible composition of different encoding techniques to create
rich quantum feature representations.
"""

import logging
from typing import Union, Optional, List, Callable, Dict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister

from qudet.core.base import BaseEncoder
from qudet.core.exceptions import EncodingError, ValidationError

logger = logging.getLogger(__name__)


class CompositeEncoder(BaseEncoder):
    """Combines multiple encoding strategies sequentially.

    Allows composition of different encoders to create more expressive
    quantum feature representations through combination.

    Best for: Complex data with diverse feature characteristics.

    Example::

        enc = CompositeEncoder(n_qubits=4, encoders=[enc_a, enc_b])
        qc = enc.encode(data)
    """

    def __init__(
        self,
        n_qubits: int,
        encoders: Optional[List[BaseEncoder]] = None,
    ) -> None:
        """Initialize composite encoder.

        Args:
            n_qubits: Number of qubits in the output circuit.  Must be ≥ 1.
            encoders: List of encoder objects to combine sequentially.

        Raises:
            ValidationError: If ``n_qubits`` is not a positive integer.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        self.n_qubits: int = int(n_qubits)
        self.encoders: List[BaseEncoder] = list(encoders) if encoders else []

    def add_encoder(self, encoder: BaseEncoder) -> None:
        """Add an encoder to the composition.

        Args:
            encoder: Encoder instance to append.

        Raises:
            TypeError: If *encoder* is not a ``BaseEncoder``.
        """
        if not isinstance(encoder, BaseEncoder):
            raise TypeError(
                f"encoder must be a BaseEncoder instance, got {type(encoder).__name__}."
            )
        self.encoders.append(encoder)

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode data using all composed encoders sequentially.

        Args:
            data: Input feature vector (1-D array).

        Returns:
            A ``QuantumCircuit`` with composite encoding applied.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        if not self.encoders:
            return QuantumCircuit(self.n_qubits)

        # Start with the first encoder
        qc = self.encoders[0].encode(data)

        # Ensure the correct number of qubits
        if qc.num_qubits < self.n_qubits:
            extra_qubits = self.n_qubits - qc.num_qubits
            qc.add_register(QuantumRegister(extra_qubits))

        # Apply subsequent encoders
        for encoder in self.encoders[1:]:
            qc_partial = encoder.encode(data)

            # Extend with extra qubits if needed
            if qc_partial.num_qubits < self.n_qubits:
                extra = self.n_qubits - qc_partial.num_qubits
                qc_partial.add_register(QuantumRegister(extra))

            qc = qc.compose(qc_partial)

        return qc

    def get_encoder_info(self) -> List[Dict]:
        """Get information about composed encoders.

        Returns:
            List of dicts, each containing the encoder class name.
        """
        return [{"type": type(enc).__name__} for enc in self.encoders]


class LayeredEncoder(BaseEncoder):
    """Applies encoding in multiple layers with entanglement between layers.

    Creates a layered structure where data is encoded in each layer,
    then layers are connected through entanglement patterns.

    Supported entanglement types:

    * ``"linear"`` — nearest-neighbour CNOT chain.
    * ``"full"`` — all-to-all CNOT connections.
    * ``"chain"`` — alternating-pair CNOTs.

    Best for: Deep feature representation, hierarchical data encoding.

    Example::

        encoder = LayeredEncoder(n_qubits=4, n_layers=3, entangle_type="full")
        qc = encoder.encode(data)
    """

    _VALID_ENTANGLE_TYPES = {"linear", "full", "chain"}

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        entangle_type: str = "linear",
    ) -> None:
        """Initialize layered encoder.

        Args:
            n_qubits: Number of qubits.  Must be ≥ 1.
            n_layers: Number of encoding layers.  Must be ≥ 1.
            entangle_type: Entanglement pattern (``'linear'``, ``'full'``,
                or ``'chain'``).

        Raises:
            ValidationError: On invalid constructor arguments.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        if not isinstance(n_layers, (int, np.integer)) or n_layers < 1:
            raise ValidationError(
                f"n_layers must be a positive integer, got {n_layers!r}."
            )
        if entangle_type not in self._VALID_ENTANGLE_TYPES:
            raise ValidationError(
                f"entangle_type must be one of {self._VALID_ENTANGLE_TYPES}, "
                f"got {entangle_type!r}."
            )
        self.n_qubits: int = int(n_qubits)
        self.n_layers: int = int(n_layers)
        self.entangle_type: str = entangle_type

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode data in a layered structure.

        Args:
            data: Input feature vector (1-D array).

        Returns:
            A ``QuantumCircuit`` with layered encoding.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        qc = QuantumCircuit(self.n_qubits)

        for layer in range(self.n_layers):
            # Feature encoding
            for i in range(min(len(data), self.n_qubits)):
                angle = data[i] * (1.0 + layer * 0.5)
                qc.ry(angle, i)

            # Entanglement pattern
            if self.entangle_type == "linear":
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
            elif self.entangle_type == "full":
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qc.cx(i, j)
            elif self.entangle_type == "chain":
                for i in range(0, self.n_qubits - 1, 2):
                    qc.cx(i, i + 1)

            # Phase accumulation
            for i in range(self.n_qubits):
                phase = (
                    data[i % len(data)] * layer if len(data) > 0 else 0.0
                )
                qc.p(phase, i)

        return qc

    def set_num_layers(self, n_layers: int) -> None:
        """Set the number of encoding layers.

        Args:
            n_layers: New layer count.  Must be ≥ 1.

        Raises:
            ValidationError: If ``n_layers`` is not a positive integer.
        """
        if not isinstance(n_layers, (int, np.integer)) or n_layers < 1:
            raise ValidationError(
                f"n_layers must be a positive integer, got {n_layers!r}."
            )
        self.n_layers = int(n_layers)


class DataReuseEncoder(BaseEncoder):
    """Encodes the same data multiple times with different transformations.

    Reuses limited data by applying multiple transformations,
    effectively creating a richer feature space.

    Best for: Small datasets, data augmentation in quantum domain.

    Example::

        encoder = DataReuseEncoder(n_qubits=6, n_reuses=3)
        qc = encoder.encode(data)
    """

    def __init__(self, n_qubits: int, n_reuses: int = 2) -> None:
        """Initialize data reuse encoder.

        Args:
            n_qubits: Number of qubits.  Must be ≥ 1.
            n_reuses: Number of reuse passes.  Must be ≥ 1.

        Raises:
            ValidationError: On invalid arguments.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        if not isinstance(n_reuses, (int, np.integer)) or n_reuses < 1:
            raise ValidationError(
                f"n_reuses must be a positive integer, got {n_reuses!r}."
            )
        self.n_qubits: int = int(n_qubits)
        self.n_reuses: int = int(n_reuses)

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode data with multiple transformations.

        Args:
            data: Input feature vector (1-D array).

        Returns:
            A ``QuantumCircuit`` with data-reuse encoding.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        qc = QuantumCircuit(self.n_qubits)
        features_per_reuse = self.n_qubits // self.n_reuses

        for reuse_idx in range(self.n_reuses):
            start_qubit = reuse_idx * features_per_reuse
            end_qubit = min(
                (reuse_idx + 1) * features_per_reuse, self.n_qubits
            )

            for qubit in range(start_qubit, end_qubit):
                if len(data) > 0:
                    data_idx = (qubit - start_qubit) % len(data)
                    angle = data[data_idx] * (
                        1.0 + reuse_idx * np.pi / self.n_reuses
                    )
                else:
                    angle = 0.0

                if reuse_idx % 2 == 0:
                    qc.rx(angle, qubit)
                else:
                    qc.ry(angle, qubit)

        return qc


class AdaptiveEncoder(BaseEncoder):
    """Adapts encoding strategy based on data characteristics.

    Analyses input data properties (sparsity, magnitude, normalisation) and
    selects the most appropriate encoding strategy automatically.

    Built-in strategies:

    * ``"sparse"`` — only encode non-zero elements.
    * ``"dense"`` — normalise and encode on all qubits with entanglement.
    * ``"normalized"`` — R_y + R_z encoding assuming unit-range data.
    * ``"auto"`` — pick one of the above based on data statistics.

    Best for: Heterogeneous data, adaptive quantum algorithms.

    Example::

        encoder = AdaptiveEncoder(n_qubits=4)
        qc = encoder.encode(data)
    """

    def __init__(
        self,
        n_qubits: int,
        strategies: Optional[Dict[str, Callable]] = None,
    ) -> None:
        """Initialize adaptive encoder.

        Args:
            n_qubits: Number of qubits.  Must be ≥ 1.
            strategies: Mapping of strategy names to callables
                ``(np.ndarray) -> QuantumCircuit``.  Defaults to the four
                built-in strategies.

        Raises:
            ValidationError: If ``n_qubits`` is invalid.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        self.n_qubits: int = int(n_qubits)
        self.strategies: Dict[str, Callable] = (
            strategies if strategies is not None else self._default_strategies()
        )
        self.selected_strategy: str = "auto"

    def _default_strategies(self) -> Dict[str, Callable]:
        """Return the default set of encoding strategies."""
        return {
            "sparse": self._encode_sparse,
            "dense": self._encode_dense,
            "normalized": self._encode_normalized,
            "auto": self._encode_auto,
        }

    def encode(
        self,
        data: np.ndarray,
        strategy: Optional[str] = None,
    ) -> QuantumCircuit:
        """Encode data with adaptive strategy selection.

        Args:
            data: Input feature vector (1-D array).
            strategy: Force a specific strategy name, or ``None`` to use
                the currently selected strategy.

        Returns:
            A ``QuantumCircuit`` with adaptive encoding.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        use_strategy = strategy or self.selected_strategy
        if use_strategy not in self.strategies:
            logger.warning(
                "Unknown strategy %r — falling back to 'auto'.", use_strategy
            )
            use_strategy = "auto"

        return self.strategies[use_strategy](data)

    # ------------------------------------------------------------------
    # Built-in strategies
    # ------------------------------------------------------------------

    def _encode_sparse(self, data: np.ndarray) -> QuantumCircuit:
        """Encoding optimised for sparse data (many zeros)."""
        qc = QuantumCircuit(self.n_qubits)
        nonzero_indices = np.where(data != 0)[0]
        for idx in nonzero_indices:
            if idx < self.n_qubits:
                qc.rx(data[idx], idx)
        return qc

    def _encode_dense(self, data: np.ndarray) -> QuantumCircuit:
        """Encoding for dense (non-sparse) data."""
        qc = QuantumCircuit(self.n_qubits)

        if len(data) == 0:
            return qc

        norm = np.linalg.norm(data)
        normalized = data / norm if norm > 0 else data

        for i in range(self.n_qubits):
            data_idx = i % len(normalized)
            qc.ry(normalized[data_idx], i)

        # Add entanglement
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)

        return qc

    def _encode_normalized(self, data: np.ndarray) -> QuantumCircuit:
        """Encoding assuming data is already in a normalised range."""
        qc = QuantumCircuit(self.n_qubits)
        for i in range(min(len(data), self.n_qubits)):
            qc.ry(data[i] * np.pi, i)
            qc.rz(data[i] * np.pi, i)
        return qc

    def _encode_auto(self, data: np.ndarray) -> QuantumCircuit:
        """Auto-select encoding strategy based on data statistics."""
        if len(data) == 0:
            return QuantumCircuit(self.n_qubits)

        sparsity = np.sum(data == 0) / len(data)

        if sparsity > 0.7:
            return self._encode_sparse(data)
        elif np.max(np.abs(data)) > 2:
            return self._encode_normalized(data)
        else:
            return self._encode_dense(data)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_data(self, data: np.ndarray) -> Dict[str, float]:
        """Analyse data characteristics for strategy selection.

        Args:
            data: Input feature vector.

        Returns:
            Dictionary with sparsity, mean, std, max, and norm.
        """
        data = np.asarray(data, dtype=float)
        if len(data) == 0:
            return {
                "sparsity": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0,
                "norm": 0.0,
            }
        return {
            "sparsity": float(np.sum(data == 0) / len(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "max": float(np.max(np.abs(data))),
            "norm": float(np.linalg.norm(data)),
        }


class HierarchicalEncoder(BaseEncoder):
    """Encodes data in a hierarchical coarse-to-fine structure.

    Global features are encoded first, then progressively finer local
    details, creating a multi-resolution quantum representation.

    Best for: Hierarchical data, multi-scale features.

    Example::

        encoder = HierarchicalEncoder(n_qubits=8, hierarchy_levels=3)
        qc = encoder.encode(data)
    """

    def __init__(
        self,
        n_qubits: int,
        hierarchy_levels: int = 2,
    ) -> None:
        """Initialize hierarchical encoder.

        Args:
            n_qubits: Number of qubits.  Must be ≥ 1.
            hierarchy_levels: Number of hierarchy levels.  Must be ≥ 1
                and ≤ ``n_qubits``.

        Raises:
            ValidationError: On invalid constructor arguments.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        if (
            not isinstance(hierarchy_levels, (int, np.integer))
            or hierarchy_levels < 1
        ):
            raise ValidationError(
                f"hierarchy_levels must be a positive integer, "
                f"got {hierarchy_levels!r}."
            )
        self.n_qubits: int = int(n_qubits)
        self.hierarchy_levels: int = int(hierarchy_levels)
        self.level_gates: List[str] = ["rx", "ry", "rz"]

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode data hierarchically.

        Args:
            data: Input feature vector (1-D array).

        Returns:
            A ``QuantumCircuit`` with hierarchical encoding.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        qc = QuantumCircuit(self.n_qubits)

        if len(data) == 0:
            logger.warning("Empty data vector; returning identity circuit.")
            return qc

        qubits_per_level = self.n_qubits // self.hierarchy_levels

        for level in range(self.hierarchy_levels):
            start = level * qubits_per_level
            end = min((level + 1) * qubits_per_level, self.n_qubits)

            gate_type = self.level_gates[level % len(self.level_gates)]

            # Encode at this level
            for qubit in range(start, end):
                data_idx = (qubit - start) % len(data)
                angle = data[data_idx] / (level + 1)

                if gate_type == "rx":
                    qc.rx(angle, qubit)
                elif gate_type == "ry":
                    qc.ry(angle, qubit)
                elif gate_type == "rz":
                    qc.rz(angle, qubit)

            # Hierarchical entanglement
            step = 2 ** (self.hierarchy_levels - level - 1)
            for qubit in range(start, end - step, step):
                qc.cx(qubit, qubit + step)

        return qc

    def get_hierarchy_info(self) -> Dict:
        """Get hierarchy structure information.

        Returns:
            Dictionary with levels, qubits_per_level, and gate_types.
        """
        return {
            "levels": self.hierarchy_levels,
            "qubits_per_level": self.n_qubits // self.hierarchy_levels,
            "gate_types": list(self.level_gates),
        }
