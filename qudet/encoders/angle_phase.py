"""Advanced angle and phase encoding techniques for quantum feature embedding.

Provides sophisticated methods for encoding classical features into quantum
rotation angles and phase information.
"""

import logging
from typing import Optional, List

import numpy as np
from qiskit import QuantumCircuit

from qudet.core.base import BaseEncoder
from qudet.core.exceptions import EncodingError, ValidationError

logger = logging.getLogger(__name__)


class AngleEncoder(BaseEncoder):
    """Encodes classical features into rotation angles of quantum gates.

    Maps features ``x_i`` to rotation angles on one of three axes:

    * ``R_x(x_i)`` — rotation around the X-axis.
    * ``R_y(x_i)`` — rotation around the Y-axis.
    * ``R_z(x_i)`` — rotation around the Z-axis.
    * ``auto`` — cycle through X / Y / Z based on qubit index.

    Multiple repetition layers may be applied to deepen the encoding.

    Best for: Simple feature encoding, fast feature loading,
    parametrised circuits.

    Example::

        encoder = AngleEncoder(n_qubits=4, angle_type="ry")
        qc = encoder.encode(np.array([0.1, 0.5, 1.2, 0.8]))
    """

    _VALID_ANGLE_TYPES = {"rx", "ry", "rz", "auto"}

    def __init__(
        self,
        n_qubits: int,
        angle_type: str = "rx",
        reps: int = 1,
    ) -> None:
        """Initialize angle encoder.

        Args:
            n_qubits: Number of qubits.  Must be ≥ 1.
            angle_type: Rotation axis (``'rx'``, ``'ry'``, ``'rz'``,
                or ``'auto'``).
            reps: Number of encoding repetitions.  Must be ≥ 1.

        Raises:
            ValidationError: On invalid arguments.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        angle_type = angle_type.lower()
        if angle_type not in self._VALID_ANGLE_TYPES:
            raise ValidationError(
                f"angle_type must be one of {self._VALID_ANGLE_TYPES}, "
                f"got {angle_type!r}."
            )
        if not isinstance(reps, (int, np.integer)) or reps < 1:
            raise ValidationError(
                f"reps must be a positive integer, got {reps!r}."
            )
        self.n_qubits: int = int(n_qubits)
        self.angle_type: str = angle_type
        self.reps: int = int(reps)

    def encode(
        self,
        data: np.ndarray,
        angle_type: Optional[str] = None,
    ) -> QuantumCircuit:
        """Encode data into rotation angles.

        Args:
            data: Input feature vector (1-D array).
            angle_type: Override the default angle type for this call.

        Returns:
            A ``QuantumCircuit`` with angle-encoded data.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        atype = angle_type or self.angle_type
        qc = QuantumCircuit(self.n_qubits)

        for _rep in range(self.reps):
            for i in range(min(len(data), self.n_qubits)):
                angle = data[i]

                if atype == "rx":
                    qc.rx(angle, i)
                elif atype == "ry":
                    qc.ry(angle, i)
                elif atype == "rz":
                    qc.rz(angle, i)
                elif atype == "auto":
                    rotation_choice = i % 3
                    if rotation_choice == 0:
                        qc.rx(angle, i)
                    elif rotation_choice == 1:
                        qc.ry(angle, i)
                    else:
                        qc.rz(angle, i)

        return qc

    def encode_scaled(
        self,
        data: np.ndarray,
        scale_factor: float = np.pi,
    ) -> QuantumCircuit:
        """Encode data with a scaling factor applied to all angles.

        Args:
            data: Input features.
            scale_factor: Multiplicative scale for angles (default: π).

        Returns:
            A ``QuantumCircuit`` with scaled angle encoding.
        """
        scaled_data = np.asarray(data, dtype=float) * scale_factor
        return self.encode(scaled_data)


class PhaseEncoder(BaseEncoder):
    """Encodes data into quantum phase information.

    Uses phase-shift gates and controlled-phase gates to embed features.
    Phase encoding is efficient and preserves quantum coherence.

    Best for: Phase-sensitive applications, quantum interference, QAOA.

    Example::

        encoder = PhaseEncoder(n_qubits=4)
        qc = encoder.encode(np.array([0.1, 0.5, 1.2, 0.8]))
    """

    def __init__(
        self,
        n_qubits: int,
        global_phase: bool = False,
    ) -> None:
        """Initialize phase encoder.

        Args:
            n_qubits: Number of qubits.  Must be ≥ 1.
            global_phase: Whether to apply a global phase (not
                measurable, but useful for interference).

        Raises:
            ValidationError: If ``n_qubits`` is not a positive integer.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        self.n_qubits: int = int(n_qubits)
        self.global_phase: bool = global_phase

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode data into quantum phases.

        Args:
            data: Input feature vector (1-D array).

        Returns:
            A ``QuantumCircuit`` with phase encoding.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        qc = QuantumCircuit(self.n_qubits)

        # Normalise data to [0, 2π]
        norm = np.max(np.abs(data)) if len(data) > 0 else 0.0
        if norm > 0:
            normalized = (data / norm) * 2 * np.pi
        else:
            normalized = data

        n_encoded = min(len(normalized), self.n_qubits)

        # Apply single-qubit phase gates
        for i in range(n_encoded):
            qc.p(normalized[i], i)

        # Add entanglement with controlled-phase (guard index bounds)
        for i in range(self.n_qubits - 1):
            if i < len(normalized):
                qc.cp(normalized[i] * 0.5, i, i + 1)

        return qc

    def apply_global_phase(
        self,
        qc: QuantumCircuit,
        phase: float,
    ) -> QuantumCircuit:
        """Apply a global phase to an existing circuit.

        Args:
            qc: Input circuit.
            phase: Phase to apply (in radians).

        Returns:
            The same circuit with the global phase set.
        """
        qc.global_phase = phase
        return qc


class HybridAnglePhaseEncoder(BaseEncoder):
    """Combines angle and phase encoding for comprehensive feature mapping.

    Uses both rotation angles and phase information to encode features,
    providing a richer feature-space representation.

    Best for: Complex feature interactions, hybrid quantum-classical
    algorithms.

    Example::

        encoder = HybridAnglePhaseEncoder(n_qubits=4)
        qc = encoder.encode(data)
    """

    def __init__(
        self,
        n_qubits: int,
        angle_weight: float = 0.5,
        phase_weight: float = 0.5,
    ) -> None:
        """Initialize hybrid encoder.

        Args:
            n_qubits: Number of qubits.  Must be ≥ 1.
            angle_weight: Relative weight for angle encoding (≥ 0).
            phase_weight: Relative weight for phase encoding (≥ 0).

        Raises:
            ValidationError: If ``n_qubits`` is invalid or both weights
                are zero.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        total_weight = angle_weight + phase_weight
        if total_weight == 0:
            raise ValidationError(
                "angle_weight and phase_weight cannot both be zero."
            )
        self.n_qubits: int = int(n_qubits)
        self.angle_weight: float = angle_weight / total_weight
        self.phase_weight: float = phase_weight / total_weight

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode data using a hybrid angle-phase approach.

        Args:
            data: Input feature vector (1-D array).

        Returns:
            A ``QuantumCircuit`` with hybrid encoding.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        qc = QuantumCircuit(self.n_qubits)

        for i in range(min(len(data), self.n_qubits)):
            # Angle encoding with weight
            angle = data[i] * self.angle_weight
            qc.ry(angle, i)

            # Phase encoding with weight
            phase = data[i] * self.phase_weight
            qc.p(phase, i)

        # Add entanglement
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)

        return qc

    def get_encoding_weights(self) -> dict:
        """Get current encoding weights.

        Returns:
            Dictionary with ``angle_weight`` and ``phase_weight``.
        """
        return {
            "angle_weight": self.angle_weight,
            "phase_weight": self.phase_weight,
        }


class MultiAxisRotationEncoder(BaseEncoder):
    """Encodes using combined rotations around multiple axes.

    Applies rotations around X, Y, and/or Z axes in sequence,
    creating a more expressive encoding space.

    Best for: Multi-axis feature spaces, rotational symmetry.

    Example::

        encoder = MultiAxisRotationEncoder(n_qubits=4, axes=["x", "y"])
        qc = encoder.encode(data)
    """

    _VALID_AXES = {"x", "y", "z"}

    def __init__(
        self,
        n_qubits: int,
        axes: Optional[List[str]] = None,
    ) -> None:
        """Initialize multi-axis encoder.

        Args:
            n_qubits: Number of qubits.  Must be ≥ 1.
            axes: List of rotation axes (``'x'``, ``'y'``, ``'z'``).
                Defaults to ``['x', 'y', 'z']``.

        Raises:
            ValidationError: On invalid arguments.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        axes = axes or ["x", "y", "z"]
        for axis in axes:
            if axis not in self._VALID_AXES:
                raise ValidationError(
                    f"Each axis must be one of {self._VALID_AXES}, "
                    f"got {axis!r}."
                )
        self.n_qubits: int = int(n_qubits)
        self.axes: List[str] = list(axes)

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """Encode data using multi-axis rotations.

        Args:
            data: Input feature vector (1-D array).

        Returns:
            A ``QuantumCircuit`` with multi-axis encoding.

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

        for axis_idx, axis in enumerate(self.axes):
            for i in range(n_features):
                angle = data[i] * (1.0 + axis_idx * 0.25)

                if axis == "x":
                    qc.rx(angle, i)
                elif axis == "y":
                    qc.ry(angle, i)
                elif axis == "z":
                    qc.rz(angle, i)

            # Entanglement after each axis
            for i in range(n_features - 1):
                qc.cx(i, i + 1)

        return qc

    def get_available_axes(self) -> List[str]:
        """Get list of available rotation axes.

        Returns:
            ``['x', 'y', 'z']``.
        """
        return ["x", "y", "z"]


class ParametricAngleEncoder(BaseEncoder):
    """Creates parametric angle encoding for variational circuits.

    Encodes data with trainable parameters that can be optimised,
    useful for hybrid quantum-classical machine learning.

    Best for: Variational quantum algorithms, parametric circuits, VQA.

    Example::

        encoder = ParametricAngleEncoder(n_qubits=4)
        qc = encoder.encode(data)
        encoder.update_parameters(optimised_params)
    """

    def __init__(
        self,
        n_qubits: int,
        n_params: Optional[int] = None,
        param_sharing: bool = False,
    ) -> None:
        """Initialize parametric encoder.

        Args:
            n_qubits: Number of qubits.  Must be ≥ 1.
            n_params: Number of trainable parameters (default: ``n_qubits``).
            param_sharing: Whether to share parameters across qubits.

        Raises:
            ValidationError: On invalid arguments.
        """
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValidationError(
                f"n_qubits must be a positive integer, got {n_qubits!r}."
            )
        self.n_qubits: int = int(n_qubits)
        self.n_params: int = int(n_params) if n_params is not None else self.n_qubits
        self.param_sharing: bool = param_sharing
        self.parameters: np.ndarray = np.random.randn(self.n_params) * 0.1

    def encode(
        self,
        data: np.ndarray,
        parameters: Optional[np.ndarray] = None,
    ) -> QuantumCircuit:
        """Encode data with parametric angles.

        Args:
            data: Input feature vector (1-D array).
            parameters: Optional parameter values to use instead of the
                stored parameters.

        Returns:
            A ``QuantumCircuit`` with parametric encoding.

        Raises:
            EncodingError: If *data* is not a 1-D numeric array.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise EncodingError(
                f"data must be a 1-D array, got shape {data.shape}."
            )

        params = parameters if parameters is not None else self.parameters
        qc = QuantumCircuit(self.n_qubits)

        # Feature encoding
        for i in range(min(len(data), self.n_qubits)):
            qc.ry(data[i], i)

        # Parametric rotations
        for i in range(self.n_qubits):
            if self.param_sharing:
                param_idx = i % len(params)
            else:
                param_idx = min(i, len(params) - 1)
            qc.rz(params[param_idx], i)

        return qc

    def update_parameters(self, new_params: np.ndarray) -> None:
        """Update the trainable parameters.

        Args:
            new_params: New parameter values (must match ``n_params``).

        Raises:
            ValueError: If length of *new_params* does not match ``n_params``.
        """
        new_params = np.asarray(new_params, dtype=float)
        if len(new_params) != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {len(new_params)}."
            )
        self.parameters = new_params

    def get_parameters(self) -> np.ndarray:
        """Get a copy of the current trainable parameters.

        Returns:
            1-D array of parameter values.
        """
        return self.parameters.copy()
