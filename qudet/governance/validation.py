"""
Quantum capacity validation for hardware constraints.

Provides pre-flight checks to ensure a dataset fits within the qubit
limits of the target quantum hardware before circuit construction.
"""

from typing import Tuple


def check_quantum_capacity(
    data_shape: Tuple[int, int],
    max_qubits: int = 127,
) -> bool:
    """Check whether a dataset fits within the qubit limit of the target QPU.

    Args:
        data_shape: Tuple of ``(n_samples, n_features)``.  The number
            of features determines the number of qubits required.
        max_qubits: Maximum qubit count on the target hardware.
            Default: 127 (IBM Brisbane).

    Returns:
        ``True`` if the dataset fits.

    Raises:
        ValueError: If the number of features exceeds *max_qubits*.
    """
    n_features = data_shape[1]

    if n_features > max_qubits:
        raise ValueError(
            f"Dataset has {n_features} features, but hardware only has "
            f"{max_qubits} qubits. Please use "
            "qudet.reduction.RandomProjector to reduce dimensions first."
        )

    return True
