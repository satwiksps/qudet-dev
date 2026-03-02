"""
Visualisation helpers for quantum data reduction and kernel matrices.

Requires the optional ``matplotlib`` package.  Functions raise
``ImportError`` with installation instructions when the package is
not available.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _require_matplotlib() -> None:
    """Raise ``ImportError`` if ``matplotlib`` is not installed."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install it with: pip install matplotlib"
        )


def plot_reduction_2d(
    original_data: np.ndarray,
    reduced_data: np.ndarray,
) -> None:
    """Visualise the 'Smart Funnel' data-reduction effect.

    Plots the original dataset (grey) against the quantum coreset (red).
    Uses the first two columns of each array.  The original data is
    capped at 5 000 points for readability.

    Args:
        original_data: Original dataset, shape ``(n_samples, n_features)``.
            Must have at least 2 columns.
        reduced_data: Reduced coreset, shape ``(n_coreset, n_features)``.
            Must have at least 2 columns.

    Raises:
        ImportError: If ``matplotlib`` is not installed.
        ValueError: If either array has fewer than 2 columns.
    """
    _require_matplotlib()

    if original_data.ndim != 2 or original_data.shape[1] < 2:
        raise ValueError("original_data must be 2-D with at least 2 columns")
    if reduced_data.ndim != 2 or reduced_data.shape[1] < 2:
        raise ValueError("reduced_data must be 2-D with at least 2 columns")

    plt.figure(figsize=(10, 6))

    limit = min(len(original_data), 5000)
    plt.scatter(
        original_data[:limit, 0],
        original_data[:limit, 1],
        c="lightgrey",
        alpha=0.5,
        label="Classical Big Data",
    )

    plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c="red",
        s=100,
        marker="x",
        label="Quantum Coreset",
    )

    plt.title("Visualizing Data Reduction for QPU Handoff")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_kernel_matrix(matrix: np.ndarray) -> None:
    """Plot a Quantum Kernel Matrix (Gram Matrix).

    The diagonal should be 1.0 (self-similarity).

    Args:
        matrix: Square kernel matrix of shape ``(n, n)``.

    Raises:
        ImportError: If ``matplotlib`` is not installed.
        ValueError: If *matrix* is not 2-D or not square.
    """
    _require_matplotlib()

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square 2-D array")

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Quantum Fidelity")
    plt.title("Quantum Kernel Matrix")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.show()
