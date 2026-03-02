"""
Quantum Drift Detector using Maximum Mean Discrepancy (MMD).

Detects data drift by comparing reference and current distributions
in a reproducing kernel Hilbert space (RKHS) using quantum-inspired
kernel approximations.

The detector stores a baseline (training) distribution and then tests
incoming batches against it.  If the two-sample MMD statistic exceeds a
configurable threshold, drift is declared.

References:
    Gretton, A. et al. (2012).  *A Kernel Two-Sample Test.*
    Journal of Machine Learning Research 13, pp. 723–773.
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class QuantumDriftDetector:
    """Detects data drift via Quantum Maximum Mean Discrepancy (MMD).

    Compares a stored *reference* dataset against incoming *current* data
    by evaluating the MMD statistic in a feature space induced by a
    quantum kernel.

    The quantum kernel is computed via the
    :class:`~qudet.analytics.anomaly.QuantumKernelAnomalyDetector` helper,
    which approximates the RBF (Gaussian) kernel using random Fourier
    features mapped onto parameterised quantum circuits.  Specifically:

    .. math::

        K(x, x') \\approx \\langle \\phi(x) | \\phi(x') \\rangle

    where :math:`|\\phi(x)\\rangle` is a quantum state produced by an
    IQP-style encoding circuit parameterised by *x*.  The overlap is
    evaluated classically via statevector simulation for speed.

    The two-sample MMD² statistic is then:

    .. math::

        \\mathrm{MMD}^2 = \\mathbb{E}[K(x,x')]
                          - 2\\,\\mathbb{E}[K(x,y)]
                          + \\mathbb{E}[K(y,y')]

    A value above ``threshold`` signals distribution drift.

    Args:
        n_qubits: Number of qubits for the kernel circuit.  Default: 4.
        threshold: MMD threshold for drift detection.  Default: 0.1.
            Higher values reduce false alarms at the cost of missing
            subtle drift.

    Attributes:
        reference_data\_: Stored reference (training) data, set by
            :meth:`fit_reference`.
        threshold: Decision boundary for the MMD test.

    Examples:
        >>> detector = QuantumDriftDetector(n_qubits=4, threshold=0.15)
        >>> detector.fit_reference(X_train)
        >>> result = detector.detect_drift(X_current)
        >>> if result['drift_detected']:
        ...     print(f"Drift! MMD={result['mmd_score']:.4f}")
    """

    def __init__(
        self,
        n_qubits: int = 4,
        threshold: float = 0.1,
    ) -> None:
        """Initialize Quantum Drift Detector.

        Args:
            n_qubits: Number of qubits for kernel computation.
            threshold: MMD decision threshold (must be ≥ 0).

        Raises:
            ValueError: If *n_qubits* < 1 or *threshold* < 0.
        """
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

        self.n_qubits = n_qubits
        self.threshold = threshold
        self.reference_data_: Optional[np.ndarray] = None

        self._kernel_computer = None

    @property
    def kernel_computer(self):
        """Lazy-load the quantum kernel computer.

        The kernel computer is an instance of
        :class:`~qudet.analytics.anomaly.QuantumKernelAnomalyDetector`
        configured with ``self.n_qubits``.  It is created on first
        access to avoid circular-import overhead.

        Returns:
            Configured kernel computer instance.
        """
        if self._kernel_computer is None:
            from ..analytics.anomaly import QuantumKernelAnomalyDetector
            self._kernel_computer = QuantumKernelAnomalyDetector(
                n_qubits=self.n_qubits
            )
        return self._kernel_computer

    def fit_reference(self, X: np.ndarray) -> "QuantumDriftDetector":
        """Store the baseline (reference) data.

        Typically this is your training data.  Subsequent calls to
        :meth:`detect_drift` compare new data against this reference.

        Args:
            X: Reference data of shape ``(n_samples, n_features)``.

        Returns:
            self

        Raises:
            ValueError: If *X* is not a 2-D array.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        self.reference_data_ = X
        logger.info(
            "Reference data stored: shape=%s (samples=%d, features=%d)",
            X.shape, X.shape[0], X.shape[1],
        )
        return self

    def detect_drift(self, X_new: np.ndarray) -> Dict:
        """Compute the MMD statistic and test for drift.

        The kernel matrices ``K(ref, ref)``, ``K(new, new)`` and
        ``K(ref, new)`` are computed via the quantum kernel approximation,
        and the MMD² statistic is derived as:

        .. math::

            \\text{MMD}^2 = \\bar K_{xx} - 2\\bar K_{xy} + \\bar K_{yy}

        where the bars denote element-wise means.

        Args:
            X_new: New data of shape ``(n_samples, n_features)``.

        Returns:
            Dictionary with keys:
                - ``drift_detected`` (bool): ``True`` if MMD > threshold.
                - ``mmd_score`` (float): Computed MMD value.
                - ``threshold`` (float): Decision threshold.
                - ``status`` (str): ``"DRIFT DETECTED"`` or ``"STABLE"``.
                - ``reference_size`` (int): Number of reference samples.
                - ``new_size`` (int): Number of new samples.

        Raises:
            ValueError: If reference data is not set or feature dimensions
                do not match.
        """
        if self.reference_data_ is None:
            raise ValueError(
                "Reference data not set.  Call fit_reference() first."
            )
        if X_new.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_new.shape}")
        if X_new.shape[1] != self.reference_data_.shape[1]:
            raise ValueError(
                f"Feature mismatch: reference has {self.reference_data_.shape[1]} "
                f"features, new data has {X_new.shape[1]}"
            )

        logger.info(
            "Computing quantum drift (MMD): ref=%d samples, new=%d samples",
            self.reference_data_.shape[0], X_new.shape[0],
        )

        k_xx = self.kernel_computer._compute_kernel_matrix(
            self.reference_data_, self.reference_data_
        )
        mean_xx = float(np.mean(k_xx))

        k_yy = self.kernel_computer._compute_kernel_matrix(X_new, X_new)
        mean_yy = float(np.mean(k_yy))

        k_xy = self.kernel_computer._compute_kernel_matrix(
            self.reference_data_, X_new
        )
        mean_xy = float(np.mean(k_xy))

        mmd_score = max(0.0, mean_xx + mean_yy - 2.0 * mean_xy)
        is_drift = mmd_score > self.threshold

        result = {
            "drift_detected": bool(is_drift),
            "mmd_score": round(mmd_score, 4),
            "threshold": self.threshold,
            "status": "DRIFT DETECTED" if is_drift else "STABLE",
            "reference_size": self.reference_data_.shape[0],
            "new_size": X_new.shape[0],
        }

        logger.info(
            "MMD result: score=%.4f, threshold=%.4f, status=%s",
            result["mmd_score"], result["threshold"], result["status"],
        )
        return result

    def set_threshold(self, threshold: float) -> "QuantumDriftDetector":
        """Update the MMD decision threshold.

        Args:
            threshold: New threshold value (must be ≥ 0).

        Returns:
            self

        Raises:
            ValueError: If *threshold* is negative.
        """
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")

        self.threshold = threshold
        logger.info("Threshold updated to %.4f", threshold)
        return self

    def get_config(self) -> Dict:
        """Return the current detector configuration.

        Returns:
            Dictionary with ``n_qubits``, ``threshold``, and
            ``reference_shape``.
        """
        return {
            "n_qubits": self.n_qubits,
            "threshold": self.threshold,
            "reference_shape": (
                self.reference_data_.shape
                if self.reference_data_ is not None
                else None
            ),
        }
