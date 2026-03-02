"""
Noise profilers for quantum hardware simulation and stress testing.

Generates realistic noise models to validate QuDET pipelines against
real quantum hardware conditions without accessing actual QPUs.

Requires the optional ``qiskit-aer`` package.  All public methods
raise ``ImportError`` with installation instructions when the package
is not available.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
    )
    HAS_AER = True
except ImportError:
    HAS_AER = False


def _require_aer() -> None:
    """Raise ``ImportError`` if ``qiskit-aer`` is not installed."""
    if not HAS_AER:
        raise ImportError(
            "qiskit-aer is required for noise simulation. "
            "Install it with: pip install qiskit-aer"
        )


# Maximum allowed depolarizing error probability (must be < 1.0 to be
# physically meaningful for 2-qubit gates where the limit is 15/16).
_MAX_2Q_ERROR = 0.75


class NoiseSimulator:
    """Generates realistic noise models to stress-test QuDET pipelines.

    Quantum hardware experiences various types of noise:

    * **Depolarizing errors** — random bit/phase flips on every gate.
    * **Thermal relaxation** — energy loss characterised by T1/T2 times.
    * **Gate errors** — imperfect control pulses.

    This class helps answer: *"Will my algorithm survive on real hardware?"*

    By testing locally on noisy simulators you can:

    1. Identify algorithmic weak points.
    2. Add error-mitigation techniques.
    3. Estimate expected accuracy on real devices.
    4. Benchmark across different noise profiles.

    All methods are static; no internal state is required.
    """

    @staticmethod
    def get_noisy_backend(error_prob: float = 0.01):
        """Return a simulator with depolarizing noise.

        Depolarizing noise is the most common noise model:

        * Single-qubit gates — error probability ``error_prob``.
        * Two-qubit gates — ``min(error_prob × 10, 0.75)``.
        * Readout — ``min(error_prob × 5, 0.75)``.

        Args:
            error_prob: Probability of single-qubit gate error.
                Typical hardware values: 0.001–0.01 (0.1 %–1 %).
                Default: 0.01 (1 %).

        Returns:
            ``AerSimulator`` configured with the noise model.

        Raises:
            ImportError: If ``qiskit-aer`` is not installed.
            ValueError: If *error_prob* is outside (0, 1).
        """
        _require_aer()
        _validate_error_prob(error_prob)

        noise_model = NoiseModel()

        error_1q = depolarizing_error(error_prob, 1)
        noise_model.add_all_qubit_quantum_error(
            error_1q, ["u1", "u2", "u3", "rx", "ry", "rz"]
        )

        error_2q = depolarizing_error(
            min(error_prob * 10, _MAX_2Q_ERROR), 2
        )
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz"])

        error_readout = depolarizing_error(
            min(error_prob * 5, _MAX_2Q_ERROR), 1
        )
        noise_model.add_all_qubit_quantum_error(error_readout, ["measure"])

        return AerSimulator(noise_model=noise_model)

    @staticmethod
    def get_thermal_backend(
        t1: float = 50e-6,
        t2: float = 70e-6,
        gate_time: float = 100e-9,
    ):
        """Return a simulator with T1/T2 thermal relaxation errors.

        Physical qubits lose energy (relax) over time:

        * **T1** — relaxation time (energy decay).
        * **T2** — dephasing time (phase randomisation).

        Args:
            t1: T1 relaxation time in seconds.  Default: 50 µs.
            t2: T2 dephasing time in seconds.  Default: 70 µs.
            gate_time: Typical gate duration in seconds.  Default: 100 ns.

        Returns:
            ``AerSimulator`` with thermal-relaxation noise.

        Raises:
            ImportError: If ``qiskit-aer`` is not installed.
            ValueError: If *t2* > 2 × *t1* or any time is non-positive.
        """
        _require_aer()
        if t1 <= 0 or t2 <= 0 or gate_time <= 0:
            raise ValueError("All time parameters must be positive.")
        if t2 > 2 * t1:
            raise ValueError(
                f"t2 ({t2}) must be <= 2*t1 ({2*t1}) by physics constraint."
            )

        noise_model = NoiseModel()

        thermal_error = thermal_relaxation_error(t1, t2, gate_time)
        noise_model.add_all_qubit_quantum_error(
            thermal_error, ["u3", "rx", "ry", "rz"]
        )

        error_2q = depolarizing_error(0.01, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz"])

        return AerSimulator(noise_model=noise_model)

    @staticmethod
    def get_ibm_like_backend(error_prob: float = 0.005):
        """Return a simulator mimicking IBM quantum hardware noise.

        IBM devices typically exhibit:

        * Single-qubit error: 0.3 %–1 %
        * Two-qubit error: 1 %–5 %
        * Readout error: 2 %–5 %

        Args:
            error_prob: Base error probability.  Default: 0.005 (0.5 %).

        Returns:
            ``AerSimulator`` with IBM-like noise characteristics.

        Raises:
            ImportError: If ``qiskit-aer`` is not installed.
            ValueError: If *error_prob* is outside (0, 1).
        """
        _require_aer()
        _validate_error_prob(error_prob)

        noise_model = NoiseModel()

        error_1q = depolarizing_error(error_prob, 1)
        noise_model.add_all_qubit_quantum_error(
            error_1q, ["u1", "u2", "u3"]
        )

        error_2q = depolarizing_error(
            min(error_prob * 15, _MAX_2Q_ERROR), 2
        )
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

        error_readout = depolarizing_error(
            min(error_prob * 8, _MAX_2Q_ERROR), 1
        )
        noise_model.add_all_qubit_quantum_error(error_readout, ["measure"])

        return AerSimulator(noise_model=noise_model)

    @staticmethod
    def get_high_noise_backend(error_prob: float = 0.05):
        """Return a simulator with HIGH noise levels for stress testing.

        Useful to answer: *"Does this still work on a bad day?"*

        Two-qubit gate error is capped at ``min(error_prob × 5, 0.75)``
        to keep the noise model physically valid.

        Args:
            error_prob: High error probability.  Default: 0.05 (5 %).

        Returns:
            ``AerSimulator`` with high-noise model.

        Raises:
            ImportError: If ``qiskit-aer`` is not installed.
            ValueError: If *error_prob* is outside (0, 1).
        """
        _require_aer()
        _validate_error_prob(error_prob)

        noise_model = NoiseModel()

        error_1q = depolarizing_error(error_prob, 1)
        noise_model.add_all_qubit_quantum_error(
            error_1q, ["u1", "u2", "u3", "rx", "ry", "rz"]
        )

        error_2q_prob = min(error_prob * 5, _MAX_2Q_ERROR)
        error_2q = depolarizing_error(error_2q_prob, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz"])

        error_readout_prob = min(error_prob * 10, _MAX_2Q_ERROR)
        error_readout = depolarizing_error(error_readout_prob, 1)
        noise_model.add_all_qubit_quantum_error(error_readout, ["measure"])

        logger.info(
            "High-noise backend: 1q=%.3f, 2q=%.3f, readout=%.3f",
            error_prob, error_2q_prob, error_readout_prob,
        )

        return AerSimulator(noise_model=noise_model)

    @staticmethod
    def get_noiseless_backend():
        """Return a noiseless (ideal) simulator as a performance baseline.

        Useful for comparison: *"How much does noise hurt performance?"*

        Returns:
            ``AerSimulator`` with no noise.

        Raises:
            ImportError: If ``qiskit-aer`` is not installed.
        """
        _require_aer()
        return AerSimulator()

    @staticmethod
    def estimate_accuracy_degradation(
        baseline_accuracy: float,
        error_prob: float = 0.01,
    ) -> float:
        """Rough estimate of accuracy loss due to hardware noise.

        Uses the rule of thumb that each gate error accumulates
        proportionally with circuit depth:
        ``accuracy ≈ baseline × (1 − error_prob × depth)``.

        An average circuit depth of 30 gates is assumed.

        Args:
            baseline_accuracy: Accuracy on a noiseless simulator (0–1).
            error_prob: Error probability per gate.

        Returns:
            Estimated accuracy on a noisy device, clamped to [0, 1].

        Raises:
            ValueError: If *baseline_accuracy* is outside [0, 1] or
                *error_prob* is outside (0, 1).
        """
        if not 0.0 <= baseline_accuracy <= 1.0:
            raise ValueError(
                f"baseline_accuracy must be in [0, 1], got {baseline_accuracy}"
            )
        _validate_error_prob(error_prob)

        avg_circuit_depth = 30
        expected_loss = error_prob * avg_circuit_depth
        degraded_accuracy = baseline_accuracy * (1 - expected_loss)
        return float(np.clip(degraded_accuracy, 0.0, 1.0))


def _validate_error_prob(error_prob: float) -> None:
    """Validate that an error probability is in (0, 1).

    Args:
        error_prob: Value to validate.

    Raises:
        ValueError: If *error_prob* is outside the open interval (0, 1).
    """
    if not 0 < error_prob < 1:
        raise ValueError(
            f"error_prob must be in (0, 1), got {error_prob}"
        )
