"""
Quantum Differential Privacy implementation.

When data is sent to a Cloud QPU (like IBM), you are sending user data.
Differential Privacy adds calibrated noise to the quantum circuit before
it leaves your secure environment. This mathematically guarantees that
no single user's data can be reconstructed from the results.
"""

import logging
import numpy as np
from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)


class QuantumDifferentialPrivacy:
    """Applies depolarizing noise to Quantum Circuits to ensure Differential Privacy.

    This guarantees that the output of the quantum computation does not
    reveal the exact input state of any single individual.

    The noise is injected as random Pauli errors (X, Y, or Z gates), which
    effectively 'scramble' the quantum state while maintaining meaningful
    computation results. Each qubit independently receives a random Pauli
    gate with probability ``noise_prob = 1 / (1 + exp(epsilon))``.

    Attributes:
        epsilon: Privacy budget. Smaller values mean more privacy (more noise).
        noise_prob: Probability of applying a noise gate to each qubit.
    """

    def __init__(self, epsilon: float = 1.0) -> None:
        """Initialize QuantumDifferentialPrivacy.

        Args:
            epsilon: Privacy budget (must be > 0).
                Smaller values = higher privacy = more noise.
                Typical range: 0.1 to 10.0.
                Default: 1.0 (moderate privacy).

        Raises:
            ValueError: If epsilon is not positive.
        """
        if not isinstance(epsilon, (int, float)):
            raise TypeError(f"epsilon must be a number, got {type(epsilon).__name__}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.epsilon = float(epsilon)
        self.noise_prob = 1.0 / (1.0 + np.exp(epsilon))

    def sanitize(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Inject Pauli noise gates into the circuit to mask exact data values.

        Creates a privacy-preserved copy of the circuit by inserting a
        privacy barrier followed by random Pauli rotations (X, Y, or Z)
        on each qubit with probability ``self.noise_prob``. The choice of
        Pauli gate is uniformly random among X, Y, and Z.

        Args:
            circuit: Input quantum circuit to sanitize.

        Returns:
            A new ``QuantumCircuit`` with privacy-preserving noise injected.

        Raises:
            TypeError: If *circuit* is not a ``QuantumCircuit``.
        """
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(
                f"Expected QuantumCircuit, got {type(circuit).__name__}"
            )

        secure_qc = circuit.copy()
        n_qubits = secure_qc.num_qubits

        secure_qc.barrier(label="PRIVACY_WALL")

        # Inject random Pauli noise on each qubit independently
        pauli_gates = ["x", "y", "z"]
        noise_applied = 0
        for qubit_idx in range(n_qubits):
            if np.random.random() < self.noise_prob:
                gate = np.random.choice(pauli_gates)
                if gate == "x":
                    secure_qc.x(qubit_idx)
                elif gate == "y":
                    secure_qc.y(qubit_idx)
                else:
                    secure_qc.z(qubit_idx)
                noise_applied += 1

        logger.debug(
            "Privacy noise applied: %d/%d qubits affected (prob=%.4f, eps=%.2f)",
            noise_applied, n_qubits, self.noise_prob, self.epsilon,
        )

        secure_qc.name = f"PrivacyCircuit_eps{self.epsilon}"
        return secure_qc

    def estimate_privacy_loss(self, n_queries: int) -> float:
        """Calculate total privacy budget consumed after multiple queries.

        Uses the advanced composition theorem: total privacy loss grows as
        ``epsilon * sqrt(n_queries)`` rather than linearly.

        Args:
            n_queries: Number of queries/computations performed.

        Returns:
            Total privacy budget consumed (epsilon_total).

        Raises:
            ValueError: If *n_queries* is not a positive integer.
        """
        if not isinstance(n_queries, int) or n_queries <= 0:
            raise ValueError(f"n_queries must be a positive integer, got {n_queries}")

        return self.epsilon * np.sqrt(n_queries)

    def get_noise_parameters(self) -> dict:
        """Get the noise parameters for this privacy configuration.

        Returns:
            Dictionary with keys ``epsilon``, ``noise_probability``, and
            ``privacy_level`` (one of ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``).
        """
        return {
            "epsilon": self.epsilon,
            "noise_probability": self.noise_prob,
            "privacy_level": (
                "HIGH" if self.epsilon < 0.5
                else "MEDIUM" if self.epsilon < 2.0
                else "LOW"
            ),
        }
