"""
Resource estimation for quantum circuit execution.

Provides cost and feasibility analysis for quantum jobs to help
with budgeting and capacity planning in data pipelines.
"""

import logging

from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)


class ResourceEstimator:
    """Estimates the cost and complexity of a quantum job before execution.

    Analyses circuit depth, gate counts, and shot count to estimate
    wall-clock time and cloud-QPU cost.  Also provides a simple
    feasibility check for dataset dimensions against current NISQ
    hardware limits.

    All methods are static — no internal state is required.
    """

    @staticmethod
    def estimate_circuit_cost(
        circuit: QuantumCircuit,
        shots: int = 1024,
        hardware_rate: float = 0.5,
    ) -> dict:
        """Estimate execution time and cost for a quantum circuit.

        The time estimate is a rough model:
        ``(depth × 1 µs × shots) + 0.1 s overhead``.

        Args:
            circuit: Quantum circuit to analyse.
            shots: Number of measurement shots.  Default: 1024.
            hardware_rate: Cost per second in USD.  Default: 0.50.

        Returns:
            Dictionary with keys:
                - ``qubits_used`` (int)
                - ``circuit_depth`` (int)
                - ``cnot_count`` (int)
                - ``total_shots`` (int)
                - ``est_runtime_sec`` (float)
                - ``est_cost_usd`` (float)

        Raises:
            TypeError: If *circuit* is not a ``QuantumCircuit``.
            ValueError: If *shots* or *hardware_rate* is not positive.
        """
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(
                f"Expected QuantumCircuit, got {type(circuit).__name__}"
            )
        if shots <= 0:
            raise ValueError(f"shots must be positive, got {shots}")
        if hardware_rate < 0:
            raise ValueError(
                f"hardware_rate must be non-negative, got {hardware_rate}"
            )

        depth = circuit.depth()
        ops = circuit.count_ops()
        n_cnots = ops.get("cx", 0)
        n_qubits = circuit.num_qubits

        estimated_exec_time = (depth * 1e-6 * shots) + 0.1
        estimated_price = estimated_exec_time * hardware_rate

        return {
            "qubits_used": n_qubits,
            "circuit_depth": depth,
            "cnot_count": n_cnots,
            "total_shots": shots,
            "est_runtime_sec": round(estimated_exec_time, 4),
            "est_cost_usd": round(estimated_price, 4),
        }

    @staticmethod
    def check_pipeline_feasibility(
        n_samples: int,
        n_features: int,
    ) -> str:
        """Advise whether a dataset fits current NISQ hardware limits.

        Args:
            n_samples: Number of data samples.
            n_features: Number of features (maps to qubits).

        Returns:
            A human-readable feasibility assessment string:
                - ``"INFEASIBLE: …"`` when features exceed 127 qubits.
                - ``"EXPENSIVE: …"`` when samples exceed 10 000.
                - ``"FEASIBLE: …"`` otherwise.

        Raises:
            ValueError: If *n_samples* or *n_features* is not positive.
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")

        if n_features > 127:
            return (
                "INFEASIBLE: Too many features. "
                "Use RandomProjector first."
            )

        if n_samples > 10_000:
            return (
                "EXPENSIVE: >10k samples. "
                "Recommended to use CoresetReducer."
            )

        return "FEASIBLE: Job fits within standard NISQ limits."
