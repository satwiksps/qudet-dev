"""
Quantum circuit optimization and simplification.

Reduces circuit depth and gate count through transpilation passes and
gate cancellation, lowering execution cost and noise on quantum hardware.
"""

import logging
from typing import List

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition, CommutativeCancellation

logger = logging.getLogger(__name__)


class CircuitOptimizer:
    """Optimizes quantum circuits to reduce depth and gate count.

    Applies Qiskit transpiler passes and optimization-level transpilation to
    clean up redundant gates (e.g., consecutive H gates that cancel out).
    This reduces QPU cost and noise before execution.

    Example:
        >>> optimizer = CircuitOptimizer(level=3)
        >>> optimized_circuit = optimizer.optimize(circuit)
        >>> optimized_circuits = optimizer.optimize_batch(circuit_list)
    """

    def __init__(self, level: int = 3) -> None:
        """Initialize the circuit optimizer.

        Args:
            level: Optimization level (0–3).
                - 0: No optimization (fastest, lowest quality).
                - 1: Light optimization.
                - 2: Medium optimization.
                - 3: Heavy optimization (slowest, best quality). **Default.**

        Raises:
            ValueError: If *level* is not in the range 0–3.
        """
        if not 0 <= level <= 3:
            raise ValueError(
                f"Optimization level must be 0-3, got {level}"
            )

        self.level = level

        self.pm = PassManager([
            Optimize1qGatesDecomposition(),
            CommutativeCancellation()
        ])

        logger.info("Circuit optimizer initialized (level %d)", level)

    def optimize(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Optimize a single quantum circuit.

        Applies the following passes in order:

        1. **Optimize1qGatesDecomposition** — combine adjacent single-qubit gates.
        2. **CommutativeCancellation** — cancel gates that commute and annihilate.
        3. **Transpilation** — map to basis gates at the configured level.

        Args:
            circuit: Quantum circuit to optimize.

        Returns:
            Optimized circuit with reduced depth and gate count.
        """
        optimized_qc = transpile(circuit, optimization_level=self.level)
        optimized_qc = self.pm.run(optimized_qc)
        return optimized_qc

    def optimize_batch(self, circuits: List[QuantumCircuit]) -> List[QuantumCircuit]:
        """Optimize a batch of quantum circuits.

        Useful for processing large numbers of circuits (e.g., from a
        ``QuantumDataLoader``) before QPU submission.

        Args:
            circuits: List of quantum circuits to optimize.

        Returns:
            List of optimized circuits in the same order as the input.
        """
        logger.info("Optimizing batch of %d circuits", len(circuits))
        optimized_circuits = [self.optimize(qc) for qc in circuits]
        logger.info("Batch optimization complete")
        return optimized_circuits

    def estimate_savings(
        self,
        original_circuit: QuantumCircuit,
        optimized_circuit: QuantumCircuit,
    ) -> dict:
        """Estimate cost and depth savings from optimization.

        Args:
            original_circuit: Circuit before optimization.
            optimized_circuit: Circuit after optimization.

        Returns:
            Dictionary containing depth, gate-count, and percentage-reduction
            metrics.
        """
        orig_depth = original_circuit.depth()
        opt_depth = optimized_circuit.depth()

        orig_gates = len(original_circuit)
        opt_gates = len(optimized_circuit)

        depth_saved = (1 - opt_depth / orig_depth) * 100 if orig_depth > 0 else 0
        gates_saved = (1 - opt_gates / orig_gates) * 100 if orig_gates > 0 else 0

        return {
            "original_depth": orig_depth,
            "optimized_depth": opt_depth,
            "depth_reduction_%": depth_saved,
            "original_gates": orig_gates,
            "optimized_gates": opt_gates,
            "gates_reduction_%": gates_saved,
        }
