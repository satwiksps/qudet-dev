"""
Quantum circuit compilation and native-gate transpilation.

Provides tools to compile abstract quantum algorithms to device-native gate
sets, optimise compiled circuits by cancelling inverse pairs and merging
compatible rotations, and transpile circuits for specific backends.
"""

import numpy as np
from typing import Dict, List


class QuantumCircuitCompiler:
    """Compile abstract quantum algorithms to native gate sets.

    Takes a circuit specification dictionary and decomposes non-native gates
    into sequences of native gates supported by the target backend.

    Example:
        >>> compiler = QuantumCircuitCompiler(target_gateset="ibm")
        >>> compiled = compiler.compile(circuit_spec, ["rx", "ry", "rz", "cx"])
    """

    def __init__(self, target_gateset: str = "ibm", optimization_level: int = 2) -> None:
        """Initialize the compiler.

        Args:
            target_gateset: Target native gate set identifier
                (``'ibm'``, ``'rigetti'``, ``'ionq'``).
            optimization_level: Compilation optimization level (0–3).
        """
        self.target_gateset = target_gateset
        self.optimization_level = optimization_level
        self.compilation_stats: Dict = {}

    def compile(self, circuit_spec: Dict, native_gates: List[str]) -> Dict:
        """Compile a circuit specification to native gates.

        Args:
            circuit_spec: Circuit specification with ``'gates'`` and
                ``'qubits'`` keys.
            native_gates: List of allowed native gate type strings.

        Returns:
            Compiled circuit dictionary with native-gate decomposition.
        """
        self.compilation_stats = {
            'original_gates': len(circuit_spec.get('gates', [])),
            'native_gates': len(native_gates),
            'depth_reduction': 0.0
        }

        compiled = self._decompose_to_natives(circuit_spec, native_gates)
        compiled['optimization_level'] = self.optimization_level
        return compiled

    def _decompose_to_natives(self, circuit_spec: Dict, native_gates: List[str]) -> Dict:
        """Decompose circuit gates to native gate set."""
        gate_mapping = self._build_gate_decomposition(native_gates)
        decomposed_gates = []

        for gate in circuit_spec.get('gates', []):
            if gate['type'] in native_gates:
                decomposed_gates.append(gate)
            elif gate['type'] in gate_mapping:
                decomposed_gates.extend(gate_mapping[gate['type']])
            else:
                decomposed_gates.append(gate)

        return {
            'gates': decomposed_gates,
            'qubits': circuit_spec.get('qubits', 2),
            'depth': len(decomposed_gates)
        }

    def _build_gate_decomposition(self, native_gates: List[str]) -> Dict:
        """Build decomposition rules for non-native gates."""
        decompositions = {
            'h': [{'type': 'ry', 'param': np.pi / 2}, {'type': 'rz', 'param': np.pi}],
            'x': [{'type': 'rx', 'param': np.pi}],
            'y': [{'type': 'ry', 'param': np.pi}],
            'z': [{'type': 'rz', 'param': np.pi}],
        }
        return {k: v for k, v in decompositions.items() if k not in native_gates}

    def get_compilation_stats(self) -> Dict:
        """Return compilation statistics from the last ``compile()`` call."""
        return self.compilation_stats


class QuantumCircuitOptimizer:
    """Optimize compiled quantum circuits for execution.

    Applies multiple optimization passes that cancel inverse gate pairs and
    merge consecutive same-type single-qubit rotations.

    Example:
        >>> optimizer = QuantumCircuitOptimizer(optimization_passes=5)
        >>> optimized = optimizer.optimize(compiled_circuit)
    """

    def __init__(self, optimization_passes: int = 5) -> None:
        """Initialize the optimizer.

        Args:
            optimization_passes: Number of optimization passes to apply.
        """
        self.optimization_passes = optimization_passes

    def optimize(self, compiled_circuit: Dict) -> Dict:
        """Apply optimization passes to a compiled circuit.

        Args:
            compiled_circuit: Compiled circuit dictionary with a ``'gates'``
                key.

        Returns:
            Optimized circuit dictionary.
        """
        optimized = compiled_circuit.copy()

        for pass_num in range(self.optimization_passes):
            optimized = self._cancel_inverse_pairs(optimized)
            optimized = self._merge_single_qubit_gates(optimized)

        return optimized

    def _cancel_inverse_pairs(self, circuit: Dict) -> Dict:
        """Cancel adjacent inverse gate pairs."""
        gates = circuit.get('gates', [])
        optimized_gates = []
        i = 0

        while i < len(gates):
            if i + 1 < len(gates):
                current = gates[i]
                next_gate = gates[i + 1]

                if self._are_inverses(current, next_gate):
                    i += 2
                    continue

            optimized_gates.append(gates[i])
            i += 1

        return {**circuit, 'gates': optimized_gates}

    def _merge_single_qubit_gates(self, circuit: Dict) -> Dict:
        """Merge consecutive same-type single-qubit gates."""
        gates = circuit.get('gates', [])
        merged = []

        for gate in gates:
            if merged and self._can_merge(merged[-1], gate):
                merged[-1] = self._merge_gates(merged[-1], gate)
            else:
                merged.append(gate)

        return {**circuit, 'gates': merged}

    def _are_inverses(self, gate1: Dict, gate2: Dict) -> bool:
        """Check if two gates are inverses."""
        inverse_pairs = [('rx', 'rx'), ('ry', 'ry'), ('rz', 'rz')]
        if (gate1['type'], gate2['type']) in inverse_pairs:
            param1 = gate1.get('param', 0)
            param2 = gate2.get('param', 0)
            return np.isclose(param1 + param2, 2 * np.pi, atol=1e-6)
        return False

    def _can_merge(self, gate1: Dict, gate2: Dict) -> bool:
        """Check if two gates can be merged.

        Gates can only be merged when they are the **same type** of
        single-qubit rotation acting on the **same qubit**.
        """
        return (gate1['type'] == gate2['type'] and
                gate1.get('qubit') == gate2.get('qubit') and
                gate1['type'] in ['rx', 'ry', 'rz'])

    def _merge_gates(self, gate1: Dict, gate2: Dict) -> Dict:
        """Merge two compatible same-type gates."""
        return {
            'type': gate1['type'],
            'param': gate1.get('param', 0) + gate2.get('param', 0),
            'qubit': gate1.get('qubit')
        }


class QuantumNativeGateTranspiler:
    """Transpile quantum circuits to a specific native gate set.

    Converts non-native gates to equivalent sequences of native gates using
    known decomposition rules.

    Example:
        >>> transpiler = QuantumNativeGateTranspiler(["rx", "ry", "rz", "cx"])
        >>> native_circuit = transpiler.transpile(circuit_spec)
    """

    def __init__(self, native_gates: List[str]) -> None:
        """Initialize the transpiler.

        Args:
            native_gates: List of native gate type strings supported by
                the target backend.
        """
        self.native_gates = native_gates

    def transpile(self, circuit_spec: Dict) -> Dict:
        """Transpile a circuit to the native gate set.

        Args:
            circuit_spec: Circuit specification dictionary.

        Returns:
            Transpiled circuit using only native gates.
        """
        native_circuit = self._convert_to_native(circuit_spec)
        return native_circuit

    def _convert_to_native(self, circuit_spec: Dict) -> Dict:
        """Convert circuit gates to native gate set."""
        native_gates_list = []

        for gate in circuit_spec.get('gates', []):
            if gate['type'] in self.native_gates:
                native_gates_list.append(gate)
            else:
                native_gates_list.extend(self._decompose_gate(gate))

        return {
            'gates': native_gates_list,
            'qubits': circuit_spec.get('qubits', 2),
            'native': True
        }

    def _decompose_gate(self, gate: Dict) -> List[Dict]:
        """Decompose a non-native gate to native gates."""
        gate_type = gate['type']

        if gate_type == 'h' and 'h' not in self.native_gates:
            return [
                {'type': 'ry', 'param': np.pi / 2},
                {'type': 'rz', 'param': np.pi}
            ]
        elif gate_type == 't' and 't' not in self.native_gates:
            return [{'type': 'rz', 'param': np.pi / 4}]
        else:
            return [gate]
