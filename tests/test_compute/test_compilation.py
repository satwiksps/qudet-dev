import pytest
import numpy as np
from qudet.compute.compilation import (
    QuantumCircuitCompiler,
    QuantumCircuitOptimizer,
    QuantumNativeGateTranspiler
)


class TestQuantumCircuitCompiler:
    """Test suite for Quantum Circuit Compiler."""

    def test_initialization(self):
        """Test compiler initialization."""
        compiler = QuantumCircuitCompiler(target_gateset="ibm", optimization_level=2)
        assert compiler.target_gateset == "ibm"
        assert compiler.optimization_level == 2

    def test_compile_simple_circuit(self):
        """Test compilation of simple circuit."""
        compiler = QuantumCircuitCompiler()
        circuit_spec = {
            'gates': [
                {'type': 'h', 'qubit': 0},
                {'type': 'cx', 'qubits': [0, 1]}
            ],
            'qubits': 2
        }
        native_gates = ['rx', 'ry', 'rz', 'cx']
        
        compiled = compiler.compile(circuit_spec, native_gates)
        
        assert 'gates' in compiled
        assert compiled['qubits'] == 2

    def test_compile_with_decomposition(self):
        """Test circuit compilation with gate decomposition."""
        compiler = QuantumCircuitCompiler()
        circuit_spec = {
            'gates': [{'type': 'h', 'qubit': 0}],
            'qubits': 1
        }
        native_gates = ['rx', 'ry', 'rz']
        
        compiled = compiler.compile(circuit_spec, native_gates)
        
        assert len(compiled['gates']) > 0

    def test_get_compilation_stats(self):
        """Test retrieval of compilation statistics."""
        compiler = QuantumCircuitCompiler()
        circuit_spec = {
            'gates': [{'type': 'h'}, {'type': 'x'}, {'type': 'cx'}],
            'qubits': 2
        }
        
        compiled = compiler.compile(circuit_spec, ['cx'])
        stats = compiler.get_compilation_stats()
        
        assert 'original_gates' in stats
        assert 'native_gates' in stats


class TestQuantumCircuitOptimizer:
    """Test suite for Quantum Circuit Optimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = QuantumCircuitOptimizer(optimization_passes=3)
        assert optimizer.optimization_passes == 3

    def test_optimize_circuit(self):
        """Test basic circuit optimization."""
        optimizer = QuantumCircuitOptimizer()
        circuit = {
            'gates': [
                {'type': 'rx', 'param': 1.57},
                {'type': 'ry', 'param': 0.0}
            ],
            'qubits': 1
        }
        
        optimized = optimizer.optimize(circuit)
        
        assert 'gates' in optimized
        assert optimized['qubits'] == 1

    def test_cancel_inverse_pairs(self):
        """Test cancellation of inverse gate pairs."""
        optimizer = QuantumCircuitOptimizer()
        circuit = {
            'gates': [
                {'type': 'rx', 'param': np.pi, 'qubit': 0},
                {'type': 'rx', 'param': np.pi, 'qubit': 0}
            ]
        }
        
        optimized = optimizer._cancel_inverse_pairs(circuit)
        
        assert len(optimized['gates']) == 0

    def test_merge_single_qubit_gates(self):
        """Test merging of consecutive single-qubit gates."""
        optimizer = QuantumCircuitOptimizer()
        circuit = {
            'gates': [
                {'type': 'rx', 'param': 0.5, 'qubit': 0},
                {'type': 'rx', 'param': 0.5, 'qubit': 0}
            ]
        }
        
        merged = optimizer._merge_single_qubit_gates(circuit)
        
        assert len(merged['gates']) <= len(circuit['gates'])

    def test_multiple_optimization_passes(self):
        """Test optimization with multiple passes."""
        optimizer = QuantumCircuitOptimizer(optimization_passes=5)
        circuit = {
            'gates': [
                {'type': 'h', 'qubit': 0},
                {'type': 'x', 'qubit': 0},
                {'type': 'h', 'qubit': 0}
            ]
        }
        
        optimized = optimizer.optimize(circuit)
        
        assert 'gates' in optimized


class TestQuantumNativeGateTranspiler:
    """Test suite for Quantum Native Gate Transpiler."""

    def test_initialization(self):
        """Test transpiler initialization."""
        native_gates = ['rx', 'ry', 'rz', 'cx']
        transpiler = QuantumNativeGateTranspiler(native_gates)
        assert transpiler.native_gates == native_gates

    def test_transpile_simple_circuit(self):
        """Test transpilation of simple circuit."""
        transpiler = QuantumNativeGateTranspiler(['rx', 'ry', 'rz', 'cx'])
        circuit_spec = {
            'gates': [
                {'type': 'rx', 'param': 1.57},
                {'type': 'cx', 'qubits': [0, 1]}
            ],
            'qubits': 2
        }
        
        transpiled = transpiler.transpile(circuit_spec)
        
        assert transpiled['native'] == True
        assert len(transpiled['gates']) > 0

    def test_transpile_with_gate_decomposition(self):
        """Test transpilation with non-native gate decomposition."""
        transpiler = QuantumNativeGateTranspiler(['rx', 'ry', 'rz'])
        circuit_spec = {
            'gates': [{'type': 'h', 'qubit': 0}],
            'qubits': 1
        }
        
        transpiled = transpiler.transpile(circuit_spec)
        
        assert len(transpiled['gates']) > 0
        assert all(g['type'] in ['rx', 'ry', 'rz'] for g in transpiled['gates'])

    def test_transpile_preserves_native_gates(self):
        """Test that native gates are preserved."""
        native_gates = ['cx', 'h']
        transpiler = QuantumNativeGateTranspiler(native_gates)
        circuit_spec = {
            'gates': [
                {'type': 'h', 'qubit': 0},
                {'type': 'cx', 'qubits': [0, 1]}
            ],
            'qubits': 2
        }
        
        transpiled = transpiler.transpile(circuit_spec)
        
        assert any(g['type'] == 'h' for g in transpiled['gates'])
        assert any(g['type'] == 'cx' for g in transpiled['gates'])

    def test_transpile_t_gate_decomposition(self):
        """Test T gate decomposition."""
        transpiler = QuantumNativeGateTranspiler(['rz'])
        circuit_spec = {
            'gates': [{'type': 't', 'qubit': 0}],
            'qubits': 1
        }
        
        transpiled = transpiler.transpile(circuit_spec)
        
        assert all(g['type'] == 'rz' for g in transpiled['gates'])
