# File: tests/test_encoding/test_iqp.py

import numpy as np
import pytest
from qudet.encoders.iqp import IQPEncoder


class TestIQPEncoder:
    """Test suite for IQP (Instantaneous Quantum Polynomial) Encoding."""
    
    def test_iqp_basic_circuit_structure(self):
        """Verify IQP creates circuit with correct structure."""
        encoder = IQPEncoder(n_qubits=3, reps=1)
        data = np.array([0.5, 0.3, 0.8])
        circuit = encoder.encode(data)
        
        # Check basic properties
        assert circuit.num_qubits == 3
        assert circuit.num_clbits == 0  # No classical bits
        
    def test_iqp_multi_rep_depth(self):
        """Verify circuit depth increases with repetitions."""
        data = np.array([0.5, 0.3])
        
        encoder1 = IQPEncoder(n_qubits=2, reps=1)
        circuit1 = encoder1.encode(data)
        depth1 = circuit1.depth()
        
        encoder2 = IQPEncoder(n_qubits=2, reps=3)
        circuit2 = encoder2.encode(data)
        depth2 = circuit2.depth()
        
        # More reps → deeper circuit
        assert depth2 > depth1
        
    def test_iqp_truncates_oversized_data(self):
        """Verify IQP truncates data larger than n_qubits."""
        encoder = IQPEncoder(n_qubits=2, reps=1)
        # 5 features, but only 2 qubits
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        circuit = encoder.encode(data)
        
        # Should still work (uses only first 2 features)
        assert circuit.num_qubits == 2
        
    def test_iqp_with_single_feature(self):
        """Test edge case: single feature encoding."""
        encoder = IQPEncoder(n_qubits=1, reps=1)
        data = np.array([0.7])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 1
        # Single feature → no entangling gates (need 2+ features for interaction)
        
    def test_iqp_entanglement_gates_present(self):
        """Verify entangling (ZZ) gates are created."""
        encoder = IQPEncoder(n_qubits=3, reps=1)
        data = np.array([0.5, 0.3, 0.8])
        circuit = encoder.encode(data)
        
        # Check that ZZ gates (entanglement) are present
        ops = circuit.count_ops()
        assert 'rzz' in ops  # RZZ gates for entanglement
        
    def test_iqp_rotation_gates_present(self):
        """Verify single-qubit rotation gates (RZ) are created."""
        encoder = IQPEncoder(n_qubits=2, reps=1)
        data = np.array([0.5, 0.3])
        circuit = encoder.encode(data)
        
        ops = circuit.count_ops()
        # RZ for feature encoding + H for superposition
        assert 'rz' in ops
        assert 'h' in ops
        
    def test_iqp_different_qubit_counts(self):
        """Test IQP with various qubit counts."""
        for n_qubits in [1, 2, 5, 10]:
            encoder = IQPEncoder(n_qubits=n_qubits, reps=1)
            data = np.random.rand(n_qubits)
            circuit = encoder.encode(data)
            assert circuit.num_qubits == n_qubits
            
    def test_iqp_zero_data(self):
        """Test with all-zero input data."""
        encoder = IQPEncoder(n_qubits=2, reps=1)
        data = np.array([0.0, 0.0])
        circuit = encoder.encode(data)
        
        # Should still create valid circuit
        assert circuit.num_qubits == 2
        
    def test_iqp_large_values(self):
        """Test with larger angle values (beyond [0,1])."""
        encoder = IQPEncoder(n_qubits=2, reps=1)
        data = np.array([2.0, 3.14159])
        circuit = encoder.encode(data)
        
        # Qiskit handles angles in radians
        assert circuit.num_qubits == 2
