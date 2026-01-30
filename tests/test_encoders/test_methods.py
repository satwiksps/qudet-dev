# File: tests/test_encoding/test_methods.py

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qudet.encoders import RotationEncoder, StatevectorEncoder


class TestRotationEncoder:
    """Test Angle Encoding: linear rotation-based encoding."""
    
    def test_circuit_structure(self):
        """Test if 5 features generate a 5-qubit circuit with 5 parameters."""
        encoder = RotationEncoder(n_qubits=5)
        # Create dummy data (1 row, 5 cols)
        data = np.random.rand(5)
        circuit = encoder.encode(data)
        
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 5
        # Check if we have Rotation Y gates
        assert 'ry' in circuit.count_ops()

    def test_parameter_count(self):
        """Verify that 10 features create 10 parameters."""
        encoder = RotationEncoder(n_qubits=10)
        data = np.random.rand(10)
        circuit = encoder.encode(data)
        
        # Count ry gates (one per qubit)
        op_count = circuit.count_ops()
        assert op_count.get('ry', 0) == 10

    def test_different_qubits(self):
        """Test various qubit counts."""
        for n_qubits in [1, 3, 8, 16]:
            encoder = RotationEncoder(n_qubits=n_qubits)
            data = np.random.rand(n_qubits)
            circuit = encoder.encode(data)
            assert circuit.num_qubits == n_qubits


class TestStatevectorEncoder:
    """Test Amplitude Encoding: amplitude-based state initialization."""
    
    def test_normalization_logic(self):
        """
        Amplitude encoding requires the input vector to have L2 norm = 1.
        The encoder should handle un-normalized data automatically.
        """
        encoder = StatevectorEncoder()
        # Input: [1, 1] -> Norm is sqrt(2). Encoder should scale this.
        data = np.array([1.0, 1.0]) 
        
        circuit = encoder.encode(data)
        
        # 2 numbers need 1 qubit (log2(2) = 1)
        assert circuit.num_qubits == 1
        
        # Verify the state vector was initialized
        # (Instruction is 'initialize')
        assert 'initialize' in circuit.count_ops()

    def test_padding_logic(self):
        """
        If input is size 3, it must pad to size 4 (power of 2) before encoding.
        """
        encoder = StatevectorEncoder()
        data = np.array([0.5, 0.5, 0.5])  # Length 3
        
        circuit = encoder.encode(data)
        
        # ceil(log2(3)) = 2 qubits
        assert circuit.num_qubits == 2

    def test_small_input(self):
        """Test with minimal input (2 elements, smallest power of 2)."""
        encoder = StatevectorEncoder()
        data = np.array([0.6, 0.8])  # 2 elements
        
        circuit = encoder.encode(data)
        
        # 2 elements requires 1 qubit (log2(2) = 1)
        assert circuit.num_qubits == 1
        assert isinstance(circuit, QuantumCircuit)

    def test_large_input(self):
        """Test with larger input (256 elements = 8 qubits)."""
        encoder = StatevectorEncoder()
        data = np.random.rand(256)
        # Normalize
        data = data / np.linalg.norm(data)
        
        circuit = encoder.encode(data)
        
        # 256 = 2^8
        assert circuit.num_qubits == 8
