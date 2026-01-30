# File: tests/test_optimization/test_simplify.py

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qudet.compute.simplify import CircuitOptimizer


class TestCircuitOptimizer:
    """Test suite for CircuitOptimizer."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = CircuitOptimizer(level=2)
        assert optimizer.level == 2
        
    def test_optimizer_invalid_level(self):
        """Test that invalid optimization level raises error."""
        with pytest.raises(ValueError):
            CircuitOptimizer(level=5)
            
    def test_optimize_simple_circuit(self):
        """Test optimization of a simple circuit."""
        optimizer = CircuitOptimizer(level=2)
        
        # Create a simple circuit with redundant gates
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)  # These cancel out
        qc.x(1)
        qc.x(1)  # These cancel out
        
        optimized = optimizer.optimize(qc)
        
        # Optimized circuit should have fewer operations
        assert len(optimized) <= len(qc)
        
    def test_optimize_returns_circuit(self):
        """Test that optimize returns a QuantumCircuit."""
        optimizer = CircuitOptimizer(level=1)
        qc = QuantumCircuit(1)
        qc.h(0)
        
        optimized = optimizer.optimize(qc)
        
        assert isinstance(optimized, QuantumCircuit)
        
    def test_optimize_batch(self):
        """Test batch optimization."""
        optimizer = CircuitOptimizer(level=2)
        
        circuits = []
        for i in range(3):
            qc = QuantumCircuit(1)
            qc.h(0)
            qc.h(0)
            circuits.append(qc)
        
        optimized_circuits = optimizer.optimize_batch(circuits)
        
        assert len(optimized_circuits) == 3
        assert all(isinstance(qc, QuantumCircuit) for qc in optimized_circuits)
        
    def test_optimize_preserves_qubit_count(self):
        """Test that optimization preserves qubit count."""
        optimizer = CircuitOptimizer(level=3)
        
        qc = QuantumCircuit(4)
        qc.h(range(4))
        qc.cx(0, 1)
        qc.cx(1, 2)
        
        optimized = optimizer.optimize(qc)
        
        # Qubit count should be preserved
        assert optimized.num_qubits == qc.num_qubits
        
    def test_estimate_savings(self):
        """Test circuit savings estimation."""
        optimizer = CircuitOptimizer(level=2)
        
        # Original circuit with redundancy
        qc_original = QuantumCircuit(2)
        qc_original.h(0)
        qc_original.h(0)
        qc_original.x(1)
        qc_original.x(1)
        
        # Optimize
        qc_optimized = optimizer.optimize(qc_original)
        
        savings = optimizer.estimate_savings(qc_original, qc_optimized)
        
        assert "original_depth" in savings
        assert "optimized_depth" in savings
        assert "depth_reduction_%" in savings
        assert savings["optimized_depth"] <= savings["original_depth"]
        
    def test_optimization_levels(self):
        """Test different optimization levels."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)
        qc.cx(0, 1)
        
        for level in range(4):
            optimizer = CircuitOptimizer(level=level)
            optimized = optimizer.optimize(qc)
            assert isinstance(optimized, QuantumCircuit)
