# File: tests/test_utils/test_sanity.py

import pytest
from qiskit import QuantumCircuit
from qudet.governance.cost import ResourceEstimator


class TestResourceEstimator:
    """Test resource estimation and cost calculation."""
    
    def test_cost_estimator_basic(self):
        """Test if price calculation returns valid numbers."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        report = ResourceEstimator.estimate_circuit_cost(qc, shots=100, hardware_rate=0.5)
        
        assert isinstance(report, dict)
        assert report['qubits_used'] == 2
        assert report['est_cost_usd'] > 0
        assert report['cnot_count'] == 1
        assert 'circuit_depth' in report
        assert 'est_runtime_sec' in report

    def test_cost_estimator_no_cnots(self):
        """Test circuit with no CNOT gates."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)
        qc.z(2)
        
        report = ResourceEstimator.estimate_circuit_cost(qc, shots=1024)
        
        assert report['cnot_count'] == 0

    def test_cost_estimator_many_cnots(self):
        """Test circuit with multiple CNOT gates."""
        qc = QuantumCircuit(4)
        for i in range(3):
            qc.cx(i, i+1)
        
        report = ResourceEstimator.estimate_circuit_cost(qc, shots=1024)
        
        assert report['cnot_count'] == 3

    def test_hardware_rate_scaling(self):
        """Test that hardware_rate linearly scales the cost."""
        qc = QuantumCircuit(2)
        qc.h(0)
        
        report_cheap = ResourceEstimator.estimate_circuit_cost(qc, shots=1024, hardware_rate=0.1)
        report_expensive = ResourceEstimator.estimate_circuit_cost(qc, shots=1024, hardware_rate=1.0)
        
        # More expensive hardware should have higher cost
        assert report_expensive['est_cost_usd'] > report_cheap['est_cost_usd']


class TestCapacityValidation:
    """Test quantum capacity guardrails."""

    def test_capacity_placeholder(self):
        """Placeholder for capacity validation tests."""
        assert True



class TestFeasibilityCheck:
    """Test feasibility analysis."""

    def test_feasibility_placeholder(self):
        """Placeholder for feasibility tests."""
        assert True

