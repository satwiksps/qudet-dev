import pytest
import numpy as np
from qudet.compute.resource_management import (
    QuantumResourceAllocator,
    QuantumPriorityScheduler,
    QuantumCostEstimator
)


class TestQuantumResourceAllocator:
    """Test suite for Quantum Resource Allocator."""

    def test_initialization(self):
        """Test resource allocator initialization."""
        allocator = QuantumResourceAllocator(total_qubits=127, max_circuit_depth=1000)
        assert allocator.total_qubits == 127
        assert allocator.max_circuit_depth == 1000

    def test_allocate_qubits(self):
        """Test qubit allocation."""
        allocator = QuantumResourceAllocator(total_qubits=10)
        qubits = allocator.allocate_qubits("task1", n_qubits=5)
        
        assert len(qubits) == 5
        assert "task1" in allocator.allocated_qubits

    def test_allocate_insufficient_qubits_raises_error(self):
        """Test error when insufficient qubits available."""
        allocator = QuantumResourceAllocator(total_qubits=5)
        
        with pytest.raises(ValueError):
            allocator.allocate_qubits("task1", n_qubits=10)

    def test_deallocate_qubits(self):
        """Test qubit deallocation."""
        allocator = QuantumResourceAllocator(total_qubits=10)
        allocator.allocate_qubits("task1", n_qubits=5)
        allocator.deallocate_qubits("task1")
        
        assert "task1" not in allocator.allocated_qubits

    def test_update_resource_usage(self):
        """Test resource usage update."""
        allocator = QuantumResourceAllocator()
        allocator.allocate_qubits("task1", n_qubits=3)
        allocator.update_resource_usage("task1", depth=50, n_gates=100)
        
        assert allocator.resource_usage["task1"]["depth"] == 50
        assert allocator.resource_usage["task1"]["gates"] == 100

    def test_update_exceeding_max_depth_raises_error(self):
        """Test error when depth exceeds maximum."""
        allocator = QuantumResourceAllocator(max_circuit_depth=100)
        allocator.allocate_qubits("task1", n_qubits=3)
        
        with pytest.raises(ValueError):
            allocator.update_resource_usage("task1", depth=200, n_gates=50)

    def test_get_resource_summary(self):
        """Test resource summary retrieval."""
        allocator = QuantumResourceAllocator(total_qubits=20)
        allocator.allocate_qubits("task1", n_qubits=5)
        allocator.allocate_qubits("task2", n_qubits=7)
        allocator.update_resource_usage("task1", depth=10, n_gates=20)
        
        summary = allocator.get_resource_summary()
        
        assert summary['total_qubits'] == 20
        assert summary['allocated_qubits'] == 12
        assert summary['available_qubits'] == 8
        assert summary['total_gates'] == 20

    def test_multiple_task_allocation(self):
        """Test allocation of multiple tasks."""
        allocator = QuantumResourceAllocator(total_qubits=20)
        qubits1 = allocator.allocate_qubits("task1", n_qubits=5)
        qubits2 = allocator.allocate_qubits("task2", n_qubits=7)
        
        assert len(set(qubits1) & set(qubits2)) == 0


class TestQuantumPriorityScheduler:
    """Test suite for Quantum Priority Scheduler."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = QuantumPriorityScheduler(max_queue_size=50)
        assert scheduler.max_queue_size == 50

    def test_enqueue_task(self):
        """Test task enqueuing."""
        scheduler = QuantumPriorityScheduler()
        scheduler.enqueue_task("task1", priority=5)
        
        assert len(scheduler.task_queue) == 1

    def test_enqueue_full_queue_raises_error(self):
        """Test error when queue is full."""
        scheduler = QuantumPriorityScheduler(max_queue_size=2)
        scheduler.enqueue_task("task1", priority=5)
        scheduler.enqueue_task("task2", priority=5)
        
        with pytest.raises(ValueError):
            scheduler.enqueue_task("task3", priority=5)

    def test_dequeue_task(self):
        """Test task dequeuing."""
        scheduler = QuantumPriorityScheduler()
        scheduler.enqueue_task("task1", priority=5)
        
        task = scheduler.dequeue_task()
        
        assert task['id'] == "task1"
        assert task['status'] == 'executing'

    def test_priority_ordering(self):
        """Test tasks ordered by priority."""
        scheduler = QuantumPriorityScheduler()
        scheduler.enqueue_task("low", priority=2)
        scheduler.enqueue_task("high", priority=9)
        scheduler.enqueue_task("med", priority=5)
        
        task1 = scheduler.dequeue_task()
        task2 = scheduler.dequeue_task()
        task3 = scheduler.dequeue_task()
        
        assert task1['id'] == "high"
        assert task2['id'] == "med"
        assert task3['id'] == "low"

    def test_dequeue_empty_queue_raises_error(self):
        """Test error when dequeuing from empty queue."""
        scheduler = QuantumPriorityScheduler()
        
        with pytest.raises(ValueError):
            scheduler.dequeue_task()

    def test_get_queue_status(self):
        """Test queue status retrieval."""
        scheduler = QuantumPriorityScheduler(max_queue_size=10)
        scheduler.enqueue_task("task1", priority=5)
        scheduler.enqueue_task("task2", priority=3)
        
        status = scheduler.get_queue_status()
        
        assert status['queue_size'] == 2
        assert status['max_size'] == 10
        assert len(status['tasks']) == 2

    def test_execution_history(self):
        """Test execution history tracking."""
        scheduler = QuantumPriorityScheduler()
        scheduler.enqueue_task("task1", priority=5)
        scheduler.dequeue_task()
        
        assert len(scheduler.execution_history) == 1


class TestQuantumCostEstimator:
    """Test suite for Quantum Cost Estimator."""

    def test_initialization(self):
        """Test cost estimator initialization."""
        estimator = QuantumCostEstimator(cost_per_gate=0.01, cost_per_qubit=0.001)
        assert estimator.cost_per_gate == 0.01
        assert estimator.cost_per_qubit == 0.001

    def test_estimate_circuit_cost(self):
        """Test circuit cost estimation."""
        estimator = QuantumCostEstimator()
        circuit = {
            'gates': [{'type': 'h'}, {'type': 'cx'}, {'type': 'rz'}],
            'qubits': 2,
            'depth': 3
        }
        
        cost = estimator.estimate_circuit_cost(circuit)
        
        assert 'n_gates' in cost
        assert 'n_qubits' in cost
        assert 'total_cost' in cost
        assert cost['n_gates'] == 3
        assert cost['n_qubits'] == 2

    def test_cost_breakdown(self):
        """Test cost breakdown components."""
        estimator = QuantumCostEstimator(cost_per_gate=0.1, cost_per_qubit=0.01)
        circuit = {
            'gates': [{'type': 'h'}, {'type': 'x'}],
            'qubits': 1,
            'depth': 2
        }
        
        cost = estimator.estimate_circuit_cost(circuit)
        
        assert cost['gate_cost'] == 0.2
        assert cost['qubit_cost'] == 0.01

    def test_store_cost(self):
        """Test storing cost estimates."""
        estimator = QuantumCostEstimator()
        circuit = {
            'gates': [{'type': 'h'}],
            'qubits': 1
        }
        cost_data = estimator.estimate_circuit_cost(circuit)
        
        estimator.store_cost("task1", cost_data)
        
        assert "task1" in estimator.cost_history

    def test_get_total_cost(self):
        """Test total cost retrieval."""
        estimator = QuantumCostEstimator(cost_per_gate=0.01, cost_per_qubit=0.001)
        circuit1 = {
            'gates': [{'type': 'h'}, {'type': 'x'}],
            'qubits': 2
        }
        circuit2 = {
            'gates': [{'type': 'z'}],
            'qubits': 1
        }
        
        estimator.store_cost("task1", estimator.estimate_circuit_cost(circuit1))
        estimator.store_cost("task2", estimator.estimate_circuit_cost(circuit2))
        
        total = estimator.get_total_cost()
        
        assert total > 0

    def test_get_cost_breakdown(self):
        """Test cost breakdown across tasks."""
        estimator = QuantumCostEstimator(cost_per_gate=0.1, cost_per_qubit=0.01)
        circuit = {
            'gates': [{'type': 'h'}],
            'qubits': 1
        }
        
        estimator.store_cost("task1", estimator.estimate_circuit_cost(circuit))
        estimator.store_cost("task2", estimator.estimate_circuit_cost(circuit))
        
        breakdown = estimator.get_cost_breakdown()
        
        assert breakdown['num_tasks'] == 2
        assert breakdown['total_cost'] > 0

    def test_cost_scales_with_gates(self):
        """Test that cost increases with gate count."""
        estimator = QuantumCostEstimator(cost_per_gate=1.0)
        circuit_small = {'gates': [{'type': 'h'}], 'qubits': 1}
        circuit_large = {'gates': [{'type': 'h'}] * 10, 'qubits': 1}
        
        cost_small = estimator.estimate_circuit_cost(circuit_small)
        cost_large = estimator.estimate_circuit_cost(circuit_large)
        
        assert cost_large['total_cost'] > cost_small['total_cost']
