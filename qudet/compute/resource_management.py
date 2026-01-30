import numpy as np
from typing import Optional, Dict, List


class QuantumResourceAllocator:
    """Allocate and manage quantum computing resources."""

    def __init__(self, total_qubits: int = 127, max_circuit_depth: int = 1000):
        """
        Args:
            total_qubits: Total available qubits on backend
            max_circuit_depth: Maximum circuit depth allowed
        """
        self.total_qubits = total_qubits
        self.max_circuit_depth = max_circuit_depth
        self.allocated_qubits = {}
        self.resource_usage = {}

    def allocate_qubits(self, task_id: str, n_qubits: int) -> List[int]:
        """
        Allocate qubits for a task.
        
        Args:
            task_id: Task identifier
            n_qubits: Number of qubits needed
            
        Returns:
            List of allocated qubit indices
        """
        available_qubits = self.total_qubits - sum(len(q) for q in self.allocated_qubits.values())
        
        if n_qubits > available_qubits:
            raise ValueError(f"Insufficient qubits: need {n_qubits}, available {available_qubits}")
        
        all_allocated = [q for qubits in self.allocated_qubits.values() for q in qubits]
        available = [i for i in range(self.total_qubits) if i not in all_allocated]
        
        allocated = available[:n_qubits]
        self.allocated_qubits[task_id] = allocated
        
        self.resource_usage[task_id] = {
            'qubits': len(allocated),
            'depth': 0,
            'gates': 0
        }
        
        return allocated

    def deallocate_qubits(self, task_id: str):
        """
        Release qubits allocated to a task.
        
        Args:
            task_id: Task identifier
        """
        if task_id in self.allocated_qubits:
            del self.allocated_qubits[task_id]
        if task_id in self.resource_usage:
            del self.resource_usage[task_id]

    def update_resource_usage(self, task_id: str, depth: int, n_gates: int):
        """
        Update resource usage for a task.
        
        Args:
            task_id: Task identifier
            depth: Circuit depth
            n_gates: Number of gates
        """
        if task_id not in self.resource_usage:
            raise ValueError(f"Task {task_id} not allocated")
        
        if depth > self.max_circuit_depth:
            raise ValueError(f"Circuit depth {depth} exceeds limit {self.max_circuit_depth}")
        
        self.resource_usage[task_id]['depth'] = depth
        self.resource_usage[task_id]['gates'] = n_gates

    def get_resource_summary(self) -> Dict:
        """Get summary of all resource allocations."""
        total_allocated = sum(len(q) for q in self.allocated_qubits.values())
        total_gates = sum(r['gates'] for r in self.resource_usage.values())
        
        return {
            'total_qubits': self.total_qubits,
            'allocated_qubits': total_allocated,
            'available_qubits': self.total_qubits - total_allocated,
            'active_tasks': len(self.allocated_qubits),
            'total_gates': total_gates
        }


class QuantumPriorityScheduler:
    """Schedule quantum tasks based on priority and resource availability."""

    def __init__(self, max_queue_size: int = 100):
        """
        Args:
            max_queue_size: Maximum queue size
        """
        self.max_queue_size = max_queue_size
        self.task_queue = []
        self.execution_history = []

    def enqueue_task(self, task_id: str, priority: int = 5, resources: Dict = None):
        """
        Enqueue a quantum task.
        
        Args:
            task_id: Task identifier
            priority: Priority level (1-10, higher is urgent)
            resources: Resource requirements
        """
        if len(self.task_queue) >= self.max_queue_size:
            raise ValueError(f"Queue full: max size {self.max_queue_size}")
        
        task = {
            'id': task_id,
            'priority': priority,
            'resources': resources or {},
            'status': 'queued'
        }
        
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda x: -x['priority'])

    def dequeue_task(self) -> Dict:
        """
        Dequeue next task from queue.
        
        Returns:
            Next task to execute
        """
        if not self.task_queue:
            raise ValueError("Queue is empty")
        
        task = self.task_queue.pop(0)
        task['status'] = 'executing'
        self.execution_history.append(task)
        return task

    def get_queue_status(self) -> Dict:
        """Get current queue status."""
        return {
            'queue_size': len(self.task_queue),
            'max_size': self.max_queue_size,
            'tasks': [{'id': t['id'], 'priority': t['priority']} for t in self.task_queue],
            'total_executed': len(self.execution_history)
        }


class QuantumCostEstimator:
    """Estimate computational costs for quantum circuits."""

    def __init__(self, cost_per_gate: float = 0.01, cost_per_qubit: float = 0.001):
        """
        Args:
            cost_per_gate: Cost per quantum gate
            cost_per_qubit: Cost per qubit used
        """
        self.cost_per_gate = cost_per_gate
        self.cost_per_qubit = cost_per_qubit
        self.cost_history = {}

    def estimate_circuit_cost(self, circuit_spec: Dict) -> Dict:
        """
        Estimate cost for a circuit.
        
        Args:
            circuit_spec: Circuit specification
            
        Returns:
            Cost breakdown
        """
        n_gates = len(circuit_spec.get('gates', []))
        n_qubits = circuit_spec.get('qubits', 2)
        depth = circuit_spec.get('depth', 0)
        
        gate_cost = n_gates * self.cost_per_gate
        qubit_cost = n_qubits * self.cost_per_qubit
        total_cost = gate_cost + qubit_cost
        
        return {
            'n_gates': n_gates,
            'n_qubits': n_qubits,
            'depth': depth,
            'gate_cost': gate_cost,
            'qubit_cost': qubit_cost,
            'total_cost': total_cost
        }

    def store_cost(self, task_id: str, cost_data: Dict):
        """Store cost estimate for a task."""
        self.cost_history[task_id] = cost_data

    def get_total_cost(self) -> float:
        """Get total cost from all stored estimates."""
        return sum(c.get('total_cost', 0) for c in self.cost_history.values())

    def get_cost_breakdown(self) -> Dict:
        """Get breakdown of costs."""
        all_gate_costs = sum(c.get('gate_cost', 0) for c in self.cost_history.values())
        all_qubit_costs = sum(c.get('qubit_cost', 0) for c in self.cost_history.values())
        
        return {
            'total_gate_cost': all_gate_costs,
            'total_qubit_cost': all_qubit_costs,
            'total_cost': all_gate_costs + all_qubit_costs,
            'num_tasks': len(self.cost_history)
        }
