"""
Quantum resource allocation, scheduling, and cost estimation.

Provides tools for managing qubit allocation across concurrent tasks,
priority-based job scheduling, and estimating execution costs.
"""

import numpy as np
from typing import Dict, List, Optional


class QuantumResourceAllocator:
    """Allocate and manage quantum computing resources.

    Tracks which qubits are assigned to which tasks and enforces capacity
    limits.

    Example:
        >>> allocator = QuantumResourceAllocator(total_qubits=127)
        >>> qubits = allocator.allocate_qubits("task_1", n_qubits=5)
        >>> allocator.deallocate_qubits("task_1")
    """

    def __init__(self, total_qubits: int = 127, max_circuit_depth: int = 1000) -> None:
        """Initialize the resource allocator.

        Args:
            total_qubits: Total number of qubits available on the backend.
            max_circuit_depth: Maximum allowed circuit depth.
        """
        self.total_qubits = total_qubits
        self.max_circuit_depth = max_circuit_depth
        self.allocated_qubits: Dict[str, List[int]] = {}
        self.resource_usage: Dict[str, Dict] = {}

    def allocate_qubits(self, task_id: str, n_qubits: int) -> List[int]:
        """Allocate qubits for a task.

        Args:
            task_id: Unique task identifier.
            n_qubits: Number of qubits to allocate.

        Returns:
            List of allocated qubit indices.

        Raises:
            ValueError: If *task_id* is already allocated, or if there are
                insufficient free qubits.
        """
        if task_id in self.allocated_qubits:
            raise ValueError(
                f"Task '{task_id}' already has allocated qubits. "
                "Deallocate first or use a different task_id."
            )

        available_qubits = self.total_qubits - sum(
            len(q) for q in self.allocated_qubits.values()
        )

        if n_qubits > available_qubits:
            raise ValueError(
                f"Insufficient qubits: need {n_qubits}, available {available_qubits}"
            )

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

    def deallocate_qubits(self, task_id: str) -> None:
        """Release qubits allocated to a task.

        Args:
            task_id: Task identifier to deallocate.
        """
        if task_id in self.allocated_qubits:
            del self.allocated_qubits[task_id]
        if task_id in self.resource_usage:
            del self.resource_usage[task_id]

    def update_resource_usage(self, task_id: str, depth: int, n_gates: int) -> None:
        """Update resource usage metrics for a task.

        Args:
            task_id: Task identifier.
            depth: Current circuit depth.
            n_gates: Number of gates used.

        Raises:
            ValueError: If the task is not allocated or circuit depth exceeds
                the maximum.
        """
        if task_id not in self.resource_usage:
            raise ValueError(f"Task '{task_id}' not allocated")

        if depth > self.max_circuit_depth:
            raise ValueError(
                f"Circuit depth {depth} exceeds limit {self.max_circuit_depth}"
            )

        self.resource_usage[task_id]['depth'] = depth
        self.resource_usage[task_id]['gates'] = n_gates

    def get_resource_summary(self) -> Dict:
        """Get a summary of all resource allocations.

        Returns:
            Dictionary with total, allocated, and available qubit counts,
            active task count, and total gate count.
        """
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
    """Schedule quantum tasks based on priority and resource availability.

    Maintains a priority queue of tasks sorted by descending priority (higher
    values execute first).

    Example:
        >>> scheduler = QuantumPriorityScheduler(max_queue_size=50)
        >>> scheduler.enqueue_task("job_1", priority=8)
        >>> task = scheduler.dequeue_task()
    """

    def __init__(self, max_queue_size: int = 100) -> None:
        """Initialize the priority scheduler.

        Args:
            max_queue_size: Maximum number of tasks allowed in the queue.
        """
        self.max_queue_size = max_queue_size
        self.task_queue: List[Dict] = []
        self.execution_history: List[Dict] = []

    def enqueue_task(
        self, task_id: str, priority: int = 5, resources: Optional[Dict] = None
    ) -> None:
        """Add a quantum task to the priority queue.

        Args:
            task_id: Unique task identifier.
            priority: Priority level (1–10; higher is more urgent).
            resources: Optional resource requirements dictionary.

        Raises:
            ValueError: If the queue is full.
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
        """Dequeue the highest-priority task.

        Returns:
            The next task dictionary to execute.

        Raises:
            ValueError: If the queue is empty.
        """
        if not self.task_queue:
            raise ValueError("Queue is empty")

        task = self.task_queue.pop(0)
        task['status'] = 'executing'
        self.execution_history.append(task)
        return task

    def get_queue_status(self) -> Dict:
        """Get the current queue status.

        Returns:
            Dictionary with queue size, capacity, task list, and execution
            history count.
        """
        return {
            'queue_size': len(self.task_queue),
            'max_size': self.max_queue_size,
            'tasks': [{'id': t['id'], 'priority': t['priority']} for t in self.task_queue],
            'total_executed': len(self.execution_history)
        }


class QuantumCostEstimator:
    """Estimate computational costs for quantum circuits.

    Uses configurable per-gate and per-qubit cost rates to produce cost
    breakdowns and track cumulative spending across tasks.

    Example:
        >>> estimator = QuantumCostEstimator(cost_per_gate=0.01)
        >>> cost = estimator.estimate_circuit_cost(circuit_spec)
    """

    def __init__(self, cost_per_gate: float = 0.01, cost_per_qubit: float = 0.001) -> None:
        """Initialize the cost estimator.

        Args:
            cost_per_gate: Cost charged per quantum gate.
            cost_per_qubit: Cost charged per qubit used.
        """
        self.cost_per_gate = cost_per_gate
        self.cost_per_qubit = cost_per_qubit
        self.cost_history: Dict[str, Dict] = {}

    def estimate_circuit_cost(self, circuit_spec: Dict) -> Dict:
        """Estimate the cost for a single circuit.

        Args:
            circuit_spec: Circuit specification dictionary with ``'gates'``,
                ``'qubits'``, and optionally ``'depth'`` keys.

        Returns:
            Cost breakdown dictionary.
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

    def store_cost(self, task_id: str, cost_data: Dict) -> None:
        """Store a cost estimate for a task.

        Args:
            task_id: Task identifier.
            cost_data: Cost data dictionary (typically from
                ``estimate_circuit_cost``).
        """
        self.cost_history[task_id] = cost_data

    def get_total_cost(self) -> float:
        """Get the total cost across all stored estimates.

        Returns:
            Cumulative total cost.
        """
        return sum(c.get('total_cost', 0) for c in self.cost_history.values())

    def get_cost_breakdown(self) -> Dict:
        """Get an aggregate cost breakdown across all tasks.

        Returns:
            Dictionary with total gate cost, qubit cost, combined total,
            and number of tasks.
        """
        all_gate_costs = sum(c.get('gate_cost', 0) for c in self.cost_history.values())
        all_qubit_costs = sum(c.get('qubit_cost', 0) for c in self.cost_history.values())

        return {
            'total_gate_cost': all_gate_costs,
            'total_qubit_cost': all_qubit_costs,
            'total_cost': all_gate_costs + all_qubit_costs,
            'num_tasks': len(self.cost_history)
        }
