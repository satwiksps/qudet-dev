"""
Workflow orchestration and execution management for quantum data pipelines.

Provides workflow definition, task scheduling, dependency resolution,
execution, and error handling for complex quantum-classical data
processing pipelines.
"""

import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Execution status of an individual task.

    Values:
        PENDING: Task is waiting to be executed.
        RUNNING: Task is currently executing.
        SUCCESS: Task completed successfully.
        FAILED: Task encountered an error.
        SKIPPED: Task was skipped (e.g. dependency failure).
        CANCELLED: Task was explicitly cancelled.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    """Execution status of a workflow.

    Values:
        DEFINED: Workflow has been created but not scheduled.
        SCHEDULED: Workflow is scheduled for future execution.
        RUNNING: Workflow is currently executing.
        COMPLETED: All tasks finished successfully.
        FAILED: One or more tasks failed.
        PAUSED: Workflow execution is paused.
    """

    DEFINED = "defined"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class Task:
    """Represents a single task within a workflow.

    A task encapsulates an operation, its parameters, and any
    dependency constraints on other tasks.

    Attributes:
        task_id: Unique identifier.
        name: Human-readable task name.
        operation: Operation to perform (string key).
        params: Parameter dictionary passed to the executor.
        dependencies: Task IDs that must complete before this task.
        status: Current execution status.
        created_at: ISO-8601 creation timestamp.
        started_at: ISO-8601 start timestamp (or ``None``).
        completed_at: ISO-8601 completion timestamp (or ``None``).
        result: Return value of the executor (or ``None``).
        error: Error message if the task failed (or ``None``).
    """

    def __init__(
        self,
        task_id: str,
        name: str,
        operation: str,
        params: Optional[Dict] = None,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Initialize a task.

        Args:
            task_id: Unique task identifier.
            name: Human-readable task name.
            operation: Operation key describing what to execute.
            params: Parameters passed to the executor function.
            dependencies: List of task IDs this task depends on.
        """
        self.task_id = task_id
        self.name = name
        self.operation = operation
        self.params = params or {}
        self.dependencies = dependencies or []
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.result: Optional[Any] = None
        self.error: Optional[str] = None


class Workflow:
    """Orchestrates execution of interdependent quantum tasks.

    Manages task scheduling, dependency resolution, error handling,
    and parallel-execution eligibility.

    Attributes:
        workflow_id: Unique identifier (UUID).
        workflow_name: Human-readable workflow name.
        description: Workflow description.
        tasks: Mapping of task_id → ``Task``.
        status: Current workflow status.
        created_at: ISO-8601 creation timestamp.
        started_at: ISO-8601 start timestamp (or ``None``).
        completed_at: ISO-8601 completion timestamp (or ``None``).
        execution_history: Chronological execution log entries.

    Examples:
        >>> wf = Workflow("my_pipeline", description="Weekly ETL")
        >>> t1 = wf.add_task("reduce", "coreset_reduce", params={"k": 100})
        >>> t2 = wf.add_task("encode", "iqp_encode", dependencies=[t1])
        >>> ready = wf.get_executable_tasks()
    """

    def __init__(
        self,
        workflow_name: str,
        description: str = "",
    ) -> None:
        """Initialize a workflow.

        Args:
            workflow_name: Name of the workflow.
            description: Optional description.
        """
        self.workflow_id = str(uuid.uuid4())
        self.workflow_name = workflow_name
        self.description = description
        self.tasks: Dict[str, Task] = {}
        self.status = WorkflowStatus.DEFINED
        self.created_at = datetime.now().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.execution_history: List[Dict] = []

    def add_task(
        self,
        name: str,
        operation: str,
        params: Optional[Dict] = None,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """Add a task to the workflow.

        Args:
            name: Human-readable task name.
            operation: Operation key (passed to the executor).
            params: Task parameters.
            dependencies: List of task IDs this task depends on.

        Returns:
            The new task's unique identifier.

        Raises:
            ValueError: If any dependency ID is not in the workflow.
        """
        if dependencies:
            unknown = [d for d in dependencies if d not in self.tasks]
            if unknown:
                raise ValueError(
                    f"Unknown dependency task IDs: {unknown}"
                )

        task_id = str(uuid.uuid4())
        task = Task(task_id, name, operation, params, dependencies)
        self.tasks[task_id] = task
        logger.debug("Task '%s' (%s) added to workflow '%s'.", name, task_id, self.workflow_name)
        return task_id

    def get_executable_tasks(self) -> List[str]:
        """Get task IDs that are ready to execute.

        A task is executable when it is ``PENDING`` and all of its
        dependencies have status ``SUCCESS``.

        Returns:
            List of executable task IDs.
        """
        executable: List[str] = []

        for task_id, task in self.tasks.items():
            if task.status != TaskStatus.PENDING:
                continue

            all_deps_done = all(
                self.tasks[dep_id].status == TaskStatus.SUCCESS
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )

            if all_deps_done:
                executable.append(task_id)

        return executable

    def execute_task(
        self, task_id: str, executor: Callable
    ) -> Tuple[bool, Any]:
        """Execute a single task using the provided executor callable.

        The executor is called as ``executor(operation, params)`` and
        should return the task result.

        Args:
            task_id: Identifier of the task to execute.
            executor: Callable that performs the actual work.

        Returns:
            Tuple of ``(success, result)``.  *result* is ``None``
            on failure or if the task is unknown.
        """
        if task_id not in self.tasks:
            return False, None

        task = self.tasks[task_id]

        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()

            result = executor(task.operation, task.params)

            task.status = TaskStatus.SUCCESS
            task.result = result
            task.completed_at = datetime.now().isoformat()

            logger.info("Task '%s' completed successfully.", task.name)
            return True, result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now().isoformat()
            logger.error("Task '%s' failed: %s", task.name, e)
            return False, None

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get the status of a task.

        Args:
            task_id: Task identifier.

        Returns:
            Status dictionary, or ``None`` if the task is unknown.
        """
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status.value,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "error": task.error,
        }

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get the overall workflow status.

        Returns:
            Dictionary with workflow metadata and per-status task counts.
        """
        task_statuses = {
            status.value: sum(
                1 for t in self.tasks.values() if t.status == status
            )
            for status in TaskStatus
        }

        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_tasks": len(self.tasks),
            "task_statuses": task_statuses,
        }

    def export_workflow(self, filename: str) -> None:
        """Export the workflow definition to a JSON file.

        Args:
            filename: Path to the output JSON file.
        """
        workflow_def = {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "description": self.description,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "name": task.name,
                    "operation": task.operation,
                    "params": task.params,
                    "dependencies": task.dependencies,
                }
                for task in self.tasks.values()
            ],
        }

        with open(filename, "w") as f:
            json.dump(workflow_def, f, indent=2)

        logger.info("Workflow exported to %s.", filename)


class ResourceScheduler:
    """Schedules and manages resource allocation for quantum tasks.

    Prevents over-allocation of memory and parallel-task slots,
    and supports priority-based scheduling.

    Attributes:
        max_parallel_tasks: Maximum concurrent tasks allowed.
        max_memory_gb: Maximum available memory in GB.
        allocated_resources: Currently allocated resources per task.
        schedules: List of scheduled execution entries.
    """

    def __init__(
        self,
        max_parallel_tasks: int = 4,
        max_memory_gb: float = 32.0,
    ) -> None:
        """Initialize resource scheduler.

        Args:
            max_parallel_tasks: Maximum number of parallel tasks.
            max_memory_gb: Maximum available memory in GB.

        Raises:
            ValueError: If parameters are not positive.
        """
        if max_parallel_tasks <= 0:
            raise ValueError(
                f"max_parallel_tasks must be positive, got {max_parallel_tasks}"
            )
        if max_memory_gb <= 0:
            raise ValueError(
                f"max_memory_gb must be positive, got {max_memory_gb}"
            )

        self.max_parallel_tasks = max_parallel_tasks
        self.max_memory_gb = max_memory_gb
        self.allocated_resources: Dict[str, Dict] = {}
        self.schedules: List[Dict] = []

    def allocate_resources(
        self,
        task_id: str,
        memory_gb: float,
        num_cpus: int,
        priority: int = 0,
    ) -> Tuple[bool, str]:
        """Allocate resources to a task.

        Args:
            task_id: Task identifier.
            memory_gb: Memory needed in GB.
            num_cpus: Number of CPUs needed.
            priority: Task priority (0–100, higher = more important).

        Returns:
            Tuple of ``(success, message)``.
        """
        if task_id in self.allocated_resources:
            return False, "Task already has allocated resources"

        total_allocated = sum(
            r.get("memory_gb", 0) for r in self.allocated_resources.values()
        )

        if total_allocated + memory_gb > self.max_memory_gb:
            available = self.max_memory_gb - total_allocated
            return False, f"Insufficient memory. Available: {available:.2f}GB"

        if len(self.allocated_resources) >= self.max_parallel_tasks:
            return (
                False,
                f"Maximum parallel tasks ({self.max_parallel_tasks}) exceeded",
            )

        self.allocated_resources[task_id] = {
            "memory_gb": memory_gb,
            "num_cpus": num_cpus,
            "priority": priority,
            "allocated_at": datetime.now().isoformat(),
        }

        return True, f"Resources allocated: {memory_gb}GB, {num_cpus} CPUs"

    def deallocate_resources(self, task_id: str) -> bool:
        """Release resources allocated to a task.

        Args:
            task_id: Task identifier.

        Returns:
            ``True`` if resources were deallocated, ``False`` if the
            task had no allocation.
        """
        if task_id in self.allocated_resources:
            del self.allocated_resources[task_id]
            return True
        return False

    def get_resource_availability(self) -> Dict[str, Any]:
        """Get current resource availability.

        Returns:
            Dictionary with total, allocated, and available memory
            and task-slot information.
        """
        allocated_memory = sum(
            r.get("memory_gb", 0) for r in self.allocated_resources.values()
        )
        available_memory = self.max_memory_gb - allocated_memory
        available_slots = self.max_parallel_tasks - len(self.allocated_resources)

        return {
            "total_memory_gb": self.max_memory_gb,
            "allocated_memory_gb": allocated_memory,
            "available_memory_gb": available_memory,
            "max_parallel_tasks": self.max_parallel_tasks,
            "active_tasks": len(self.allocated_resources),
            "available_slots": available_slots,
        }

    def schedule_task(
        self,
        task_id: str,
        scheduled_time: str,
        estimated_duration_seconds: int,
    ) -> Tuple[bool, str]:
        """Schedule a task for future execution.

        Args:
            task_id: Task identifier.
            scheduled_time: ISO-8601 formatted scheduled time.
            estimated_duration_seconds: Estimated execution duration.

        Returns:
            Tuple of ``(success, message)``.
        """
        schedule_entry = {
            "task_id": task_id,
            "scheduled_time": scheduled_time,
            "estimated_duration_seconds": estimated_duration_seconds,
            "created_at": datetime.now().isoformat(),
        }

        self.schedules.append(schedule_entry)
        return True, f"Task {task_id} scheduled for {scheduled_time}"

    def get_schedule(self) -> List[Dict]:
        """Get the current schedule, sorted by scheduled time.

        Returns:
            List of schedule entries in chronological order.
        """
        return sorted(self.schedules, key=lambda x: x["scheduled_time"])
