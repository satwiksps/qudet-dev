"""
Workflow orchestration and execution management for quantum data pipelines.

Provides workflow definition, scheduling, execution, and error handling
for complex quantum-classical data processing pipelines.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
import json


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    """Workflow execution status."""
    DEFINED = "defined"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class Task:
    """Represents a single task in a workflow."""
    
    def __init__(self, task_id: str, name: str, operation: str,
                 params: Optional[Dict] = None, dependencies: Optional[List[str]] = None):
        """
        Initialize task.
        
        Args:
            task_id: Unique task identifier
            name: Human-readable task name
            operation: Operation to perform
            params: Task parameters
            dependencies: Task IDs this task depends on
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
    """
    Orchestrates execution of interdependent quantum tasks.
    
    Manages task scheduling, dependency resolution, error handling,
    and parallel execution where possible.
    
    Best for: Pipeline orchestration, complex workflows, job management.
    """
    
    def __init__(self, workflow_name: str, description: str = ""):
        """
        Initialize workflow.
        
        Args:
            workflow_name: Name of workflow
            description: Workflow description
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

    def add_task(self, name: str, operation: str,
                params: Optional[Dict] = None,
                dependencies: Optional[List[str]] = None) -> str:
        """
        Add task to workflow.
        
        Args:
            name: Task name
            operation: Operation to perform
            params: Task parameters
            dependencies: Task dependencies
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        task = Task(task_id, name, operation, params, dependencies)
        self.tasks[task_id] = task
        return task_id

    def get_executable_tasks(self) -> List[str]:
        """
        Get tasks that are ready to execute (all dependencies satisfied).
        
        Returns:
            List of executable task IDs
        """
        executable = []
        
        for task_id, task in self.tasks.items():
            # Task must be pending
            if task.status != TaskStatus.PENDING:
                continue
            
            # All dependencies must be completed
            all_deps_done = all(
                self.tasks[dep_id].status == TaskStatus.SUCCESS
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )
            
            if all_deps_done:
                executable.append(task_id)
        
        return executable

    def execute_task(self, task_id: str, executor: Callable) -> Tuple[bool, Any]:
        """
        Execute a single task.
        
        Args:
            task_id: Task to execute
            executor: Function to execute task
            
        Returns:
            Tuple of (success, result)
        """
        if task_id not in self.tasks:
            return False, None
        
        task = self.tasks[task_id]
        
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
            
            # Execute task
            result = executor(task.operation, task.params)
            
            task.status = TaskStatus.SUCCESS
            task.result = result
            task.completed_at = datetime.now().isoformat()
            
            return True, result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now().isoformat()
            return False, None

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a task."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status.value,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "error": task.error
        }

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get overall workflow status."""
        task_statuses = {
            status.value: sum(1 for t in self.tasks.values() if t.status == status)
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
            "task_statuses": task_statuses
        }

    def export_workflow(self, filename: str) -> None:
        """Export workflow definition to JSON."""
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
                    "dependencies": task.dependencies
                }
                for task in self.tasks.values()
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(workflow_def, f, indent=2)


class ResourceScheduler:
    """
    Schedules and manages resource allocation for quantum tasks.
    
    Optimizes resource usage, prevents overallocation, and
    prioritizes critical tasks.
    
    Best for: Resource optimization, scheduling, priority management.
    """
    
    def __init__(self, max_parallel_tasks: int = 4, max_memory_gb: float = 32.0):
        """
        Initialize resource scheduler.
        
        Args:
            max_parallel_tasks: Maximum parallel task execution
            max_memory_gb: Maximum available memory in GB
        """
        self.max_parallel_tasks = max_parallel_tasks
        self.max_memory_gb = max_memory_gb
        self.allocated_resources: Dict[str, Dict] = {}
        self.schedules: List[Dict] = []

    def allocate_resources(self, task_id: str, memory_gb: float,
                          num_cpus: int, priority: int = 0) -> Tuple[bool, str]:
        """
        Allocate resources to task.
        
        Args:
            task_id: Task identifier
            memory_gb: Memory needed in GB
            num_cpus: Number of CPUs needed
            priority: Task priority (0-100)
            
        Returns:
            Tuple of (success, message)
        """
        if task_id in self.allocated_resources:
            return False, "Task already has allocated resources"
        
        # Check if resources available
        total_allocated = sum(r.get("memory_gb", 0) for r in self.allocated_resources.values())
        
        if total_allocated + memory_gb > self.max_memory_gb:
            return False, f"Insufficient memory. Available: {self.max_memory_gb - total_allocated:.2f}GB"
        
        if len(self.allocated_resources) >= self.max_parallel_tasks:
            return False, f"Maximum parallel tasks ({self.max_parallel_tasks}) exceeded"
        
        self.allocated_resources[task_id] = {
            "memory_gb": memory_gb,
            "num_cpus": num_cpus,
            "priority": priority,
            "allocated_at": datetime.now().isoformat()
        }
        
        return True, f"Resources allocated: {memory_gb}GB, {num_cpus} CPUs"

    def deallocate_resources(self, task_id: str) -> bool:
        """
        Deallocate resources from task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Success status
        """
        if task_id in self.allocated_resources:
            del self.allocated_resources[task_id]
            return True
        
        return False

    def get_resource_availability(self) -> Dict[str, Any]:
        """Get current resource availability."""
        allocated_memory = sum(r.get("memory_gb", 0) 
                              for r in self.allocated_resources.values())
        available_memory = self.max_memory_gb - allocated_memory
        available_slots = self.max_parallel_tasks - len(self.allocated_resources)
        
        return {
            "total_memory_gb": self.max_memory_gb,
            "allocated_memory_gb": allocated_memory,
            "available_memory_gb": available_memory,
            "max_parallel_tasks": self.max_parallel_tasks,
            "active_tasks": len(self.allocated_resources),
            "available_slots": available_slots
        }

    def schedule_task(self, task_id: str, scheduled_time: str,
                     estimated_duration_seconds: int) -> Tuple[bool, str]:
        """
        Schedule task execution.
        
        Args:
            task_id: Task identifier
            scheduled_time: ISO format scheduled time
            estimated_duration_seconds: Estimated execution time
            
        Returns:
            Tuple of (success, message)
        """
        schedule_entry = {
            "task_id": task_id,
            "scheduled_time": scheduled_time,
            "estimated_duration_seconds": estimated_duration_seconds,
            "created_at": datetime.now().isoformat()
        }
        
        self.schedules.append(schedule_entry)
        
        return True, f"Task {task_id} scheduled for {scheduled_time}"

    def get_schedule(self) -> List[Dict]:
        """Get current schedule."""
        return sorted(self.schedules, key=lambda x: x["scheduled_time"])
