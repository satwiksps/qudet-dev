"""
Comprehensive tests for workflow orchestration and resource scheduling.

Tests workflow management, task execution, and resource allocation.
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from qudet.governance.orchestration import (
    Workflow,
    ResourceScheduler,
    Task,
    TaskStatus,
    WorkflowStatus
)


class TestTask:
    """Test task creation and management."""

    def test_task_initialization(self):
        """Test task initialization."""
        task = Task("task1", "process_data", "quantum_algorithm", 
                   params={"n_qubits": 5}, dependencies=[])
        
        assert task.task_id == "task1"
        assert task.name == "process_data"
        assert task.status == TaskStatus.PENDING

    def test_task_with_dependencies(self):
        """Test task with dependencies."""
        task = Task("task2", "aggregate", "aggregation", 
                   dependencies=["task1"])
        
        assert "task1" in task.dependencies

    def test_task_attributes(self):
        """Test task attributes."""
        task = Task("task1", "test", "operation")
        
        assert task.created_at is not None
        assert task.started_at is None
        assert task.completed_at is None
        assert task.result is None
        assert task.error is None


class TestWorkflow:
    """Test workflow management."""

    def test_workflow_initialization(self):
        """Test workflow initialization."""
        workflow = Workflow("test_workflow", "Test workflow")
        
        assert workflow.workflow_name == "test_workflow"
        assert workflow.status == WorkflowStatus.DEFINED
        assert len(workflow.tasks) == 0

    def test_add_single_task(self):
        """Test adding single task."""
        workflow = Workflow("test")
        task_id = workflow.add_task("process", "algorithm", {"param": "value"})
        
        assert task_id in workflow.tasks
        assert workflow.tasks[task_id].name == "process"

    def test_add_multiple_tasks(self):
        """Test adding multiple tasks."""
        workflow = Workflow("test")
        
        ids = []
        for i in range(5):
            task_id = workflow.add_task(f"task{i}", "operation", {})
            ids.append(task_id)
        
        assert len(workflow.tasks) == 5
        assert all(tid in workflow.tasks for tid in ids)

    def test_task_dependencies(self):
        """Test task dependencies."""
        workflow = Workflow("test")
        
        task1_id = workflow.add_task("task1", "op1")
        task2_id = workflow.add_task("task2", "op2", dependencies=[task1_id])
        task3_id = workflow.add_task("task3", "op3", dependencies=[task1_id, task2_id])
        
        assert task1_id in workflow.tasks[task2_id].dependencies
        assert task2_id in workflow.tasks[task3_id].dependencies

    def test_get_executable_tasks_no_deps(self):
        """Test getting executable tasks without dependencies."""
        workflow = Workflow("test")
        
        id1 = workflow.add_task("task1", "op1")
        id2 = workflow.add_task("task2", "op2")
        
        executable = workflow.get_executable_tasks()
        
        assert len(executable) == 2
        assert id1 in executable
        assert id2 in executable

    def test_get_executable_tasks_with_deps(self):
        """Test getting executable tasks with dependencies."""
        workflow = Workflow("test")
        
        id1 = workflow.add_task("task1", "op1")
        id2 = workflow.add_task("task2", "op2", dependencies=[id1])
        
        executable = workflow.get_executable_tasks()
        
        # Only task1 should be executable
        assert id1 in executable
        assert id2 not in executable

    def test_execute_task_success(self):
        """Test executing task successfully."""
        workflow = Workflow("test")
        task_id = workflow.add_task("task1", "operation")
        
        def mock_executor(operation, params):
            return {"status": "success"}
        
        success, result = workflow.execute_task(task_id, mock_executor)
        
        assert success is True
        assert result["status"] == "success"
        assert workflow.tasks[task_id].status == TaskStatus.SUCCESS

    def test_execute_task_failure(self):
        """Test executing task with failure."""
        workflow = Workflow("test")
        task_id = workflow.add_task("task1", "operation")
        
        def mock_executor(operation, params):
            raise ValueError("Operation failed")
        
        success, result = workflow.execute_task(task_id, mock_executor)
        
        assert success is False
        assert workflow.tasks[task_id].status == TaskStatus.FAILED
        assert workflow.tasks[task_id].error is not None

    def test_get_task_status(self):
        """Test getting task status."""
        workflow = Workflow("test")
        task_id = workflow.add_task("task1", "operation")
        
        status = workflow.get_task_status(task_id)
        
        assert status["name"] == "task1"
        assert status["status"] == TaskStatus.PENDING.value

    def test_get_workflow_status(self):
        """Test getting workflow status."""
        workflow = Workflow("test")
        
        for i in range(3):
            workflow.add_task(f"task{i}", "operation")
        
        status = workflow.get_workflow_status()
        
        assert status["workflow_name"] == "test"
        assert status["total_tasks"] == 3
        assert status["status"] == WorkflowStatus.DEFINED.value

    def test_export_workflow(self):
        """Test exporting workflow."""
        workflow = Workflow("test_workflow", "Test")
        
        id1 = workflow.add_task("task1", "operation1", {"param": "value"})
        id2 = workflow.add_task("task2", "operation2", dependencies=[id1])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filename = f.name
        
        workflow.export_workflow(filename)
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        assert data["workflow_name"] == "test_workflow"
        assert len(data["tasks"]) == 2


class TestResourceScheduler:
    """Test resource scheduling."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = ResourceScheduler(max_parallel_tasks=4, max_memory_gb=64.0)
        
        assert scheduler.max_parallel_tasks == 4
        assert scheduler.max_memory_gb == 64.0

    def test_allocate_resources(self):
        """Test resource allocation."""
        scheduler = ResourceScheduler()
        
        success, message = scheduler.allocate_resources("task1", memory_gb=8, num_cpus=4)
        
        assert success is True
        assert "task1" in scheduler.allocated_resources

    def test_allocate_insufficient_memory(self):
        """Test allocation with insufficient memory."""
        scheduler = ResourceScheduler(max_memory_gb=10.0)
        
        success1, _ = scheduler.allocate_resources("task1", memory_gb=8, num_cpus=2)
        assert success1 is True
        
        success2, _ = scheduler.allocate_resources("task2", memory_gb=5, num_cpus=2)
        assert success2 is False

    def test_allocate_max_parallel_exceeded(self):
        """Test allocation when max parallel exceeded."""
        scheduler = ResourceScheduler(max_parallel_tasks=2)
        
        success1, _ = scheduler.allocate_resources("task1", 4, 2)
        success2, _ = scheduler.allocate_resources("task2", 4, 2)
        success3, _ = scheduler.allocate_resources("task3", 4, 2)
        
        assert success1 is True
        assert success2 is True
        assert success3 is False

    def test_allocate_duplicate_task(self):
        """Test allocating resources for duplicate task."""
        scheduler = ResourceScheduler()
        
        success1, _ = scheduler.allocate_resources("task1", 4, 2)
        success2, _ = scheduler.allocate_resources("task1", 4, 2)
        
        assert success1 is True
        assert success2 is False

    def test_deallocate_resources(self):
        """Test deallocating resources."""
        scheduler = ResourceScheduler()
        
        scheduler.allocate_resources("task1", 8, 4)
        assert len(scheduler.allocated_resources) == 1
        
        success = scheduler.deallocate_resources("task1")
        assert success is True
        assert len(scheduler.allocated_resources) == 0

    def test_deallocate_nonexistent_task(self):
        """Test deallocating resources for nonexistent task."""
        scheduler = ResourceScheduler()
        
        success = scheduler.deallocate_resources("nonexistent")
        assert success is False

    def test_get_resource_availability(self):
        """Test getting resource availability."""
        scheduler = ResourceScheduler(max_parallel_tasks=4, max_memory_gb=32.0)
        
        scheduler.allocate_resources("task1", 10, 2)
        
        availability = scheduler.get_resource_availability()
        
        assert availability["total_memory_gb"] == 32.0
        assert availability["allocated_memory_gb"] == 10
        assert availability["available_memory_gb"] == 22.0
        assert availability["available_slots"] == 3

    def test_schedule_task(self):
        """Test scheduling task."""
        scheduler = ResourceScheduler()
        
        scheduled_time = (datetime.now() + timedelta(hours=1)).isoformat()
        success, message = scheduler.schedule_task("task1", scheduled_time, 3600)
        
        assert success is True
        assert len(scheduler.schedules) == 1

    def test_get_schedule(self):
        """Test getting schedule."""
        scheduler = ResourceScheduler()
        
        time1 = (datetime.now() + timedelta(hours=2)).isoformat()
        time2 = (datetime.now() + timedelta(hours=1)).isoformat()
        
        scheduler.schedule_task("task1", time1, 1000)
        scheduler.schedule_task("task2", time2, 2000)
        
        schedule = scheduler.get_schedule()
        
        # Should be sorted by time
        assert schedule[0]["scheduled_time"] < schedule[1]["scheduled_time"]


class TestOrchestrationIntegration:
    """Integration tests for orchestration."""

    def test_workflow_with_scheduler(self):
        """Test workflow with resource scheduler."""
        workflow = Workflow("test")
        scheduler = ResourceScheduler()
        
        id1 = workflow.add_task("task1", "operation1")
        id2 = workflow.add_task("task2", "operation2", dependencies=[id1])
        
        # Allocate resources
        scheduler.allocate_resources(id1, 8, 4)
        scheduler.allocate_resources(id2, 8, 4)
        
        assert len(scheduler.allocated_resources) == 2

    def test_complex_workflow_execution(self):
        """Test complex workflow execution."""
        workflow = Workflow("complex")
        scheduler = ResourceScheduler()
        
        # Create workflow DAG
        id1 = workflow.add_task("load", "load_data")
        id2 = workflow.add_task("process1", "process", dependencies=[id1])
        id3 = workflow.add_task("process2", "process", dependencies=[id1])
        id4 = workflow.add_task("aggregate", "aggregate", dependencies=[id2, id3])
        
        # Allocate resources
        for tid in [id1, id2, id3, id4]:
            scheduler.allocate_resources(tid, 4, 2, priority=0)
        
        # Check executable tasks
        executable = workflow.get_executable_tasks()
        assert id1 in executable
        
        # Execute first task
        def mock_executor(op, params):
            return {"result": "success"}
        
        success, result = workflow.execute_task(id1, mock_executor)
        assert success is True

    def test_resource_constraint_workflow(self):
        """Test workflow with resource constraints."""
        scheduler = ResourceScheduler(max_parallel_tasks=2, max_memory_gb=16.0)
        
        # Try to allocate beyond limits
        success1, _ = scheduler.allocate_resources("task1", 10, 2)
        success2, _ = scheduler.allocate_resources("task2", 4, 2)  # 10 + 4 = 14, fits
        success3, _ = scheduler.allocate_resources("task3", 4, 2)  # Would exceed memory
        
        assert success1 is True
        assert success2 is True  # Should succeed (14GB total)
        assert success3 is False  # Should fail (would be 18GB)
