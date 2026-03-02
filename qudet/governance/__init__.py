"""Governance module for QuDET.

Provides monitoring, security, auditing, privacy, drift detection, cost
estimation, orchestration, and validation tools for quantum data pipelines.
"""

from .audit import AuditEvent, AuditLogger, ComplianceChecker, DataGovernance
from .cost import ResourceEstimator
from .drift import QuantumDriftDetector
from .integrity import DataIntegrityCheck
from .monitor import JobMonitor
from .orchestration import (
    ResourceScheduler,
    Task,
    TaskStatus,
    Workflow,
    WorkflowStatus,
)
from .privacy import QuantumDifferentialPrivacy
from .security import (
    AccessLevel,
    EncryptionManager,
    SecureAccessControl,
    SecurityMonitor,
)
from .simulation import NoiseSimulator
from .validation import check_quantum_capacity
from .visualization import plot_kernel_matrix, plot_reduction_2d

__all__ = [
    # Cost & drift
    "ResourceEstimator",
    "QuantumDriftDetector",
    # Integrity & monitoring
    "DataIntegrityCheck",
    "JobMonitor",
    # Privacy & simulation
    "QuantumDifferentialPrivacy",
    "NoiseSimulator",
    # Visualization
    "plot_reduction_2d",
    "plot_kernel_matrix",
    # Audit
    "AuditLogger",
    "ComplianceChecker",
    "DataGovernance",
    "AuditEvent",
    # Security
    "SecureAccessControl",
    "EncryptionManager",
    "SecurityMonitor",
    "AccessLevel",
    # Orchestration
    "Workflow",
    "ResourceScheduler",
    "Task",
    "TaskStatus",
    "WorkflowStatus",
    # Validation
    "check_quantum_capacity",
]
