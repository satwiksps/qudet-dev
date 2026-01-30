"""
Audit and compliance tracking for quantum data engineering pipelines.

Provides comprehensive audit logging, compliance verification, and
governance workflows for data processing and algorithm execution.
"""

import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class AuditEvent:
    """Represents a single audit event."""
    timestamp: str
    event_type: str
    user: str
    action: str
    resource: str
    status: str
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class AuditLogger:
    """
    Records all operations on quantum data and algorithms.
    
    Tracks who did what, when, and what resources were used.
    Essential for compliance and forensic analysis.
    
    Best for: Compliance, audit trails, forensics.
    """
    
    def __init__(self, max_events: int = 10000):
        """
        Initialize audit logger.
        
        Args:
            max_events: Maximum events to store in memory
        """
        self.max_events = max_events
        self.events: List[AuditEvent] = []
        self.checksums: Dict[str, str] = {}

    def log_event(self, event_type: str, user: str, action: str, 
                  resource: str, status: str, details: Optional[Dict] = None) -> None:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event (data_access, algorithm_run, config_change)
            user: User performing action
            action: Description of action
            resource: Resource affected
            status: Success/failure/warning
            details: Additional details
        """
        event = AuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            user=user,
            action=action,
            resource=resource,
            status=status,
            details=details or {}
        )
        
        self.events.append(event)
        
        # Maintain max size
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def get_user_activity(self, user: str) -> List[AuditEvent]:
        """Get all events for a specific user."""
        return [e for e in self.events if e.user == user]

    def get_resource_access(self, resource: str) -> List[AuditEvent]:
        """Get all access to a specific resource."""
        return [e for e in self.events if e.resource == resource]

    def export_audit_trail(self, filename: str) -> None:
        """Export audit trail to JSON file."""
        with open(filename, 'w') as f:
            json.dump([e.to_dict() for e in self.events], f, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics."""
        return {
            "total_events": len(self.events),
            "unique_users": len(set(e.user for e in self.events)),
            "unique_resources": len(set(e.resource for e in self.events)),
            "event_types": list(set(e.event_type for e in self.events)),
            "success_rate": self._calculate_success_rate()
        }

    def _calculate_success_rate(self) -> float:
        """Calculate success rate of operations."""
        if not self.events:
            return 0.0
        successful = sum(1 for e in self.events if e.status == "success")
        return successful / len(self.events)


class ComplianceChecker:
    """
    Verifies compliance with governance policies.
    
    Checks data processing, algorithm usage, and resource constraints
    against defined compliance rules.
    
    Best for: Policy enforcement, compliance validation.
    """
    
    def __init__(self):
        """Initialize compliance checker."""
        self.policies: Dict[str, Dict] = {}
        self.violations: List[Dict] = []

    def add_policy(self, policy_name: str, policy_config: Dict) -> None:
        """
        Add a compliance policy.
        
        Args:
            policy_name: Name of policy
            policy_config: Policy configuration with rules
        """
        self.policies[policy_name] = policy_config

    def check_data_sensitivity(self, data_labels: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if data contains sensitive information.
        
        Args:
            data_labels: Column/feature labels
            
        Returns:
            Tuple of (is_compliant, issues)
        """
        sensitive_keywords = ['ssn', 'password', 'credit_card', 'phone', 'email', 'address']
        issues = []
        
        for label in data_labels:
            if any(keyword in label.lower() for keyword in sensitive_keywords):
                issues.append(f"Sensitive field detected: {label}")
        
        is_compliant = len(issues) == 0
        
        if not is_compliant:
            self.violations.append({
                "type": "data_sensitivity",
                "timestamp": datetime.now().isoformat(),
                "issues": issues
            })
        
        return is_compliant, issues

    def check_resource_limits(self, usage: Dict[str, float], limits: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Check if resource usage is within limits.
        
        Args:
            usage: Current resource usage
            limits: Resource limits
            
        Returns:
            Tuple of (is_compliant, violations)
        """
        violations = []
        
        for resource, limit in limits.items():
            if resource in usage and usage[resource] > limit:
                violations.append(f"{resource} usage ({usage[resource]}) exceeds limit ({limit})")
        
        is_compliant = len(violations) == 0
        
        if not is_compliant:
            self.violations.append({
                "type": "resource_limit",
                "timestamp": datetime.now().isoformat(),
                "violations": violations
            })
        
        return is_compliant, violations

    def check_data_retention(self, data_age_days: float, max_retention_days: int) -> Tuple[bool, Optional[str]]:
        """
        Check if data retention is compliant.
        
        Args:
            data_age_days: Age of data in days
            max_retention_days: Maximum allowed retention
            
        Returns:
            Tuple of (is_compliant, message)
        """
        is_compliant = data_age_days <= max_retention_days
        message = None
        
        if not is_compliant:
            message = f"Data exceeds retention period: {data_age_days:.1f} > {max_retention_days} days"
            self.violations.append({
                "type": "retention_policy",
                "timestamp": datetime.now().isoformat(),
                "message": message
            })
        
        return is_compliant, message

    def get_compliance_report(self) -> Dict[str, Any]:
        """Get comprehensive compliance report."""
        return {
            "total_violations": len(self.violations),
            "policies_defined": len(self.policies),
            "violations": self.violations,
            "timestamp": datetime.now().isoformat()
        }

    def clear_violations(self) -> None:
        """Clear violation history."""
        self.violations = []


class DataGovernance:
    """
    Manages data governance policies and metadata tracking.
    
    Maintains data lineage, ownership, and governance metadata
    for all datasets in the pipeline.
    
    Best for: Data catalog, lineage tracking, governance.
    """
    
    def __init__(self):
        """Initialize data governance system."""
        self.datasets: Dict[str, Dict] = {}
        self.lineage: Dict[str, List[str]] = {}

    def register_dataset(self, dataset_id: str, metadata: Dict) -> None:
        """
        Register a dataset with governance metadata.
        
        Args:
            dataset_id: Unique dataset identifier
            metadata: Governance metadata
        """
        required_fields = ['owner', 'classification', 'created_date', 'source']
        
        for field in required_fields:
            if field not in metadata:
                metadata[field] = "unknown"
        
        self.datasets[dataset_id] = metadata
        if dataset_id not in self.lineage:
            self.lineage[dataset_id] = []

    def track_lineage(self, dataset_id: str, source_datasets: List[str], 
                     transformation: str) -> None:
        """
        Track data lineage and transformations.
        
        Args:
            dataset_id: Current dataset ID
            source_datasets: Source dataset IDs
            transformation: Transformation applied
        """
        if dataset_id not in self.lineage:
            self.lineage[dataset_id] = []
        
        lineage_entry = {
            "sources": source_datasets,
            "transformation": transformation,
            "timestamp": datetime.now().isoformat()
        }
        
        self.lineage[dataset_id].append(lineage_entry)

    def get_dataset_lineage(self, dataset_id: str) -> List[Dict]:
        """Get complete lineage for a dataset."""
        return self.lineage.get(dataset_id, [])

    def get_dataset_owners(self) -> Dict[str, List[str]]:
        """Get datasets grouped by owner."""
        owners = {}
        
        for dataset_id, metadata in self.datasets.items():
            owner = metadata.get('owner', 'unknown')
            if owner not in owners:
                owners[owner] = []
            owners[owner].append(dataset_id)
        
        return owners

    def get_governance_report(self) -> Dict[str, Any]:
        """Get data governance report."""
        return {
            "total_datasets": len(self.datasets),
            "datasets_by_owner": self.get_dataset_owners(),
            "total_lineage_entries": sum(len(v) for v in self.lineage.values()),
            "timestamp": datetime.now().isoformat()
        }
