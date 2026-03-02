"""
Audit and compliance tracking for quantum data engineering pipelines.

Provides comprehensive audit logging, compliance verification, and
governance workflows for data processing and algorithm execution.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Represents a single audit event.

    Attributes:
        timestamp: ISO-8601 timestamp of the event.
        event_type: Category (e.g. ``"data_access"``, ``"algorithm_run"``).
        user: User who performed the action.
        action: Description of the action.
        resource: Resource affected by the action.
        status: Outcome (``"success"``, ``"failure"``, ``"warning"``).
        details: Arbitrary additional metadata.
    """

    timestamp: str
    event_type: str
    user: str
    action: str
    resource: str
    status: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict:
        """Convert this event to a plain dictionary.

        Returns:
            Serialisable dictionary representation.
        """
        return asdict(self)


class AuditLogger:
    """Records all operations on quantum data and algorithms.

    Tracks who did what, when, and what resources were used.
    Essential for compliance and forensic analysis.

    Attributes:
        max_events: Maximum number of events retained in memory.
        events: Chronological list of audit events.
        checksums: Mapping of resource → checksum for integrity tracking.
    """

    def __init__(self, max_events: int = 10_000) -> None:
        """Initialize audit logger.

        Args:
            max_events: Maximum events to store in memory before the
                oldest events are discarded.

        Raises:
            ValueError: If *max_events* is not positive.
        """
        if max_events <= 0:
            raise ValueError(f"max_events must be positive, got {max_events}")
        self.max_events = max_events
        self.events: List[AuditEvent] = []
        self.checksums: Dict[str, str] = {}

    def log_event(
        self,
        event_type: str,
        user: str,
        action: str,
        resource: str,
        status: str,
        details: Optional[Dict] = None,
    ) -> None:
        """Log an audit event.

        Args:
            event_type: Type of event (e.g. ``"data_access"``).
            user: User performing the action.
            action: Description of the action.
            resource: Resource affected.
            status: ``"success"`` / ``"failure"`` / ``"warning"``.
            details: Additional metadata dictionary.
        """
        event = AuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            user=user,
            action=action,
            resource=resource,
            status=status,
            details=details or {},
        )
        self.events.append(event)

        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def get_user_activity(self, user: str) -> List[AuditEvent]:
        """Get all events for a specific user.

        Args:
            user: Username to filter by.

        Returns:
            List of matching events.
        """
        return [e for e in self.events if e.user == user]

    def get_resource_access(self, resource: str) -> List[AuditEvent]:
        """Get all access events for a specific resource.

        Args:
            resource: Resource identifier.

        Returns:
            List of matching events.
        """
        return [e for e in self.events if e.resource == resource]

    def export_audit_trail(self, filename: str) -> None:
        """Export the audit trail to a JSON file.

        Args:
            filename: Path to the output JSON file.
        """
        with open(filename, "w") as f:
            json.dump([e.to_dict() for e in self.events], f, indent=2)
        logger.info("Audit trail exported to %s (%d events).", filename, len(self.events))

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate audit statistics.

        Returns:
            Dictionary with ``total_events``, ``unique_users``,
            ``unique_resources``, ``event_types``, and ``success_rate``.
        """
        return {
            "total_events": len(self.events),
            "unique_users": len(set(e.user for e in self.events)),
            "unique_resources": len(set(e.resource for e in self.events)),
            "event_types": list(set(e.event_type for e in self.events)),
            "success_rate": self._calculate_success_rate(),
        }

    def _calculate_success_rate(self) -> float:
        """Calculate the fraction of successful operations."""
        if not self.events:
            return 0.0
        successful = sum(1 for e in self.events if e.status == "success")
        return successful / len(self.events)


class ComplianceChecker:
    """Verifies compliance with governance policies.

    Checks data processing, algorithm usage, and resource constraints
    against defined compliance rules.

    Attributes:
        policies: Registered compliance policies.
        violations: Chronological list of detected violations.
    """

    def __init__(self) -> None:
        """Initialize compliance checker."""
        self.policies: Dict[str, Dict] = {}
        self.violations: List[Dict] = []

    def add_policy(self, policy_name: str, policy_config: Dict) -> None:
        """Register a compliance policy.

        Args:
            policy_name: Unique policy name.
            policy_config: Policy configuration with rules.
        """
        self.policies[policy_name] = policy_config

    def check_data_sensitivity(
        self, data_labels: List[str]
    ) -> Tuple[bool, List[str]]:
        """Check if data columns contain sensitive information.

        Scans column/feature labels for keywords indicative of PII
        (e.g. ``ssn``, ``password``, ``credit_card``).

        Args:
            data_labels: Column or feature labels to scan.

        Returns:
            Tuple of ``(is_compliant, issues)`` where *issues* lists
            all detected sensitive fields.
        """
        sensitive_keywords = [
            "ssn", "password", "credit_card", "phone", "email", "address",
        ]
        issues: List[str] = []

        for label in data_labels:
            if any(kw in label.lower() for kw in sensitive_keywords):
                issues.append(f"Sensitive field detected: {label}")

        is_compliant = len(issues) == 0

        if not is_compliant:
            self.violations.append({
                "type": "data_sensitivity",
                "timestamp": datetime.now().isoformat(),
                "issues": issues,
            })

        return is_compliant, issues

    def check_resource_limits(
        self,
        usage: Dict[str, float],
        limits: Dict[str, float],
    ) -> Tuple[bool, List[str]]:
        """Check if resource usage is within defined limits.

        Args:
            usage: Current resource usage mapping.
            limits: Maximum allowed resource values.

        Returns:
            Tuple of ``(is_compliant, violations_list)``.
        """
        violation_msgs: List[str] = []

        for resource, limit in limits.items():
            if resource in usage and usage[resource] > limit:
                violation_msgs.append(
                    f"{resource} usage ({usage[resource]}) exceeds limit ({limit})"
                )

        is_compliant = len(violation_msgs) == 0

        if not is_compliant:
            self.violations.append({
                "type": "resource_limit",
                "timestamp": datetime.now().isoformat(),
                "violations": violation_msgs,
            })

        return is_compliant, violation_msgs

    def check_data_retention(
        self, data_age_days: float, max_retention_days: int
    ) -> Tuple[bool, Optional[str]]:
        """Check if data retention is compliant.

        Args:
            data_age_days: Age of the data in days.
            max_retention_days: Maximum allowed retention period.

        Returns:
            Tuple of ``(is_compliant, message)``.  *message* is
            ``None`` when compliant.
        """
        is_compliant = data_age_days <= max_retention_days
        message: Optional[str] = None

        if not is_compliant:
            message = (
                f"Data exceeds retention period: "
                f"{data_age_days:.1f} > {max_retention_days} days"
            )
            self.violations.append({
                "type": "retention_policy",
                "timestamp": datetime.now().isoformat(),
                "message": message,
            })

        return is_compliant, message

    def get_compliance_report(self) -> Dict[str, Any]:
        """Get a comprehensive compliance report.

        Returns:
            Dictionary with violation counts, policy counts, details,
            and timestamp.
        """
        return {
            "total_violations": len(self.violations),
            "policies_defined": len(self.policies),
            "violations": self.violations,
            "timestamp": datetime.now().isoformat(),
        }

    def clear_violations(self) -> None:
        """Clear the violation history."""
        self.violations = []


class DataGovernance:
    """Manages data governance policies and metadata tracking.

    Maintains data lineage, ownership, and governance metadata
    for all datasets in the pipeline.

    Attributes:
        datasets: Registry of dataset_id → governance metadata.
        lineage: Mapping of dataset_id → list of lineage entries.
    """

    def __init__(self) -> None:
        """Initialize data governance system."""
        self.datasets: Dict[str, Dict] = {}
        self.lineage: Dict[str, List[str]] = {}

    def register_dataset(self, dataset_id: str, metadata: Dict) -> None:
        """Register a dataset with governance metadata.

        Missing required fields (``owner``, ``classification``,
        ``created_date``, ``source``) are filled with ``"unknown"``.

        Args:
            dataset_id: Unique dataset identifier.
            metadata: Governance metadata dictionary.
        """
        required_fields = ["owner", "classification", "created_date", "source"]

        for field in required_fields:
            if field not in metadata:
                metadata[field] = "unknown"

        self.datasets[dataset_id] = metadata
        if dataset_id not in self.lineage:
            self.lineage[dataset_id] = []

    def track_lineage(
        self,
        dataset_id: str,
        source_datasets: List[str],
        transformation: str,
    ) -> None:
        """Record a data lineage entry.

        Args:
            dataset_id: Current dataset identifier.
            source_datasets: Identifiers of source datasets.
            transformation: Description of the transformation applied.
        """
        if dataset_id not in self.lineage:
            self.lineage[dataset_id] = []

        self.lineage[dataset_id].append({
            "sources": source_datasets,
            "transformation": transformation,
            "timestamp": datetime.now().isoformat(),
        })

    def get_dataset_lineage(self, dataset_id: str) -> List[Dict]:
        """Get the complete lineage for a dataset.

        Args:
            dataset_id: Dataset identifier.

        Returns:
            List of lineage entries (may be empty).
        """
        return self.lineage.get(dataset_id, [])

    def get_dataset_owners(self) -> Dict[str, List[str]]:
        """Get datasets grouped by owner.

        Returns:
            Mapping of owner → list of dataset identifiers.
        """
        owners: Dict[str, List[str]] = {}

        for dataset_id, metadata in self.datasets.items():
            owner = metadata.get("owner", "unknown")
            if owner not in owners:
                owners[owner] = []
            owners[owner].append(dataset_id)

        return owners

    def get_governance_report(self) -> Dict[str, Any]:
        """Get a summary data governance report.

        Returns:
            Dictionary with dataset counts, ownership breakdown,
            lineage statistics, and timestamp.
        """
        return {
            "total_datasets": len(self.datasets),
            "datasets_by_owner": self.get_dataset_owners(),
            "total_lineage_entries": sum(
                len(v) for v in self.lineage.values()
            ),
            "timestamp": datetime.now().isoformat(),
        }
