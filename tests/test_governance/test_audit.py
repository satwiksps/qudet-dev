"""
Comprehensive tests for audit and compliance governance features.

Tests audit logging, compliance checking, and data governance.
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from qudet.governance.audit import (
    AuditLogger,
    ComplianceChecker,
    DataGovernance,
    AuditEvent
)


class TestAuditLogger:
    """Test audit logging functionality."""

    def test_initialization(self):
        """Test audit logger initialization."""
        logger = AuditLogger(max_events=5000)
        assert logger.max_events == 5000
        assert len(logger.events) == 0

    def test_log_event(self):
        """Test logging an event."""
        logger = AuditLogger()
        logger.log_event("data_access", "user1", "read_data", "dataset1", "success")
        
        assert len(logger.events) == 1
        assert logger.events[0].event_type == "data_access"
        assert logger.events[0].user == "user1"

    def test_log_multiple_events(self):
        """Test logging multiple events."""
        logger = AuditLogger()
        
        for i in range(10):
            logger.log_event("algorithm_run", f"user{i}", f"run_algo_{i}", 
                            f"algo{i}", "success")
        
        assert len(logger.events) == 10

    def test_get_user_activity(self):
        """Test retrieving user activity."""
        logger = AuditLogger()
        
        logger.log_event("data_access", "alice", "read", "dataset1", "success")
        logger.log_event("data_access", "bob", "read", "dataset2", "success")
        logger.log_event("data_access", "alice", "write", "dataset1", "success")
        
        alice_events = logger.get_user_activity("alice")
        assert len(alice_events) == 2
        assert all(e.user == "alice" for e in alice_events)

    def test_get_resource_access(self):
        """Test retrieving resource access."""
        logger = AuditLogger()
        
        logger.log_event("data_access", "user1", "read", "dataset1", "success")
        logger.log_event("data_access", "user2", "read", "dataset2", "success")
        logger.log_event("data_access", "user3", "write", "dataset1", "success")
        
        dataset1_access = logger.get_resource_access("dataset1")
        assert len(dataset1_access) == 2

    def test_export_audit_trail(self):
        """Test exporting audit trail."""
        logger = AuditLogger()
        logger.log_event("data_access", "user1", "read", "dataset1", "success")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filename = f.name
        
        logger.export_audit_trail(filename)
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]["user"] == "user1"

    def test_statistics(self):
        """Test audit statistics."""
        logger = AuditLogger()
        
        logger.log_event("data_access", "user1", "read", "dataset1", "success")
        logger.log_event("data_access", "user2", "read", "dataset2", "success")
        logger.log_event("algorithm_run", "user1", "run", "algo1", "failed")
        
        stats = logger.get_statistics()
        
        assert stats["total_events"] == 3
        assert stats["unique_users"] == 2
        assert stats["unique_resources"] == 3  # dataset1, dataset2, algo1

    def test_max_events_limit(self):
        """Test maximum events limit."""
        logger = AuditLogger(max_events=5)
        
        for i in range(10):
            logger.log_event("test", "user", "action", "resource", "success")
        
        assert len(logger.events) <= 5


class TestComplianceChecker:
    """Test compliance checking."""

    def test_initialization(self):
        """Test compliance checker initialization."""
        checker = ComplianceChecker()
        assert len(checker.policies) == 0
        assert len(checker.violations) == 0

    def test_add_policy(self):
        """Test adding compliance policy."""
        checker = ComplianceChecker()
        policy = {"data_retention": 365, "encryption_required": True}
        
        checker.add_policy("data_policy", policy)
        assert "data_policy" in checker.policies

    def test_check_data_sensitivity_clean(self):
        """Test data sensitivity check with clean data."""
        checker = ComplianceChecker()
        labels = ["age", "salary", "region"]
        
        is_compliant, issues = checker.check_data_sensitivity(labels)
        assert is_compliant is True
        assert len(issues) == 0

    def test_check_data_sensitivity_sensitive(self):
        """Test data sensitivity check with sensitive data."""
        checker = ComplianceChecker()
        labels = ["name", "ssn", "email_address", "phone_number"]
        
        is_compliant, issues = checker.check_data_sensitivity(labels)
        assert is_compliant is False
        assert len(issues) > 0

    def test_check_resource_limits_compliant(self):
        """Test resource limit check - compliant."""
        checker = ComplianceChecker()
        usage = {"memory_gb": 16, "cpu_percent": 50}
        limits = {"memory_gb": 32, "cpu_percent": 80}
        
        is_compliant, violations = checker.check_resource_limits(usage, limits)
        assert is_compliant is True

    def test_check_resource_limits_exceeded(self):
        """Test resource limit check - exceeded."""
        checker = ComplianceChecker()
        usage = {"memory_gb": 40, "cpu_percent": 90}
        limits = {"memory_gb": 32, "cpu_percent": 80}
        
        is_compliant, violations = checker.check_resource_limits(usage, limits)
        assert is_compliant is False
        assert len(violations) > 0

    def test_check_retention_compliant(self):
        """Test data retention check - compliant."""
        checker = ComplianceChecker()
        
        is_compliant, message = checker.check_data_retention(100, 365)
        assert is_compliant is True
        assert message is None

    def test_check_retention_exceeded(self):
        """Test data retention check - exceeded."""
        checker = ComplianceChecker()
        
        is_compliant, message = checker.check_data_retention(400, 365)
        assert is_compliant is False
        assert message is not None

    def test_compliance_report(self):
        """Test compliance report."""
        checker = ComplianceChecker()
        checker.check_data_sensitivity(["ssn"])
        
        report = checker.get_compliance_report()
        assert "total_violations" in report
        assert "policies_defined" in report
        assert "timestamp" in report

    def test_clear_violations(self):
        """Test clearing violations."""
        checker = ComplianceChecker()
        checker.check_data_sensitivity(["ssn"])
        
        assert len(checker.violations) > 0
        checker.clear_violations()
        assert len(checker.violations) == 0


class TestDataGovernance:
    """Test data governance."""

    def test_initialization(self):
        """Test data governance initialization."""
        governance = DataGovernance()
        assert len(governance.datasets) == 0
        assert len(governance.lineage) == 0

    def test_register_dataset(self):
        """Test registering dataset."""
        governance = DataGovernance()
        metadata = {
            "owner": "data_team",
            "classification": "confidential",
            "created_date": "2024-01-01",
            "source": "api"
        }
        
        governance.register_dataset("dataset1", metadata)
        assert "dataset1" in governance.datasets

    def test_register_dataset_missing_fields(self):
        """Test registering dataset with missing fields."""
        governance = DataGovernance()
        metadata = {"owner": "team"}  # Missing other fields
        
        governance.register_dataset("dataset1", metadata)
        assert governance.datasets["dataset1"]["classification"] == "unknown"

    def test_track_lineage(self):
        """Test tracking data lineage."""
        governance = DataGovernance()
        
        governance.register_dataset("dataset1", {"owner": "team"})
        governance.register_dataset("dataset2", {"owner": "team"})
        
        governance.track_lineage("dataset2", ["dataset1"], "normalization")
        
        lineage = governance.get_dataset_lineage("dataset2")
        assert len(lineage) == 1
        assert lineage[0]["transformation"] == "normalization"

    def test_complex_lineage(self):
        """Test complex lineage tracking."""
        governance = DataGovernance()
        
        for i in range(3):
            governance.register_dataset(f"dataset{i}", {"owner": "team"})
        
        governance.track_lineage("dataset1", ["dataset0"], "filtering")
        governance.track_lineage("dataset2", ["dataset1"], "aggregation")
        
        lineage = governance.get_dataset_lineage("dataset2")
        assert len(lineage) == 1

    def test_get_dataset_owners(self):
        """Test getting datasets by owner."""
        governance = DataGovernance()
        
        governance.register_dataset("dataset1", {"owner": "alice"})
        governance.register_dataset("dataset2", {"owner": "alice"})
        governance.register_dataset("dataset3", {"owner": "bob"})
        
        owners = governance.get_dataset_owners()
        assert len(owners["alice"]) == 2
        assert len(owners["bob"]) == 1

    def test_governance_report(self):
        """Test governance report."""
        governance = DataGovernance()
        
        governance.register_dataset("dataset1", {"owner": "team"})
        governance.track_lineage("dataset1", [], "creation")
        
        report = governance.get_governance_report()
        assert report["total_datasets"] == 1
        assert report["total_lineage_entries"] >= 1


class TestAuditIntegration:
    """Integration tests for audit functionality."""

    def test_audit_compliance_workflow(self):
        """Test audit with compliance checking."""
        logger = AuditLogger()
        checker = ComplianceChecker()
        
        # Log access to sensitive data
        logger.log_event("data_access", "user1", "access", "ssn_data", "success")
        
        # Check compliance
        is_compliant, issues = checker.check_data_sensitivity(["ssn_data"])
        
        assert not is_compliant
        assert len(logger.events) == 1

    def test_governance_audit_lineage(self):
        """Test governance with audit tracking."""
        governance = DataGovernance()
        logger = AuditLogger()
        
        governance.register_dataset("raw_data", {"owner": "data_team"})
        governance.register_dataset("processed_data", {"owner": "data_team"})
        
        logger.log_event("data_access", "analyst", "read", "raw_data", "success")
        
        governance.track_lineage("processed_data", ["raw_data"], "cleaning")
        
        assert len(governance.datasets) == 2
        assert len(logger.events) == 1
