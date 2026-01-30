"""
Comprehensive tests for security and access control features.

Tests access control, encryption, and security monitoring.
"""

import pytest
from qudet.governance.security import (
    SecureAccessControl,
    EncryptionManager,
    SecurityMonitor,
    AccessLevel
)


class TestSecureAccessControl:
    """Test secure access control."""

    def test_initialization(self):
        """Test access control initialization."""
        control = SecureAccessControl()
        assert len(control.users) == 0
        assert len(control.roles) == 4  # Default roles

    def test_add_user(self):
        """Test adding user."""
        control = SecureAccessControl()
        success = control.add_user("alice", "password123", AccessLevel.USER)
        
        assert success is True
        assert "alice" in control.users

    def test_add_duplicate_user(self):
        """Test adding duplicate user fails."""
        control = SecureAccessControl()
        control.add_user("alice", "password123", AccessLevel.USER)
        success = control.add_user("alice", "password456", AccessLevel.ADMIN)
        
        assert success is False

    def test_authenticate_success(self):
        """Test successful authentication."""
        control = SecureAccessControl()
        control.add_user("alice", "password123", AccessLevel.USER)
        
        success, token = control.authenticate("alice", "password123")
        assert success is True
        assert token is not None

    def test_authenticate_wrong_password(self):
        """Test authentication with wrong password."""
        control = SecureAccessControl()
        control.add_user("alice", "password123", AccessLevel.USER)
        
        success, token = control.authenticate("alice", "wrongpassword")
        assert success is False
        assert token is None

    def test_authenticate_nonexistent_user(self):
        """Test authentication of nonexistent user."""
        control = SecureAccessControl()
        
        success, token = control.authenticate("bob", "password")
        assert success is False

    def test_check_permission_viewer(self):
        """Test viewer permissions."""
        control = SecureAccessControl()
        control.add_user("alice", "password", AccessLevel.VIEWER)
        
        assert control.check_permission("alice", "data:read") is True
        assert control.check_permission("alice", "data:write") is False

    def test_check_permission_user(self):
        """Test user permissions."""
        control = SecureAccessControl()
        control.add_user("alice", "password", AccessLevel.USER)
        
        assert control.check_permission("alice", "data:read") is True
        assert control.check_permission("alice", "data:write") is True
        assert control.check_permission("alice", "users:manage") is False

    def test_check_permission_admin(self):
        """Test admin permissions."""
        control = SecureAccessControl()
        control.add_user("alice", "password", AccessLevel.ADMIN)
        
        assert control.check_permission("alice", "data:read") is True
        assert control.check_permission("alice", "users:manage") is True
        assert control.check_permission("alice", "policies:edit") is True

    def test_check_permission_superuser(self):
        """Test superuser permissions."""
        control = SecureAccessControl()
        control.add_user("alice", "password", AccessLevel.SUPERUSER)
        
        assert control.check_permission("alice", "data:read") is True
        assert control.check_permission("alice", "users:manage") is True
        assert control.check_permission("alice", "any_permission") is True

    def test_get_user_info(self):
        """Test getting user information."""
        control = SecureAccessControl()
        control.add_user("alice", "password123", AccessLevel.USER)
        
        info = control.get_user_info("alice")
        assert info is not None
        assert info["role"] == AccessLevel.USER.value
        assert "password_hash" not in info  # Should not return password

    def test_get_nonexistent_user_info(self):
        """Test getting info for nonexistent user."""
        control = SecureAccessControl()
        info = control.get_user_info("bob")
        assert info is None


class TestEncryptionManager:
    """Test encryption management."""

    def test_initialization(self):
        """Test encryption manager initialization."""
        manager = EncryptionManager()
        assert len(manager.keys) == 0
        assert len(manager.encrypted_data) == 0

    def test_generate_key(self):
        """Test key generation."""
        manager = EncryptionManager()
        key = manager.generate_key("key1", key_size=256)
        
        assert key is not None
        assert len(key) > 0
        assert "key1" in manager.keys

    def test_generate_multiple_keys(self):
        """Test generating multiple keys."""
        manager = EncryptionManager()
        
        for i in range(5):
            manager.generate_key(f"key{i}")
        
        assert len(manager.keys) == 5

    def test_encrypt_data(self):
        """Test encrypting data."""
        manager = EncryptionManager()
        manager.generate_key("key1")
        
        success, data_id = manager.encrypt_data("sensitive_data", "key1")
        
        assert success is True
        assert data_id is not None
        assert data_id in manager.encrypted_data

    def test_encrypt_with_invalid_key(self):
        """Test encrypting with invalid key."""
        manager = EncryptionManager()
        
        success, data_id = manager.encrypt_data("data", "invalid_key")
        
        assert success is False
        assert data_id == ""

    def test_decrypt_data(self):
        """Test decrypting data."""
        manager = EncryptionManager()
        manager.generate_key("key1")
        
        success, data_id = manager.encrypt_data("sensitive_data", "key1")
        assert success is True
        
        success, decrypted = manager.decrypt_data(data_id, "key1")
        assert success is True

    def test_decrypt_with_wrong_key(self):
        """Test decryption with wrong key."""
        manager = EncryptionManager()
        manager.generate_key("key1")
        manager.generate_key("key2")
        
        success, data_id = manager.encrypt_data("data", "key1")
        
        success, decrypted = manager.decrypt_data(data_id, "key2")
        assert success is False

    def test_rotate_key(self):
        """Test key rotation."""
        manager = EncryptionManager()
        manager.generate_key("key1")
        
        success, new_key_id = manager.rotate_key("key1")
        
        assert success is True
        assert new_key_id is not None
        assert "rotated_to" in manager.keys["key1"]

    def test_get_key_info(self):
        """Test getting key information."""
        manager = EncryptionManager()
        manager.generate_key("key1", key_size=256)
        
        info = manager.get_key_info("key1")
        
        assert info is not None
        assert "key" not in info  # Should not return actual key
        assert info["size"] == 256
        assert info["algorithm"] == "AES"


class TestSecurityMonitor:
    """Test security monitoring."""

    def test_initialization(self):
        """Test security monitor initialization."""
        monitor = SecurityMonitor(alert_threshold=5)
        assert monitor.alert_threshold == 5
        assert len(monitor.security_events) == 0

    def test_log_security_event(self):
        """Test logging security event."""
        monitor = SecurityMonitor()
        monitor.log_security_event("unauthorized_access", "user1", "Attempted database access")
        
        assert len(monitor.security_events) == 1

    def test_log_critical_event_generates_alert(self):
        """Test that critical events generate alerts."""
        monitor = SecurityMonitor()
        monitor.log_security_event("breach", "system", "Data breach detected", severity="critical")
        
        assert len(monitor.alerts) > 0

    def test_detect_excessive_failed_attempts(self):
        """Test detecting excessive failed attempts."""
        monitor = SecurityMonitor()
        
        for i in range(7):
            monitor.log_security_event("failed_login", "user1", "Login attempt failed")
        
        anomalies = monitor.detect_anomalies("user1", max_failed_attempts=5)
        
        assert len(anomalies) > 0
        assert anomalies[0]["type"] == "excessive_failed_attempts"

    def test_multiple_users(self):
        """Test monitoring multiple users."""
        monitor = SecurityMonitor()
        
        monitor.log_security_event("login", "alice", "User alice logged in")
        monitor.log_security_event("login", "bob", "User bob logged in")
        monitor.log_security_event("login", "charlie", "User charlie logged in")
        
        assert len(monitor.security_events) == 3

    def test_security_report(self):
        """Test security report."""
        monitor = SecurityMonitor()
        
        monitor.log_security_event("login", "user1", "Login successful")
        monitor.log_security_event("breach", "system", "Alert", severity="critical")
        
        report = monitor.get_security_report()
        
        assert report["total_events"] == 2
        assert report["total_alerts"] >= 1
        assert "timestamp" in report

    def test_clear_events(self):
        """Test clearing event history."""
        monitor = SecurityMonitor()
        
        monitor.log_security_event("login", "user1", "Login")
        assert len(monitor.security_events) > 0
        
        monitor.clear_events()
        assert len(monitor.security_events) == 0


class TestSecurityIntegration:
    """Integration tests for security features."""

    def test_access_control_with_encryption(self):
        """Test access control with encryption."""
        access_control = SecureAccessControl()
        encryption = EncryptionManager()
        
        # Create user
        access_control.add_user("alice", "password", AccessLevel.USER)
        
        # Authenticate
        success, token = access_control.authenticate("alice", "password")
        assert success is True
        
        # Encrypt sensitive data
        encryption.generate_key("user_key")
        success, data_id = encryption.encrypt_data("sensitive", "user_key")
        
        assert success is True

    def test_full_security_workflow(self):
        """Test full security workflow."""
        access_control = SecureAccessControl()
        encryption = EncryptionManager()
        monitor = SecurityMonitor()
        
        # Add user and authenticate
        access_control.add_user("alice", "password123", AccessLevel.ADMIN)
        
        # Log authentication attempt
        success, token = access_control.authenticate("alice", "password123")
        monitor.log_security_event("authentication", "alice", "User authenticated successfully")
        
        # Encrypt data
        encryption.generate_key("secure_key")
        success, data_id = encryption.encrypt_data("data123", "secure_key")
        
        # Generate report
        report = monitor.get_security_report()
        
        assert success is True
        assert len(report["recent_events"]) > 0
