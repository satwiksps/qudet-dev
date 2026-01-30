"""
Security and access control mechanisms for quantum data engineering.

Provides authentication, authorization, encryption, and security
monitoring for data pipelines and quantum algorithms.
"""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum


class AccessLevel(Enum):
    """Access control levels."""
    VIEWER = "viewer"
    USER = "user"
    ADMIN = "admin"
    SUPERUSER = "superuser"


class SecureAccessControl:
    """
    Manages user authentication and role-based access control (RBAC).
    
    Enforces fine-grained permissions on resources and operations
    with audit trail of access attempts.
    
    Best for: Access management, permissions, role assignment.
    """
    
    def __init__(self):
        """Initialize access control system."""
        self.users: Dict[str, Dict] = {}
        self.roles: Dict[str, Set[str]] = {}
        self.access_log: List[Dict] = []
        self._init_default_roles()

    def _init_default_roles(self) -> None:
        """Initialize default roles."""
        self.roles = {
            AccessLevel.VIEWER.value: {"data:read", "reports:view"},
            AccessLevel.USER.value: {"data:read", "data:write", "algorithms:run", "reports:view"},
            AccessLevel.ADMIN.value: {"data:read", "data:write", "algorithms:run", "reports:view", 
                                      "users:manage", "policies:edit"},
            AccessLevel.SUPERUSER.value: {"*"}  # All permissions
        }

    def add_user(self, username: str, password: str, role: AccessLevel) -> bool:
        """
        Add a user to the system.
        
        Args:
            username: User identifier
            password: User password
            role: User role/access level
            
        Returns:
            Success status
        """
        if username in self.users:
            return False
        
        password_hash = self._hash_password(password)
        
        self.users[username] = {
            "password_hash": password_hash,
            "role": role.value,
            "created": datetime.now().isoformat(),
            "last_access": None,
            "access_attempts": 0
        }
        
        return True

    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        Authenticate a user.
        
        Args:
            username: User identifier
            password: User password
            
        Returns:
            Tuple of (success, token)
        """
        if username not in self.users:
            self._log_access_attempt(username, "authentication", "failure", "user_not_found")
            return False, None
        
        user = self.users[username]
        password_hash = self._hash_password(password)
        
        if not hmac.compare_digest(user["password_hash"], password_hash):
            user["access_attempts"] += 1
            self._log_access_attempt(username, "authentication", "failure", "invalid_password")
            return False, None
        
        # Reset access attempts on successful auth
        user["access_attempts"] = 0
        user["last_access"] = datetime.now().isoformat()
        
        # Generate token
        token = self._generate_token(username)
        
        self._log_access_attempt(username, "authentication", "success", "authenticated")
        
        return True, token

    def check_permission(self, username: str, permission: str) -> bool:
        """
        Check if user has permission.
        
        Args:
            username: User identifier
            permission: Permission to check
            
        Returns:
            Permission granted status
        """
        if username not in self.users:
            return False
        
        user = self.users[username]
        role = user["role"]
        
        permissions = self.roles.get(role, set())
        
        # Superuser has all permissions
        if "*" in permissions:
            return True
        
        return permission in permissions

    def _hash_password(self, password: str) -> str:
        """Hash a password."""
        return hashlib.pbkdf2_hmac('sha256', password.encode(), 
                                    b'quantum_salt', 100000).hex()

    def _generate_token(self, username: str) -> str:
        """Generate authentication token."""
        return secrets.token_urlsafe(32)

    def _log_access_attempt(self, username: str, action: str, 
                           status: str, details: str) -> None:
        """Log access attempt."""
        self.access_log.append({
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "action": action,
            "status": status,
            "details": details
        })

    def get_user_info(self, username: str) -> Optional[Dict]:
        """Get user information."""
        if username not in self.users:
            return None
        
        user = self.users[username].copy()
        # Don't return password hash
        user.pop("password_hash", None)
        return user


class EncryptionManager:
    """
    Manages encryption and decryption of sensitive data.
    
    Provides encryption for data at rest and in transit,
    with key management and rotation policies.
    
    Best for: Data encryption, key management, confidentiality.
    """
    
    def __init__(self):
        """Initialize encryption manager."""
        self.keys: Dict[str, Dict] = {}
        self.encrypted_data: Dict[str, Dict] = {}

    def generate_key(self, key_id: str, key_size: int = 256) -> str:
        """
        Generate encryption key.
        
        Args:
            key_id: Key identifier
            key_size: Key size in bits
            
        Returns:
            Generated key
        """
        key = secrets.token_hex(key_size // 8)
        
        self.keys[key_id] = {
            "key": key,
            "created": datetime.now().isoformat(),
            "algorithm": "AES",
            "size": key_size
        }
        
        return key

    def encrypt_data(self, data: str, key_id: str) -> Tuple[bool, str]:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            key_id: Key identifier to use
            
        Returns:
            Tuple of (success, encrypted_data)
        """
        if key_id not in self.keys:
            return False, ""
        
        # Simplified encryption (in practice would use proper crypto)
        key = self.keys[key_id]["key"]
        encrypted = hashlib.sha256((data + key).encode()).hexdigest()
        
        data_id = secrets.token_hex(8)
        self.encrypted_data[data_id] = {
            "encrypted": encrypted,
            "key_id": key_id,
            "created": datetime.now().isoformat()
        }
        
        return True, data_id

    def decrypt_data(self, data_id: str, key_id: str) -> Tuple[bool, Optional[str]]:
        """
        Decrypt data (note: simplified for demo).
        
        Args:
            data_id: Data identifier
            key_id: Key identifier
            
        Returns:
            Tuple of (success, decrypted_data)
        """
        if data_id not in self.encrypted_data:
            return False, None
        
        if self.encrypted_data[data_id]["key_id"] != key_id:
            return False, None
        
        # In real implementation, would perform actual decryption
        return True, "decrypted_data"

    def rotate_key(self, key_id: str) -> Tuple[bool, Optional[str]]:
        """
        Rotate encryption key.
        
        Args:
            key_id: Key identifier
            
        Returns:
            Tuple of (success, new_key_id)
        """
        if key_id not in self.keys:
            return False, None
        
        # Generate new key
        new_key_id = f"{key_id}_rotated_{int(datetime.now().timestamp())}"
        new_key = self.generate_key(new_key_id)
        
        # Mark old key as rotated
        self.keys[key_id]["rotated"] = datetime.now().isoformat()
        self.keys[key_id]["rotated_to"] = new_key_id
        
        return True, new_key_id

    def get_key_info(self, key_id: str) -> Optional[Dict]:
        """Get key information (without actual key)."""
        if key_id not in self.keys:
            return None
        
        info = self.keys[key_id].copy()
        # Don't return actual key
        info.pop("key", None)
        return info


class SecurityMonitor:
    """
    Monitors security events and detects anomalies.
    
    Tracks suspicious activities, unauthorized access attempts,
    and security policy violations.
    
    Best for: Threat detection, anomaly detection, security alerts.
    """
    
    def __init__(self, alert_threshold: int = 5):
        """
        Initialize security monitor.
        
        Args:
            alert_threshold: Failed attempts before alert
        """
        self.alert_threshold = alert_threshold
        self.security_events: List[Dict] = []
        self.alerts: List[Dict] = []

    def log_security_event(self, event_type: str, source: str, 
                          details: str, severity: str = "info") -> None:
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            source: Source of event (user, system, etc.)
            details: Event details
            severity: Severity level (info, warning, critical)
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "source": source,
            "details": details,
            "severity": severity
        }
        
        self.security_events.append(event)
        
        # Generate alert if needed
        if severity == "critical":
            self._generate_alert(event)

    def detect_anomalies(self, user: str, max_failed_attempts: int = 5) -> List[Dict]:
        """
        Detect anomalous behavior.
        
        Args:
            user: User to analyze
            max_failed_attempts: Maximum allowed failed attempts
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check for excessive failed attempts
        user_events = [e for e in self.security_events if e["source"] == user]
        failed_attempts = sum(1 for e in user_events 
                             if "failed" in e["event_type"].lower())
        
        if failed_attempts > max_failed_attempts:
            anomaly = {
                "type": "excessive_failed_attempts",
                "user": user,
                "count": failed_attempts,
                "timestamp": datetime.now().isoformat()
            }
            anomalies.append(anomaly)
            self._generate_alert(anomaly)
        
        return anomalies

    def _generate_alert(self, event: Dict) -> None:
        """Generate security alert."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "alert_level": "HIGH"
        }
        self.alerts.append(alert)

    def get_security_report(self) -> Dict:
        """Get security report."""
        return {
            "total_events": len(self.security_events),
            "total_alerts": len(self.alerts),
            "recent_events": self.security_events[-10:],
            "recent_alerts": self.alerts[-5:],
            "timestamp": datetime.now().isoformat()
        }

    def clear_events(self) -> None:
        """Clear event history."""
        self.security_events = []
