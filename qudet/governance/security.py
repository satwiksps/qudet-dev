"""
Security and access control mechanisms for quantum data engineering.

Provides authentication, authorization, encryption, and security
monitoring for data pipelines and quantum algorithms.

The ``EncryptionManager`` uses Fernet symmetric encryption from the
``cryptography`` package when available.  If the package is not
installed, calling encrypt/decrypt will raise an ``ImportError`` with
installation instructions.
"""

import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class AccessLevel(Enum):
    """Role-based access control levels.

    Each level grants a progressively wider set of permissions:

    * ``VIEWER`` – read-only access to data and reports.
    * ``USER`` – read/write data and execute algorithms.
    * ``ADMIN`` – user management and policy editing.
    * ``SUPERUSER`` – unrestricted access (wildcard).
    """

    VIEWER = "viewer"
    USER = "user"
    ADMIN = "admin"
    SUPERUSER = "superuser"


class SecureAccessControl:
    """Manages user authentication and role-based access control (RBAC).

    Enforces fine-grained permissions on resources and operations
    with an audit trail of access attempts.

    Passwords are hashed with PBKDF2-HMAC-SHA256 using a unique random
    salt per user.  Authentication tokens are time-limited (default 1 hour).

    Attributes:
        users: Mapping of username → user record.
        roles: Mapping of role name → set of permission strings.
        access_log: Chronological list of access-attempt records.
    """

    _TOKEN_LIFETIME = timedelta(hours=1)

    def __init__(self) -> None:
        """Initialize access control system."""
        self.users: Dict[str, Dict] = {}
        self.roles: Dict[str, Set[str]] = {}
        self.access_log: List[Dict] = []
        self._tokens: Dict[str, Dict] = {}  # token → {username, expires}
        self._init_default_roles()

    def _init_default_roles(self) -> None:
        """Initialize default role → permission mapping."""
        self.roles = {
            AccessLevel.VIEWER.value: {"data:read", "reports:view"},
            AccessLevel.USER.value: {
                "data:read", "data:write", "algorithms:run", "reports:view",
            },
            AccessLevel.ADMIN.value: {
                "data:read", "data:write", "algorithms:run", "reports:view",
                "users:manage", "policies:edit",
            },
            AccessLevel.SUPERUSER.value: {"*"},
        }

    def add_user(
        self, username: str, password: str, role: AccessLevel
    ) -> bool:
        """Add a user to the system.

        Args:
            username: Unique user identifier.
            password: User password (will be hashed before storage).
            role: Access level to assign.

        Returns:
            ``True`` if the user was created, ``False`` if the username
            already exists.

        Raises:
            ValueError: If *username* or *password* is empty.
        """
        if not username:
            raise ValueError("username must not be empty")
        if not password:
            raise ValueError("password must not be empty")
        if username in self.users:
            return False

        salt = os.urandom(32)
        password_hash = self._hash_password(password, salt)

        self.users[username] = {
            "password_hash": password_hash,
            "salt": salt.hex(),
            "role": role.value,
            "created": datetime.now().isoformat(),
            "last_access": None,
            "access_attempts": 0,
        }

        logger.info("User '%s' created with role '%s'.", username, role.value)
        return True

    def authenticate(
        self, username: str, password: str
    ) -> Tuple[bool, Optional[str]]:
        """Authenticate a user and return a time-limited token.

        Args:
            username: User identifier.
            password: User password.

        Returns:
            Tuple of ``(success, token)``.  *token* is ``None`` on failure.
        """
        if username not in self.users:
            self._log_access_attempt(
                username, "authentication", "failure", "user_not_found"
            )
            return False, None

        user = self.users[username]
        salt = bytes.fromhex(user["salt"])
        password_hash = self._hash_password(password, salt)

        if not hmac.compare_digest(user["password_hash"], password_hash):
            user["access_attempts"] += 1
            self._log_access_attempt(
                username, "authentication", "failure", "invalid_password"
            )
            return False, None

        user["access_attempts"] = 0
        user["last_access"] = datetime.now().isoformat()

        token = self._generate_token(username)
        self._log_access_attempt(
            username, "authentication", "success", "authenticated"
        )
        return True, token

    def validate_token(self, token: str) -> Optional[str]:
        """Validate an authentication token.

        Args:
            token: Token string previously returned by :meth:`authenticate`.

        Returns:
            The associated username if the token is valid and not expired,
            or ``None`` otherwise.
        """
        record = self._tokens.get(token)
        if record is None:
            return None
        if datetime.now() > record["expires"]:
            del self._tokens[token]
            return None
        return record["username"]

    def check_permission(self, username: str, permission: str) -> bool:
        """Check if a user has a specific permission.

        Args:
            username: User identifier.
            permission: Permission string (e.g. ``"data:read"``).

        Returns:
            ``True`` if the user has the requested permission.
        """
        if username not in self.users:
            return False

        role = self.users[username]["role"]
        permissions = self.roles.get(role, set())

        if "*" in permissions:
            return True

        return permission in permissions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_password(password: str, salt: bytes) -> str:
        """Hash a password with PBKDF2-HMAC-SHA256.

        Args:
            password: Plaintext password.
            salt: Per-user random salt (32 bytes recommended).

        Returns:
            Hex-encoded hash string.
        """
        return hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt, 100_000
        ).hex()

    def _generate_token(self, username: str) -> str:
        """Generate a time-limited authentication token.

        Args:
            username: User the token is associated with.

        Returns:
            URL-safe token string.
        """
        token = secrets.token_urlsafe(32)
        self._tokens[token] = {
            "username": username,
            "expires": datetime.now() + self._TOKEN_LIFETIME,
        }
        return token

    def _log_access_attempt(
        self, username: str, action: str, status: str, details: str
    ) -> None:
        """Log an access attempt to the audit trail."""
        self.access_log.append({
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "action": action,
            "status": status,
            "details": details,
        })

    def get_user_info(self, username: str) -> Optional[Dict]:
        """Get user information (excluding sensitive fields).

        Args:
            username: User identifier.

        Returns:
            Dictionary with user metadata, or ``None`` if not found.
        """
        if username not in self.users:
            return None

        user = self.users[username].copy()
        user.pop("password_hash", None)
        user.pop("salt", None)
        return user


class EncryptionManager:
    """Manages symmetric encryption and decryption of sensitive data.

    Uses ``Fernet`` (AES-128-CBC with HMAC-SHA256) from the
    ``cryptography`` package.  Encryption is *reversible* — ciphertext
    can be decrypted back to the original plaintext using the same key.

    Raises:
        ImportError: On any encrypt/decrypt call when ``cryptography``
            is not installed.

    Attributes:
        keys: Mapping of key_id → key metadata (Fernet key stored internally).
        encrypted_store: Mapping of data_id → encrypted record.
    """

    def __init__(self) -> None:
        """Initialize encryption manager."""
        self.keys: Dict[str, Dict] = {}
        self.encrypted_store: Dict[str, Dict] = {}

    @staticmethod
    def _require_cryptography() -> None:
        """Raise ImportError if the ``cryptography`` package is missing."""
        if not HAS_CRYPTOGRAPHY:
            raise ImportError(
                "The 'cryptography' package is required for encryption. "
                "Install it with: pip install cryptography"
            )

    def generate_key(self, key_id: str) -> str:
        """Generate a Fernet encryption key.

        Args:
            key_id: Unique key identifier.

        Returns:
            The generated Fernet key (URL-safe base64-encoded string).

        Raises:
            ImportError: If ``cryptography`` is not installed.
            ValueError: If *key_id* is empty or already exists.
        """
        self._require_cryptography()

        if not key_id:
            raise ValueError("key_id must not be empty")
        if key_id in self.keys:
            raise ValueError(f"Key '{key_id}' already exists")

        key = Fernet.generate_key()

        self.keys[key_id] = {
            "key": key,
            "created": datetime.now().isoformat(),
            "algorithm": "Fernet (AES-128-CBC + HMAC-SHA256)",
        }

        logger.info("Encryption key '%s' generated.", key_id)
        return key.decode()

    def encrypt_data(self, data: str, key_id: str) -> Tuple[bool, str]:
        """Encrypt a string using the specified key.

        Args:
            data: Plaintext string to encrypt.
            key_id: Identifier of a previously generated key.

        Returns:
            Tuple of ``(True, data_id)`` on success, where *data_id*
            can be used with :meth:`decrypt_data` to retrieve the
            original plaintext.  Returns ``(False, "")`` if the key
            does not exist.

        Raises:
            ImportError: If ``cryptography`` is not installed.
        """
        self._require_cryptography()

        if key_id not in self.keys:
            return False, ""

        fernet = Fernet(self.keys[key_id]["key"])
        ciphertext = fernet.encrypt(data.encode())

        data_id = secrets.token_hex(8)
        self.encrypted_store[data_id] = {
            "ciphertext": ciphertext,
            "key_id": key_id,
            "created": datetime.now().isoformat(),
        }

        return True, data_id

    def decrypt_data(
        self, data_id: str, key_id: str
    ) -> Tuple[bool, Optional[str]]:
        """Decrypt previously encrypted data.

        Args:
            data_id: Identifier returned by :meth:`encrypt_data`.
            key_id: Key identifier used during encryption.

        Returns:
            Tuple of ``(True, plaintext)`` on success, or
            ``(False, None)`` if the *data_id* is unknown or the
            *key_id* does not match.

        Raises:
            ImportError: If ``cryptography`` is not installed.
        """
        self._require_cryptography()

        record = self.encrypted_store.get(data_id)
        if record is None:
            return False, None

        if record["key_id"] != key_id:
            return False, None

        if key_id not in self.keys:
            return False, None

        fernet = Fernet(self.keys[key_id]["key"])
        plaintext = fernet.decrypt(record["ciphertext"]).decode()
        return True, plaintext

    def rotate_key(self, key_id: str) -> Tuple[bool, Optional[str]]:
        """Rotate an encryption key.

        Generates a new key and marks the old key as rotated.

        Note:
            Existing data encrypted with the old key is **not**
            re-encrypted automatically.  Callers should decrypt and
            re-encrypt if needed.

        Args:
            key_id: Identifier of the key to rotate.

        Returns:
            Tuple of ``(True, new_key_id)`` on success, or
            ``(False, None)`` if the key does not exist.
        """
        if key_id not in self.keys:
            return False, None

        new_key_id = f"{key_id}_rotated_{int(datetime.now().timestamp())}"
        self.generate_key(new_key_id)

        self.keys[key_id]["rotated"] = datetime.now().isoformat()
        self.keys[key_id]["rotated_to"] = new_key_id

        logger.info("Key '%s' rotated to '%s'.", key_id, new_key_id)
        return True, new_key_id

    def get_key_info(self, key_id: str) -> Optional[Dict]:
        """Get key metadata (without the actual key material).

        Args:
            key_id: Key identifier.

        Returns:
            Dictionary with key metadata, or ``None`` if not found.
        """
        if key_id not in self.keys:
            return None

        info = self.keys[key_id].copy()
        info.pop("key", None)
        return info


class SecurityMonitor:
    """Monitors security events and detects anomalies.

    Tracks suspicious activities, unauthorized access attempts,
    and security policy violations.

    Attributes:
        alert_threshold: Number of failed attempts before an alert is raised.
        security_events: Chronological list of security events.
        alerts: List of generated security alerts.
    """

    def __init__(self, alert_threshold: int = 5) -> None:
        """Initialize security monitor.

        Args:
            alert_threshold: Number of failed attempts before generating
                an alert.  Must be positive.

        Raises:
            ValueError: If *alert_threshold* is not positive.
        """
        if alert_threshold <= 0:
            raise ValueError(
                f"alert_threshold must be positive, got {alert_threshold}"
            )
        self.alert_threshold = alert_threshold
        self.security_events: List[Dict] = []
        self.alerts: List[Dict] = []

    def log_security_event(
        self,
        event_type: str,
        source: str,
        details: str,
        severity: str = "info",
    ) -> None:
        """Log a security event.

        Args:
            event_type: Type of security event (e.g. ``"auth_failure"``).
            source: Source of event (user, system, IP, etc.).
            details: Human-readable event details.
            severity: One of ``"info"``, ``"warning"``, ``"critical"``.
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "source": source,
            "details": details,
            "severity": severity,
        }
        self.security_events.append(event)

        if severity == "critical":
            self._generate_alert(event)

    def detect_anomalies(
        self, user: str, max_failed_attempts: int = 5
    ) -> List[Dict]:
        """Detect anomalous behaviour for a user.

        Args:
            user: User identifier to analyse.
            max_failed_attempts: Threshold for excessive failures.

        Returns:
            List of detected anomaly records.
        """
        anomalies: List[Dict] = []

        user_events = [e for e in self.security_events if e["source"] == user]
        failed_attempts = sum(
            1 for e in user_events if "failed" in e["event_type"].lower()
        )

        if failed_attempts > max_failed_attempts:
            anomaly = {
                "type": "excessive_failed_attempts",
                "user": user,
                "count": failed_attempts,
                "timestamp": datetime.now().isoformat(),
            }
            anomalies.append(anomaly)
            self._generate_alert(anomaly)

        return anomalies

    def _generate_alert(self, event: Dict) -> None:
        """Generate a security alert from an event."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "alert_level": "HIGH",
        }
        self.alerts.append(alert)
        logger.warning("Security alert: %s", event)

    def get_security_report(self) -> Dict:
        """Get a summary security report.

        Returns:
            Dictionary with event/alert counts and recent entries.
        """
        return {
            "total_events": len(self.security_events),
            "total_alerts": len(self.alerts),
            "recent_events": self.security_events[-10:],
            "recent_alerts": self.alerts[-5:],
            "timestamp": datetime.now().isoformat(),
        }

    def clear_events(self) -> None:
        """Clear the security event history."""
        self.security_events = []
