"""Core exceptions for the QuDET library.

All QuDET-specific exceptions inherit from :class:`QuDETError`, making it easy
to catch any library-level error with a single ``except QuDETError`` clause.
"""


class QuDETError(Exception):
    """Base exception class for all QuDET errors."""


class QuantumCapacityError(QuDETError):
    """Raised when data dimensions exceed the capacity of the target qubit register."""


class DriftDetectedError(QuDETError):
    """Raised when statistical drift is detected between reference and new data."""


class EncodingError(QuDETError):
    """Raised when classical-to-quantum data encoding fails."""


class CircuitError(QuDETError):
    """Raised when a quantum circuit operation fails."""


class ValidationError(QuDETError):
    """Raised when input data fails validation checks."""


class BackendError(QuDETError):
    """Raised when a quantum backend cannot be initialized or is unavailable."""


class NotFittedError(QuDETError):
    """Raised when ``transform`` or ``predict`` is called before ``fit``."""
