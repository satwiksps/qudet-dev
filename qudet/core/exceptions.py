class QuDETError(Exception):
    """Base exception class for QuDET library."""
    pass


class QuantumCapacityError(QuDETError):
    """Raised when data exceeds qubit limits."""
    pass


class DriftDetectedError(QuDETError):
    """Raised by governance checks when data drift is found."""
    pass
