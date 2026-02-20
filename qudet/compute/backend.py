"""
Quantum backend management.

Provides a factory for obtaining quantum backend instances, supporting both
local simulation via Qiskit Aer and cloud QPU access via qiskit-ibm-runtime.
"""

import logging
from typing import Optional

from qiskit_aer import AerSimulator

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
    HAS_IBM = True
except ImportError:
    HAS_IBM = False

logger = logging.getLogger(__name__)


class BackendManager:
    """Centralized controller for quantum hardware connections.

    Handles the switch between local simulation and cloud QPU (IBM).
    Uses the Factory Method pattern for backend creation with graceful
    fallback to simulator when cloud connectivity fails.

    Example:
        >>> backend = BackendManager.get_backend("simulator")
        >>> # or connect to real quantum computer:
        >>> backend = BackendManager.get_backend("ibm_brisbane", token="YOUR_API_TOKEN")
    """

    @staticmethod
    def get_backend(name: str = "simulator", token: Optional[str] = None):
        """Factory method to get a quantum backend instance.

        Args:
            name: Backend name. Use ``'simulator'`` for local Aer simulation,
                or an IBM backend name (e.g. ``'ibm_brisbane'``) for cloud QPU.
            token: IBM Quantum API token. If ``None``, looks for saved
                credentials via ``QiskitRuntimeService``.

        Returns:
            A quantum backend instance suitable for circuit execution.

        Raises:
            ImportError: If requesting an IBM backend but ``qiskit-ibm-runtime``
                is not installed.
        """
        logger.info("Connecting to backend: %s", name)

        if name == "simulator":
            return AerSimulator(method='statevector')

        if not HAS_IBM:
            raise ImportError(
                "qiskit-ibm-runtime is not installed. "
                "Install with: pip install qiskit-ibm-runtime"
            )

        try:
            if token:
                service = QiskitRuntimeService(channel="ibm_quantum", token=token)
            else:
                service = QiskitRuntimeService(channel="ibm_quantum")

            real_backend = service.backend(name)
            n_qubits = real_backend.num_qubits
            logger.info(
                "Connected to QPU: %s (%d qubits)", real_backend.name, n_qubits
            )
            return real_backend

        except Exception as e:
            logger.warning("Connection failed: %s — falling back to simulator", e)
            return AerSimulator()

    @staticmethod
    def optimize_level(backend_name: str) -> int:
        """Return the recommended transpilation optimization level for a backend.

        Optimization levels:
            - 0: No optimization (fastest transpile, may use more gates)
            - 1: Light optimization (balance speed/quality)
            - 2: Medium optimization
            - 3: Heavy optimization (slowest transpile, fewest gates)

        Args:
            backend_name: Name of the quantum backend.

        Returns:
            Recommended optimization level (0–3).
        """
        if "simulator" in backend_name.lower():
            return 1
        return 3
