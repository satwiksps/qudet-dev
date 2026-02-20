"""
Hardware-aware quantum circuit layout optimization.

Selects the best physical qubits on a quantum device to minimize error rates.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HardwareLayoutSelector:
    """Select the best physical qubits on a quantum device.

    On a real quantum chip (like IBM Brisbane), some qubits are noisier
    (higher error rate) than others.  This selector queries backend
    properties and picks the qubits with the lowest readout error rates.

    Attributes:
        backend: Qiskit backend object with device properties.

    Example:
        >>> selector = HardwareLayoutSelector(backend)
        >>> best = selector.find_best_subgraph(n_qubits=5)
    """

    def __init__(self, backend) -> None:
        """Initialize the hardware layout selector.

        Args:
            backend: Qiskit backend object (e.g., ``AerSimulator``,
                ``IBMBackend``).
        """
        self.backend = backend

    def find_best_subgraph(self, n_qubits: int) -> List[int]:
        """Find the *n_qubits* with the lowest average readout error rate.

        Algorithm:
            1. Query the backend for per-qubit readout error rates.
            2. Sort qubits by error rate (lowest first).
            3. Return the top *n_qubits*.

        Args:
            n_qubits: Number of physical qubits needed.

        Returns:
            Indices of the best qubits to use.
        """
        if hasattr(self.backend, 'name') and 'aer_simulator' in self.backend.name.lower():
            return list(range(min(n_qubits, self.backend.num_qubits)))

        try:
            props = self.backend.properties()
            if not props:
                return list(range(min(n_qubits, self.backend.num_qubits)))

            readout_errors: Dict[int, float] = {}
            for i in range(self.backend.num_qubits):
                try:
                    err = props.readout_error(i)
                    readout_errors[i] = err
                except Exception:
                    readout_errors[i] = 1.0

            sorted_qubits = sorted(readout_errors, key=readout_errors.get)

            best_qubits = sorted_qubits[:min(n_qubits, len(sorted_qubits))]

            avg_error = np.mean([readout_errors[q] for q in best_qubits])
            logger.info(
                "Selected best qubits: %s (avg error: %.4f)",
                best_qubits, avg_error,
            )

            return best_qubits

        except Exception as e:
            logger.warning(
                "Could not query backend properties (%s). Using default layout.", e
            )
            return list(range(min(n_qubits, self.backend.num_qubits)))

    def get_qubit_error_rates(self) -> Dict[int, Optional[float]]:
        """Get the readout error rate for every qubit on the backend.

        Returns:
            Mapping of qubit index to error rate (``None`` if unavailable).
        """
        error_rates: Dict[int, Optional[float]] = {}

        try:
            props = self.backend.properties()
            if not props:
                return error_rates

            for i in range(self.backend.num_qubits):
                try:
                    error_rates[i] = props.readout_error(i)
                except Exception:
                    error_rates[i] = None

        except Exception as e:
            logger.warning("Could not retrieve error rates: %s", e)

        return error_rates

    def get_best_qubits_sorted(self) -> List[int]:
        """Get all qubits sorted by error rate (best first).

        Returns:
            Qubit indices sorted by ascending readout error rate.
        """
        try:
            error_rates = self.get_qubit_error_rates()
            if not error_rates:
                return list(range(self.backend.num_qubits))

            valid_rates = {q: err for q, err in error_rates.items() if err is not None}
            sorted_qubits = sorted(valid_rates, key=valid_rates.get)

            return sorted_qubits

        except Exception as e:
            logger.warning("Error sorting qubits: %s", e)
            return list(range(self.backend.num_qubits))
