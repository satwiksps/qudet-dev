"""
Quantum error mitigation and simplified noise modelling.

Provides tools for calibrating error mitigation, applying zero-noise
extrapolation, and simulating simplified noise effects on state vectors.

Note:
    The noise models in this module are **simplified pedagogical
    approximations**.  For production-grade noise simulation use Qiskit Aer's
    ``NoiseModel`` or equivalent libraries.
"""

import numpy as np
from typing import Dict, List


class QuantumErrorMitigation:
    """Mitigate errors in quantum circuit execution results.

    Supports zero-noise extrapolation (ZNE) as the primary mitigation
    strategy. The workflow is:

    1. **Calibrate** with known test circuits and their ideal results.
    2. **Mitigate** noisy measurement outcomes using the learned error rate.

    Note:
        This is a simplified ZNE implementation suitable for quick
        prototyping. For rigorous error mitigation consider dedicated
        libraries such as *Mitiq*.

    Example:
        >>> mitigator = QuantumErrorMitigation()
        >>> mitigator.calibrate(test_circuits, ideal_results)
        >>> clean = mitigator.mitigate(noisy_results)
    """

    def __init__(self, mitigation_method: str = "zero_noise_extrapolation") -> None:
        """Initialize the error mitigator.

        Args:
            mitigation_method: Error mitigation technique to use.
                Currently only ``'zero_noise_extrapolation'`` is supported.
        """
        self.mitigation_method = mitigation_method
        self.calibration_data = None

    def calibrate(self, test_circuits: List[Dict], results: np.ndarray) -> "QuantumErrorMitigation":
        """Calibrate the mitigator using test circuits and known-good results.

        Args:
            test_circuits: List of test circuit specification dictionaries.
            results: Ideal / reference measurement results as a NumPy array.

        Returns:
            ``self``, for method chaining.
        """
        self.calibration_data = {
            'test_circuits': len(test_circuits),
            'error_rate': float(np.mean(1.0 - results)),
            'method': self.mitigation_method
        }
        return self

    def mitigate(self, noisy_results: np.ndarray) -> np.ndarray:
        """Apply error mitigation to noisy measurement results.

        Args:
            noisy_results: Noisy measurement results as a NumPy array.

        Returns:
            Mitigated results (clipped to [0, 1]).

        Raises:
            ValueError: If ``calibrate()`` has not been called yet.
        """
        if self.calibration_data is None:
            raise ValueError("Must calibrate before mitigating")

        if self.mitigation_method == "zero_noise_extrapolation":
            return self._zne_mitigation(noisy_results)
        else:
            return noisy_results

    def _zne_mitigation(self, results: np.ndarray) -> np.ndarray:
        """Zero-noise extrapolation mitigation (simplified)."""
        error_rate = self.calibration_data['error_rate']
        correction_factor = 1.0 / (1.0 - error_rate) if error_rate < 1.0 else 1.0
        mitigated = results * correction_factor
        return np.clip(mitigated, 0.0, 1.0)


class QuantumNoiseModel:
    """Simplified quantum noise model for pedagogical use.

    Provides depolarizing and amplitude-damping noise channels that can be
    applied to state vectors at specified qubit positions.

    Note:
        These are **simplified approximations** that operate directly on
        state-vector amplitudes.  They do not implement full Kraus-operator
        or density-matrix noise channels.  For production noise simulation
        use ``qiskit_aer.noise.NoiseModel``.

    Example:
        >>> noise = QuantumNoiseModel("depolarizing", error_rate=0.02)
        >>> noisy_state = noise.apply_noise(state_vector, qubit_index=1)
    """

    def __init__(self, noise_type: str = "depolarizing", error_rate: float = 0.01) -> None:
        """Initialize the noise model.

        Args:
            noise_type: Type of noise channel. Supported values:
                ``'depolarizing'``, ``'amplitude_damping'``.
            error_rate: Noise strength parameter in the range [0, 1].

        Raises:
            ValueError: If *error_rate* is outside [0, 1].
        """
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError(
                f"error_rate must be in [0, 1], got {error_rate}"
            )
        self.noise_type = noise_type
        self.error_rate = error_rate

    def apply_noise(self, state_vector: np.ndarray, qubit_index: int = 0) -> np.ndarray:
        """Apply noise to a state vector at the specified qubit.

        This is a **simplified model**: the noise is localised to amplitude
        pairs associated with ``qubit_index`` rather than using full
        Kraus operators.

        Args:
            state_vector: Input state vector of length 2^n.
            qubit_index: Index of the qubit to apply noise to (0-indexed).

        Returns:
            Noisy state vector (normalised).

        Raises:
            ValueError: If *qubit_index* is out of range for the state vector.
        """
        n_qubits = int(np.log2(len(state_vector)))
        if qubit_index < 0 or qubit_index >= n_qubits:
            raise ValueError(
                f"qubit_index {qubit_index} out of range for "
                f"{n_qubits}-qubit state vector"
            )

        if self.noise_type == "depolarizing":
            return self._apply_depolarizing_noise(state_vector, qubit_index)
        elif self.noise_type == "amplitude_damping":
            return self._apply_amplitude_damping(state_vector, qubit_index)
        else:
            return state_vector

    def _apply_depolarizing_noise(
        self, state: np.ndarray, qubit_index: int
    ) -> np.ndarray:
        """Apply simplified depolarizing noise to a specific qubit.

        For each basis-state pair that differs only at ``qubit_index``,
        randomly swap the amplitudes with probability ``error_rate``.
        """
        noisy_state = state.copy().astype(complex)
        n_qubits = int(np.log2(len(state)))

        for i in range(len(state)):
            # Find the paired index that differs at qubit_index
            j = i ^ (1 << (n_qubits - 1 - qubit_index))
            if i < j and np.random.rand() < self.error_rate:
                noisy_state[i], noisy_state[j] = noisy_state[j], noisy_state[i]

        norm = np.linalg.norm(noisy_state)
        return noisy_state / norm if norm > 0 else noisy_state

    def _apply_amplitude_damping(
        self, state: np.ndarray, qubit_index: int
    ) -> np.ndarray:
        """Apply simplified amplitude-damping noise to a specific qubit.

        Damps the |1⟩ component of the target qubit towards |0⟩.
        """
        damped_state = state.copy().astype(complex)
        n_qubits = int(np.log2(len(state)))
        gamma = self.error_rate

        for i in range(len(state)):
            # Check if qubit_index is |1⟩ in basis state i
            if (i >> (n_qubits - 1 - qubit_index)) & 1:
                damped_state[i] *= np.sqrt(1.0 - gamma)

        norm = np.linalg.norm(damped_state)
        return damped_state / norm if norm > 0 else damped_state

    def get_noise_parameters(self) -> Dict:
        """Return the noise model parameters.

        Returns:
            Dictionary with ``'type'`` and ``'error_rate'`` keys.
        """
        return {
            'type': self.noise_type,
            'error_rate': self.error_rate
        }


class QuantumCalibrationalAnalyzer:
    """Analyse and store quantum hardware calibration data.

    Tracks per-backend gate error rates and readout errors, and provides
    a simple fidelity estimator for circuits of a given gate count.

    Example:
        >>> analyzer = QuantumCalibrationalAnalyzer()
        >>> analyzer.store_calibration("ibm_brisbane", cal_data)
        >>> fidelity = analyzer.estimate_circuit_fidelity("ibm_brisbane", n_gates=50)
    """

    def __init__(self) -> None:
        """Initialize the calibration analyzer."""
        self.calibration_data: Dict[str, Dict] = {}
        self.gate_errors: Dict[str, Dict] = {}
        self.readout_errors: Dict[str, float] = {}

    def store_calibration(self, backend_name: str, cal_data: Dict) -> None:
        """Store calibration data for a backend.

        Args:
            backend_name: Name of the quantum backend.
            cal_data: Calibration data dictionary. Expected keys include
                ``'single_qubit_error'``, ``'two_qubit_error'``, and
                ``'readout_error'``.
        """
        self.calibration_data[backend_name] = cal_data
        self._extract_gate_errors(backend_name, cal_data)
        self._extract_readout_errors(backend_name, cal_data)

    def _extract_gate_errors(self, backend_name: str, cal_data: Dict) -> None:
        """Extract single- and two-qubit gate errors."""
        single = cal_data.get('single_qubit_error', 1e-3)
        two = cal_data.get('two_qubit_error', 5e-3)
        self.gate_errors[backend_name] = {
            'single_qubit': single,
            'two_qubit': two,
            'average': (single + two) / 2
        }

    def _extract_readout_errors(self, backend_name: str, cal_data: Dict) -> None:
        """Extract readout error rates."""
        self.readout_errors[backend_name] = cal_data.get('readout_error', 1e-2)

    def get_gate_error(self, backend_name: str, gate_type: str = 'average') -> float:
        """Get the gate error rate for a backend.

        Args:
            backend_name: Backend name.
            gate_type: One of ``'single_qubit'``, ``'two_qubit'``, or
                ``'average'``.

        Returns:
            Gate error rate.
        """
        return self.gate_errors.get(backend_name, {}).get(gate_type, 0.01)

    def get_readout_error(self, backend_name: str) -> float:
        """Get the readout error rate for a backend.

        Args:
            backend_name: Backend name.

        Returns:
            Readout error rate.
        """
        return self.readout_errors.get(backend_name, 0.01)

    def estimate_circuit_fidelity(self, backend_name: str, n_gates: int) -> float:
        """Estimate circuit fidelity given the number of gates.

        Uses a simple independent-error model:
        ``fidelity ≈ (1 - gate_error)^n_gates * (1 - readout_error)``.

        Args:
            backend_name: Backend name.
            n_gates: Number of gates in the circuit.

        Returns:
            Estimated fidelity in [0, 1].
        """
        gate_error = self.get_gate_error(backend_name, 'average')
        readout_error = self.get_readout_error(backend_name)

        circuit_error = 1.0 - (1.0 - gate_error) ** n_gates
        total_error = circuit_error + readout_error * (1.0 - circuit_error)

        return max(0.0, 1.0 - total_error)
