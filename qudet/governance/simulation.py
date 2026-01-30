"""
Noise profilers for quantum hardware simulation and stress testing.

Generates realistic noise models to validate QuDET pipelines against
real quantum hardware conditions without accessing actual QPUs.
"""

import numpy as np
from typing import Optional

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
    HAS_AER = True
except ImportError:
    HAS_AER = False


class NoiseSimulator:
    """
    Generates realistic noise models to stress-test QuDET pipelines.
    
    Quantum hardware experiences various types of noise:
    - Depolarizing errors: Random bit/phase flips
    - Thermal relaxation: Energy loss over time (T1/T2 times)
    - Gate errors: Imperfect control pulses
    
    This class helps answer: "Will my algorithm survive on real hardware?"
    
    By testing on noisy simulators locally, you can:
    1. Identify weak points in your algorithm
    2. Add error mitigation techniques
    3. Estimate expected accuracy on real devices
    4. Benchmark against different noise profiles
    """
    
    @staticmethod
    def get_noisy_backend(error_prob: float = 0.01):
        """
        Returns a simulator with depolarizing noise.
        
        Depolarizing noise is the most common noise model:
        - Single-qubit gates: Random bit flip or phase flip
        - Two-qubit gates: Same but 10x worse (more complex)
        
        Args:
            error_prob (float): Probability of single-qubit gate error.
                Typical hardware: 0.001-0.01 (0.1%-1%)
                Default: 0.01 (1%)
                
        Returns:
            AerSimulator: Noisy backend configured with errors
            
        Raises:
            ImportError: If qiskit-aer not installed
        """
        if not HAS_AER:
            raise ImportError("qiskit-aer not installed. Run 'pip install qiskit-aer'.")
            
        noise_model = NoiseModel()
        
        error_1q = depolarizing_error(error_prob, 1)
        noise_model.add_all_qubit_quantum_error(
            error_1q, 
            ['u1', 'u2', 'u3', 'rx', 'ry', 'rz']
        )
        
        error_2q = depolarizing_error(error_prob * 10, 2)
        noise_model.add_all_qubit_quantum_error(
            error_2q, 
            ['cx', 'cz']
        )
        
        error_readout = depolarizing_error(error_prob * 5, 1)
        noise_model.add_all_qubit_quantum_error(error_readout, ['measure'])
        
        return AerSimulator(noise_model=noise_model)

    @staticmethod
    def get_thermal_backend(
        t1: float = 50e-6, 
        t2: float = 70e-6, 
        gate_time: float = 100e-9
    ):
        """
        Returns a simulator with T1/T2 thermal relaxation errors.
        
        Physical qubits lose energy (relax) over time:
        - T1: Relaxation time (energy decay)
        - T2: Dephasing time (phase randomization)
        
        Args:
            t1 (float): T1 relaxation time in seconds. Default: 50 microseconds
            t2 (float): T2 dephasing time in seconds. Default: 70 microseconds
            gate_time (float): Typical gate duration in seconds. Default: 100 nanoseconds
                
        Returns:
            AerSimulator: Noisy backend with thermal errors
            
        Raises:
            ImportError: If qiskit-aer not installed
        """
        if not HAS_AER:
            raise ImportError("qiskit-aer not installed. Run 'pip install qiskit-aer'.")
            
        noise_model = NoiseModel()
        
        thermal_error = thermal_relaxation_error(t1, t2, gate_time)
        noise_model.add_all_qubit_quantum_error(thermal_error, ['u3', 'rx', 'ry', 'rz'])
        
        error_2q = depolarizing_error(0.01, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
        
        return AerSimulator(noise_model=noise_model)

    @staticmethod
    def get_ibm_like_backend(error_prob: float = 0.005):
        """
        Returns a simulator mimicking IBM quantum hardware.
        
        IBM devices typically have:
        - Single-qubit error: 0.3%-1%
        - Two-qubit error: 1%-5%
        - Readout error: 2%-5%
        
        Args:
            error_prob (float): Base error probability. Default: 0.005 (0.5%)
            
        Returns:
            AerSimulator: IBM-like noisy backend
        """
        if not HAS_AER:
            raise ImportError("qiskit-aer not installed. Run 'pip install qiskit-aer'.")
            
        noise_model = NoiseModel()
        
        error_1q = depolarizing_error(error_prob, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3'])
        
        error_2q = depolarizing_error(error_prob * 15, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
        
        error_readout = depolarizing_error(error_prob * 8, 1)
        noise_model.add_all_qubit_quantum_error(error_readout, ['measure'])
        
        return AerSimulator(noise_model=noise_model)

    @staticmethod
    def get_high_noise_backend(error_prob: float = 0.05):
        """
        Returns a simulator with HIGH noise levels.
        
        Useful for stress testing: "Does this still work on a bad day?"
        
        Args:
            error_prob (float): High error probability. Default: 0.05 (5%)
            
        Returns:
            AerSimulator: High-noise backend
        """
        if not HAS_AER:
            raise ImportError("qiskit-aer not installed. Run 'pip install qiskit-aer'.")
            
        noise_model = NoiseModel()
        
        error_1q = depolarizing_error(error_prob, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
        
        error_2q = depolarizing_error(error_prob * 20, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
        
        error_readout = depolarizing_error(error_prob * 10, 1)
        noise_model.add_all_qubit_quantum_error(error_readout, ['measure'])
        
        return AerSimulator(noise_model=noise_model)

    @staticmethod
    def get_noiseless_backend():
        """
        Returns a noiseless simulator (ideal baseline).
        
        Useful for comparison: "How much does noise hurt performance?"
        
        Returns:
            AerSimulator: Perfect noiseless backend
        """
        if not HAS_AER:
            raise ImportError("qiskit-aer not installed. Run 'pip install qiskit-aer'.")
            
        return AerSimulator()

    @staticmethod
    def estimate_accuracy_degradation(
        baseline_accuracy: float,
        error_prob: float = 0.01
    ) -> float:
        """
        Rough estimate of accuracy loss due to noise.
        
        Rule of thumb: Each error_prob causes ~(error_prob * circuit_depth) accuracy loss
        
        Args:
            baseline_accuracy (float): Accuracy on noiseless simulator
            error_prob (float): Error probability per gate
            
        Returns:
            float: Estimated accuracy on noisy device
        """
        avg_circuit_depth = 30
        expected_loss = error_prob * avg_circuit_depth
        
        degraded_accuracy = baseline_accuracy * (1 - expected_loss)
        return max(0, degraded_accuracy)
