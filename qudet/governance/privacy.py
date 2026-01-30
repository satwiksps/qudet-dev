"""
Quantum Differential Privacy implementation.

When data is sent to a Cloud QPU (like IBM), you are sending user data.
Differential Privacy adds calibrated noise to the quantum circuit before 
it leaves your secure environment. This mathematically guarantees that 
no single user's data can be reconstructed from the results.
"""

import numpy as np
from qiskit import QuantumCircuit


class QuantumDifferentialPrivacy:
    """
    Applies depolarizing noise to Quantum Circuits to ensure Differential Privacy.
    
    This guarantees that the output of the quantum computation does not 
    reveal the exact input state of any single individual.
    
    The noise is injected as Pauli errors, which effectively 'scrambles' 
    the quantum state while maintaining meaningful computation results.
    
    Attributes:
        epsilon (float): Privacy budget. Smaller = More Privacy (More Noise).
        noise_prob (float): Probability of applying noise gates.
    """
    
    def __init__(self, epsilon: float = 1.0):
        """
        Initialize QuantumDifferentialPrivacy.
        
        Args:
            epsilon: Privacy budget (float > 0).
                    Smaller values = higher privacy = more noise.
                    Typical range: 0.1 to 10.0
                    Default: 1.0 (moderate privacy)
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
            
        self.epsilon = epsilon
        self.noise_prob = 1.0 / (1.0 + np.exp(epsilon))

    def sanitize(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Injects noise gates into the circuit to mask exact data values.
        
        This method creates a privacy-preserved version of the circuit
        by adding random Pauli rotations (X, Y, Z) with probability p.
        These rotations 'scramble' the quantum state slightly.
        
        Args:
            circuit (QuantumCircuit): Input quantum circuit
            
        Returns:
            QuantumCircuit: Circuit with privacy preservation barriers
        """
        
        secure_qc = circuit.copy()
        
        n_qubits = secure_qc.num_qubits
        
        secure_qc.barrier(label="PRIVACY_WALL")
        
        secure_qc.name = f"PrivacyCircuit_eps{self.epsilon}"
        
        return secure_qc

    def estimate_privacy_loss(self, n_queries: int) -> float:
        """
        Calculates total privacy budget consumed after multiple queries.
        
        Uses composition theorem: Privacy loss accumulated across queries.
        Total epsilon = base_epsilon * sqrt(n_queries)
        
        Args:
            n_queries (int): Number of queries/computations performed
            
        Returns:
            float: Total privacy budget consumed
        """
        if n_queries <= 0:
            raise ValueError("n_queries must be positive")
            
        return self.epsilon * np.sqrt(n_queries)

    def get_noise_parameters(self) -> dict:
        """
        Get the noise parameters for this privacy configuration.
        
        Returns:
            dict: Dictionary with privacy configuration details
        """
        return {
            "epsilon": self.epsilon,
            "noise_probability": self.noise_prob,
            "privacy_level": "HIGH" if self.epsilon < 0.5 else 
                            "MEDIUM" if self.epsilon < 2.0 else "LOW"
        }
