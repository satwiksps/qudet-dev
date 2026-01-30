
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qdet.core.base import BaseEncoder
from typing import Tuple


class DataIntegrityCheck:
    """
    Verifies that the Quantum Encoding process preserved data information.
    
    Philosophy:
    Data Integrity is the core of data engineering. How do you know your
    "Quantum Encoding" didn't lose information?
    
    This module performs a 'Round Trip' check:
    Data → Quantum State → Measured Probabilities → Compare with Original
    
    This is a simulation-only check (Unit Test helper), useful for:
    - Validating encoder implementations
    - Debugging encoding issues
    - Unit testing encoder correctness
    
    Example:
        >>> checker = DataIntegrityCheck()
        >>> is_valid = checker.verify_encoding(data, encoder, tolerance=1e-5)
        >>> stats = checker.compute_encoding_fidelity(data, encoder)
    """
    
    @staticmethod
    def verify_encoding(original_data: np.ndarray, encoder: BaseEncoder, tolerance: float = 1e-5) -> bool:
        """
        Checks if the data matches the encoded state information.
        
        Verification strategy depends on encoder type:
        
        - StatevectorEncoder: Data stored directly in state amplitudes
          Check: |ψ(data)⟩ amplitudes ≈ normalized(data)
          
        - IQPEncoder/RotationEncoder: Data stored in rotation angles
          Check: Encoding succeeds without errors
        
        Parameters
        ----------
        original_data : np.ndarray
            Original input data (1D array)
        encoder : BaseEncoder
            Encoder instance to verify
        tolerance : float
            Absolute tolerance for floating-point comparison (default: 1e-5)
            
        Returns
        -------
        bool
            True if data integrity verified, False otherwise
            
        Raises
        ------
        ValueError
            If integrity check fails with diagnostic information
        """
        try:
            qc = encoder.encode(original_data)
            state = Statevector(qc)
            probs = state.probabilities()
            
        except Exception as e:
            raise ValueError(f"Encoding failed: {str(e)}")
        
        encoder_name = encoder.__class__.__name__
        
        if "StatevectorEncoder" in encoder_name:
            try:
                norm_data = original_data / np.linalg.norm(original_data)
                squared_data = np.abs(norm_data) ** 2
                
                n = len(squared_data)
                
                if np.allclose(probs[:n], squared_data, atol=tolerance):
                    return True
                else:
                    expected_str = squared_data[:5]
                    actual_str = probs[:5]
                    raise ValueError(
                        f"Integrity Fail for StatevectorEncoder:\n"
                        f"  Expected probs (first 5): {expected_str}\n"
                        f"  Actual probs (first 5): {actual_str}\n"
                        f"  Tolerance: {tolerance}"
                    )
            except Exception as e:
                raise ValueError(f"StatevectorEncoder integrity check failed: {str(e)}")
        
        else:
            return True

    @staticmethod
    def compute_encoding_fidelity(original_data: np.ndarray, encoder: BaseEncoder) -> dict:
        """
        Computes detailed fidelity statistics of the encoding.
        
        Fidelity measures how well the quantum state represents the data:
        F = |⟨data|ψ⟩|² = overlap between data and quantum state
        
        Parameters
        ----------
        original_data : np.ndarray
            Original input data (1D array)
        encoder : BaseEncoder
            Encoder instance
            
        Returns
        -------
        dict
            Dictionary with fidelity statistics:
            - 'fidelity': Overall fidelity (0-1)
            - 'min_prob': Minimum state probability
            - 'max_prob': Maximum state probability
            - 'entropy': Shannon entropy of state
            - 'purity': State purity (1 = pure, < 1 = mixed)
        """
        qc = encoder.encode(original_data)
        state = Statevector(qc)
        probs = state.probabilities()
        
        encoder_name = encoder.__class__.__name__
        
        if "StatevectorEncoder" in encoder_name:
            norm_data = original_data / np.linalg.norm(original_data)
            squared_data = np.abs(norm_data) ** 2
            
            n = min(len(squared_data), len(probs))
            fidelity = np.sum(np.sqrt(squared_data[:n] * probs[:n]))
        else:
            fidelity = np.sum(probs ** 2)
        
        probs_nonzero = probs[probs > 0]
        entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero + 1e-10))
        
        purity = np.sum(probs ** 2)
        
        return {
            "encoder": encoder_name,
            "fidelity": float(fidelity),
            "min_probability": float(np.min(probs)),
            "max_probability": float(np.max(probs)),
            "shannon_entropy": float(entropy),
            "purity": float(purity),
            "num_qubits": encoder.n_qubits if hasattr(encoder, 'n_qubits') else None,
        }

    @staticmethod
    def verify_batch(data_batch: np.ndarray, encoder: BaseEncoder, tolerance: float = 1e-5) -> Tuple[int, int]:
        """
        Verifies integrity for a batch of samples.
        
        Parameters
        ----------
        data_batch : np.ndarray
            Batch of data samples (2D array, shape: n_samples x n_features)
        encoder : BaseEncoder
            Encoder instance
        tolerance : float
            Tolerance for verification
            
        Returns
        -------
        Tuple[int, int]
            (num_passed, num_failed) - Counts of passed and failed checks
        """
        num_passed = 0
        num_failed = 0
        
        for i, sample in enumerate(data_batch):
            try:
                DataIntegrityCheck.verify_encoding(sample, encoder, tolerance)
                num_passed += 1
            except Exception as e:
                num_failed += 1
                print(f"  Sample {i} failed: {str(e)[:100]}")
        
        print(f"--- Batch Verification: {num_passed}/{len(data_batch)} passed ---")
        return num_passed, num_failed
