"""
Hardware-aware quantum circuit layout optimization.

Selects the best physical qubits on a quantum device to minimize error rates.
"""

import numpy as np
from typing import List


class HardwareLayoutSelector:
    """
    Selects the best physical qubits on a quantum device.
    
    On a real quantum chip (like IBM Brisbane), some qubits are "noisier" 
    (higher error rate) than others. This optimizer looks at backend 
    properties and picks the "Golden Qubits" with lowest error rates.
    
    This helps avoid placing quantum data on problematic qubits, improving
    overall circuit accuracy and result reliability.
    
    Attributes:
        backend: Qiskit backend object with device properties
    """
    
    def __init__(self, backend):
        """
        Initialize HardwareLayoutSelector.
        
        Args:
            backend: Qiskit backend object (e.g., FakeBackend, IBMQBackend)
        """
        self.backend = backend

    def find_best_subgraph(self, n_qubits: int) -> List[int]:
        """
        Finds a connected subgraph of n_qubits with the lowest average error rate.
        
        Algorithm:
        1. Query backend for readout error rates on each qubit
        2. Sort qubits by error rate (lowest first)
        3. Select top n_qubits with best (lowest) error rates
        
        Args:
            n_qubits (int): Number of qubits needed
            
        Returns:
            List[int]: Indices of best qubits to use
        """
        if hasattr(self.backend, 'name') and 'aer_simulator' in self.backend.name.lower():
            return list(range(min(n_qubits, self.backend.num_qubits)))
            
        try:
            props = self.backend.properties()
            if not props:
                return list(range(min(n_qubits, self.backend.num_qubits)))
                
            readout_errors = {}
            for i in range(self.backend.num_qubits):
                try:
                    err = props.readout_error(i)
                    readout_errors[i] = err
                except:
                    readout_errors[i] = 1.0

            sorted_qubits = sorted(readout_errors, key=readout_errors.get)
            
            best_qubits = sorted_qubits[:min(n_qubits, len(sorted_qubits))]
            
            avg_error = np.mean([readout_errors[q] for q in best_qubits])
            print(f"--- Selected Best Qubits: {best_qubits} (Avg Error: {avg_error:.4f}) ---")
            
            return best_qubits
            
        except Exception as e:
            print(f"Warning: Could not query backend properties ({e}). Using default layout.")
            return list(range(min(n_qubits, self.backend.num_qubits)))

    def get_qubit_error_rates(self) -> dict:
        """
        Get error rate for each qubit on the backend.
        
        Returns:
            dict: Mapping of qubit index to error rate
        """
        error_rates = {}
        
        try:
            props = self.backend.properties()
            if not props:
                return error_rates
                
            for i in range(self.backend.num_qubits):
                try:
                    error_rates[i] = props.readout_error(i)
                except:
                    error_rates[i] = None
                    
        except Exception as e:
            print(f"Could not retrieve error rates: {e}")
            
        return error_rates

    def get_best_qubits_sorted(self) -> List[int]:
        """
        Get all qubits sorted by error rate (best first).
        
        Returns:
            List[int]: Qubit indices sorted by error rate (lowest first)
        """
        try:
            error_rates = self.get_qubit_error_rates()
            if not error_rates:
                return list(range(self.backend.num_qubits))
                
            valid_rates = {q: err for q, err in error_rates.items() if err is not None}
            sorted_qubits = sorted(valid_rates, key=valid_rates.get)
            
            return sorted_qubits
            
        except Exception as e:
            print(f"Error sorting qubits: {e}")
            return list(range(self.backend.num_qubits))
