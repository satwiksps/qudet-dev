
import numpy as np
from qiskit import QuantumCircuit
from qudet.core.base import BaseEncoder

class StatevectorEncoder(BaseEncoder):
    """
    Encodes a data vector into the amplitudes of a quantum state.
    Logarithmic compression: Encodes N features into log2(N) qubits.
    
    Constraint: Data vector must be normalized (L2 norm = 1).
    This class handles normalization automatically.
    """
    
    def __init__(self, n_qubits: int = None):
        self.n_qubits = n_qubits

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Input: 1D numpy array of features.
        Output: QuantumCircuit initialized to state |x>.
        """
        n_features = len(data)
        required_len = 2 ** np.ceil(np.log2(n_features)).astype(int)
        
        if len(data) < required_len:
            padded_data = np.pad(data, (0, required_len - len(data)))
        else:
            padded_data = data
            
        state_vector = self._normalize(padded_data)
        
        n_qubits = int(np.log2(len(state_vector)))
        
        qc = QuantumCircuit(n_qubits)
        qc.initialize(state_vector, range(n_qubits))
        
        return qc
