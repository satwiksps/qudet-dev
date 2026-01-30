
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qudet.core.base import BaseEncoder


class IQPEncoder(BaseEncoder):
    """
    Implements IQP (Instantaneous Quantum Polynomial) Encoding.
    Unlike Angle Encoding, this creates ENTANGLEMENT between features.
    
    It encodes data into:
    1. Single qubit rotations (Z-rotations based on x_i)
    2. Two-qubit entangling gates (ZZ-rotations based on x_i * x_j)
    
    Best for: Quantum Support Vector Machines (QSVM) and complex feature interactions.
    
    Mathematical Model:
    - Single-qubit: R_z(x_i) rotations
    - Entangling: R_zz(x_i * x_j) creates correlation structure
    - Multiple repetitions allow deeper feature mixing
    """
    
    def __init__(self, n_qubits: int, reps: int = 2):
        """
        Initialize IQP Encoder.
        
        Parameters
        ----------
        n_qubits : int
            Number of qubits in the encoding circuit
        reps : int
            Number of repetition layers (depth of circuit)
        """
        self.n_qubits = n_qubits
        self.reps = reps

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encodes classical data using IQP gates.
        
        Parameters
        ----------
        data : np.ndarray
            Input features to encode (1D array)
            
        Returns
        -------
        QuantumCircuit
            Parameterized quantum circuit encoding the data
        """
        qc = QuantumCircuit(self.n_qubits)
        
        n_features = min(len(data), self.n_qubits)
        
        for r in range(self.reps):
            qc.h(range(self.n_qubits))
            
            for i in range(n_features):
                qc.rz(data[i], i)
            
            for i in range(n_features - 1):
                interaction_strength = data[i] * data[i+1]
                qc.rzz(interaction_strength, i, i+1)

        return qc
