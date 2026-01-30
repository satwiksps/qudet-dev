
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qudet.core.base import BaseEncoder

class RotationEncoder(BaseEncoder):
    """
    Encodes classical features into qubit rotation angles.
    Each feature x_i becomes a rotation R_y(x_i) on qubit i.
    """
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Creates a parameterized circuit for the input data.
        """
        qc = QuantumCircuit(self.n_qubits)
        
        params = ParameterVector('x', self.n_qubits)
        
        for i in range(self.n_qubits):
            qc.ry(params[i], i)
            
        return qc
