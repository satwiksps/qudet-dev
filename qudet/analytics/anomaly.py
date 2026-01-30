
import numpy as np
from sklearn.svm import OneClassSVM
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorSampler as Sampler
from qudet.core.base import BaseQuantumEstimator

class QuantumKernelAnomalyDetector(BaseQuantumEstimator):
    """
    Detects anomalies by mapping data to a high-dimensional Quantum Hilbert Space.
    Uses a Precomputed Quantum Kernel fed into a One-Class SVM.
    """

    def __init__(self, n_qubits: int, nu: float = 0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.nu = nu
        self.svm_ = OneClassSVM(kernel="precomputed", nu=nu)
        self.train_data_ = None

    def _get_encoding_circuit(self, x_data):
        """Helper to create a simple Angle Encoding circuit."""
        qc = QuantumCircuit(self.n_qubits)
        for i in range(min(len(x_data), self.n_qubits)):
            qc.ry(x_data[i], i)
        return qc

    def _compute_kernel_matrix(self, X1, X2):
        """
        Computes the Fidelity Kernel: K(x, y) = |<psi(x)|psi(y)>|^2
        For MVP, we use simple statevector simulation (scales to ~20 qubits).
        """
        n_samples_1 = len(X1)
        n_samples_2 = len(X2)
        kernel_matrix = np.zeros((n_samples_1, n_samples_2))
        
        from qiskit.quantum_info import Statevector

        states_1 = []
        for x in X1:
            qc = self._get_encoding_circuit(x)
            states_1.append(Statevector.from_instruction(qc))
            
        states_2 = []
        if X2 is X1:
            states_2 = states_1
        else:
            for x in X2:
                qc = self._get_encoding_circuit(x)
                states_2.append(Statevector.from_instruction(qc))

        for i in range(n_samples_1):
            for j in range(n_samples_2):
                fidelity = np.abs(states_1[i].inner(states_2[j]))**2
                kernel_matrix[i, j] = fidelity
                
        return kernel_matrix

    def fit(self, X, y=None):
        """Compute the kernel matrix for training data and fit SVM."""
        self.train_data_ = X
        kernel_matrix = self._compute_kernel_matrix(X, X)
        self.svm_.fit(kernel_matrix)
        return self

    def predict(self, X):
        """Predict if new data is an anomaly."""
        kernel_matrix = self._compute_kernel_matrix(X, self.train_data_)
        return self.svm_.predict(kernel_matrix)
