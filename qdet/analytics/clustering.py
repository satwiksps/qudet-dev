
import numpy as np
from sklearn.base import ClusterMixin
from qdet.core.base import BaseQuantumEstimator

class QuantumKMeans(BaseQuantumEstimator, ClusterMixin):
    """
    K-Means clustering that uses a Quantum Kernel to estimate distances.
    Points are clustered based on their similarity in Hilbert Space.
    """
    
    def __init__(self, n_clusters: int = 3, n_qubits: int = 4, max_iter: int = 10):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_qubits = n_qubits
        self.max_iter = max_iter
        self.centroids_ = None
        self.labels_ = None

    def _quantum_distance(self, x1, x2):
        """
        Calculates distance in Hilbert Space: d(u,v) = sqrt(2 - 2*|<u|v>|^2)
        Uses the 'Swap Test' logic or simple overlap for MVP.
        """
        from qiskit.quantum_info import Statevector
        from qiskit import QuantumCircuit
        
        qc1 = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(x1):
            if i < self.n_qubits: qc1.ry(val, i)
            
        qc2 = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(x2):
            if i < self.n_qubits: qc2.ry(val, i)
            
        sv1 = Statevector.from_instruction(qc1)
        sv2 = Statevector.from_instruction(qc2)
        
        overlap = np.abs(sv1.inner(sv2))**2
        return np.sqrt(2 - 2 * overlap)

    def fit(self, X, y=None):
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids_ = X.iloc[indices].values if hasattr(X, 'iloc') else X[indices]
        
        for iteration in range(self.max_iter):
            print(f"--- Q-Means Iteration {iteration+1}/{self.max_iter} ---")
            
            labels = []
            for row in X.values if hasattr(X, 'iloc') else X:
                distances = [self._quantum_distance(row, c) for c in self.centroids_]
                labels.append(np.argmin(distances))
            
            self.labels_ = np.array(labels)
            
            new_centroids = []
            for k in range(self.n_clusters):
                cluster_points = X.values[self.labels_ == k] if hasattr(X, 'iloc') else X[self.labels_ == k]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis=0))
                else:
                    new_centroids.append(self.centroids_[k])
            
            if np.allclose(self.centroids_, new_centroids):
                print("--- Converged Early ---")
                break
                
            self.centroids_ = np.array(new_centroids)
            
        return self

    def predict(self, X):
        """Predict cluster assignment for X using quantum distances."""
        if self.centroids_ is None:
            raise RuntimeError("QuantumKMeans must be fit() before predict()")
        
        labels = []
        for row in X.values if hasattr(X, 'iloc') else X:
            distances = [self._quantum_distance(row, c) for c in self.centroids_]
            labels.append(np.argmin(distances))
        
        return np.array(labels)
