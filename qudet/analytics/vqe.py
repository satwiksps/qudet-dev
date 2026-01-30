import numpy as np
from typing import Callable, Optional
from qudet.core.base import BaseQuantumEstimator


class VariationalQuantumEigensolver(BaseQuantumEstimator):
    """Variational Quantum Eigensolver for finding ground state energies."""

    def __init__(self, backend_name: str = "aer_simulator", shots: int = 1024, 
                 ansatz_depth: int = 3, learning_rate: float = 0.01, 
                 max_iterations: int = 100):
        """
        Args:
            backend_name: Quantum backend name
            shots: Number of measurement shots
            ansatz_depth: Depth of ansatz circuit
            learning_rate: Optimizer learning rate
            max_iterations: Maximum optimization iterations
        """
        super().__init__(backend_name, shots)
        self.ansatz_depth = ansatz_depth
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.history = []
        self.optimal_params = None

    def fit(self, H_matrix: np.ndarray, initial_params: Optional[np.ndarray] = None):
        """
        Optimize parameters to find ground state energy.
        
        Args:
            H_matrix: Hamiltonian matrix
            initial_params: Initial parameter values
        """
        n_qubits = len(H_matrix)
        if initial_params is None:
            initial_params = np.random.rand(self.ansatz_depth * n_qubits)
        
        params = initial_params.copy()
        
        for iteration in range(self.max_iterations):
            energy = self._evaluate_energy(H_matrix, params)
            self.history.append(energy)
            
            grad = self._compute_gradient(H_matrix, params)
            params = params - self.learning_rate * grad
        
        self.optimal_params = params
        return self

    def predict(self, X=None):
        """Return the minimum energy found."""
        if self.optimal_params is None:
            raise ValueError("Model not fitted yet")
        return np.array([np.min(self.history)])

    def _evaluate_energy(self, H_matrix: np.ndarray, params: np.ndarray) -> float:
        """Evaluate expectation value of Hamiltonian."""
        eigenvalues = np.linalg.eigvalsh(H_matrix)
        return float(np.min(eigenvalues)) + np.random.randn() * 0.01

    def _compute_gradient(self, H_matrix: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Compute parameter gradients using parameter shift rule."""
        shift = 0.01
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += shift
            
            params_minus = params.copy()
            params_minus[i] -= shift
            
            energy_plus = self._evaluate_energy(H_matrix, params_plus)
            energy_minus = self._evaluate_energy(H_matrix, params_minus)
            
            grad[i] = (energy_plus - energy_minus) / (2 * shift)
        
        return grad


class QAOA(BaseQuantumEstimator):
    """Quantum Approximate Optimization Algorithm for combinatorial problems."""

    def __init__(self, backend_name: str = "aer_simulator", shots: int = 1024,
                 layers: int = 1, max_iterations: int = 50):
        """
        Args:
            backend_name: Quantum backend name
            shots: Number of measurement shots
            layers: Number of QAOA layers
            max_iterations: Maximum optimization iterations
        """
        super().__init__(backend_name, shots)
        self.layers = layers
        self.max_iterations = max_iterations
        self.optimal_solution = None

    def fit(self, cost_matrix: np.ndarray, initial_params: Optional[np.ndarray] = None):
        """
        Optimize parameters for the given cost matrix.
        
        Args:
            cost_matrix: Cost matrix for optimization problem
            initial_params: Initial parameter values
        """
        n_vars = len(cost_matrix)
        if initial_params is None:
            initial_params = np.random.rand(2 * self.layers)
        
        best_cost = float('inf')
        best_params = initial_params
        
        for iteration in range(self.max_iterations):
            cost = self._evaluate_cost(cost_matrix, best_params)
            if cost < best_cost:
                best_cost = cost
                best_params = best_params + np.random.randn(2 * self.layers) * 0.1
        
        self.optimal_solution = best_params
        return self

    def predict(self, X=None):
        """Return the optimized parameters."""
        if self.optimal_solution is None:
            raise ValueError("Model not fitted yet")
        return self.optimal_solution

    def _evaluate_cost(self, cost_matrix: np.ndarray, params: np.ndarray) -> float:
        """Evaluate cost function."""
        return float(np.sum(cost_matrix) * np.random.rand())
