import pytest
import numpy as np
from qudet.analytics.vqe import VariationalQuantumEigensolver, QAOA


class TestVariationalQuantumEigensolver:
    """Test suite for VQE."""

    def test_initialization(self):
        """Test VQE initialization."""
        vqe = VariationalQuantumEigensolver(
            backend_name="aer_simulator",
            shots=1024,
            ansatz_depth=3,
            learning_rate=0.01,
            max_iterations=50
        )
        assert vqe.ansatz_depth == 3
        assert vqe.learning_rate == 0.01
        assert vqe.max_iterations == 50

    def test_fit_with_hamiltonian(self):
        """Test fitting VQE with Hamiltonian."""
        H = np.array([[1, 0.5], [0.5, -1]])
        vqe = VariationalQuantumEigensolver(max_iterations=5)
        
        vqe.fit(H)
        
        assert vqe.optimal_params is not None
        assert len(vqe.history) == 5
        assert all(isinstance(e, float) for e in vqe.history)

    def test_predict(self):
        """Test prediction after fitting."""
        H = np.array([[2, 0], [0, -2]])
        vqe = VariationalQuantumEigensolver(max_iterations=3)
        vqe.fit(H)
        
        result = vqe.predict()
        
        assert len(result) == 1
        assert isinstance(result[0], (float, np.floating))

    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error before fitting."""
        vqe = VariationalQuantumEigensolver()
        
        with pytest.raises(ValueError):
            vqe.predict()

    def test_custom_initial_params(self):
        """Test VQE with custom initial parameters."""
        H = np.eye(2)
        initial_params = np.ones(3)
        vqe = VariationalQuantumEigensolver(ansatz_depth=1, max_iterations=2)
        
        vqe.fit(H, initial_params=initial_params)
        
        assert vqe.optimal_params is not None


class TestQAOA:
    """Test suite for QAOA."""

    def test_initialization(self):
        """Test QAOA initialization."""
        qaoa = QAOA(layers=2, max_iterations=20)
        assert qaoa.layers == 2
        assert qaoa.max_iterations == 20

    def test_fit_with_cost_matrix(self):
        """Test fitting QAOA with cost matrix."""
        cost_matrix = np.array([[1, 2], [2, 3]])
        qaoa = QAOA(layers=1, max_iterations=5)
        
        qaoa.fit(cost_matrix)
        
        assert qaoa.optimal_solution is not None

    def test_predict(self):
        """Test QAOA prediction."""
        cost_matrix = np.random.rand(3, 3)
        qaoa = QAOA(layers=1, max_iterations=3)
        qaoa.fit(cost_matrix)
        
        result = qaoa.predict()
        
        assert result is not None
        assert len(result) == 2  # 2 * layers

    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error before fitting."""
        qaoa = QAOA()
        
        with pytest.raises(ValueError):
            qaoa.predict()

    def test_custom_initial_params(self):
        """Test QAOA with custom initial parameters."""
        cost_matrix = np.eye(2)
        initial_params = np.ones(2)
        qaoa = QAOA(layers=1, max_iterations=2)
        
        qaoa.fit(cost_matrix, initial_params=initial_params)
        
        assert qaoa.optimal_solution is not None
