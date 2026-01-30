"""Tests for Hardware Layout Optimizer"""

import pytest
from qudet.compute.layout import HardwareLayoutSelector


class MockBackend:
    """Mock backend for testing."""
    def __init__(self, num_qubits=5, error_rates=None):
        self.num_qubits = num_qubits
        self.name = "mock_backend"
        self._error_rates = error_rates or {i: 0.01 * (i % 3 + 1) for i in range(num_qubits)}
    
    def properties(self):
        """Return mock properties."""
        return MockProperties(self._error_rates)


class MockProperties:
    """Mock properties object."""
    def __init__(self, error_rates):
        self._error_rates = error_rates
    
    def readout_error(self, qubit):
        return self._error_rates.get(qubit, 0.01)


class TestHardwareLayoutSelector:
    """Test suite for HardwareLayoutSelector."""
    
    @pytest.fixture
    def mock_backend(self):
        """Create mock backend."""
        return MockBackend(
            num_qubits=5,
            error_rates={0: 0.005, 1: 0.01, 2: 0.02, 3: 0.015, 4: 0.008}
        )
    
    def test_initialization(self, mock_backend):
        """Test initialization."""
        selector = HardwareLayoutSelector(mock_backend)
        assert selector.backend is mock_backend
    
    def test_find_best_subgraph(self, mock_backend):
        """Test finding best qubits."""
        selector = HardwareLayoutSelector(mock_backend)
        best_qubits = selector.find_best_subgraph(3)
        
        # Should return 3 qubits
        assert len(best_qubits) == 3
        # Should be valid qubit indices
        assert all(0 <= q < 5 for q in best_qubits)
    
    def test_find_best_subgraph_orders_by_error(self, mock_backend):
        """Test that qubits are ordered by error rate."""
        selector = HardwareLayoutSelector(mock_backend)
        best_qubits = selector.find_best_subgraph(5)
        
        # Get error rates
        error_rates = [mock_backend.properties().readout_error(q) for q in best_qubits]
        
        # Should be sorted (lowest error first)
        assert error_rates == sorted(error_rates)
    
    def test_get_qubit_error_rates(self, mock_backend):
        """Test getting all error rates."""
        selector = HardwareLayoutSelector(mock_backend)
        error_rates = selector.get_qubit_error_rates()
        
        assert len(error_rates) == 5
        assert all(0 <= q < 5 for q in error_rates.keys())
    
    def test_get_best_qubits_sorted(self, mock_backend):
        """Test getting sorted qubit list."""
        selector = HardwareLayoutSelector(mock_backend)
        sorted_qubits = selector.get_best_qubits_sorted()
        
        assert len(sorted_qubits) == 5
        
        # Get error rates and check ordering
        error_rates = [mock_backend.properties().readout_error(q) for q in sorted_qubits]
        assert error_rates == sorted(error_rates)
    
    def test_more_qubits_than_available(self, mock_backend):
        """Test requesting more qubits than available."""
        selector = HardwareLayoutSelector(mock_backend)
        best_qubits = selector.find_best_subgraph(10)
        
        # Should return what's available
        assert len(best_qubits) <= 5
