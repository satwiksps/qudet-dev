"""
Comprehensive tests for composite and hybrid encoding methods.

Tests composite, layered, data reuse, adaptive, and hierarchical encoders.
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qudet.encoders.composite import (
    CompositeEncoder,
    LayeredEncoder,
    DataReuseEncoder,
    AdaptiveEncoder,
    HierarchicalEncoder
)


class TestCompositeEncoder:
    """Test composite encoding functionality."""

    def test_initialization(self):
        """Test composite encoder initialization."""
        encoder = CompositeEncoder(n_qubits=2)
        assert encoder.n_qubits == 2
        assert len(encoder.encoders) == 0

    def test_add_encoder(self):
        """Test adding encoders."""
        composite = CompositeEncoder(n_qubits=2)
        
        # Create mock encoder (minimal implementation)
        class MockEncoder:
            def encode(self, data):
                return QuantumCircuit(2)
        
        mock = MockEncoder()
        composite.add_encoder(mock)
        
        assert len(composite.encoders) == 1

    def test_encode_empty(self):
        """Test encoding with no encoders."""
        encoder = CompositeEncoder(n_qubits=2)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_get_encoder_info(self):
        """Test getting encoder information."""
        encoder = CompositeEncoder(n_qubits=2)
        
        class MockEncoder:
            def encode(self, data):
                return QuantumCircuit(2)
        
        encoder.add_encoder(MockEncoder())
        info = encoder.get_encoder_info()
        
        assert len(info) == 1


class TestLayeredEncoder:
    """Test layered encoding."""

    def test_initialization(self):
        """Test layered encoder initialization."""
        encoder = LayeredEncoder(n_qubits=2, n_layers=2)
        assert encoder.n_qubits == 2
        assert encoder.n_layers == 2

    def test_encode_linear_entangle(self):
        """Test encoding with linear entanglement."""
        encoder = LayeredEncoder(n_qubits=3, n_layers=2, entangle_type="linear")
        data = np.array([0.5, 0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 3

    def test_encode_full_entangle(self):
        """Test encoding with full entanglement."""
        encoder = LayeredEncoder(n_qubits=3, n_layers=2, entangle_type="full")
        data = np.array([0.5, 0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 3

    def test_encode_chain_entangle(self):
        """Test encoding with chain entanglement."""
        encoder = LayeredEncoder(n_qubits=3, n_layers=2, entangle_type="chain")
        data = np.array([0.5, 0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 3

    def test_single_layer(self):
        """Test with single layer."""
        encoder = LayeredEncoder(n_qubits=2, n_layers=1)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_many_layers(self):
        """Test with many layers."""
        encoder = LayeredEncoder(n_qubits=2, n_layers=5)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_set_num_layers(self):
        """Test setting number of layers."""
        encoder = LayeredEncoder(n_qubits=2, n_layers=2)
        encoder.set_num_layers(5)
        
        assert encoder.n_layers == 5


class TestDataReuseEncoder:
    """Test data reuse encoding."""

    def test_initialization(self):
        """Test data reuse encoder initialization."""
        encoder = DataReuseEncoder(n_qubits=4, n_reuses=2)
        assert encoder.n_qubits == 4
        assert encoder.n_reuses == 2

    def test_encode_basic(self):
        """Test basic data reuse encoding."""
        encoder = DataReuseEncoder(n_qubits=4, n_reuses=2)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 4

    def test_single_reuse(self):
        """Test with single reuse."""
        encoder = DataReuseEncoder(n_qubits=2, n_reuses=1)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_multiple_reuses(self):
        """Test with multiple reuses."""
        encoder = DataReuseEncoder(n_qubits=8, n_reuses=4)
        data = np.array([0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 8

    def test_empty_data(self):
        """Test with empty data."""
        encoder = DataReuseEncoder(n_qubits=4, n_reuses=2)
        data = np.array([])
        circuit = encoder.encode(data)
        
        # Should create identity state with no rotations
        assert circuit.num_qubits == 4


class TestAdaptiveEncoder:
    """Test adaptive encoding."""

    def test_initialization(self):
        """Test adaptive encoder initialization."""
        encoder = AdaptiveEncoder(n_qubits=2)
        assert encoder.n_qubits == 2
        assert encoder.selected_strategy == "auto"

    def test_sparse_strategy(self):
        """Test sparse encoding strategy."""
        encoder = AdaptiveEncoder(n_qubits=3)
        data = np.array([1.0, 0.0, 2.0])
        circuit = encoder.encode(data, strategy="sparse")
        
        assert circuit.num_qubits == 3

    def test_dense_strategy(self):
        """Test dense encoding strategy."""
        encoder = AdaptiveEncoder(n_qubits=3)
        data = np.array([0.5, 0.5, 0.5])
        circuit = encoder.encode(data, strategy="dense")
        
        assert circuit.num_qubits == 3

    def test_normalized_strategy(self):
        """Test normalized encoding strategy."""
        encoder = AdaptiveEncoder(n_qubits=2)
        data = np.array([1.0, 2.0])
        circuit = encoder.encode(data, strategy="normalized")
        
        assert circuit.num_qubits == 2

    def test_auto_strategy(self):
        """Test auto strategy selection."""
        encoder = AdaptiveEncoder(n_qubits=2)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data, strategy="auto")
        
        assert circuit.num_qubits == 2

    def test_analyze_data(self):
        """Test data analysis."""
        encoder = AdaptiveEncoder(n_qubits=2)
        data = np.array([1.0, 2.0, 0.0])
        analysis = encoder.analyze_data(data)
        
        assert "sparsity" in analysis
        assert "mean" in analysis
        assert "std" in analysis
        assert "max" in analysis
        assert "norm" in analysis

    def test_sparsity_detection(self):
        """Test sparsity detection in analysis."""
        encoder = AdaptiveEncoder(n_qubits=2)
        sparse_data = np.array([1.0, 0.0, 0.0, 0.0])
        dense_data = np.array([1.0, 1.0, 1.0, 1.0])
        
        sparse_analysis = encoder.analyze_data(sparse_data)
        dense_analysis = encoder.analyze_data(dense_data)
        
        assert sparse_analysis["sparsity"] > dense_analysis["sparsity"]


class TestHierarchicalEncoder:
    """Test hierarchical encoding."""

    def test_initialization(self):
        """Test hierarchical encoder initialization."""
        encoder = HierarchicalEncoder(n_qubits=4, hierarchy_levels=2)
        assert encoder.n_qubits == 4
        assert encoder.hierarchy_levels == 2

    def test_encode_two_levels(self):
        """Test encoding with two levels."""
        encoder = HierarchicalEncoder(n_qubits=4, hierarchy_levels=2)
        data = np.array([0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 4

    def test_encode_three_levels(self):
        """Test encoding with three levels."""
        encoder = HierarchicalEncoder(n_qubits=8, hierarchy_levels=3)
        data = np.array([0.5, 0.5, 0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 8

    def test_single_level(self):
        """Test with single level."""
        encoder = HierarchicalEncoder(n_qubits=2, hierarchy_levels=1)
        data = np.array([0.5])
        circuit = encoder.encode(data)
        
        assert circuit.num_qubits == 2

    def test_get_hierarchy_info(self):
        """Test getting hierarchy information."""
        encoder = HierarchicalEncoder(n_qubits=6, hierarchy_levels=3)
        info = encoder.get_hierarchy_info()
        
        assert info["levels"] == 3
        assert "qubits_per_level" in info
        assert "gate_types" in info


# Integration tests
class TestCompositeIntegration:
    """Integration tests for composite encoders."""

    def test_layered_with_varied_data(self):
        """Test layered encoder with varied data."""
        encoder = LayeredEncoder(n_qubits=3, n_layers=2)
        
        test_data = [
            np.array([0.5, 0.5, 0.5]),
            np.array([1.0, 0.0, 0.5]),
            np.array([0.1, 0.2, 0.3])
        ]
        
        for data in test_data:
            circuit = encoder.encode(data)
            assert circuit.num_qubits == 3

    def test_adaptive_strategy_selection(self):
        """Test adaptive encoder strategy selection."""
        encoder = AdaptiveEncoder(n_qubits=4)
        
        sparse_data = np.array([1.0, 0.0, 0.0, 0.0])
        dense_data = np.array([0.5, 0.5, 0.5, 0.5])
        
        circuit_sparse = encoder.encode(sparse_data)
        circuit_dense = encoder.encode(dense_data)
        
        assert circuit_sparse.num_qubits == 4
        assert circuit_dense.num_qubits == 4

    def test_hierarchical_scaling(self):
        """Test hierarchical encoder with different scales."""
        for n_levels in [1, 2, 3, 4]:
            encoder = HierarchicalEncoder(n_qubits=2**n_levels, hierarchy_levels=n_levels)
            data = np.random.randn(n_levels)
            circuit = encoder.encode(data)
            
            assert circuit.num_qubits == 2**n_levels

    def test_encoder_composition(self):
        """Test composing multiple encoding strategies."""
        composite = CompositeEncoder(n_qubits=2)
        
        # Test with empty encoders
        data = np.array([0.5, 0.5])
        circuit = composite.encode(data)
        
        assert circuit.num_qubits == 2
