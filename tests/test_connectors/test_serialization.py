"""Tests for Quantum I/O Module (Serialization)"""

import os
import json
import tempfile
import pytest
from qiskit import QuantumCircuit
from qudet.connectors.serialization import QuantumSerializer


class TestQuantumSerializer:
    """Test suite for QuantumSerializer class."""
    
    @pytest.fixture
    def sample_circuits(self):
        """Create sample quantum circuits."""
        circuits = []
        for i in range(3):
            qc = QuantumCircuit(2, name=f"circuit_{i}")
            qc.h(0)
            qc.cx(0, 1)
            circuits.append(qc)
        return circuits
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_save_circuits(self, sample_circuits, temp_dir):
        """Test saving circuits to JSON."""
        filepath = os.path.join(temp_dir, "circuits.json")
        
        QuantumSerializer.save_circuits(sample_circuits, filepath)
        
        # Check file exists
        assert os.path.exists(filepath)
        
        # Check contents
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 3
        for i, record in enumerate(data):
            assert record['id'] == i
            assert record['n_qubits'] == 2
            assert 'qasm' in record
    
    def test_load_circuits(self, sample_circuits, temp_dir):
        """Test loading circuits from JSON."""
        filepath = os.path.join(temp_dir, "circuits.json")
        
        # Save first
        QuantumSerializer.save_circuits(sample_circuits, filepath)
        
        # Load
        loaded = QuantumSerializer.load_circuits(filepath)
        
        assert len(loaded) == len(sample_circuits)
        assert all(isinstance(qc, QuantumCircuit) for qc in loaded)
        assert all(qc.num_qubits == 2 for qc in loaded)
    
    def test_save_load_roundtrip(self, sample_circuits, temp_dir):
        """Test save-load roundtrip preserves circuits."""
        filepath = os.path.join(temp_dir, "circuits.json")
        
        # Save
        QuantumSerializer.save_circuits(sample_circuits, filepath)
        
        # Load
        loaded = QuantumSerializer.load_circuits(filepath)
        
        # Compare
        for original, loaded_qc in zip(sample_circuits, loaded):
            assert loaded_qc.num_qubits == original.num_qubits
            assert loaded_qc.num_clbits == original.num_clbits
    
    def test_save_model(self, temp_dir):
        """Test saving a mock model."""
        filepath = os.path.join(temp_dir, "model.pkl")
        
        # Create a simple mock model
        mock_model = {"type": "QuantumKMeans", "n_clusters": 3, "centroids": [[1, 2], [3, 4]]}
        
        QuantumSerializer.save_model(mock_model, filepath)
        
        assert os.path.exists(filepath)
    
    def test_load_model(self, temp_dir):
        """Test loading a pickled model."""
        filepath = os.path.join(temp_dir, "model.pkl")
        
        # Save a model
        mock_model = {"type": "QuantumKMeans", "n_clusters": 3, "centroids": [[1, 2], [3, 4]]}
        QuantumSerializer.save_model(mock_model, filepath)
        
        # Load it
        loaded = QuantumSerializer.load_model(filepath)
        
        assert loaded == mock_model
    
    def test_save_load_model_roundtrip(self, temp_dir):
        """Test save-load roundtrip for models."""
        filepath = os.path.join(temp_dir, "model.pkl")
        
        original = {
            "type": "QuantumKMeans",
            "n_clusters": 3,
            "centroids": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "fitted": True
        }
        
        QuantumSerializer.save_model(original, filepath)
        loaded = QuantumSerializer.load_model(filepath)
        
        assert loaded == original
    
    def test_export_circuit_qasm(self, sample_circuits, temp_dir):
        """Test exporting a single circuit to QASM."""
        filepath = os.path.join(temp_dir, "circuit.qasm")
        
        QuantumSerializer.export_circuit_qasm(sample_circuits[0], filepath)
        
        assert os.path.exists(filepath)
        
        # Check it's QASM format
        with open(filepath, 'r') as f:
            content = f.read()
        assert "OPENQASM" in content or "qasm" in filepath
    
    def test_import_circuit_qasm(self, sample_circuits, temp_dir):
        """Test importing a circuit from QASM."""
        filepath = os.path.join(temp_dir, "circuit.qasm")
        
        # Export first
        QuantumSerializer.export_circuit_qasm(sample_circuits[0], filepath)
        
        # Import
        loaded = QuantumSerializer.import_circuit_qasm(filepath)
        
        assert isinstance(loaded, QuantumCircuit)
    
    def test_save_empty_circuit_list(self, temp_dir):
        """Test saving empty circuit list."""
        filepath = os.path.join(temp_dir, "empty.json")
        
        QuantumSerializer.save_circuits([], filepath)
        
        loaded = QuantumSerializer.load_circuits(filepath)
        assert loaded == []
    
    def test_save_circuits_with_measurements(self, temp_dir):
        """Test saving circuits with classical bits."""
        circuits = []
        for i in range(2):
            qc = QuantumCircuit(2, 2, name=f"measured_{i}")
            qc.h(0)
            qc.measure(range(2), range(2))
            circuits.append(qc)
        
        filepath = os.path.join(temp_dir, "measured.json")
        QuantumSerializer.save_circuits(circuits, filepath)
        
        loaded = QuantumSerializer.load_circuits(filepath)
        assert len(loaded) == 2
        assert all(qc.num_clbits == 2 for qc in loaded)
