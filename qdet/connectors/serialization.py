"""
Quantum Circuit and Model Serialization.

Data pipelines crash. You need to save your work. This module saves 
Quantum Circuits (which are objects) into QASM (Quantum Assembly) or JSON 
so they can be stored in a database or S3.
"""

import json
import pickle
from typing import List, Union
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps, loads


class QuantumSerializer:
    """
    Handles saving/loading of QDET artifacts.
    
    Supports:
    - QASM (Quantum Assembly Language): Standard format for quantum circuits
    - JSON: Metadata and circuit descriptions
    - Pickle: Full Python objects (models, trained instances)
    
    This enables persistence of quantum computations for 
    reproducibility and pipeline recovery.
    """
    
    @staticmethod
    def save_circuits(circuits: List[QuantumCircuit], filepath: str) -> None:
        """
        Saves a list of circuits to a file.
        Format: JSON list of QASM strings.
        
        Args:
            circuits (List[QuantumCircuit]): List of quantum circuits to save
            filepath (str): Path to output JSON file
            
        Returns:
            None
        """
        data = []
        for i, qc in enumerate(circuits):
            qasm_str = dumps(qc)
                
            record = {
                "id": i,
                "n_qubits": qc.num_qubits,
                "n_clbits": qc.num_clbits,
                "name": qc.name,
                "qasm": qasm_str
            }
            data.append(record)
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(circuits)} circuits to {filepath}")

    @staticmethod
    def load_circuits(filepath: str) -> List[QuantumCircuit]:
        """
        Loads circuits from a QDET JSON file.
        
        Args:
            filepath (str): Path to JSON file with saved circuits
            
        Returns:
            List[QuantumCircuit]: List of loaded quantum circuits
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        circuits = []
        for record in data:
            qasm_str = record['qasm']
            try:
                qc = loads(qasm_str)
            except Exception as e:
                print(f"Warning: Could not load circuit {record['id']}: {e}")
                continue
                
            circuits.append(qc)
            
        print(f"Loaded {len(circuits)} circuits from {filepath}")
        return circuits

    @staticmethod
    def save_model(model, filepath: str) -> None:
        """
        Saves a trained QDET model (like QuantumKMeans) via Pickle.
        
        This allows you to save the entire fitted model state,
        including centroids, metadata, and fitted parameters.
        
        Args:
            model: Trained QDET model instance
            filepath (str): Path to output pickle file
            
        Returns:
            None
        """
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model to {filepath}")
            
    @staticmethod
    def load_model(filepath: str):
        """
        Loads a pickled QDET model.
        
        Args:
            filepath (str): Path to pickle file
            
        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded model from {filepath}")
        return model

    @staticmethod
    def export_circuit_qasm(circuit: QuantumCircuit, filepath: str) -> None:
        """
        Export a single circuit to QASM format.
        
        Args:
            circuit (QuantumCircuit): Circuit to export
            filepath (str): Output QASM file path
            
        Returns:
            None
        """
        qasm_str = dumps(circuit)
            
        with open(filepath, 'w') as f:
            f.write(qasm_str)
        print(f"Exported circuit to {filepath}")

    @staticmethod
    def import_circuit_qasm(filepath: str) -> QuantumCircuit:
        """
        Import a circuit from QASM format.
        
        Args:
            filepath (str): Path to QASM file
            
        Returns:
            QuantumCircuit: Imported circuit
        """
        with open(filepath, 'r') as f:
            qasm_str = f.read()
            
        circuit = loads(qasm_str)
        print(f"Imported circuit from {filepath}")
        return circuit
