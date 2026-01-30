"""
Amplitude encoding implementations for quantum feature space representation.

Amplitude encoding directly maps classical data into quantum state amplitudes,
providing efficient data loading for quantum algorithms.
"""

import numpy as np
from typing import Union, Optional, Tuple
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import CircuitInstruction
from qudet.core.base import BaseEncoder


class AmplitudeEncoder(BaseEncoder):
    """
    Encodes normalized data into quantum state amplitudes.
    
    Maps classical data vector directly into amplitudes of quantum state:
    |ψ⟩ = ∑_i amplitude_i |i⟩
    
    Requires data to be normalized to unit norm.
    Uses logarithmic number of qubits (2^n_qubits for n_qubits).
    
    Best for: Quantum state preparation, quantum machine learning, data loading.
    """
    
    def __init__(self, n_qubits: int, normalize: bool = True):
        """
        Initialize amplitude encoder.
        
        Args:
            n_qubits: Number of qubits for encoding
            normalize: Whether to normalize input data
        """
        self.n_qubits = n_qubits
        self.normalize = normalize
        self.max_features = 2 ** n_qubits

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encode classical data as quantum state amplitudes.
        
        Args:
            data: Feature vector of size ≤ 2^n_qubits
            
        Returns:
            QuantumCircuit with amplitude encoding
        """
        # Pad to exact size needed
        padded_data = np.zeros(self.max_features)
        if len(data) > 0:
            padded_data[:min(len(data), self.max_features)] = data[:self.max_features]
        
        # Normalize if requested (required for state initialization)
        if self.normalize or np.linalg.norm(padded_data) == 0:
            norm = np.linalg.norm(padded_data)
            if norm > 0:
                padded_data = padded_data / norm
            else:
                # Handle empty data - create |0...0⟩ state
                padded_data[0] = 1.0
        
        qc = QuantumCircuit(self.n_qubits)
        
        # Use basic state preparation (simplified approach)
        # In practice, would use more efficient circuits
        qc.initialize(padded_data, qc.qubits)
        
        return qc

    def get_features_supported(self) -> int:
        """Get maximum number of features that can be encoded."""
        return self.max_features


class DensityMatrixEncoder(BaseEncoder):
    """
    Encodes data using quantum density matrix representation.
    
    Captures both classical information and quantum correlations.
    Useful for encoding mixed quantum states.
    
    Best for: Quantum kernel methods, metric learning, correlation matrices.
    """
    
    def __init__(self, n_qubits: int, mixed_state: bool = False):
        """
        Initialize density matrix encoder.
        
        Args:
            n_qubits: Number of qubits
            mixed_state: Whether to allow mixed states
        """
        self.n_qubits = n_qubits
        self.mixed_state = mixed_state
        self.state_vector = None

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encode data into density matrix structure.
        
        Args:
            data: Input feature vector
            
        Returns:
            QuantumCircuit with density matrix encoding
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Normalize features
        norm = np.linalg.norm(data)
        if norm > 0:
            normalized = data / norm
        else:
            normalized = data
        
        # Encode as parametrized rotations creating coherence
        for i in range(min(len(normalized), self.n_qubits)):
            angle = np.pi * normalized[i]
            qc.ry(angle, i)
        
        # Add entanglement for correlation capture
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        
        return qc

    def compute_density_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Compute density matrix from encoding.
        
        Args:
            data: Input features
            
        Returns:
            Density matrix representation
        """
        # Simplified computation - in practice would use statevector
        dim = 2 ** self.n_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        
        # Create outer product of normalized data
        norm = np.linalg.norm(data)
        if norm > 0:
            psi = data / norm
            padded = np.zeros(dim)
            padded[:len(psi)] = psi
            rho = np.outer(padded, padded.conj())
        
        return rho


class BasisChangeEncoder(BaseEncoder):
    """
    Encodes data by applying basis transformations to quantum states.
    
    Enables encoding in different computational bases:
    - X-basis (Hadamard)
    - Y-basis (Y-rotation)
    - Z-basis (computational basis)
    
    Best for: Basis-adaptive encoding, flexibility in feature representation.
    """
    
    def __init__(self, n_qubits: int, basis: str = "z"):
        """
        Initialize basis change encoder.
        
        Args:
            n_qubits: Number of qubits
            basis: Measurement basis ('z', 'x', 'y')
        """
        self.n_qubits = n_qubits
        self.basis = basis.lower()
        if self.basis not in ['z', 'x', 'y']:
            raise ValueError("Basis must be 'z', 'x', or 'y'")

    def encode(self, data: np.ndarray, basis: Optional[str] = None) -> QuantumCircuit:
        """
        Encode data with basis transformation.
        
        Args:
            data: Input features
            basis: Override basis for encoding
            
        Returns:
            QuantumCircuit with basis encoding
        """
        use_basis = basis or self.basis
        qc = QuantumCircuit(self.n_qubits)
        
        # First, prepare initial state with rotation angles
        for i in range(min(len(data), self.n_qubits)):
            angle = np.pi * data[i] / (np.max(np.abs(data)) + 1e-10)
            qc.ry(angle, i)
        
        # Apply basis transformation
        if use_basis == 'x':
            qc.h(range(self.n_qubits))
        elif use_basis == 'y':
            qc.sdg(range(self.n_qubits))
            qc.h(range(self.n_qubits))
        # z-basis needs no transformation (computational basis)
        
        return qc

    def get_supported_bases(self) -> list:
        """Get list of supported measurement bases."""
        return ['z', 'x', 'y']


class FeatureMapEncoder(BaseEncoder):
    """
    General-purpose feature map encoder for flexible data encoding.
    
    Combines multiple encoding strategies:
    - Linear mapping
    - Polynomial mapping
    - Trigonometric mapping
    
    Best for: Adaptable encoding for various data types and distributions.
    """
    
    def __init__(self, n_qubits: int, mapping_type: str = "linear", 
                 power: int = 1, reps: int = 1):
        """
        Initialize feature map encoder.
        
        Args:
            n_qubits: Number of qubits
            mapping_type: Type of mapping ('linear', 'polynomial', 'trigonometric')
            power: Power for polynomial mapping
            reps: Number of repetitions
        """
        self.n_qubits = n_qubits
        self.mapping_type = mapping_type
        self.power = power
        self.reps = reps

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encode data using selected feature map.
        
        Args:
            data: Input features
            
        Returns:
            QuantumCircuit with feature map encoding
        """
        qc = QuantumCircuit(self.n_qubits)
        
        for rep in range(self.reps):
            # Apply Hadamard layer
            qc.h(range(self.n_qubits))
            
            # Apply feature-dependent rotations
            for i in range(min(len(data), self.n_qubits)):
                if self.mapping_type == "linear":
                    angle = data[i]
                elif self.mapping_type == "polynomial":
                    angle = (data[i] ** self.power)
                elif self.mapping_type == "trigonometric":
                    angle = np.sin(data[i])
                else:
                    angle = data[i]
                
                qc.rz(angle, i)
            
            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        
        return qc

    def apply_mapping(self, data: np.ndarray) -> np.ndarray:
        """
        Apply feature map transformation to classical data.
        
        Args:
            data: Input features
            
        Returns:
            Transformed features
        """
        if self.mapping_type == "linear":
            return data
        elif self.mapping_type == "polynomial":
            return np.power(data, self.power)
        elif self.mapping_type == "trigonometric":
            return np.sin(data)
        else:
            return data

    def get_mapping_info(self) -> dict:
        """Get information about current mapping."""
        return {
            "type": self.mapping_type,
            "power": self.power,
            "repetitions": self.reps,
            "n_qubits": self.n_qubits
        }
