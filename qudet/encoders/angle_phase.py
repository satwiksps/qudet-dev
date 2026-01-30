"""
Advanced angle and phase encoding techniques for quantum feature embedding.

Provides sophisticated methods for encoding classical features into quantum
rotation angles and phase information.
"""

import numpy as np
from typing import Union, Optional, List
from qiskit import QuantumCircuit
from qudet.core.base import BaseEncoder


class AngleEncoder(BaseEncoder):
    """
    Encodes classical features into rotation angles of quantum gates.
    
    Maps features x_i to rotation angles:
    - R_x(x_i) rotations around X-axis
    - R_y(x_i) rotations around Y-axis  
    - R_z(x_i) rotations around Z-axis
    
    Best for: Simple feature encoding, fast feature loading, parametrized circuits.
    """
    
    def __init__(self, n_qubits: int, angle_type: str = "rx", reps: int = 1):
        """
        Initialize angle encoder.
        
        Args:
            n_qubits: Number of qubits
            angle_type: Type of rotation ('rx', 'ry', 'rz', 'auto')
            reps: Number of encoding repetitions
        """
        self.n_qubits = n_qubits
        self.angle_type = angle_type.lower()
        self.reps = reps
        
        if self.angle_type not in ['rx', 'ry', 'rz', 'auto']:
            raise ValueError("angle_type must be 'rx', 'ry', 'rz', or 'auto'")

    def encode(self, data: np.ndarray, angle_type: Optional[str] = None) -> QuantumCircuit:
        """
        Encode data into rotation angles.
        
        Args:
            data: Input feature vector
            angle_type: Override angle type for this encoding
            
        Returns:
            QuantumCircuit with angle-encoded data
        """
        atype = angle_type or self.angle_type
        qc = QuantumCircuit(self.n_qubits)
        
        for rep in range(self.reps):
            for i in range(min(len(data), self.n_qubits)):
                angle = data[i]
                
                if atype == "rx":
                    qc.rx(angle, i)
                elif atype == "ry":
                    qc.ry(angle, i)
                elif atype == "rz":
                    qc.rz(angle, i)
                elif atype == "auto":
                    # Use different axes for different qubits
                    rotation_choice = i % 3
                    if rotation_choice == 0:
                        qc.rx(angle, i)
                    elif rotation_choice == 1:
                        qc.ry(angle, i)
                    else:
                        qc.rz(angle, i)
        
        return qc

    def encode_scaled(self, data: np.ndarray, scale_factor: float = np.pi) -> QuantumCircuit:
        """
        Encode data with scaling factor.
        
        Args:
            data: Input features
            scale_factor: Scaling for angles
            
        Returns:
            QuantumCircuit with scaled angles
        """
        scaled_data = data * scale_factor
        return self.encode(scaled_data)


class PhaseEncoder(BaseEncoder):
    """
    Encodes data into quantum phase information.
    
    Uses controlled phase gates and phase shift operations to embed features.
    Phase encoding is efficient and preserves quantum coherence.
    
    Best for: Phase-sensitive applications, quantum interference, QAOA.
    """
    
    def __init__(self, n_qubits: int, global_phase: bool = False):
        """
        Initialize phase encoder.
        
        Args:
            n_qubits: Number of qubits
            global_phase: Whether to apply global phase (not measurable)
        """
        self.n_qubits = n_qubits
        self.global_phase = global_phase

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encode data into quantum phases.
        
        Args:
            data: Input feature vector
            
        Returns:
            QuantumCircuit with phase encoding
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Normalize data to [0, 2π]
        norm = np.max(np.abs(data))
        if norm > 0:
            normalized = (data / norm) * 2 * np.pi
        else:
            normalized = data
        
        # Apply phase gates
        for i in range(min(len(normalized), self.n_qubits)):
            qc.p(normalized[i], i)
        
        # Add entanglement with controlled phase
        for i in range(self.n_qubits - 1):
            qc.cp(normalized[i] * 0.5, i, i + 1)
        
        return qc

    def apply_global_phase(self, qc: QuantumCircuit, phase: float) -> QuantumCircuit:
        """
        Apply global phase to circuit.
        
        Args:
            qc: Input circuit
            phase: Phase to apply (in radians)
            
        Returns:
            Circuit with global phase applied
        """
        qc.global_phase = phase
        return qc


class HybridAnglePhaseEncoder(BaseEncoder):
    """
    Combines angle and phase encoding for comprehensive feature representation.
    
    Uses both rotation angles and phase information to encode features,
    providing richer feature space representation.
    
    Best for: Complex feature interactions, hybrid quantum-classical algorithms.
    """
    
    def __init__(self, n_qubits: int, angle_weight: float = 0.5, 
                 phase_weight: float = 0.5):
        """
        Initialize hybrid encoder.
        
        Args:
            n_qubits: Number of qubits
            angle_weight: Weight for angle encoding
            phase_weight: Weight for phase encoding
        """
        self.n_qubits = n_qubits
        self.angle_weight = angle_weight / (angle_weight + phase_weight)
        self.phase_weight = phase_weight / (angle_weight + phase_weight)

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encode data using hybrid angle-phase approach.
        
        Args:
            data: Input feature vector
            
        Returns:
            QuantumCircuit with hybrid encoding
        """
        qc = QuantumCircuit(self.n_qubits)
        
        for i in range(min(len(data), self.n_qubits)):
            # Apply angle encoding with weight
            angle = data[i] * self.angle_weight
            qc.ry(angle, i)
            
            # Apply phase encoding with weight
            phase = data[i] * self.phase_weight
            qc.p(phase, i)
        
        # Add entanglement
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        
        return qc

    def get_encoding_weights(self) -> dict:
        """Get current encoding weights."""
        return {
            "angle_weight": self.angle_weight,
            "phase_weight": self.phase_weight
        }


class MultiAxisRotationEncoder(BaseEncoder):
    """
    Encodes using combined rotations around multiple axes.
    
    Applies rotations around X, Y, and Z axes in sequence,
    creating a more expressive encoding space.
    
    Best for: Multi-axis feature spaces, rotational symmetry.
    """
    
    def __init__(self, n_qubits: int, axes: List[str] = None):
        """
        Initialize multi-axis encoder.
        
        Args:
            n_qubits: Number of qubits
            axes: List of axes to use ('x', 'y', 'z')
        """
        self.n_qubits = n_qubits
        self.axes = axes or ['x', 'y', 'z']
        
        for axis in self.axes:
            if axis not in ['x', 'y', 'z']:
                raise ValueError("Each axis must be 'x', 'y', or 'z'")

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encode data using multi-axis rotations.
        
        Args:
            data: Input feature vector
            
        Returns:
            QuantumCircuit with multi-axis encoding
        """
        qc = QuantumCircuit(self.n_qubits)
        
        n_features = min(len(data), self.n_qubits)
        
        # Cycle through axes for each feature
        for axis_idx, axis in enumerate(self.axes):
            for i in range(n_features):
                # Modulate angle based on axis position
                angle = data[i] * (1.0 + axis_idx * 0.25)
                
                if axis == 'x':
                    qc.rx(angle, i)
                elif axis == 'y':
                    qc.ry(angle, i)
                elif axis == 'z':
                    qc.rz(angle, i)
            
            # Entanglement after each axis
            for i in range(n_features - 1):
                qc.cx(i, i + 1)
        
        return qc

    def get_available_axes(self) -> List[str]:
        """Get available rotation axes."""
        return ['x', 'y', 'z']


class ParametricAngleEncoder(BaseEncoder):
    """
    Creates parametric angle encoding for variational circuits.
    
    Encodes data with trainable parameters that can be optimized,
    useful for hybrid quantum-classical machine learning.
    
    Best for: Variational quantum algorithms, parametric circuits, VQA.
    """
    
    def __init__(self, n_qubits: int, n_params: Optional[int] = None,
                 param_sharing: bool = False):
        """
        Initialize parametric encoder.
        
        Args:
            n_qubits: Number of qubits
            n_params: Number of parameters (default: n_qubits)
            param_sharing: Whether to share parameters across qubits
        """
        self.n_qubits = n_qubits
        self.n_params = n_params or n_qubits
        self.param_sharing = param_sharing
        self.parameters = np.random.randn(self.n_params) * 0.1

    def encode(self, data: np.ndarray, parameters: Optional[np.ndarray] = None) -> QuantumCircuit:
        """
        Encode data with parametric angles.
        
        Args:
            data: Input feature vector
            parameters: Optional parameter values to use
            
        Returns:
            QuantumCircuit with parametric encoding
        """
        params = parameters if parameters is not None else self.parameters
        qc = QuantumCircuit(self.n_qubits)
        
        # Feature encoding
        for i in range(min(len(data), self.n_qubits)):
            qc.ry(data[i], i)
        
        # Parametric rotations
        for i in range(self.n_qubits):
            param_idx = i % len(params) if self.param_sharing else min(i, len(params) - 1)
            qc.rz(params[param_idx], i)
        
        return qc

    def update_parameters(self, new_params: np.ndarray) -> None:
        """
        Update trainable parameters.
        
        Args:
            new_params: New parameter values
        """
        if len(new_params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(new_params)}")
        self.parameters = new_params

    def get_parameters(self) -> np.ndarray:
        """Get current parameters."""
        return self.parameters.copy()
