"""
Composite and hybrid encoding methods combining multiple encoding strategies.

Provides flexible composition of different encoding techniques to create
rich quantum feature representations.
"""

import numpy as np
from typing import Union, Optional, List, Callable, Dict
from qiskit import QuantumCircuit
from qudet.core.base import BaseEncoder


class CompositeEncoder(BaseEncoder):
    """
    Combines multiple encoding strategies sequentially.
    
    Allows composition of different encoders to create more expressive
    quantum feature representations through combination.
    
    Best for: Complex data with diverse feature characteristics.
    """
    
    def __init__(self, n_qubits: int, encoders: Optional[List[BaseEncoder]] = None):
        """
        Initialize composite encoder.
        
        Args:
            n_qubits: Number of qubits
            encoders: List of encoder objects to combine
        """
        self.n_qubits = n_qubits
        self.encoders = encoders or []

    def add_encoder(self, encoder: BaseEncoder) -> None:
        """
        Add an encoder to the composition.
        
        Args:
            encoder: Encoder to add
        """
        self.encoders.append(encoder)

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encode data using all composed encoders sequentially.
        
        Args:
            data: Input feature vector
            
        Returns:
            QuantumCircuit with composite encoding
        """
        if not self.encoders:
            qc = QuantumCircuit(self.n_qubits)
            return qc
        
        # Start with first encoder
        qc = self.encoders[0].encode(data)
        
        # Ensure correct number of qubits
        if qc.num_qubits < self.n_qubits:
            extra_qubits = self.n_qubits - qc.num_qubits
            qc.add_register(QuantumCircuit(extra_qubits))
        
        # Apply subsequent encoders
        for encoder in self.encoders[1:]:
            qc_partial = encoder.encode(data)
            
            # Extend with extra qubits if needed
            if qc_partial.num_qubits < self.n_qubits:
                extra = self.n_qubits - qc_partial.num_qubits
                qc_partial.add_register(QuantumCircuit(extra))
            
            qc = qc.compose(qc_partial)
        
        return qc

    def get_encoder_info(self) -> List[Dict]:
        """Get information about composed encoders."""
        return [
            {"type": type(enc).__name__}
            for enc in self.encoders
        ]


class LayeredEncoder(BaseEncoder):
    """
    Applies encoding in multiple layers with entanglement between layers.
    
    Creates a layered structure where data is encoded in each layer,
    then layers are connected through entanglement patterns.
    
    Best for: Deep feature representation, hierarchical data encoding.
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 2,
                 entangle_type: str = "linear"):
        """
        Initialize layered encoder.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of encoding layers
            entangle_type: Type of entanglement ('linear', 'full', 'chain')
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangle_type = entangle_type

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encode data in layered structure.
        
        Args:
            data: Input feature vector
            
        Returns:
            QuantumCircuit with layered encoding
        """
        qc = QuantumCircuit(self.n_qubits)
        
        for layer in range(self.n_layers):
            # Feature encoding
            for i in range(min(len(data), self.n_qubits)):
                angle = data[i] * (1.0 + layer * 0.5)
                qc.ry(angle, i)
            
            # Entanglement pattern
            if self.entangle_type == "linear":
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
            elif self.entangle_type == "full":
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qc.cx(i, j)
            elif self.entangle_type == "chain":
                for i in range(0, self.n_qubits - 1, 2):
                    qc.cx(i, i + 1)
            
            # Phase accumulation
            for i in range(self.n_qubits):
                phase = (data[i % len(data)] if len(data) > 0 else 0) * layer
                qc.p(phase, i)
        
        return qc

    def set_num_layers(self, n_layers: int) -> None:
        """Set number of layers."""
        self.n_layers = n_layers


class DataReuseEncoder(BaseEncoder):
    """
    Encodes same data multiple times with different transformations.
    
    Reuses limited data by applying multiple transformations,
    effectively creating a richer feature space.
    
    Best for: Small datasets, data augmentation in quantum domain.
    """
    
    def __init__(self, n_qubits: int, n_reuses: int = 2):
        """
        Initialize data reuse encoder.
        
        Args:
            n_qubits: Number of qubits
            n_reuses: Number of times to reuse data with different transforms
        """
        self.n_qubits = n_qubits
        self.n_reuses = n_reuses

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encode data with multiple transformations.
        
        Args:
            data: Input feature vector
            
        Returns:
            QuantumCircuit with data reuse encoding
        """
        qc = QuantumCircuit(self.n_qubits)
        
        features_per_reuse = self.n_qubits // self.n_reuses
        
        for reuse_idx in range(self.n_reuses):
            start_qubit = reuse_idx * features_per_reuse
            end_qubit = min((reuse_idx + 1) * features_per_reuse, self.n_qubits)
            
            for qubit in range(start_qubit, end_qubit):
                # Handle empty data safely
                if len(data) > 0:
                    data_idx = (qubit - start_qubit) % len(data)
                    angle = data[data_idx] * (1.0 + reuse_idx * np.pi / self.n_reuses)
                else:
                    angle = 0.0
                
                if reuse_idx % 2 == 0:
                    qc.rx(angle, qubit)
                else:
                    qc.ry(angle, qubit)
        
        return qc


class AdaptiveEncoder(BaseEncoder):
    """
    Adapts encoding strategy based on data characteristics.
    
    Analyzes input data properties and selects or modifies
    encoding strategy accordingly.
    
    Best for: Heterogeneous data, adaptive quantum algorithms.
    """
    
    def __init__(self, n_qubits: int, strategies: Optional[Dict] = None):
        """
        Initialize adaptive encoder.
        
        Args:
            n_qubits: Number of qubits
            strategies: Dictionary of strategy names to strategy functions
        """
        self.n_qubits = n_qubits
        self.strategies = strategies or self._default_strategies()
        self.selected_strategy = "auto"

    def _default_strategies(self) -> Dict[str, Callable]:
        """Get default encoding strategies."""
        return {
            "sparse": self._encode_sparse,
            "dense": self._encode_dense,
            "normalized": self._encode_normalized,
            "auto": self._encode_auto
        }

    def encode(self, data: np.ndarray, strategy: Optional[str] = None) -> QuantumCircuit:
        """
        Encode data with adaptive strategy selection.
        
        Args:
            data: Input feature vector
            strategy: Force specific strategy or None for auto
            
        Returns:
            QuantumCircuit with adaptive encoding
        """
        use_strategy = strategy or self.selected_strategy
        
        if use_strategy not in self.strategies:
            use_strategy = "auto"
        
        return self.strategies[use_strategy](data)

    def _encode_sparse(self, data: np.ndarray) -> QuantumCircuit:
        """Encoding for sparse data."""
        qc = QuantumCircuit(self.n_qubits)
        
        nonzero_indices = np.where(data != 0)[0]
        
        for idx in nonzero_indices:
            if idx < self.n_qubits:
                qc.rx(data[idx], idx)
        
        return qc

    def _encode_dense(self, data: np.ndarray) -> QuantumCircuit:
        """Encoding for dense data."""
        qc = QuantumCircuit(self.n_qubits)
        
        # Normalize and apply to all qubits
        norm = np.linalg.norm(data)
        if norm > 0:
            normalized = data / norm
        else:
            normalized = data
        
        for i in range(self.n_qubits):
            data_idx = i % len(normalized) if len(normalized) > 0 else 0
            qc.ry(normalized[data_idx], i)
        
        # Add entanglement
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        
        return qc

    def _encode_normalized(self, data: np.ndarray) -> QuantumCircuit:
        """Encoding assuming normalized data."""
        qc = QuantumCircuit(self.n_qubits)
        
        for i in range(min(len(data), self.n_qubits)):
            qc.ry(data[i] * np.pi, i)
            qc.rz(data[i] * np.pi, i)
        
        return qc

    def _encode_auto(self, data: np.ndarray) -> QuantumCircuit:
        """Auto-select encoding strategy based on data."""
        sparsity = np.sum(data == 0) / len(data) if len(data) > 0 else 0
        
        if sparsity > 0.7:
            return self._encode_sparse(data)
        elif np.max(np.abs(data)) > 2:
            return self._encode_normalized(data)
        else:
            return self._encode_dense(data)

    def analyze_data(self, data: np.ndarray) -> Dict[str, float]:
        """
        Analyze data characteristics.
        
        Args:
            data: Input feature vector
            
        Returns:
            Dictionary with data statistics
        """
        return {
            "sparsity": float(np.sum(data == 0) / len(data)) if len(data) > 0 else 0.0,
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "max": float(np.max(np.abs(data))),
            "norm": float(np.linalg.norm(data))
        }


class HierarchicalEncoder(BaseEncoder):
    """
    Encodes data in hierarchical structure with coarse-to-fine approach.
    
    Encodes global features first, then local details,
    creating a multi-resolution representation.
    
    Best for: Hierarchical data, multi-scale features.
    """
    
    def __init__(self, n_qubits: int, hierarchy_levels: int = 2):
        """
        Initialize hierarchical encoder.
        
        Args:
            n_qubits: Number of qubits
            hierarchy_levels: Number of hierarchy levels
        """
        self.n_qubits = n_qubits
        self.hierarchy_levels = hierarchy_levels
        self.level_gates = ["rx", "ry", "rz"]

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encode data hierarchically.
        
        Args:
            data: Input feature vector
            
        Returns:
            QuantumCircuit with hierarchical encoding
        """
        qc = QuantumCircuit(self.n_qubits)
        
        qubits_per_level = self.n_qubits // self.hierarchy_levels
        
        for level in range(self.hierarchy_levels):
            start = level * qubits_per_level
            end = min((level + 1) * qubits_per_level, self.n_qubits)
            
            gate_type = self.level_gates[level % len(self.level_gates)]
            
            # Encode at this level
            for qubit in range(start, end):
                data_idx = (qubit - start) % len(data) if len(data) > 0 else 0
                angle = data[data_idx] / (level + 1)
                
                if gate_type == "rx":
                    qc.rx(angle, qubit)
                elif gate_type == "ry":
                    qc.ry(angle, qubit)
                elif gate_type == "rz":
                    qc.rz(angle, qubit)
            
            # Hierarchical entanglement
            step = 2 ** (self.hierarchy_levels - level - 1)
            for qubit in range(start, end - step, step):
                qc.cx(qubit, qubit + step)
        
        return qc

    def get_hierarchy_info(self) -> Dict:
        """Get hierarchy structure information."""
        return {
            "levels": self.hierarchy_levels,
            "qubits_per_level": self.n_qubits // self.hierarchy_levels,
            "gate_types": self.level_gates
        }
