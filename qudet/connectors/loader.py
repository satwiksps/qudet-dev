
import pandas as pd
import numpy as np
from typing import Iterator, Tuple
from ..encoders.rotation import RotationEncoder
from ..encoders.statevector import StatevectorEncoder

class QuantumDataLoader:
    """
    An iterator that loads classical data, batches it, and yields 
    Ready-to-Run Quantum Circuits.
    
    Acts like PyTorch's DataLoader, but for Quantum Backends.
    """
    
    def __init__(self, data: pd.DataFrame, batch_size: int = 32, encoder_type: str = 'angle'):
        self.data = data
        self.batch_size = batch_size
        self.encoder_type = encoder_type
        self.n_features = data.shape[1]
        
        if encoder_type == 'angle':
            self.encoder = RotationEncoder(n_qubits=self.n_features)
        elif encoder_type == 'amplitude':
            self.encoder = StatevectorEncoder()
        else:
            raise ValueError("Unknown encoder type. Choose 'angle' or 'amplitude'.")

    def __iter__(self) -> Iterator[Tuple[np.ndarray, list]]:
        """
        Yields:
            (raw_batch_data, list_of_quantum_circuits)
        """
        n_samples = len(self.data)
        indices = np.arange(n_samples)
        
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_values = self.data.iloc[batch_indices].values
            
            circuits = []
            for row in batch_values:
                circuits.append(self.encoder.encode(row))
            
            yield batch_values, circuits

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
