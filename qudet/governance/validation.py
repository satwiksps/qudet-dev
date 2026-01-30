
import numpy as np
from typing import Tuple

def check_quantum_capacity(data_shape: Tuple[int, int], max_qubits: int = 127) -> bool:
    """
    Checks if the dataset fits within the qubit limit of the target hardware.
    
    Parameters
    ----------
    data_shape : (n_samples, n_features)
    max_qubits : int (default 127 for IBM Brisbane)
    
    Returns
    -------
    bool : True if it fits, raises ValueError otherwise.
    """
    n_features = data_shape[1]
    
    if n_features > max_qubits:
        raise ValueError(
            f"Dataset has {n_features} features, but hardware only has {max_qubits} qubits. "
            "Please use qudet.reduction.RandomProjector to reduce dimensions first."
        )
    
    return True
