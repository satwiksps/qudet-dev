"""
Quantum Drift Detector using Maximum Mean Discrepancy (MMD).

Detects data drift by comparing reference and current distributions
in Hilbert space using quantum kernels.
"""

import numpy as np
from typing import Dict, Optional


class QuantumDriftDetector:
    """
    Detects Data Drift using Quantum Maximum Mean Discrepancy (MMD).
    
    Compares two datasets (Reference vs. Current) in Hilbert Space.
    If the 'Quantum Distance' exceeds threshold, drift is detected.
    
    This is critical for production ML pipelines where data distributions
    change over time (data drift / model rot).
    
    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits for quantum kernel computation. Default: 4
    threshold : float, optional
        MMD threshold for drift detection. Default: 0.1
        Higher threshold = fewer false alarms but misses real drift.
        
    Attributes
    ----------
    reference_data_ : np.ndarray or None
        Stored reference (training) data. Set by fit_reference().
    threshold : float
        MMD threshold for decision boundary.
        
    Examples
    --------
    >>> detector = QuantumDriftDetector(n_qubits=4, threshold=0.15)
    >>> detector.fit_reference(X_train)
    >>> result = detector.detect_drift(X_current)
    >>> if result['drift_detected']:
    ...     print("Data drift detected! Retrain model!")
    ...     print(f"MMD score: {result['mmd_score']:.4f}")
    
    References
    ----------
    Maximum Mean Discrepancy (MMD): Gretton et al., 2012
        Measures discrepancy between two distributions in Hilbert space.
        MMD^2 = E[K(x,x')] - 2E[K(x,y)] + E[K(y,y')]
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        threshold: float = 0.1
    ):
        """Initialize Quantum Drift Detector."""
        self.n_qubits = n_qubits
        self.threshold = threshold
        self.reference_data_ = None
        
        self._kernel_computer = None

    @property
    def kernel_computer(self):
        """Lazy-load quantum kernel computer."""
        if self._kernel_computer is None:
            from ..analytics.anomaly import QuantumKernelAnomalyDetector
            self._kernel_computer = QuantumKernelAnomalyDetector(n_qubits=self.n_qubits)
        return self._kernel_computer

    def fit_reference(self, X: np.ndarray) -> "QuantumDriftDetector":
        """
        Store the baseline (reference) data.
        
        Typically, this is your training data. The detector will compare
        new data against this reference to detect drift.
        
        Parameters
        ----------
        X : np.ndarray
            Reference data of shape (n_samples, n_features).
            
        Returns
        -------
        self
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        
        self.reference_data_ = X
        print(f"Reference data stored: {X.shape}")
        print(f"   • Samples: {X.shape[0]}")
        print(f"   • Features: {X.shape[1]}")
        
        return self

    def detect_drift(self, X_new: np.ndarray) -> Dict:
        """
        Calculate MMD distance between reference and new data.
        
        Maximum Mean Discrepancy (MMD) test:
            MMD^2 = E[K(x,x)] - 2E[K(x,y)] + E[K(y,y)]
        
        Where:
        - K(x,x): kernel between reference samples
        - K(y,y): kernel between new samples
        - K(x,y): kernel between reference and new samples
        
        Parameters
        ----------
        X_new : np.ndarray
            New data to test for drift, shape (n_samples, n_features).
            
        Returns
        -------
        dict
            Dictionary with keys:
            - "drift_detected" (bool): True if MMD > threshold
            - "mmd_score" (float): Computed MMD value
            - "threshold" (float): Decision threshold
            - "status" (str): "DRIFT" or "STABLE"
            
        Raises
        ------
        ValueError
            If reference data not set or X_new has wrong shape.
            
        Examples
        --------
        >>> result = detector.detect_drift(X_new)
        >>> print(f"Status: {result['status']}")
        >>> print(f"MMD Score: {result['mmd_score']:.4f}")
        """
        if self.reference_data_ is None:
            raise ValueError("Reference data not set. Call fit_reference() first.")
        
        if X_new.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_new.shape}")
        
        if X_new.shape[1] != self.reference_data_.shape[1]:
            raise ValueError(
                f"Feature mismatch: reference has {self.reference_data_.shape[1]}, "
                f"new data has {X_new.shape[1]}"
            )
        
        print(f"\nCalculating Quantum Drift (MMD)...")
        print(f"   • Reference samples: {self.reference_data_.shape[0]}")
        print(f"   • New samples: {X_new.shape[0]}")
        
        print("   • Computing K(reference, reference)...")
        k_xx = self.kernel_computer._compute_kernel_matrix(
            self.reference_data_,
            self.reference_data_
        )
        mean_xx = np.mean(k_xx)
        print(f"     → Mean: {mean_xx:.4f}")
        
        print("   • Computing K(new, new)...")
        k_yy = self.kernel_computer._compute_kernel_matrix(X_new, X_new)
        mean_yy = np.mean(k_yy)
        print(f"     → Mean: {mean_yy:.4f}")
        
        print("   • Computing K(reference, new)...")
        k_xy = self.kernel_computer._compute_kernel_matrix(
            self.reference_data_,
            X_new
        )
        mean_xy = np.mean(k_xy)
        print(f"     → Mean: {mean_xy:.4f}")
        
        mmd_score = mean_xx + mean_yy - (2 * mean_xy)
        
        mmd_score = max(0, mmd_score)
        
        is_drift = mmd_score > self.threshold
        
        result = {
            "drift_detected": bool(is_drift),
            "mmd_score": float(round(mmd_score, 4)),
            "threshold": self.threshold,
            "status": "DRIFT DETECTED" if is_drift else "STABLE",
            "reference_size": self.reference_data_.shape[0],
            "new_size": X_new.shape[0]
        }
        
        print(f"\nResults:")
        print(f"   • MMD Score: {result['mmd_score']:.4f}")
        print(f"   • Threshold: {result['threshold']:.4f}")
        print(f"   • Status: {result['status']}")
        
        return result

    def set_threshold(self, threshold: float) -> "QuantumDriftDetector":
        """
        Update the MMD threshold for drift detection.
        
        Parameters
        ----------
        threshold : float
            New threshold value.
            
        Returns
        -------
        self
        """
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")
        
        self.threshold = threshold
        print(f"Threshold updated to {threshold:.4f}")
        
        return self

    def get_config(self) -> Dict:
        """
        Return detector configuration.
        
        Returns
        -------
        dict
            Configuration parameters.
        """
        return {
            "n_qubits": self.n_qubits,
            "threshold": self.threshold,
            "reference_shape": self.reference_data_.shape if self.reference_data_ is not None else None
        }
