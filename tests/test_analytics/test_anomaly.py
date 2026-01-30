# File: tests/test_analytics/test_anomaly.py

import pytest
import numpy as np
from qudet.analytics.anomaly import QuantumKernelAnomalyDetector

def test_anomaly_detection_flow():
    """
    Test if the Quantum Anomaly Detector runs end-to-end without crashing.
    """
    # 1. Create dummy data (10 samples, 2 features)
    # 9 "Normal" points clustered around 0
    X_train = np.random.normal(0, 0.1, size=(10, 2))
    
    # 2. Initialize Detector
    detector = QuantumKernelAnomalyDetector(n_qubits=2, nu=0.1)
    
    # 3. Fit (This triggers the Quantum Kernel calculation)
    detector.fit(X_train)
    
    # 4. Predict on a clear outlier
    X_test = np.array([
        [0.0, 0.0],  # Normal
        [5.0, 5.0]   # Anomaly (far away)
    ])
    
    predictions = detector.predict(X_test)
    
    # 5. Assertions
    assert len(predictions) == 2
    # The SVM returns 1 for normal, -1 for anomaly
    # Note: With random data and small samples, accuracy varies, 
    # so we mostly check shape/type here for the MVP.
    assert isinstance(predictions, np.ndarray)
