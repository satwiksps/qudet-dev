"""Quantum analytics module for QuDET.

Provides quantum machine-learning estimators for classification, regression,
clustering, anomaly detection, and feature selection.
"""

from .anomaly import QuantumKernelAnomalyDetector
from .classifier import QuantumSVC
from .clustering import QuantumKMeans
from .feature_select import QuantumFeatureSelector
from .regression import QuantumKernelRegressor

__all__ = [
    "QuantumKernelAnomalyDetector",
    "QuantumSVC",
    "QuantumKMeans",
    "QuantumFeatureSelector",
    "QuantumKernelRegressor",
]
