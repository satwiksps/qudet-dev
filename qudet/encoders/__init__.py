"""Quantum data encoders for the QuDET library.

This package provides a variety of classical-to-quantum encoding strategies,
each mapping classical feature vectors into quantum circuits.

Modules:
    iqp: IQP (Instantaneous Quantum Polynomial) encoding with entanglement.
    rotation: Simple rotation-based encoding (R_y gates).
    statevector: Amplitude-based statevector encoding.
    amplitude: Amplitude and density-matrix encoders.
    angle_phase: Angle, phase, hybrid, and parametric encoders.
    composite: Composite, layered, adaptive, and hierarchical encoders.
"""

from .iqp import IQPEncoder
from .rotation import RotationEncoder
from .statevector import StatevectorEncoder

# Amplitude encoding
from .amplitude import (
    AmplitudeEncoder,
    DensityMatrixEncoder,
    BasisChangeEncoder,
    FeatureMapEncoder,
)

# Angle and phase encoding
from .angle_phase import (
    AngleEncoder,
    PhaseEncoder,
    HybridAnglePhaseEncoder,
    MultiAxisRotationEncoder,
    ParametricAngleEncoder,
)

# Composite encoding
from .composite import (
    CompositeEncoder,
    LayeredEncoder,
    DataReuseEncoder,
    AdaptiveEncoder,
    HierarchicalEncoder,
)

__all__ = [
    # Core encoders
    "IQPEncoder",
    "RotationEncoder",
    "StatevectorEncoder",
    # Amplitude encoders
    "AmplitudeEncoder",
    "DensityMatrixEncoder",
    "BasisChangeEncoder",
    "FeatureMapEncoder",
    # Angle & phase encoders
    "AngleEncoder",
    "PhaseEncoder",
    "HybridAnglePhaseEncoder",
    "MultiAxisRotationEncoder",
    "ParametricAngleEncoder",
    # Composite encoders
    "CompositeEncoder",
    "LayeredEncoder",
    "DataReuseEncoder",
    "AdaptiveEncoder",
    "HierarchicalEncoder",
]
