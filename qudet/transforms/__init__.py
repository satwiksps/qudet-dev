"""Data transformation and reduction utilities for QuDET.

This package provides classical data transformations that prepare raw
datasets for quantum encoding and execution.  It includes:

* **Dimensionality reduction** — :class:`AutoReducer`, :class:`QuantumPCA`,
  :class:`RandomProjector`, :class:`CoresetReducer`, :class:`StreamingHasher`.
* **Feature engineering** — :class:`FeatureScaler`, :class:`FeatureSelector`,
  :class:`OutlierRemover`, :class:`DataBalancer`.
* **Encoding** — :class:`CategoricalEncoder`, :class:`TargetEncoder`,
  :class:`FrequencyEncoder`, :class:`BinningEncoder`.
* **Normalisation** — :class:`QuantumNormalizer`, :class:`RangeNormalizer`,
  :class:`DecimalScaler`, :class:`LogTransformer`, :class:`PowerTransformer`.
* **Imputation** — :class:`QuantumImputer`.
"""

from .auto import AutoReducer
from .coresets import CoresetReducer
from .encoding import (
    BinningEncoder,
    CategoricalEncoder,
    FrequencyEncoder,
    TargetEncoder,
)
from .feature_engineering import (
    DataBalancer,
    FeatureScaler,
    FeatureSelector,
    OutlierRemover,
)
from .imputation import QuantumImputer
from .normalization import (
    DecimalScaler,
    LogTransformer,
    PowerTransformer,
    QuantumNormalizer,
    RangeNormalizer,
)
from .pca import QuantumPCA
from .projections import RandomProjector
from .sketching import StreamingHasher

__all__ = [
    # Dimensionality reduction
    "AutoReducer",
    "CoresetReducer",
    "QuantumPCA",
    "RandomProjector",
    "StreamingHasher",
    # Feature engineering
    "FeatureScaler",
    "FeatureSelector",
    "OutlierRemover",
    "DataBalancer",
    # Encoding
    "CategoricalEncoder",
    "TargetEncoder",
    "FrequencyEncoder",
    "BinningEncoder",
    # Normalisation
    "QuantumNormalizer",
    "RangeNormalizer",
    "DecimalScaler",
    "LogTransformer",
    "PowerTransformer",
    # Imputation
    "QuantumImputer",
]
