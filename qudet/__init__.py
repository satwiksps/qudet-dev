"""Quantum Data Engineering Toolkit (QuDET) - Enterprise Edition v0.3.0

A comprehensive, production-ready toolkit for quantum-enhanced data engineering,
machine learning, and analytics using industry-standard naming conventions.

Architecture Layers:
1. Core (qudet.core): Base classes and exceptions
2. Connectors (qudet.connectors): Data loading, SQL, Parquet, serialization
3. Transforms (qudet.transforms): Dimensionality reduction, cleaning, scaling
4. Encoders (qudet.encoders): Feature encoding into Hilbert space
5. Analytics (qudet.analytics): ML algorithms (SVC, KMeans, anomaly detection, etc.)
6. Compute (qudet.compute): Execution, backend management, circuit optimization
7. Governance (qudet.governance): Monitoring, drift detection, cost tracking
"""

__version__ = "0.3.0"

from . import core
from . import connectors
from . import transforms
from . import encoders
from . import analytics
from . import compute
from . import governance

from .connectors.loader import QuantumDataLoader
from .connectors.sql import QuantumSQLLoader
from .connectors.parquet import QuantumParquetLoader
from .connectors.serialization import QuantumSerializer
from .connectors.streaming import (
    StreamingDataBuffer,
    DataStreamIterator,
    DataValidator,
    DataCacher,
    BatchAggregator
)
from .connectors.transformation import (
    DataTransformer,
    DataMetadataTracker,
    DataQualityChecker,
    DataProfiler
)
from .connectors.utilities import (
    DataConnectorFactory,
    DataBatchProcessor,
    DataFormatConverter,
    DataSplitter,
    DataSampler
)

from .transforms.coresets import CoresetReducer
from .transforms.auto import AutoReducer
from .transforms.projections import RandomProjector
from .transforms.pca import QuantumPCA
from .transforms.imputation import QuantumImputer
from .transforms.sketching import StreamingHasher
from .transforms.feature_engineering import FeatureScaler, FeatureSelector, OutlierRemover, DataBalancer
from .transforms.encoding import CategoricalEncoder, TargetEncoder, FrequencyEncoder, BinningEncoder
from .transforms.normalization import QuantumNormalizer, RangeNormalizer, DecimalScaler, LogTransformer, PowerTransformer

from .encoders.rotation import RotationEncoder
from .encoders.statevector import StatevectorEncoder
from .encoders.iqp import IQPEncoder
from .encoders.amplitude import (
    AmplitudeEncoder,
    DensityMatrixEncoder,
    BasisChangeEncoder,
    FeatureMapEncoder
)
from .encoders.angle_phase import (
    AngleEncoder,
    PhaseEncoder,
    HybridAnglePhaseEncoder,
    MultiAxisRotationEncoder,
    ParametricAngleEncoder
)
from .encoders.composite import (
    CompositeEncoder,
    LayeredEncoder,
    DataReuseEncoder,
    AdaptiveEncoder,
    HierarchicalEncoder
)

from .analytics.anomaly import QuantumKernelAnomalyDetector
from .analytics.classifier import QuantumSVC
from .analytics.clustering import QuantumKMeans
from .analytics.regression import QuantumKernelRegressor
from .analytics.feature_select import QuantumFeatureSelector
from .analytics.autoencoder import QuantumAutoencoder
from .analytics.vqe import VariationalQuantumEigensolver, QAOA
from .analytics.neural_net import QuantumNeuralNetwork, QuantumTransferLearning
from .analytics.ensemble import QuantumEnsemble, QuantumDataAugmentation, QuantumMetaLearner
from .analytics.timeseries import QuantumTimeSeriesPredictor, QuantumOutlierDetection, QuantumDimensionalityReduction

from .compute.backend import BackendManager
from .compute.distributed import DistributedQuantumProcessor
from .compute.layout import HardwareLayoutSelector
from .compute.simplify import CircuitOptimizer
from .compute.compilation import (
    QuantumCircuitCompiler,
    QuantumCircuitOptimizer,
    QuantumNativeGateTranspiler
)
from .compute.error_mitigation import (
    QuantumErrorMitigation,
    QuantumNoiseModel,
    QuantumCalibrationalAnalyzer
)
from .compute.resource_management import (
    QuantumResourceAllocator,
    QuantumPriorityScheduler,
    QuantumCostEstimator
)

from .governance.cost import ResourceEstimator
from .governance.privacy import QuantumDifferentialPrivacy
from .governance.drift import QuantumDriftDetector
from .governance.simulation import NoiseSimulator
from .governance.visualization import plot_reduction_2d, plot_kernel_matrix
from .governance.audit import (
    AuditLogger,
    ComplianceChecker,
    DataGovernance,
    AuditEvent
)
from .governance.security import (
    SecureAccessControl,
    EncryptionManager,
    SecurityMonitor,
    AccessLevel
)
from .governance.orchestration import (
    Workflow,
    ResourceScheduler,
    Task,
    TaskStatus,
    WorkflowStatus
)

__all__ = [
    "core",
    "connectors",
    "transforms",
    "encoders",
    "analytics",
    "compute",
    "governance",
    "QuantumDataLoader",
    "QuantumSQLLoader",
    "QuantumParquetLoader",
    "QuantumSerializer",
    "StreamingDataBuffer",
    "DataStreamIterator",
    "DataValidator",
    "DataCacher",
    "BatchAggregator",
    "DataTransformer",
    "DataMetadataTracker",
    "DataQualityChecker",
    "DataProfiler",
    "DataConnectorFactory",
    "DataBatchProcessor",
    "DataFormatConverter",
    "DataSplitter",
    "DataSampler",
    "CoresetReducer",
    "AutoReducer",
    "RandomProjector",
    "QuantumPCA",
    "QuantumImputer",
    "StreamingHasher",
    "FeatureScaler",
    "FeatureSelector",
    "OutlierRemover",
    "DataBalancer",
    "CategoricalEncoder",
    "TargetEncoder",
    "FrequencyEncoder",
    "BinningEncoder",
    "QuantumNormalizer",
    "RangeNormalizer",
    "DecimalScaler",
    "LogTransformer",
    "PowerTransformer",
    "RotationEncoder",
    "StatevectorEncoder",
    "IQPEncoder",
    "AmplitudeEncoder",
    "DensityMatrixEncoder",
    "BasisChangeEncoder",
    "FeatureMapEncoder",
    "AngleEncoder",
    "PhaseEncoder",
    "HybridAnglePhaseEncoder",
    "MultiAxisRotationEncoder",
    "ParametricAngleEncoder",
    "CompositeEncoder",
    "LayeredEncoder",
    "DataReuseEncoder",
    "AdaptiveEncoder",
    "HierarchicalEncoder",
    "QuantumKernelAnomalyDetector",
    "QuantumSVC",
    "QuantumKMeans",
    "QuantumKernelRegressor",
    "QuantumFeatureSelector",
    "QuantumAutoencoder",
    "VariationalQuantumEigensolver",
    "QAOA",
    "QuantumNeuralNetwork",
    "QuantumTransferLearning",
    "QuantumEnsemble",
    "QuantumDataAugmentation",
    "QuantumMetaLearner",
    "QuantumTimeSeriesPredictor",
    "QuantumOutlierDetection",
    "QuantumDimensionalityReduction",
    "BackendManager",
    "DistributedQuantumProcessor",
    "HardwareLayoutSelector",
    "CircuitOptimizer",
    "QuantumCircuitCompiler",
    "QuantumCircuitOptimizer",
    "QuantumNativeGateTranspiler",
    "QuantumErrorMitigation",
    "QuantumNoiseModel",
    "QuantumCalibrationalAnalyzer",
    "QuantumResourceAllocator",
    "QuantumPriorityScheduler",
    "QuantumCostEstimator",
    "ResourceEstimator",
    "QuantumDifferentialPrivacy",
    "QuantumDriftDetector",
    "NoiseSimulator",
    "plot_reduction_2d",
    "plot_kernel_matrix",
    "AuditLogger",
    "ComplianceChecker",
    "DataGovernance",
    "AuditEvent",
    "SecureAccessControl",
    "EncryptionManager",
    "SecurityMonitor",
    "AccessLevel",
    "Workflow",
    "ResourceScheduler",
    "Task",
    "TaskStatus",
    "WorkflowStatus",
]
