<p align="center">
  <h1 align="center">QuDET</h1>
  <p align="center"><strong>Quantum Data Engineering Toolkit</strong></p>
  <p align="center">
    A modular Python framework for building hybrid classical–quantum data pipelines.
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/qudet/"><img src="https://img.shields.io/pypi/v/qudet?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/qudet/"><img src="https://img.shields.io/pypi/pyversions/qudet" alt="Python versions"></a>
  <a href="https://github.com/satwiksps/qudet-dev/blob/main/LICENSE"><img src="https://img.shields.io/github/license/satwiksps/qudet-dev" alt="License"></a>
</p>

---

## Overview

**QuDET** bridges the gap between classical data engineering and quantum computing. It provides a production-ready, modular framework that lets AI engineers and researchers integrate quantum algorithms into data pipelines — without deep quantum physics expertise.

QuDET follows scikit-learn conventions (`fit` / `transform` / `predict`) so you can drop quantum components into existing ML workflows with minimal friction.

## Why QuDET?

- **Practical quantum integration:** Quantum components solve real problems (kernel methods, encoding, anomaly detection) rather than existing for novelty.
- **Familiar API:** scikit-learn compatible interfaces mean minimal learning curve.
- **Modular architecture:** Use only what you need. Each module works independently.
- **Production-ready:** Input validation, proper error handling, logging, and type hints throughout.
- **Simulator-first:** Works out of the box with Qiskit Aer. Optional IBM Quantum hardware support.

## Architecture

QuDET is organized into six specialized layers:


| Module | Purpose | Key Classes |
|--------|---------|-------------|
| **Connectors** | Data ingestion & I/O | `QuantumDataLoader`, `QuantumParquetLoader`, `QuantumSQLLoader` |
| **Transforms** | Feature engineering | `QuantumPCA`, `FeatureScaler`, `QuantumNormalizer`, `CoresetReducer` |
| **Encoders** | Classical → Quantum | `AngleEncoder`, `AmplitudeEncoder`, `IQPEncoder`, `RotationEncoder` |
| **Analytics** | Quantum ML models | `QuantumSVC`, `QuantumKernelRegressor`, `QuantumKMeans` |
| **Compute** | Execution layer | `BackendManager`, `CircuitOptimizer`, `QuantumErrorMitigation` |
| **Governance** | Safety & operations | `QuantumDriftDetector`, `ResourceEstimator`, `AuditLogger` |

## Installation

```bash
pip install qudet
```

### Optional Dependencies

```bash
# Parquet file support
pip install "qudet[parquet]"

# SQL database connectors
pip install "qudet[sql]"

# Encryption and security features
pip install "qudet[crypto]"

# Distributed computing with Dask
pip install "qudet[distributed]"

# Visualization (matplotlib, seaborn)
pip install "qudet[visualization]"

# IBM Quantum hardware access
pip install "qudet[ibm]"

# Everything (including dev tools)
pip install "qudet[all]"
```

## Quick Start

### Quantum-Enhanced Classification Pipeline

```python
from qudet.transforms import QuantumPCA, FeatureScaler
from qudet.analytics import QuantumSVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale and reduce dimensions
scaler = FeatureScaler(method="standard")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = QuantumPCA(n_components=3)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

# Train quantum classifier
clf = QuantumSVC(n_qubits=3, C=1.0)
clf.fit(X_train_reduced, y_train)
print(f"Accuracy: {clf.score(X_test_reduced, y_test):.2%}")
```

### Quantum Encoding

```python
from qudet.encoders import AngleEncoder, AmplitudeEncoder, IQPEncoder
import numpy as np

data = np.array([0.1, 0.5, 0.3, 0.8])

# Angle encoding — one rotation gate per feature
encoder = AngleEncoder(n_qubits=4, angle_type="ry")
circuit = encoder.encode(data)
print(circuit.draw())

# Amplitude encoding — logarithmic qubit compression
amp_encoder = AmplitudeEncoder(n_qubits=2, normalize=True)
circuit = amp_encoder.encode(data)

# IQP encoding — with feature interactions
iqp_encoder = IQPEncoder(n_qubits=4, reps=2)
circuit = iqp_encoder.encode(data)
```

### Data Governance

```python
from qudet.governance import QuantumDriftDetector, ResourceEstimator
from qiskit import QuantumCircuit

# Monitor data drift
detector = QuantumDriftDetector(n_qubits=4, threshold=0.1)
detector.fit_reference(X_train)
result = detector.detect_drift(X_test)
print(f"Drift detected: {result['drift_detected']}")

# Estimate circuit execution cost
qc = QuantumCircuit(4)
qc.h(range(4))
cost = ResourceEstimator.estimate_circuit_cost(qc, shots=8192)
print(f"Estimated cost: ${cost['total_cost']:.4f}")
```

## Development Setup

```bash
git clone https://github.com/satwiksps/qudet-dev.git
cd qudet-dev
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -e ".[dev]"
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=qudet --cov-report=term-missing

# Run specific module tests
pytest tests/test_encoders/ -v

# Run only fast tests
pytest -m "not slow"
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Format code (`black .` and `ruff check .`)
6. Submit a pull request

## License

Distributed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.