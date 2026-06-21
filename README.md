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

> **Acknowledgment:** QuDET was developed during the **IBM Qiskit Advocate Mentorship Program** (October 2025 - January 2026). 
> 🏅 **[View Contributor Badge on Credly](https://www.credly.com/badges/64aed70f-a9ca-48b8-ae02-53a3a466b0c6/public_url)**

---

## Overview

**QuDET** bridges the gap between classical data engineering and quantum computing. It provides a production-ready, modular framework that lets AI engineers and researchers integrate quantum algorithms into data pipelines without deep quantum physics expertise.

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
| **Encoders** | Classical to Quantum | `AngleEncoder`, `AmplitudeEncoder`, `IQPEncoder`, `RotationEncoder` |
| **Analytics** | Quantum ML models | `QuantumSVC`, `QuantumKernelRegressor`, `QuantumKMeans` |
| **Compute** | Execution layer | `BackendManager`, `CircuitOptimizer`, `QuantumErrorMitigation` |
| **Governance** | Safety & operations | `QuantumDriftDetector`, `ResourceEstimator`, `AuditLogger` |

## Installation

```bash
pip install qudet
```

### Optional Dependencies

```bash
# SQL database connectors
pip install "qudet[sql]"

# Encryption and security features
pip install "qudet[crypto]"

# Distributed computing with Dask
pip install "qudet[distributed]"

# IBM Quantum hardware access
pip install "qudet[ibm]"

# Everything (including dev tools)
pip install "qudet[all]"
```

### Testing the installation

Open a Python prompt and run this minimal code to verify QuDET is installed correctly:

```python
import numpy as np
from qudet.encoders import AngleEncoder

# Create dummy data and encode it
data = np.array([0.5, 0.8])
encoder = AngleEncoder(n_qubits=2)
circuit = encoder.encode(data)

print(f"QuDET installed successfully! Generated a {circuit.num_qubits}-qubit circuit.")
```

## Examples & Tutorials

Check out the `examples/` directory in this repository! We provide a highly informative **Kickstart Guide** (`kickstart_guide.ipynb`) that walks you through an end-to-end real-world workflow using all 6 QuDET modules on a wine quality dataset.

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

## Feedback and Contributions

We welcome any feedback on how `QuDET` is working and are happy to receive bug reports, pull requests, and other feedback:

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Format code (`black .` and `ruff check .`)
6. Submit a pull request

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


## License

Distributed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.