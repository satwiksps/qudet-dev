# Changelog

All notable changes to QuDET will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-20

### Added
- Modern `pyproject.toml` build configuration (PEP 621).
- Comprehensive exception hierarchy (`QuDETError`, `EncodingError`, `CircuitError`,
  `ValidationError`, `BackendError`, `NotFittedError`).
- Optional dependency groups: `parquet`, `sql`, `crypto`, `distributed`, `visualization`, `ibm`, `dev`.
- Professional docstrings across all public APIs.
- Input validation and descriptive error messages in all modules.
- `__repr__` on all base classes for better debugging.
- `_check_is_fitted()` helper on `BaseQuantumEstimator`.

### Changed
- **Breaking:** Minimum Qiskit version is now `>=1.0`. All deprecated APIs removed.
- **Breaking:** Removed placeholder analytics modules (`ensemble`, `neural_net`, `vqe`,
  `autoencoder`, `timeseries`) that had non-functional implementations.
- Replaced `requirements.txt` with minimal `pyproject.toml` dependencies.
- All `fit()` methods now consistently return `self` for method chaining.
- All `BaseReducer.fit()` signatures now accept `y=None` parameter.
- Switched from `print()` to `logging` module throughout.
- `DataSplitter`/`DataSampler` now use `np.random.default_rng()` instead of global seed.

### Fixed
- `RotationEncoder.encode()` now actually uses input data (was ignoring it entirely).
- `CompositeEncoder` — fixed `add_register(QuantumCircuit(...))` → `add_register(QuantumRegister(...))`.
- `PhaseEncoder.encode()` — fixed IndexError in entanglement loop.
- `DataTransformer` — 'scale' and 'standardize' no longer do the same thing.
- `DataTransformer.transform()` — no longer mutates stored `fit_params`.
- `DataValidator` — `allow_nan=True` now correctly permits NaN values.
- `CoresetReducer.transform()` — now maps input points to nearest coreset representatives.
- `OutlierRemover.transform()` — recomputes mask on new data.
- `QuantumImputer` now properly extends `BaseReducer`.
- Removed broken `fit_transform()` on `BaseQuantumEstimator`.
- Fixed `security.py` encryption (was using one-way SHA-256 hash).
- Fixed `privacy.py` sanitize() (was a no-op).

### Removed
- `qudet/_base.py` (duplicate of `qudet/core/base.py`).
- `qudet/compute/airflow_ops.py` (incomplete; requires Apache Airflow).
- `qudet/analytics/ensemble.py` (predictions ignored input data).
- `qudet/analytics/neural_net.py` (gradients were always zero).
- `qudet/analytics/vqe.py` (cheated by computing exact eigenvalues).
- `qudet/analytics/autoencoder.py` (all methods were stubs).
- `qudet/analytics/timeseries.py` (purely classical, no quantum circuits).
