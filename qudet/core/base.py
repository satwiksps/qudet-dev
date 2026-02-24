"""Abstract base classes for all QuDET components.

QuDET follows a three-layer architecture:

* **Reducers** — Shrink classical data to a size compatible with quantum hardware.
* **Encoders** — Convert classical vectors into quantum circuits / states.
* **Estimators** — Run quantum algorithms (classification, regression, etc.).

Every concrete class in QuDET inherits from one of these three ABCs.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd

from .exceptions import NotFittedError


class BaseReducer(ABC):
    """Abstract base class for classical data-reduction stages.

    Subclasses must implement :meth:`fit` and :meth:`transform`.
    The convenience method :meth:`fit_transform` is provided automatically.

    All reducers follow the scikit-learn convention where ``fit()`` returns
    ``self`` so that calls can be chained.
    """

    @abstractmethod
    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None
    ) -> "BaseReducer":
        """Learn parameters from the training data.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            y: Optional target values (ignored by unsupervised reducers).

        Returns:
            self
        """
        ...

    @abstractmethod
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Apply the learned reduction to *X*.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            Reduced array of shape ``(n_samples, n_components)``.
        """
        ...

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit to *X* and return the transformed result in one step."""
        return self.fit(X, y).transform(X)

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({params})"


class BaseEncoder(ABC):
    """Abstract base class for classical-to-quantum encoders.

    Subclasses must implement :meth:`encode`, which accepts a 1-D numpy array
    and returns a :class:`qiskit.QuantumCircuit`.
    """

    @abstractmethod
    def encode(self, data: np.ndarray):
        """Encode a classical data vector into a quantum circuit.

        Args:
            data: 1-D array of shape ``(n_features,)``.

        Returns:
            A ``qiskit.QuantumCircuit`` representing the encoded state.
        """
        ...

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({params})"


class BaseQuantumEstimator(ABC):
    """Abstract base class for quantum machine-learning estimators.

    Follows the scikit-learn estimator interface (``fit`` / ``predict``).

    Args:
        backend_name: Name of the Qiskit backend to use.
        shots: Number of measurement shots per circuit execution.
    """

    def __init__(self, backend_name: str = "aer_simulator", shots: int = 1024):
        self.backend_name = backend_name
        self.shots = shots

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseQuantumEstimator":
        """Fit the estimator to training data."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for the input data."""
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy on the given test data and labels."""
        predictions = self.predict(X)
        return float(np.mean(predictions == y))

    def _check_is_fitted(self) -> None:
        """Raise :class:`NotFittedError` if the estimator has not been fitted."""
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(
                f"{self.__class__.__name__} has not been fitted yet. "
                "Call .fit() before .predict() or .transform()."
            )

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({params})"
