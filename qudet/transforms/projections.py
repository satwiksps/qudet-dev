"""Random projection for dimensionality reduction.

Provides :class:`RandomProjector`, which uses Gaussian random projection
to reduce the number of features while approximately preserving pairwise
distances (Johnson–Lindenstrauss lemma).
"""

import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection

from qudet.core.base import BaseReducer
from qudet.core.exceptions import NotFittedError, ValidationError

logger = logging.getLogger(__name__)


class RandomProjector(BaseReducer):
    """Reduces feature dimensionality using Gaussian random projection.

    High-dimensional data (e.g. 1 000 columns) can be projected into a
    much lower-dimensional space (e.g. 10 dimensions ≈ 10 qubits) while
    approximately preserving pairwise Euclidean distances.

    Args:
        n_components: Target number of output dimensions.

    Attributes:
        model\_: Fitted :class:`sklearn.random_projection.GaussianRandomProjection`
            instance.

    Example:
        >>> proj = RandomProjector(n_components=10)
        >>> proj.fit(X_train).transform(X_test)
    """

    def __init__(self, n_components: int = 8) -> None:
        if not isinstance(n_components, int) or n_components < 1:
            raise ValidationError(
                f"n_components must be a positive integer, got {n_components!r}"
            )
        self.n_components = n_components
        self.model_: GaussianRandomProjection | None = None

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y=None
    ) -> "RandomProjector":
        """Fit the random projection matrix.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        self.model_ = GaussianRandomProjection(
            n_components=self.n_components, random_state=42
        )
        self.model_.fit(X)
        logger.info(
            "RandomProjector fitted: %d → %d dimensions",
            X.shape[1] if hasattr(X, "shape") else "?",
            self.n_components,
        )
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Project *X* to the lower-dimensional space.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            Projected data of shape ``(n_samples, n_components)``.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if self.model_ is None:
            raise NotFittedError(
                "RandomProjector has not been fitted. Call fit() first."
            )
        return self.model_.transform(X)
