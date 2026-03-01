"""Coreset reduction via K-Means clustering.

Reduces large datasets to a compact set of representative points (coresets)
using K-Means clustering.  Each input point is mapped to its nearest
cluster centre during :meth:`transform`.
"""

import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from qudet.core.base import BaseReducer
from qudet.core.exceptions import NotFittedError, ValidationError

logger = logging.getLogger(__name__)


class CoresetReducer(BaseReducer):
    """Reduces a large dataset to a compact coreset using K-Means clustering.

    During ``fit``, K-Means is trained to identify ``target_size`` cluster
    centres.  During ``transform``, every input sample is replaced by the
    centre of its nearest cluster, producing a *coreset representation*
    that preserves the statistical structure of the original data.

    Args:
        target_size: Number of representative cluster centres to learn.
            Must be a positive integer.

    Attributes:
        centers\_: Cluster centres learned during ``fit``, shape
            ``(target_size, n_features)``.
        model\_: Fitted :class:`sklearn.cluster.KMeans` instance.

    Example:
        >>> reducer = CoresetReducer(target_size=50)
        >>> reducer.fit(X_train)
        >>> X_reduced = reducer.transform(X_new)
    """

    def __init__(self, target_size: int = 100) -> None:
        if not isinstance(target_size, int) or target_size < 1:
            raise ValidationError(
                f"target_size must be a positive integer, got {target_size!r}"
            )
        self.target_size = target_size
        self.centers_: np.ndarray | None = None
        self.model_: KMeans | None = None

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y=None
    ) -> "CoresetReducer":
        """Fit K-Means to learn coreset centres.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self

        Raises:
            ValidationError: If ``target_size`` exceeds the number of samples.
        """
        data = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        if data.ndim != 2:
            raise ValidationError(
                f"X must be 2-dimensional, got {data.ndim} dimensions"
            )
        if self.target_size > data.shape[0]:
            raise ValidationError(
                f"target_size ({self.target_size}) must not exceed the number "
                f"of samples ({data.shape[0]})"
            )

        logger.info(
            "Fitting CoresetReducer with %d clusters on data of shape %s",
            self.target_size,
            data.shape,
        )

        self.model_ = KMeans(
            n_clusters=self.target_size, random_state=42, n_init=10
        )
        self.model_.fit(data)
        self.centers_ = self.model_.cluster_centers_

        logger.info("CoresetReducer fitted successfully")
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Map each sample in *X* to its nearest coreset centre.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples, n_features)`` where each row is
            replaced by the nearest cluster centre learned during ``fit``.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if self.model_ is None or self.centers_ is None:
            raise NotFittedError(
                "CoresetReducer has not been fitted yet. Call fit() first."
            )

        data = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        if data.ndim != 2:
            raise ValidationError(
                f"X must be 2-dimensional, got {data.ndim} dimensions"
            )

        labels = self.model_.predict(data)
        return self.centers_[labels]
