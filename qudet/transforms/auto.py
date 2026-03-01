"""Automatic data reduction for quantum hardware constraints.

Provides :class:`AutoReducer`, a meta-reducer that inspects the shape of
incoming data and automatically selects an appropriate reduction strategy
(random projection, coreset sampling, or passthrough).
"""

import logging
from typing import Union

import numpy as np
import pandas as pd

from qudet.core.base import BaseReducer
from .coresets import CoresetReducer
from .projections import RandomProjector

logger = logging.getLogger(__name__)


class AutoReducer(BaseReducer):
    """Meta-reducer that selects the best reduction strategy automatically.

    Selection logic:

    1. **Too many features** (``n_features > target_qubits``)
       → :class:`RandomProjector` to reduce dimensionality.
    2. **Too many samples** (``n_samples > max_rows``)
       → :class:`CoresetReducer` to downsample.
    3. **Otherwise** → data passes through unchanged.

    This frees data engineers from needing to understand quantum hardware
    constraints when preparing data.

    Args:
        target_qubits: Maximum number of features (columns) allowed,
            typically matching available qubit count.
        max_rows: Maximum number of samples to retain.

    Attributes:
        pipeline\_: Ordered list of ``(name, reducer)`` tuples built
            during ``fit``.
        reduction_strategy\_: Human-readable description of the chosen
            strategy.

    Example:
        >>> ar = AutoReducer(target_qubits=10, max_rows=500)
        >>> ar.fit(big_dataframe)
        >>> reduced = ar.transform(big_dataframe)
    """

    def __init__(self, target_qubits: int = 10, max_rows: int = 500) -> None:
        self.target_qubits = target_qubits
        self.max_rows = max_rows
        self.pipeline_: list = []
        self.reduction_strategy_: str | None = None

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y=None
    ) -> "AutoReducer":
        """Analyse data shape and build the reduction pipeline.

        Args:
            X: Input data of shape ``(n_samples, n_features)``.
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        self.pipeline_ = []

        if isinstance(X, pd.DataFrame):
            n_rows, n_cols = X.shape
        else:
            n_rows, n_cols = X.shape

        logger.info(
            "AutoReducer analysing shape (%d, %d)", n_rows, n_cols
        )

        if n_cols > self.target_qubits:
            logger.info(
                "High dimensionality detected (%d > %d). "
                "Adding RandomProjector.",
                n_cols,
                self.target_qubits,
            )
            proj = RandomProjector(n_components=self.target_qubits)
            proj.fit(X)
            self.pipeline_.append(("projection", proj))
            X = proj.transform(X)

        if n_rows > self.max_rows:
            logger.info(
                "Large volume detected (%d > %d). Adding CoresetReducer.",
                n_rows,
                self.max_rows,
            )
            core = CoresetReducer(target_size=self.max_rows)
            core.fit(X)
            self.pipeline_.append(("coreset", core))

        if not self.pipeline_:
            logger.info("Data fits comfortably. No reduction needed.")
            self.reduction_strategy_ = "none"
        else:
            self.reduction_strategy_ = f"{len(self.pipeline_)} step(s)"

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Apply the reduction pipeline to data.

        Args:
            X: Input data of shape ``(n_samples, n_features)``.

        Returns:
            Reduced data as ``np.ndarray``.
        """
        data = X
        for _name, reducer in self.pipeline_:
            data = reducer.transform(data)
        return data if isinstance(data, np.ndarray) else np.asarray(data)
