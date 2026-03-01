"""Feature hashing (sketching) for streaming / high-cardinality data.

Provides :class:`StreamingHasher`, which uses the hashing trick to map
arbitrary dictionaries or DataFrame rows into fixed-width numerical vectors
suitable for quantum encoding.
"""

import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

from qudet.core.base import BaseReducer

logger = logging.getLogger(__name__)


class StreamingHasher(BaseReducer):
    """Maps high-dimensional categorical data to fixed-width hash vectors.

    Uses scikit-learn's :class:`~sklearn.feature_extraction.FeatureHasher`
    (the *hashing trick*) to produce a fixed-size numerical representation
    from variable-schema dictionaries or DataFrame rows.

    This is memory-efficient and stateless, making it ideal for streaming
    scenarios.

    Args:
        n_features: Target dimensionality of the output vectors.  Powers
            of two (e.g. 1024 = 2¹⁰) map cleanly to qubit counts via
            amplitude encoding.

    Example:
        >>> hasher = StreamingHasher(n_features=1024)
        >>> hasher.fit(data)  # no-op, stateless
        >>> vecs = hasher.transform(data)
    """

    def __init__(self, n_features: int = 1024) -> None:
        self.n_features = n_features
        self.hasher_ = FeatureHasher(
            n_features=self.n_features, input_type="dict"
        )

    def fit(
        self,
        X: Union[pd.DataFrame, List[Dict], np.ndarray],
        y=None,
    ) -> "StreamingHasher":
        """No-op fit (hashing is stateless).

        Args:
            X: Ignored.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(
        self, X: Union[pd.DataFrame, List[Dict]]
    ) -> np.ndarray:
        """Hash each record into a fixed-width vector.

        Args:
            X: Input data as a :class:`~pandas.DataFrame` (rows become
                dicts) or a list of dictionaries.

        Returns:
            Dense array of shape ``(n_records, n_features)``.
        """
        if isinstance(X, pd.DataFrame):
            data_iter = X.to_dict(orient="records")
        else:
            data_iter = X

        return self.hasher_.transform(data_iter).toarray()
