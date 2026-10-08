"""
coco_pipe/io/transform.py
-------------------------
Stateful transformers compatible with DataContainer.
Wraps scikit-learn transformers.

This module provides classes that adhere to the Scikit-Learn Transformer API
but operate natively on `~coco_pipe.io.DataContainer` objects, preserving metadata (IDs,
coordinates) throughout the transformation pipeline.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone

from .structures import DataContainer

logger = logging.getLogger(__name__)


def _rebuild_container(
    old_container: DataContainer, new_X: np.ndarray
) -> DataContainer:
    """Helper to reconstruct DataContainer with new data and propagated metadata."""
    return old_container.replace(X=new_X)


def _check_container(container: DataContainer):
    """Helper to validate input."""
    if not isinstance(container, DataContainer):
        raise TypeError(f"Transformer expects DataContainer, got {type(container)}")


class SklearnWrapper(BaseEstimator, TransformerMixin):
    """
    Generic wrapper for ANY scikit-learn transformer (Scaler, PCA, etc.).

    This wrapper applies a standard scikit-learn transformer to the `.X` data
    matrix of a `~coco_pipe.io.DataContainer`, ensuring that the resulting container has
    correctly updated data while checking for dimension compatibility.

    Parameters
    ----------
    transformer : BaseEstimator
        An instantiated scikit-learn transformer (e.g., `StandardScaler()`,
        `PCA(n_components=10)`).

    Attributes
    ----------
    estimator_ : BaseEstimator
        The fitted scikit-learn estimator.

    Examples
    --------
    >>> from sklearn.preprocessing import RobustScaler
    >>> from coco_pipe.io import DataContainer, SklearnWrapper
    >>> import numpy as np

    >>> # Create formatted data (100 obs, 10 features)
    >>> X = np.random.randn(100, 10)
    >>> container = DataContainer(X, dims=("obs", "feature"))

    >>> # Wrap a Scaler
    >>> scaler = SklearnWrapper(RobustScaler())
    >>> scaled_container = scaler.fit_transform(container)

    >>> # Metadata is preserved
    >>> scaled_container.dims == container.dims
    True
    """

    def __init__(self, transformer: BaseEstimator):
        self.transformer = transformer
        self.estimator_ = None

    def fit(self, container: DataContainer, y=None):
        _check_container(container)
        if container.X.ndim != 2:
            raise ValueError(
                f"SklearnWrapper expects 2D input (Obs, Feat). Got {container.shape} "
                f"with dims {container.dims}. Use .flatten() or .stack() first."
            )

        self.estimator_ = clone(self.transformer)
        self.estimator_.fit(container.X, y)
        return self

    def transform(self, container: DataContainer) -> DataContainer:
        _check_container(container)
        if container.X.ndim != 2:
            raise ValueError("SklearnWrapper expects 2D input.")
        if self.estimator_ is None:
            raise RuntimeError("Transformer not fitted.")

        X_new = self.estimator_.transform(container.X)
        return _rebuild_container(container, X_new)  # Use Helper

    def fit_transform(self, container: DataContainer, y=None):
        """
        Fit the wrapped estimator and transform the data.

        Parameters
        ----------
        container : DataContainer
            Input data to transform.
        y : array-like, optional
            Target values.

        Returns
        -------
        DataContainer
            Transformed data with the same dimensions.
        """
        return self.fit(container, y).transform(container)

    def inverse_transform(self, container: DataContainer) -> DataContainer:
        _check_container(container)
        if container.X.ndim != 2:
            raise ValueError("SklearnWrapper expects 2D input.")
        if self.estimator_ is None:
            raise RuntimeError("Transformer not fitted.")
        if not hasattr(self.estimator_, "inverse_transform"):
            raise NotImplementedError(
                f"Wrapped estimator {type(self.estimator_)} has no inverse_transform."
            )

        X_orig = self.estimator_.inverse_transform(container.X)
        return _rebuild_container(container, X_orig)  # Use Helper
