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
    old_container: DataContainer, new_X: np.ndarray, names=None
) -> DataContainer:
    """Put ``new_X`` in the container, keeping all that still describes it.

    With as many features as before, every coordinate is kept. With another
    number (PCA, feature selection), the feature axis is labelled ``names``
    and the coordinates of the old features are dropped.
    """
    if new_X.shape[1] == old_container.X.shape[1]:
        return old_container.replace(X=new_X)
    feature_dim = old_container.dims[1]
    return old_container.with_features(
        new_X, names=names, feature_dim=feature_dim, new_dim_name=feature_dim
    )


def _output_names(estimator: BaseEstimator):
    """Return the estimator's output feature names, or None if it has none."""
    try:
        return np.asarray(estimator.get_feature_names_out())
    except (AttributeError, ValueError, TypeError):
        return None


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
        # The labels inverse_transform gives back.
        self.feature_labels_ = container.coords.get(container.dims[1])
        return self

    def transform(self, container: DataContainer) -> DataContainer:
        _check_container(container)
        if container.X.ndim != 2:
            raise ValueError("SklearnWrapper expects 2D input.")
        if self.estimator_ is None:
            raise RuntimeError("Transformer not fitted.")

        X_new = self.estimator_.transform(container.X)
        return _rebuild_container(container, X_new, _output_names(self.estimator_))

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
        return _rebuild_container(container, X_orig, self.feature_labels_)
