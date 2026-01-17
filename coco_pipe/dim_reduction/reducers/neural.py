"""
Neural Reducers
===============

This module implements neural network-based dimensionality reduction techniques,
specifically optimizing triplet losses for scalable manifold learning.

Classes
-------
IVISReducer
    Deep learning-based dimensionality reduction (ivis wrapper).

References
----------
.. [1] Szubert, B., et al. (2019). Structure-preserving visualization of high dimensional
       single-cell datasets. Scientific reports, 9(1), 1-10.
.. [2] ivis: https://github.com/beringresearch/ivis

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
Date: 2026-01-06
"""

from typing import Optional

import numpy as np
from ivis import Ivis

from .base import ArrayLike, BaseReducer


class IVISReducer(BaseReducer):
    """
    IVIS Dimensionality Reducer.

    IVIS uses a Siamese Neural Network with a triplet loss function to learn a
    low-dimensional representation. It is highly scalable and supports
    out-of-sample extension.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in the embedding.
    **kwargs : dict
        Additional arguments passed to ivis.Ivis.
        Common arguments:
        - k : int, default=15 (Number of neighbors)
        - model : str, default='szubert' (Network architecture)
        - n_epochs_without_progress : int, default=15
        - supervise_metric : str, default='softmax_cross_entropy' (for supervised)

    Attributes
    ----------
    model : ivis.Ivis
        The underlying fitted IVIS estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.reducers.neural import IVISReducer
    >>> # Mock data
    >>> X = np.random.rand(100, 10)
    >>> reducer = IVISReducer(n_components=2, epochs=2)
    >>> X_reduced = reducer.fit_transform(X)
    >>> print(X_reduced.shape)
    (100, 2)
    """

    def __init__(self, n_components: int = 2, **kwargs):
        # IVIS uses 'embedding_dims' instead of n_components
        kwargs["embedding_dims"] = n_components
        super().__init__(n_components=n_components, **kwargs)
        self.model = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "IVISReducer":
        """
        Fit the IVIS model.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike of shape (n_samples,), optional
            Target values for supervised dimensionality reduction.

        Returns
        -------
        self : IVISReducer
            Returns the instance itself.
        """
        self.model = Ivis(**self.params)
        self.model.fit(X, Y=y)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform X.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_components)
             Transformed data.
        """
        if self.model is None:
            raise RuntimeError("IVISReducer must be fitted before calling transform().")
        return self.model.transform(X)

    @property
    def loss_history_(self) -> list:
        """
        History of loss values during training.

        Returns
        -------
        loss_history_ : list of float
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")

        # IVIS 2.0.11+ (git ver) exposes loss_history_ directly
        if hasattr(self.model, "loss_history_"):
            return self.model.loss_history_

        raise RuntimeError("Could not retrieve loss history from IVIS model.")
