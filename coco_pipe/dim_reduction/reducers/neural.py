"""
Neural-network dimensionality reduction reducers.

This module provides wrappers around neural embedding backends that follow the
shared `BaseReducer` contract. These reducers integrate with `DimReduction`,
reporting, and visualization while keeping optional deep-learning dependencies
lazy at import time.

Classes
-------
IVISReducer
    Neural triplet-loss embedding based on `ivis.Ivis`.

References
----------
.. [1] Szubert, B., Cole, J. E., Monaco, C., and Drozdov, I. (2019).
       "Structure-preserving visualization of high dimensional single-cell
       datasets". Scientific Reports, 9(1), 8914.
.. [2] IVIS documentation:
       https://github.com/beringresearch/ivis

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from typing import Optional

import numpy as np

from ...utils import import_optional_dependency
from .base import ArrayLike, BaseReducer

__all__ = ["IVISReducer"]


class IVISReducer(BaseReducer):
    """
    IVIS dimensionality reducer.

    IVIS learns a low-dimensional representation with a Siamese neural network
    trained using a triplet-loss objective. The reducer supports out-of-sample
    transformation and is suitable for large datasets when the optional
    `ivis` dependency is installed.

    Parameters
    ----------
    n_components : int, default=2
        Number of embedding dimensions to learn.
    **kwargs : dict
        Additional keyword arguments forwarded to `ivis.Ivis` after signature
        filtering. Common options include `k`, `model`, `epochs`,
        `batch_size`, `n_epochs_without_progress`, and `supervise_metric`.

    Attributes
    ----------
    model : ivis.Ivis or None
        Fitted IVIS estimator after `fit`.

    Notes
    -----
    The IVIS backend uses `embedding_dims` instead of `n_components`. This
    wrapper maps the reducer component count to the backend constructor
    automatically.

    See Also
    --------
    ParametricUMAPReducer : Neural graph-based embedding with parametric transform.
    UMAPReducer : Nonparametric graph-based nonlinear embedding.
    PHATEReducer : Diffusion-based nonlinear embedding for smooth trajectories.
    TopologicalAEReducer : Neural autoencoder with topological regularization.
    PCAReducer : Linear baseline for global variance preservation.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import IVISReducer
    >>> X = np.random.rand(100, 10).astype(np.float32)
    >>> reducer = IVISReducer(n_components=2, k=10, epochs=2, batch_size=16)
    >>> _ = reducer.fit(X)
    >>> embedding = reducer.transform(X[:8])
    >>> embedding.shape
    (8, 2)
    >>> reducer.get_diagnostics()["loss_history_"]  # doctest: +SKIP
    [...]
    >>> reducer = IVISReducer(n_components=3, epochs=2, batch_size=16)
    >>> reducer.fit_transform(X).shape  # doctest: +SKIP
    (100, 3)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for IVIS.

        Returns
        -------
        dict
            Capability mapping describing IVIS as a stochastic nonlinear
            reducer with transform support and loss-history diagnostics.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_transform=True,
            supported_diagnostics=("loss_history_",),
            supported_metadata=(),
            is_linear=False,
            is_stochastic=True,
        )

    def __init__(self, n_components: int = 2, **kwargs):
        """
        Initialize the IVIS reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of embedding dimensions to learn.
        **kwargs : dict
            Additional keyword arguments forwarded to `ivis.Ivis` after
            filtering.
        """
        super().__init__(n_components=n_components, **kwargs)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "IVISReducer":
        """
        Fit IVIS on the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike of shape (n_samples,), optional
            Optional supervision passed to IVIS for supervised training.

        Returns
        -------
        IVISReducer
            Fitted reducer instance.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import IVISReducer
        >>> X = np.random.rand(30, 6).astype(np.float32)
        >>> reducer = IVISReducer(n_components=2, epochs=2, batch_size=8)
        >>> _ = reducer.fit(X)
        >>> reducer.model is not None
        True
        """
        Ivis = import_optional_dependency(
            lambda: __import__("ivis", fromlist=["Ivis"]).Ivis,
            feature="IVISReducer",
            dependency="ivis",
            install_hint="pip install coco-pipe[ivis]",
        )

        self.model = self._build_estimator(Ivis, component_param="embedding_dims")
        self.model.fit(X, Y=y)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Project data into the fitted IVIS embedding space.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            New samples to embed.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Low-dimensional embedding of `X`.
        """
        model = self._require_fitted()
        return model.transform(X)
