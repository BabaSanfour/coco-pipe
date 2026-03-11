"""
Nonlinear manifold-learning reducers.

This module provides wrappers around scikit-learn manifold-learning estimators.
These reducers follow the shared `BaseReducer` contract so they can be used
with `DimReduction`, reporting, and visualization utilities while preserving a
consistent reducer API.

Classes
-------
IsomapReducer
    Nonlinear geodesic-distance embedding based on Isomap.
LLEReducer
    Nonlinear neighborhood-preserving embedding based on Locally Linear
    Embedding.
MDSReducer
    Distance-preserving embedding based on multidimensional scaling.
SpectralEmbeddingReducer
    Graph Laplacian embedding based on spectral decomposition.

References
----------
.. [1] Tenenbaum, J. B., de Silva, V., and Langford, J. C. (2000).
       "A global geometric framework for nonlinear dimensionality reduction".
       Science, 290(5500), 2319-2323.
.. [2] Roweis, S. T., and Saul, L. K. (2000). "Nonlinear dimensionality
       reduction by locally linear embedding". Science, 290(5500), 2323-2326.
.. [3] Borg, I., and Groenen, P. J. F. (2005). Modern multidimensional scaling:
       Theory and applications. Springer.
.. [4] Belkin, M., and Niyogi, P. (2003). "Laplacian eigenmaps for
       dimensionality reduction and data representation". Neural Computation,
       15(6), 1373-1396.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from typing import Optional

import numpy as np
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding

from .base import ArrayLike, BaseReducer

__all__ = [
    "IsomapReducer",
    "LLEReducer",
    "MDSReducer",
    "SpectralEmbeddingReducer",
]


class IsomapReducer(BaseReducer):
    """
    Isometric Mapping reducer.

    Isomap estimates geodesic distances on a nearest-neighbor graph and then
    computes a low-dimensional embedding consistent with those distances.

    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    **kwargs : dict
        Additional keyword arguments forwarded to `sklearn.manifold.Isomap`
        after signature filtering. Common options include `n_neighbors`,
        `metric`, `p`, and `eigen_solver`.

    Attributes
    ----------
    model : sklearn.manifold.Isomap or None
        Fitted Isomap estimator after `fit`.

    See Also
    --------
    LLEReducer : Nonlinear local-neighborhood manifold embedding.
    MDSReducer : Distance-preserving manifold embedding.
    SpectralEmbeddingReducer : Nonlinear graph Laplacian embedding.
    PCAReducer : Linear baseline for global variance preservation.
    UMAPReducer : Nonlinear graph-based embedding for local and global structure.
    TSNEReducer : Nonlinear neighborhood-preserving visualization method.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import IsomapReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = IsomapReducer(n_components=2, n_neighbors=5)
    >>> _ = reducer.fit(X)
    >>> reducer.transform(X[:8]).shape
    (8, 2)
    >>> reducer.n_features_in_
    10
    >>> embedding = reducer.fit_transform(X)
    >>> embedding.shape
    (100, 2)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for Isomap.

        Returns
        -------
        dict
            Capability mapping describing Isomap as a nonlinear reducer with
            out-of-sample transform support.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_transform=True,
            supported_metadata=("n_features_in_", "reconstruction_error_"),
            is_linear=False,
        )

    def __init__(self, n_components: int = 2, **kwargs):
        """
        Initialize the Isomap reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of coordinates for the manifold.
        **kwargs : dict
            Additional keyword arguments forwarded to `Isomap` after filtering.
        """
        super().__init__(n_components=n_components, **kwargs)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "IsomapReducer":
        """
        Fit Isomap on the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        IsomapReducer
            Fitted reducer instance.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import IsomapReducer
        >>> X = np.random.rand(30, 6)
        >>> reducer = IsomapReducer(n_components=2, n_neighbors=4)
        >>> _ = reducer.fit(X)
        >>> reducer.model is not None
        True
        """
        self.model = self._build_estimator(Isomap)
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Project data into the fitted Isomap embedding space.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Data to project.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Low-dimensional embedding coordinates.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        self._require_fitted()
        return self.model.transform(X)

    @property
    def reconstruction_error_(self) -> Optional[float]:
        """
        Return the Isomap reconstruction error.

        Returns
        -------
        float
            Reconstruction error returned by the fitted estimator.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.reconstruction_error()


class LLEReducer(BaseReducer):
    """
    Locally Linear Embedding reducer.

    LLE learns a nonlinear embedding by reconstructing each point from its
    local neighborhood in the input space and preserving those reconstruction
    weights in the low-dimensional space.

    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    **kwargs : dict
        Additional keyword arguments forwarded to
        `sklearn.manifold.LocallyLinearEmbedding` after signature filtering.
        Common options include `n_neighbors`, `method`, `eigen_solver`, and
        `random_state`.

    Attributes
    ----------
    model : sklearn.manifold.LocallyLinearEmbedding or None
        Fitted LLE estimator after `fit`.

    See Also
    --------
    IsomapReducer : Nonlinear geodesic-distance embedding.
    MDSReducer : Distance-preserving manifold embedding.
    SpectralEmbeddingReducer : Nonlinear graph Laplacian embedding.
    PCAReducer : Linear baseline for global variance preservation.
    UMAPReducer : Nonlinear graph-based embedding for local and global structure.
    TSNEReducer : Nonlinear neighborhood-preserving visualization method.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import LLEReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = LLEReducer(n_components=2, n_neighbors=10, eigen_solver="dense")
    >>> _ = reducer.fit(X)
    >>> reducer.transform(X[:6]).shape
    (6, 2)
    >>> embedding = reducer.fit_transform(X)
    >>> embedding.shape
    (100, 2)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for LLE.

        Returns
        -------
        dict
            Capability mapping describing LLE as a nonlinear reducer with
            out-of-sample transform support.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_transform=True,
            supported_metadata=("reconstruction_error_", "n_features_in_"),
            is_linear=False,
        )

    def __init__(self, n_components: int = 2, **kwargs):
        """
        Initialize the LLE reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of coordinates for the manifold.
        **kwargs : dict
            Additional keyword arguments forwarded to LLE after filtering.
        """
        super().__init__(n_components=n_components, **kwargs)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "LLEReducer":
        """
        Fit LLE on the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        LLEReducer
            Fitted reducer instance.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import LLEReducer
        >>> X = np.random.rand(30, 6)
        >>> reducer = LLEReducer(n_components=2, n_neighbors=5, eigen_solver="dense")
        >>> _ = reducer.fit(X)
        >>> reducer.model is not None
        True
        >>> reducer = LLEReducer(n_components=2, method="modified", n_neighbors=5)
        >>> _ = reducer.fit(X)
        >>> reducer.model is not None
        True
        """
        self.model = self._build_estimator(LocallyLinearEmbedding)
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Project data into the fitted LLE embedding space.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Data to project.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Low-dimensional embedding coordinates.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        self._require_fitted()
        return self.model.transform(X)

    @property
    def reconstruction_error_(self) -> float:
        """
        Return the LLE reconstruction error.

        Returns
        -------
        float
            Reconstruction error associated with the embedding.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        if self.model is None or not hasattr(self.model, "reconstruction_error_"):
            raise RuntimeError("Model is not fitted yet.")
        return self.model.reconstruction_error_


class MDSReducer(BaseReducer):
    """
    Multidimensional Scaling reducer.

    MDS seeks a low-dimensional representation whose pairwise distances best
    match the pairwise distances in the original space.

    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    **kwargs : dict
        Additional keyword arguments forwarded to `sklearn.manifold.MDS` after
        signature filtering. Common options include `metric`, `n_init`,
        `max_iter`, `dissimilarity`, and `random_state`.

    Attributes
    ----------
    model : sklearn.manifold.MDS or None
        Fitted MDS estimator after `fit` or `fit_transform`.

    Notes
    -----
    `transform` is not supported because scikit-learn MDS does not provide an
    out-of-sample projection API.

    See Also
    --------
    IsomapReducer : Nonlinear geodesic-distance embedding.
    LLEReducer : Nonlinear local-neighborhood embedding.
    SpectralEmbeddingReducer : Nonlinear graph Laplacian embedding.
    PCAReducer : Linear baseline for global variance preservation.
    UMAPReducer : Nonlinear graph-based embedding for local and global structure.
    TSNEReducer : Nonlinear neighborhood-preserving visualization method.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import MDSReducer
    >>> X = np.random.rand(60, 8)
    >>> reducer = MDSReducer(n_components=2, random_state=42)
    >>> embedding = reducer.fit_transform(X)
    >>> embedding.shape
    (60, 2)
    >>> reducer.stress_ >= 0
    True
    >>> _ = reducer.fit(X)
    >>> reducer.model is not None
    True
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for MDS.

        Returns
        -------
        dict
            Capability mapping describing MDS as a nonlinear reducer without
            out-of-sample transform support.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_transform=False,
            supported_metadata=("stress_", "n_iter_", "n_features_in_"),
            is_linear=False,
            is_stochastic=True,
        )

    def __init__(self, n_components: int = 2, **kwargs):
        """
        Initialize the MDS reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of coordinates for the manifold.
        **kwargs : dict
            Additional keyword arguments forwarded to `MDS` after filtering.
        """
        super().__init__(n_components=n_components, **kwargs)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "MDSReducer":
        """
        Fit MDS on the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        MDSReducer
            Fitted reducer instance.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import MDSReducer
        >>> X = np.random.rand(25, 5)
        >>> reducer = MDSReducer(n_components=2, random_state=0)
        >>> _ = reducer.fit(X)
        >>> reducer.model is not None
        True
        """
        self.model = self._build_estimator(MDS)
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Raise because scikit-learn MDS does not support out-of-sample transform.

        Parameters
        ----------
        X : ArrayLike
            Ignored input included for API compatibility.

        Raises
        ------
        NotImplementedError
            Always raised because MDS does not support transforming new data.
        """
        raise NotImplementedError("MDS cannot transform new data. Use fit_transform().")

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Fit MDS and return the embedding coordinates.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Embedded coordinates produced by MDS.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import MDSReducer
        >>> X = np.random.rand(20, 4)
        >>> reducer = MDSReducer(n_components=2, random_state=0)
        >>> reducer.fit_transform(X).shape
        (20, 2)
        """
        self.model = self._build_estimator(MDS)
        return self.model.fit_transform(X)

    @property
    def stress_(self) -> float:
        """
        Return the MDS stress (sum of squared distances mismatch).

        Returns
        -------
        float
            Stress value returned by the fitted MDS model.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        if self.model is None or not hasattr(self.model, "stress_"):
            raise RuntimeError("Model is not fitted yet.")
        return self.model.stress_


class SpectralEmbeddingReducer(BaseReducer):
    """
    Spectral Embedding reducer.

    Spectral Embedding computes a nonlinear embedding using eigenvectors of the
    graph Laplacian built from the data affinity graph.

    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    **kwargs : dict
        Additional keyword arguments forwarded to
        `sklearn.manifold.SpectralEmbedding` after signature filtering. Common
        options include `affinity`, `gamma`, `random_state`, `eigen_solver`,
        and `n_neighbors`.

    Attributes
    ----------
    model : sklearn.manifold.SpectralEmbedding or None
        Fitted spectral embedding estimator after `fit` or `fit_transform`.

    Notes
    -----
    `transform` is not supported because scikit-learn SpectralEmbedding does
    not provide an out-of-sample projection API.

    See Also
    --------
    IsomapReducer : Nonlinear geodesic-distance embedding.
    LLEReducer : Nonlinear local-neighborhood embedding.
    MDSReducer : Distance-preserving manifold embedding.
    PCAReducer : Linear baseline for global variance preservation.
    UMAPReducer : Nonlinear graph-based embedding for local and global structure.
    TSNEReducer : Nonlinear neighborhood-preserving visualization method.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import SpectralEmbeddingReducer
    >>> X = np.random.rand(80, 10)
    >>> reducer = SpectralEmbeddingReducer(n_components=2, random_state=42)
    >>> embedding = reducer.fit_transform(X)
    >>> embedding.shape
    (80, 2)
    >>> _ = reducer.fit(X)
    >>> reducer.model is not None
    True
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for Spectral Embedding.

        Returns
        -------
        dict
            Capability mapping describing Spectral Embedding as a nonlinear
            reducer without out-of-sample transform support.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_transform=False,
            supported_metadata=("n_features_in_",),
            is_linear=False,
        )

    def __init__(self, n_components: int = 2, **kwargs):
        """
        Initialize the Spectral Embedding reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of coordinates for the manifold.
        **kwargs : dict
            Additional keyword arguments forwarded to `SpectralEmbedding` after
            filtering.
        """
        super().__init__(n_components=n_components, **kwargs)

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] = None
    ) -> "SpectralEmbeddingReducer":
        """
        Fit Spectral Embedding on the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        SpectralEmbeddingReducer
            Fitted reducer instance.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import SpectralEmbeddingReducer
        >>> X = np.random.rand(30, 6)
        >>> reducer = SpectralEmbeddingReducer(n_components=2, random_state=0)
        >>> _ = reducer.fit(X)
        >>> reducer.model is not None
        True
        """
        self.model = self._build_estimator(SpectralEmbedding)
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Raise because scikit-learn Spectral Embedding lacks out-of-sample transform.

        Parameters
        ----------
        X : ArrayLike
            Ignored input included for API compatibility.

        Raises
        ------
        NotImplementedError
            Always raised because Spectral Embedding does not support
            transforming new data.
        """
        raise NotImplementedError(
            "SpectralEmbedding does not support out-of-sample transformation. "
            "Use fit_transform()."
        )

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Fit Spectral Embedding and return the embedding coordinates.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Embedded coordinates produced by Spectral Embedding.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import SpectralEmbeddingReducer
        >>> X = np.random.rand(20, 4)
        >>> reducer = SpectralEmbeddingReducer(n_components=2, random_state=0)
        >>> reducer.fit_transform(X).shape
        (20, 2)
        """
        self.model = self._build_estimator(SpectralEmbedding)
        return self.model.fit_transform(X)
