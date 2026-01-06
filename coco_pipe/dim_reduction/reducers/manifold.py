"""
Manifold Learning Reducers
==========================

This module implements nonlinear dimensionality reduction techniques based on
manifold learning. It wraps Scikit-Learn's manifold learning algorithms.

Classes
-------
IsomapReducer
    Isometric Mapping (Isomap).
LLEReducer
    Locally Linear Embedding (LLE).
MDSReducer
    Multidimensional Scaling (MDS).
SpectralEmbeddingReducer
    Spectral Embedding (Laplacian Eigenmaps).

References
----------
.. [1] Tenenbaum, J.B.; De Silva, V.; Langford, J.C. (2000). A global geometric framework for
       nonlinear dimensionality reduction. Science. 290 (5500): 2319-2323.
.. [2] Roweis, S. T., & Saul, L. K. (2000). Nonlinear dimensionality reduction by locally
       linear embedding. Science, 290(5500), 2323-2326.
.. [3] Borg, I., & Groenen, P. J. (2005). Modern multidimensional scaling: Theory and
       applications. Springer.
.. [4] Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality reduction
       and data representation. Neural computation, 15(6), 1373-1396.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
Date: 2026-01-06
"""

from typing import Optional, Any
import numpy as np
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding

from .base import BaseReducer, ArrayLike


class IsomapReducer(BaseReducer):
    """
    Isometric Mapping (Isomap) reducer.

    Isomap is a nonlinear dimensionality reduction method that combines the key
    features of PCA and MDS. It estimates the geodesic distance between all
    points and then uses MDS to compute the low-dimensional embedding.

    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    **kwargs : dict
        Additional arguments passed to sklearn.manifold.Isomap.
        Common arguments:
        - n_neighbors : int, default=5
        - metric : str, default='minkowski'
        - p : int, default=2
        - eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'

    Attributes
    ----------
    model : sklearn.manifold.Isomap
        The underlying fitted Isomap estimator.
    
    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.reducers.manifold import IsomapReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = IsomapReducer(n_components=2, n_neighbors=5)
    >>> X_reduced = reducer.fit_transform(X)
    >>> print(X_reduced.shape)
    (100, 2)
    >>> print(f"{reducer.n_features_in_}")
    10
    """

    def __init__(self, n_components: int = 2, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.model = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "IsomapReducer":
        """
        Fit the Isomap model.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used.

        Returns
        -------
        self : IsomapReducer
            Returns the instance itself.
        """
        self.model = Isomap(n_components=self.n_components, **self.params)
        self.model.fit(X)
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
            raise RuntimeError("IsomapReducer must be fitted before calling transform().")
        return self.model.transform(X)
    
    @property
    def reconstruction_error_(self) -> float:
        """
        Reconstruction error associated with the embedding.
        
        Returns
        -------
        reconstruction_error_ : float or None
        """
        if self.model is None:
             raise RuntimeError("Model is not fitted yet.")
        return getattr(self.model, "reconstruction_error_", None)

    @property
    def n_features_in_(self) -> int:
        """
        Number of features seen during fit.
        """
        if self.model is None:
             raise RuntimeError("Model is not fitted yet.")
        return self.model.n_features_in_


class LLEReducer(BaseReducer):
    """
    Locally Linear Embedding (LLE) reducer.

    LLE seeks a lower-dimensional projection of the data which preserves distances
    within local neighborhoods. It can be thought of as a series of local Principal
    Component Analyses which are globally compared to find the best non-linear embedding.

    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    **kwargs : dict
        Additional arguments passed to sklearn.manifold.LocallyLinearEmbedding.
        Common arguments:
        - n_neighbors : int, default=5
        - method : {'standard', 'hessian', 'modified', 'ltsa'}, default='standard'
        - eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
        - random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    model : sklearn.manifold.LocallyLinearEmbedding
        The underlying fitted LLE estimator.
    
    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.reducers.manifold import LLEReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = LLEReducer(n_components=2, n_neighbors=10)
    >>> X_reduced = reducer.fit_transform(X)
    >>> print(X_reduced.shape)
    (100, 2)
    >>> print(f"{reducer.reconstruction_error_:.4e}")
    """

    def __init__(self, n_components: int = 2, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.model = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "LLEReducer":
        """
        Fit the LLE model.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used.

        Returns
        -------
        self : LLEReducer
            Returns the instance itself.
        """
        self.model = LocallyLinearEmbedding(n_components=self.n_components, **self.params)
        self.model.fit(X)
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
            raise RuntimeError("LLEReducer must be fitted before calling transform().")
        return self.model.transform(X)

    @property
    def reconstruction_error_(self) -> float:
        """
        Reconstruction error associated with the embedding.
        
        Returns
        -------
        reconstruction_error_ : float
        """
        if self.model is None or not hasattr(self.model, "reconstruction_error_"):
             raise RuntimeError("Model is not fitted yet.")
        return self.model.reconstruction_error_


class MDSReducer(BaseReducer):
    """
    Multidimensional Scaling (MDS) reducer.

    MDS seeks a low-dimensional representation of the data in which the distances
    respect well the distances in the original high-dimensional space.

    Note: MDS is computationally expensive for large datasets.

    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    **kwargs : dict
        Additional arguments passed to sklearn.manifold.MDS.
        Common arguments:
        - metric : bool, default=True (True for metric MDS, False for non-metric MDS)
        - n_init : int, default=4
        - max_iter : int, default=300
        - dissimilarity : {'euclidean', 'precomputed'}, default='euclidean'
        - random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    model : sklearn.manifold.MDS
        The underlying MDS estimator.
    
    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.reducers.manifold import MDSReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = MDSReducer(n_components=2)
    >>> X_reduced = reducer.fit_transform(X)
    >>> print(X_reduced.shape)
    (100, 2)
    >>> print(f"{reducer.stress_:.4f}")
    """

    def __init__(self, n_components: int = 2, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.model = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "MDSReducer":
        """
        Fit the MDS model.
        
        Note: MDS does not implement a separate fit/transform paradigm in the same
        way as other estimators in scikit-learn. fit_transform remains the primary usage.
        However, for consistency, we initialize the model here.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used.

        Returns
        -------
        self : MDSReducer
            Returns the instance itself.
        """
        self.model = MDS(n_components=self.n_components, **self.params)
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform X.
        
        Raises
        ------
        NotImplementedError
            MDS does not support transforming new data.
        """
        raise NotImplementedError("MDS cannot transform new data. Use fit_transform().")
    
    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Fit the data from X, and returns the embedded coordinates.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_components)
            Transformed data (embedding).
        """
        self.model = MDS(n_components=self.n_components, **self.params)
        return self.model.fit_transform(X)

    @property
    def stress_(self) -> float:
        """
        The final value of the stress (sum of squared distance of the disparities and the 
        distances for all constrained points).
        
        Returns
        -------
        stress_ : float
        """
        if self.model is None or not hasattr(self.model, "stress_"):
             raise RuntimeError("Model is not fitted yet.")
        return self.model.stress_


class SpectralEmbeddingReducer(BaseReducer):
    """
    Spectral Embedding (Laplacian Eigenmaps) reducer.

    Uses the Laplacian of the graph (formed by the data points) to perform dimensionality
    reduction. This is a non-linear method.

    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    **kwargs : dict
        Additional arguments passed to sklearn.manifold.SpectralEmbedding.
        Common arguments:
        - affinity : str or callable, default='nearest_neighbors'
        - gamma : float, default=None
        - random_state : int, RandomState instance or None, default=None
        - eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        - n_neighbors : int, default=None

    Attributes
    ----------
    model : sklearn.manifold.SpectralEmbedding
        The underlying estimator.
    
    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.reducers.manifold import SpectralEmbeddingReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = SpectralEmbeddingReducer(n_components=2)
    >>> X_reduced = reducer.fit_transform(X)
    >>> print(X_reduced.shape)
    (100, 2)
    """

    def __init__(self, n_components: int = 2, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.model = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "SpectralEmbeddingReducer":
        """
        Fit the Spectral model.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used.

        Returns
        -------
        self : SpectralEmbeddingReducer
            Returns the instance itself.
        """
        self.model = SpectralEmbedding(n_components=self.n_components, **self.params)
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform X.

        Raises
        ------
        NotImplementedError
            SpectralEmbedding only provides fit_transform; it does not support
            out-of-sample transformation in scikit-learn.
        """
        raise NotImplementedError(
            "SpectralEmbedding does not support out-of-sample transformation. Use fit_transform()."
        )

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Fit the model from data in X and transform X.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_components)
            Transformed data (embedding).
        """
        self.model = SpectralEmbedding(n_components=self.n_components, **self.params)
        return self.model.fit_transform(X)
