"""
Linear Dimensionality Reduction
===============================

This module implements linear dimensionality reduction techniques.
Currently supports Principal Component Analysis (PCA).

Classes
-------
PCAReducer
    Wrapper for Scikit-Learn's PCA.

References
----------
.. [1] Pearson, K. (1901). "On Lines and Planes of Closest Fit to Systems of Points in Space".
       Philosophical Magazine. 2 (11): 559-572.
.. [2] Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components.
       Journal of Educational Psychology, 24(6), 417-441.
.. [3] Scikit-Learn PCA Documentation:
       https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
Date: 2026-01-06
"""

from typing import Optional, Any
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

import dask.array as da
from dask_ml.decomposition import PCA as DaskPCA
from dask_ml.decomposition import TruncatedSVD as DaskTruncatedSVD

from .base import BaseReducer, ArrayLike


class PCAReducer(BaseReducer):
    """
    Principal Component Analysis (PCA) reducer.

    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.
    **kwargs : dict
        Additional arguments passed to sklearn.decomposition.PCA.
        Common arguments include:
        - whiten : bool, default=False
        - svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        - random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    model : sklearn.decomposition.PCA
        The underlying fitted PCA estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.linear import PCAReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = PCAReducer(n_components=2, random_state=42)
    >>> X_reduced = reducer.fit_transform(X)
    >>> print(X_reduced.shape)
    (100, 2)
    >>> print(reducer.explained_variance_ratio_)
    (2,)
    >>> print(reducer.components_)
    (2, 10)
    """

    def __init__(self, n_components: int = 2, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.model = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "PCAReducer":
        """
        Fit the model with X.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : PCAReducer
            Returns the instance itself.
        """
        self.model = PCA(n_components=self.n_components, **self.params)
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_components)
            Projection of X in the first principal components.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError("PCAReducer must be fitted before calling transform().")
        
        return self.model.transform(X)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """
        Percentage of variance explained by each of the selected components.

        Returns
        -------
        explained_variance_ratio_ : np.ndarray of shape (n_components,)
            The percentage of variance explained by each of the selected components.
        
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.explained_variance_ratio_

    @property
    def components_(self) -> np.ndarray:
        """
        Principal axes in feature space, representing the directions of maximum variance.

        Returns
        -------
        components_ : np.ndarray of shape (n_components, n_features)
            The principal components.
        
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.components_


class IncrementalPCAReducer(BaseReducer):
    """
    Incremental PCA reducer.
    
    Incremental Principal Component Analysis (IPCA) is typically used as a replacement 
    for Principal Component Analysis (PCA) when the dataset to be decomposed is too 
    large to fit in memory.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.
    batch_size : int, default=None
        The number of samples to use for each batch.
    **kwargs : dict
        Additional arguments passed to sklearn.decomposition.IncrementalPCA.
    
    Attributes
    ----------
    model : sklearn.decomposition.IncrementalPCA
        The underlying fitted estimator.
    """
    
    def __init__(self, n_components: int = 2, batch_size: Optional[int] = None, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.batch_size = batch_size
        self.model = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "IncrementalPCAReducer":
        """Fit model."""
        self.model = IncrementalPCA(
            n_components=self.n_components, 
            batch_size=self.batch_size,
            **self.params
        )
        self.model.fit(X)
        return self
        
    def partial_fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "IncrementalPCAReducer":
        """Incremental fit."""
        if self.model is None:
             self.model = IncrementalPCA(
                n_components=self.n_components, 
                batch_size=self.batch_size,
                **self.params
            )
        self.model.partial_fit(X, y=y)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform X."""
        if self.model is None:
            raise RuntimeError("IncrementalPCAReducer must be fitted before calling transform().")
        return self.model.transform(X)


class DaskPCAReducer(BaseReducer):
    """
    Dask-based PCA reducer for large-scale data.
    
    Wraps dask_ml.decomposition.PCA. Expects Dask Arrays or compatible inputs.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.
    svd_solver : str, default='auto'
        Solver to use: 'auto', 'full', 'tsqr', 'randomized'.
    **kwargs : dict
        Additional arguments.
    """
    
    def __init__(self, n_components: int = 2, svd_solver: str = 'auto', **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.svd_solver = svd_solver
        self.model = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "DaskPCAReducer":             
        self.model = DaskPCA(
            n_components=self.n_components,
            svd_solver=self.svd_solver,
            **self.params
        )
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> Any:
        # Returns a dask array
        if self.model is None:
            raise RuntimeError("DaskPCAReducer must be fitted.")
        return self.model.transform(X)


class DaskTruncatedSVDReducer(BaseReducer):
    """
    Dask-based Truncated SVD reducer.
    
    Wraps dask_ml.decomposition.TruncatedSVD.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of components.
    algorithm : str, default='tsqr'
        SVD algorithm: 'tsqr', 'randomized'.
    **kwargs : dict
        Additional arguments.
    """
    
    def __init__(self, n_components: int = 2, algorithm: str = 'tsqr', **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.algorithm = algorithm
        self.model = None
        
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "DaskTruncatedSVDReducer":
        if DaskTruncatedSVD is None:
             raise ImportError("dask-ml is required.")
             
        self.model = DaskTruncatedSVD(
            n_components=self.n_components,
            algorithm=self.algorithm,
            **self.params
        )
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> Any:
        if self.model is None:
             raise RuntimeError("DaskTruncatedSVDReducer must be fitted.")
        return self.model.transform(X)
