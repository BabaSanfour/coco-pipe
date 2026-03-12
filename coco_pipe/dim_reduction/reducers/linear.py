"""
Linear dimensionality reduction reducers.

This module provides linear projection wrappers built on top of scikit-learn
and optional Dask backends. These reducers follow the shared `BaseReducer`
contract so they can be used directly with `DimReduction`, reporting, and
visualization utilities.

Classes
-------
PCAReducer
    Principal Component Analysis wrapper based on `sklearn.decomposition.PCA`.
IncrementalPCAReducer
    Incremental PCA wrapper for batch-wise fitting on larger datasets.
DaskPCAReducer
    Optional Dask-ML PCA wrapper for lazy or distributed arrays.
DaskTruncatedSVDReducer
    Optional Dask-ML Truncated SVD wrapper for lazy or distributed arrays.

References
----------
.. [1] Pearson, K. (1901). "On Lines and Planes of Closest Fit to Systems of
       Points in Space". Philosophical Magazine, 2(11), 559-572.
.. [2] Hotelling, H. (1933). "Analysis of a complex of statistical variables
       into principal components". Journal of Educational Psychology, 24(6),
       417-441.
.. [3] Scikit-learn PCA documentation:
       https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from typing import Any, Optional

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

from ...utils import import_optional_dependency
from .base import ArrayLike, BaseReducer

__all__ = [
    "PCAReducer",
    "IncrementalPCAReducer",
    "DaskPCAReducer",
    "DaskTruncatedSVDReducer",
]

_LINEAR_DIAGNOSTIC_ATTRS = (
    "explained_variance_ratio_",
    "singular_values_",
)


def _get_components(model: Any) -> np.ndarray:
    """
    Return PCA-like components from a fitted estimator.

    Parameters
    ----------
    model : Any
        Fitted estimator instance.

    Returns
    -------
    np.ndarray
        Component matrix exposed by the estimator.

    Raises
    ------
    RuntimeError
        If the estimator is not fitted or does not expose `components_`.
    """
    if model is None or not hasattr(model, "components_"):
        raise RuntimeError("Model is not fitted yet.")
    return model.components_


class PCAReducer(BaseReducer):
    """
    Principal Component Analysis reducer.

    This reducer wraps `sklearn.decomposition.PCA` and provides a linear
    low-dimensional embedding based on singular value decomposition.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components to keep.
    **kwargs : dict
        Additional keyword arguments forwarded to
        `sklearn.decomposition.PCA` after signature filtering. Common options
        include `whiten`, `svd_solver`, and `random_state`.

    Attributes
    ----------
    model : sklearn.decomposition.PCA or None
        Fitted PCA estimator after `fit`.

    Notes
    -----
    This is a deterministic linear reducer unless a randomized solver is used.

    See Also
    --------
    IncrementalPCAReducer : Linear PCA variant for batch-wise fitting.
    DaskPCAReducer : Linear PCA variant for lazy or distributed arrays.
    DaskTruncatedSVDReducer : Linear factorization alternative for lazy arrays.
    IsomapReducer : Nonlinear manifold learner based on geodesic distances.
    TSNEReducer : Nonlinear neighborhood-preserving embedding.
    UMAPReducer : Nonlinear graph-based embedding balancing local and global structure.
    PHATEReducer : Nonlinear diffusion-based embedding for smooth trajectories.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import PCAReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = PCAReducer(n_components=2, random_state=42)
    >>> _ = reducer.fit(X)
    >>> X_reduced = reducer.transform(X)
    >>> X_reduced.shape
    (100, 2)
    >>> reducer.explained_variance_ratio_.shape
    (2,)
    >>> reducer.components_.shape
    (2, 10)
    >>> reducer = PCAReducer(n_components=3, whiten=True)
    >>> reducer.fit_transform(X).shape
    (100, 3)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for PCA.

        Returns
        -------
        dict
            Capability mapping describing PCA as a linear component-based
            reducer.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_components=True,
            supported_diagnostics=_LINEAR_DIAGNOSTIC_ATTRS,
            supported_metadata=("n_components_", "noise_variance_"),
            is_linear=True,
        )

    def __init__(self, n_components: int = 2, **kwargs):
        """
        Initialize the PCA reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of principal components to keep.
        **kwargs : dict
            Additional keyword arguments forwarded to `PCA` after filtering.
        """
        super().__init__(n_components=n_components, **kwargs)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "PCAReducer":
        """
        Fit PCA on the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        PCAReducer
            Fitted reducer instance.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import PCAReducer
        >>> X = np.random.rand(20, 5)
        >>> reducer = PCAReducer(n_components=2)
        >>> _ = reducer.fit(X)
        >>> reducer.model is not None
        True
        """
        self.model = self._build_estimator(PCA)
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Project data onto the fitted principal component basis.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Data to project.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Projected coordinates in principal component space.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        self._require_fitted()
        return self.model.transform(X)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """
        Percentage of variance explained by each selected component.

        Returns
        -------
        np.ndarray of shape (n_components,)
            Explained variance ratio for each retained component.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.explained_variance_ratio_

    @property
    def components_(self) -> np.ndarray:
        """
        Principal axes in feature space.

        Returns
        -------
        np.ndarray of shape (n_components, n_features)
            Principal component loading matrix.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        return _get_components(self.model)

    def get_components(self) -> np.ndarray:
        """
        Return the principal component loading matrix.

        Returns
        -------
        np.ndarray
            Principal component loading matrix.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        return self.components_


class IncrementalPCAReducer(BaseReducer):
    """
    Incremental PCA reducer.

    This reducer wraps `sklearn.decomposition.IncrementalPCA` for batch-wise
    fitting when the full dataset is too large to process in one pass.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components to keep.
    batch_size : int, optional
        Number of samples processed per batch.
    **kwargs : dict
        Additional keyword arguments forwarded to `IncrementalPCA` after
        signature filtering.

    Attributes
    ----------
    batch_size : int or None
        Batch size used when fitting the incremental estimator.
    model : sklearn.decomposition.IncrementalPCA or None
        Fitted IncrementalPCA estimator after `fit` or `partial_fit`.

    See Also
    --------
    PCAReducer : Standard in-memory linear PCA reducer.
    DaskPCAReducer : Linear PCA variant for lazy or distributed arrays.
    DaskTruncatedSVDReducer : Linear factorization alternative for lazy arrays.
    IsomapReducer : Nonlinear manifold learner based on geodesic distances.
    TSNEReducer : Nonlinear neighborhood-preserving embedding.
    UMAPReducer : Nonlinear graph-based embedding balancing local and global structure.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import IncrementalPCAReducer
    >>> X = np.random.rand(100, 12)
    >>> reducer = IncrementalPCAReducer(n_components=3, batch_size=25)
    >>> _ = reducer.fit(X)
    >>> reducer.transform(X[:10]).shape
    (10, 3)
    >>> stream = IncrementalPCAReducer(n_components=2, batch_size=20)
    >>> _ = stream.partial_fit(X[:50])
    >>> _ = stream.partial_fit(X[50:])
    >>> stream.transform(X).shape
    (100, 2)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for Incremental PCA.

        Returns
        -------
        dict
            Capability mapping describing Incremental PCA as a linear
            component-based reducer.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_components=True,
            supported_diagnostics=_LINEAR_DIAGNOSTIC_ATTRS,
            supported_metadata=("n_components_", "noise_variance_", "n_samples_seen_"),
            is_linear=True,
        )

    def __init__(
        self, n_components: int = 2, batch_size: Optional[int] = None, **kwargs
    ):
        """
        Initialize the incremental PCA reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of principal components to keep.
        batch_size : int, optional
            Number of samples processed per batch.
        **kwargs : dict
            Additional keyword arguments forwarded to `IncrementalPCA` after
            filtering.
        """
        super().__init__(n_components=n_components, **kwargs)
        self.batch_size = batch_size

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] = None
    ) -> "IncrementalPCAReducer":
        """
        Fit Incremental PCA in batch mode.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        IncrementalPCAReducer
            Fitted reducer instance.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import IncrementalPCAReducer
        >>> X = np.random.rand(30, 6)
        >>> reducer = IncrementalPCAReducer(n_components=2, batch_size=10)
        >>> _ = reducer.fit(X)
        >>> reducer.model is not None
        True
        """
        self.model = self._build_estimator(
            IncrementalPCA,
            batch_size=self.batch_size,
        )
        self.model.fit(X)
        return self

    def partial_fit(
        self, X: ArrayLike, y: Optional[ArrayLike] = None
    ) -> "IncrementalPCAReducer":
        """
        Incrementally fit the estimator on a batch of samples.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Batch of training samples.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        IncrementalPCAReducer
            Reducer instance after updating the incremental estimator.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import IncrementalPCAReducer
        >>> X = np.random.rand(40, 6)
        >>> reducer = IncrementalPCAReducer(n_components=2, batch_size=20)
        >>> _ = reducer.partial_fit(X[:20])
        >>> _ = reducer.partial_fit(X[20:])
        >>> reducer.model is not None
        True
        """
        if self.model is None:
            self.model = self._build_estimator(
                IncrementalPCA,
                batch_size=self.batch_size,
            )
        self.model.partial_fit(X, y=y)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Project data onto the fitted incremental PCA basis.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Data to project.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Projected coordinates in component space.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        self._require_fitted()
        return self.model.transform(X)

    def get_components(self) -> np.ndarray:
        """
        Return the incremental PCA component loading matrix.

        Returns
        -------
        np.ndarray
            Principal component loading matrix.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        return _get_components(self.model)


class DaskPCAReducer(BaseReducer):
    """
    Dask-ML PCA reducer for lazy or distributed data.

    This reducer wraps `dask_ml.decomposition.PCA`. The backend is imported
    lazily so the rest of the package remains importable without `dask-ml`.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components to keep.
    svd_solver : {"auto", "full", "tsqr", "randomized"}, default="auto"
        Solver used by the Dask PCA backend.
    **kwargs : dict
        Additional keyword arguments forwarded to `dask_ml.decomposition.PCA`
        after signature filtering.

    Attributes
    ----------
    svd_solver : str
        Solver used when instantiating the Dask PCA estimator.
    model : dask_ml.decomposition.PCA or None
        Fitted Dask PCA estimator after `fit`.

    Notes
    -----
    This reducer requires the optional `dask-ml` backend.

    See Also
    --------
    PCAReducer : Standard in-memory linear PCA reducer.
    IncrementalPCAReducer : Linear PCA variant for batch-wise fitting.
    DaskTruncatedSVDReducer : Linear SVD-based alternative for lazy arrays.
    IsomapReducer : Nonlinear manifold learner based on geodesic distances.
    TSNEReducer : Nonlinear neighborhood-preserving embedding.
    UMAPReducer : Nonlinear graph-based embedding balancing local and global structure.

    Examples
    --------
    >>> import dask.array as da
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import DaskPCAReducer
    >>> X = da.from_array(np.random.rand(100, 10), chunks=(25, 10))
    >>> reducer = DaskPCAReducer(n_components=2, svd_solver="tsqr")
    >>> _ = reducer.fit(X)
    >>> reducer.transform(X).shape
    (100, 2)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for Dask PCA.

        Returns
        -------
        dict
            Capability mapping describing Dask PCA as a linear component-based
            reducer.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_components=True,
            supported_diagnostics=_LINEAR_DIAGNOSTIC_ATTRS,
            supported_metadata=("n_components_", "noise_variance_"),
            is_linear=True,
        )

    def __init__(self, n_components: int = 2, svd_solver: str = "auto", **kwargs):
        """
        Initialize the Dask PCA reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of principal components to keep.
        svd_solver : {"auto", "full", "tsqr", "randomized"}, default="auto"
            Solver used by the Dask PCA backend.
        **kwargs : dict
            Additional keyword arguments forwarded to the backend after
            filtering.
        """
        super().__init__(n_components=n_components, **kwargs)
        self.svd_solver = svd_solver

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "DaskPCAReducer":
        """
        Fit Dask PCA on the input data.

        Parameters
        ----------
        X : ArrayLike
            Training data, typically a Dask array or a compatible array-like
            object accepted by the Dask backend.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        DaskPCAReducer
            Fitted reducer instance.

        Raises
        ------
        ImportError
            If `dask-ml` is not installed.
        RuntimeError
            If `dask-ml` is installed but fails during initialization.

        Examples
        --------
        >>> import dask.array as da
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import DaskPCAReducer
        >>> X = da.from_array(np.random.rand(40, 8), chunks=(20, 8))
        >>> reducer = DaskPCAReducer(n_components=2)
        >>> _ = reducer.fit(X)
        >>> reducer.model is not None
        True
        """
        dask_pca = import_optional_dependency(
            lambda: __import__("dask_ml.decomposition", fromlist=["PCA"]).PCA,
            feature="DaskPCAReducer",
            dependency="dask-ml",
            install_hint="pip install coco-pipe[dask]",
        )

        self.model = self._build_estimator(
            dask_pca,
            svd_solver=self.svd_solver,
        )
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> Any:
        """
        Project data using the fitted Dask PCA model.

        Parameters
        ----------
        X : ArrayLike
            Data to project.

        Returns
        -------
        Any
            Backend-specific transformed output, typically a Dask array.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        self._require_fitted()
        return self.model.transform(X)

    def get_components(self) -> np.ndarray:
        """
        Return the Dask PCA component loading matrix.

        Returns
        -------
        np.ndarray
            Principal component loading matrix or Dask-backed equivalent.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        return _get_components(self.model)


class DaskTruncatedSVDReducer(BaseReducer):
    """
    Dask-ML Truncated SVD reducer.

    This reducer wraps `dask_ml.decomposition.TruncatedSVD` and provides a
    linear projection for lazy or distributed arrays.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.
    algorithm : {"tsqr", "randomized"}, default="tsqr"
        SVD algorithm used by the Dask backend.
    **kwargs : dict
        Additional keyword arguments forwarded to
        `dask_ml.decomposition.TruncatedSVD` after signature filtering.

    Attributes
    ----------
    algorithm : str
        SVD algorithm used when instantiating the backend estimator.
    model : dask_ml.decomposition.TruncatedSVD or None
        Fitted TruncatedSVD estimator after `fit`.

    Notes
    -----
    This reducer requires the optional `dask-ml` backend.

    See Also
    --------
    PCAReducer : Standard in-memory linear PCA reducer.
    IncrementalPCAReducer : Linear PCA variant for batch-wise fitting.
    DaskPCAReducer : Linear PCA variant for lazy or distributed arrays.
    IsomapReducer : Nonlinear manifold learner based on geodesic distances.
    TSNEReducer : Nonlinear neighborhood-preserving embedding.
    UMAPReducer : Nonlinear graph-based embedding balancing local and global structure.

    Examples
    --------
    >>> import dask.array as da
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import DaskTruncatedSVDReducer
    >>> X = da.from_array(np.random.rand(120, 15), chunks=(30, 15))
    >>> reducer = DaskTruncatedSVDReducer(n_components=3, algorithm="randomized")
    >>> _ = reducer.fit(X)
    >>> reducer.transform(X).shape
    (120, 3)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for Dask Truncated SVD.

        Returns
        -------
        dict
            Capability mapping describing Dask Truncated SVD as a linear
            component-based reducer.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_components=True,
            supported_diagnostics=_LINEAR_DIAGNOSTIC_ATTRS,
            supported_metadata=("algorithm",),
            is_linear=True,
        )

    def __init__(self, n_components: int = 2, algorithm: str = "tsqr", **kwargs):
        """
        Initialize the Dask Truncated SVD reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of components to keep.
        algorithm : {"tsqr", "randomized"}, default="tsqr"
            SVD algorithm used by the Dask backend.
        **kwargs : dict
            Additional keyword arguments forwarded to the backend after
            filtering.
        """
        super().__init__(n_components=n_components, **kwargs)
        self.algorithm = algorithm

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] = None
    ) -> "DaskTruncatedSVDReducer":
        """
        Fit Dask Truncated SVD on the input data.

        Parameters
        ----------
        X : ArrayLike
            Training data, typically a Dask array or compatible array-like
            object accepted by the backend.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        DaskTruncatedSVDReducer
            Fitted reducer instance.

        Raises
        ------
        ImportError
            If `dask-ml` is not installed.
        RuntimeError
            If `dask-ml` is installed but fails during initialization.

        Examples
        --------
        >>> import dask.array as da
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import DaskTruncatedSVDReducer
        >>> X = da.from_array(np.random.rand(40, 8), chunks=(20, 8))
        >>> reducer = DaskTruncatedSVDReducer(n_components=2)
        >>> _ = reducer.fit(X)
        >>> reducer.model is not None
        True
        """
        dask_truncated_svd = import_optional_dependency(
            lambda: (
                __import__(
                    "dask_ml.decomposition", fromlist=["TruncatedSVD"]
                ).TruncatedSVD
            ),
            feature="DaskTruncatedSVDReducer",
            dependency="dask-ml",
            install_hint="pip install coco-pipe[dask]",
        )

        self.model = self._build_estimator(
            dask_truncated_svd,
            algorithm=self.algorithm,
        )
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> Any:
        """
        Project data using the fitted Dask Truncated SVD model.

        Parameters
        ----------
        X : ArrayLike
            Data to project.

        Returns
        -------
        Any
            Backend-specific transformed output, typically a Dask array.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        self._require_fitted()
        return self.model.transform(X)

    def get_components(self) -> np.ndarray:
        """
        Return the Truncated SVD component loading matrix.

        Returns
        -------
        np.ndarray
            Component loading matrix or Dask-backed equivalent.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        return _get_components(self.model)
