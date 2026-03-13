"""
Neighbor-embedding and graph-based reducers.

This module provides wrappers for neighborhood-preserving and graph-based
nonlinear dimensionality reduction methods, including t-SNE, UMAP, PaCMAP,
TriMap, PHATE, and Parametric UMAP.

Classes
-------
TSNEReducer
    t-Distributed Stochastic Neighbor Embedding wrapper.
UMAPReducer
    Uniform Manifold Approximation and Projection wrapper.
PacmapReducer
    Pairwise Controlled Manifold Approximation wrapper.
TrimapReducer
    Triplet-based manifold embedding wrapper.
PHATEReducer
    Diffusion-based PHATE embedding wrapper.
ParametricUMAPReducer
    Neural-network-backed Parametric UMAP wrapper.

References
----------
.. [1] van der Maaten, L., and Hinton, G. (2008). "Visualizing data using
       t-SNE". Journal of Machine Learning Research, 9, 2579-2605.
.. [2] McInnes, L., Healy, J., and Melville, J. (2018). "UMAP: Uniform
       Manifold Approximation and Projection for Dimension Reduction". arXiv.
.. [3] Wang, Y., et al. (2021). "PaCMAP: Pairwise Controlled Manifold
       Approximation". Journal of Machine Learning Research, 22(201), 1-47.
.. [4] Amid, E., and Warmuth, M. K. (2019). "TriMap: Large-scale
       Dimensionality Reduction Using Triplets". arXiv.
.. [5] Moon, K. R., et al. (2019). "Visualizing structure and transitions in
       high-dimensional biological data". Nature Biotechnology, 37, 1482-1492.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
        Sina Esmaeili (sina.esmaeili@umontreal.ca)
"""

from typing import Any, Optional

import numpy as np
from sklearn.manifold import TSNE

from ...utils import import_optional_dependency
from .base import ArrayLike, BaseReducer

__all__ = [
    "TSNEReducer",
    "UMAPReducer",
    "PacmapReducer",
    "TrimapReducer",
    "PHATEReducer",
    "ParametricUMAPReducer",
]


class TSNEReducer(BaseReducer):
    """
    t-SNE reducer.

    t-Distributed Stochastic Neighbor Embedding (t-SNE) is a neighborhood-
    preserving method designed primarily for visualization. It optimizes a
    low-dimensional embedding by matching pairwise similarities between the
    original space and the embedding.

    Parameters
    ----------
    n_components : int, default=2
        Number of embedding dimensions.
    **kwargs : dict
        Additional keyword arguments forwarded to
        `sklearn.manifold.TSNE` after signature filtering. Common options
        include `perplexity`, `learning_rate`, `max_iter`, `init`, and
        `random_state`.

    Attributes
    ----------
    embedding_ : np.ndarray or None
        Learned training-set embedding after `fit` or `fit_transform`.
    model : sklearn.manifold.TSNE or None
        Fitted t-SNE estimator after `fit` or `fit_transform`.

    Notes
    -----
    `transform` is not supported because scikit-learn t-SNE does not provide
    an out-of-sample projection API.

    See Also
    --------
    UMAPReducer : Nonlinear graph-based embedding with transform support.
    PacmapReducer : Nonlinear embedding balancing local and global structure.
    TrimapReducer : Nonlinear triplet-based embedding preserving global layout.
    PHATEReducer : Diffusion-based embedding for continuous trajectories.
    PCAReducer : Linear baseline for global variance preservation.
    IsomapReducer : Nonlinear geodesic-distance manifold embedding.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import TSNEReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = TSNEReducer(n_components=2, perplexity=20, random_state=42)
    >>> embedding = reducer.fit_transform(X)
    >>> embedding.shape
    (100, 2)
    >>> reducer.get_quality_metadata()["kl_divergence_"] >= 0
    True
    >>> _ = reducer.fit(X)
    >>> reducer.embedding_.shape
    (100, 2)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for t-SNE.

        Returns
        -------
        dict
            Capability mapping describing t-SNE as a nonlinear stochastic
            reducer without out-of-sample transform support.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_transform=False,
            supported_metadata=("kl_divergence_", "n_iter_", "learning_rate_"),
            is_linear=False,
            is_stochastic=True,
        )

    def __init__(self, n_components: int = 2, **kwargs):
        """
        Initialize the t-SNE reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of embedding dimensions.
        **kwargs : dict
            Additional keyword arguments forwarded to `TSNE` after filtering.
        """
        super().__init__(n_components=n_components, **kwargs)
        self.embedding_ = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "TSNEReducer":
        """
        Fit t-SNE on the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        TSNEReducer
            Fitted reducer instance.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import TSNEReducer
        >>> X = np.random.rand(30, 6)
        >>> reducer = TSNEReducer(n_components=2, perplexity=5, max_iter=250)
        >>> _ = reducer.fit(X)
        >>> reducer.model is not None
        True
        """
        self.model = self._build_estimator(TSNE)
        self.embedding_ = self.model.fit_transform(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Raise because t-SNE does not support out-of-sample transformation.

        Parameters
        ----------
        X : ArrayLike
            Ignored input included for API compatibility.

        Raises
        ------
        NotImplementedError
            Always raised because t-SNE does not support transforming new data.
        """
        raise NotImplementedError(
            "TSNEReducer cannot transform new data. Use fit_transform()."
        )

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Fit t-SNE and return the embedding coordinates.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Embedded coordinates produced by t-SNE.
        """
        self.fit(X, y=y)
        return self.embedding_


class UMAPReducer(BaseReducer):
    """
    UMAP reducer.

    Uniform Manifold Approximation and Projection (UMAP) constructs a graph in
    the high-dimensional space and optimizes a low-dimensional representation of
    that graph. Unlike t-SNE, UMAP supports out-of-sample transformation.

    Parameters
    ----------
    n_components : int, default=2
        Number of embedding dimensions.
    **kwargs : dict
        Additional keyword arguments forwarded to `umap.UMAP` after signature
        filtering. Common options include `n_neighbors`, `min_dist`, `metric`,
        and `random_state`.

    Attributes
    ----------
    model : umap.UMAP or None
        Fitted UMAP estimator after `fit`.

    See Also
    --------
    TSNEReducer : Nonlinear neighborhood-preserving visualization method.
    PacmapReducer : Nonlinear embedding balancing local and global structure.
    TrimapReducer : Nonlinear triplet-based embedding preserving global layout.
    PHATEReducer : Diffusion-based embedding for continuous trajectories.
    IsomapReducer : Nonlinear geodesic-distance manifold embedding.
    PCAReducer : Linear baseline for global variance preservation.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import UMAPReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = UMAPReducer(n_components=2, n_neighbors=10, random_state=42)
    >>> _ = reducer.fit(X)
    >>> reducer.transform(X[:10]).shape
    (10, 2)
    >>> reducer.get_diagnostics()["graph_"] is not None
    True
    >>> reducer.fit_transform(X).shape
    (100, 2)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for UMAP.

        Returns
        -------
        dict
            Capability mapping describing UMAP as a nonlinear stochastic
            reducer with transform support and a native plotting path.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_transform=True,
            has_native_plot=True,
            supported_diagnostics=("graph_",),
            supported_metadata=("n_features_in_",),
            is_linear=False,
            is_stochastic=True,
        )

    def __init__(self, n_components: int = 2, **kwargs):
        """
        Initialize the UMAP reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of embedding dimensions.
        **kwargs : dict
            Additional keyword arguments forwarded to `umap.UMAP` after
            filtering.
        """
        super().__init__(n_components=n_components, **kwargs)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "UMAPReducer":
        """
        Fit UMAP on the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Optional supervision supported by UMAP.

        Returns
        -------
        UMAPReducer
            Fitted reducer instance.

        Raises
        ------
        ImportError
            If `umap-learn` is not installed.
        RuntimeError
            If `umap-learn` is installed but fails during initialization.
        """
        umap_cls = import_optional_dependency(
            lambda: __import__("umap", fromlist=["UMAP"]).UMAP,
            feature="UMAPReducer",
            dependency="umap-learn",
            install_hint="pip install coco-pipe[neighbor]",
        )

        self.model = self._build_estimator(umap_cls)
        self.model.fit(X, y=y)
        if not hasattr(self.model, "n_features_in_"):
            self.model.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Project data using the fitted UMAP model.

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


class PacmapReducer(BaseReducer):
    """
    PaCMAP reducer.

    Pairwise Controlled Manifold Approximation (PaCMAP) preserves local and
    global structure by balancing near, mid-near, and far pairs during the
    optimization.

    Parameters
    ----------
    n_components : int, default=2
        Number of embedding dimensions.
    n_neighbors : int, default=10
        Number of neighbors used to form local pairs.
    MN_ratio : float, default=0.5
        Ratio of mid-near pairs.
    FP_ratio : float, default=2.0
        Ratio of far pairs.
    nn_backend : {"faiss", "annoy", "voyager"}, default="faiss"
        Nearest-neighbor backend used by recent PaCMAP versions. Older
        PaCMAP releases that do not expose this argument will ignore it
        through signature filtering.
    init : str, default="pca"
        Initialization strategy passed to `fit_transform`.
    **kwargs : dict
        Additional keyword arguments forwarded to `pacmap.PaCMAP` after
        signature filtering.

    Attributes
    ----------
    embedding_ : np.ndarray or None
        Learned training-set embedding after `fit` or `fit_transform`.
    model : pacmap.PaCMAP or None
        Fitted PaCMAP estimator after `fit` or `fit_transform`.

    Notes
    -----
    `transform` is not supported because PaCMAP does not provide an efficient
    out-of-sample projection API.

    See Also
    --------
    UMAPReducer : Nonlinear graph-based embedding with transform support.
    TrimapReducer : Nonlinear triplet-based embedding preserving global layout.
    TSNEReducer : Nonlinear neighborhood-preserving visualization method.
    PHATEReducer : Diffusion-based embedding for continuous trajectories.
    PCAReducer : Linear baseline for global variance preservation.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import PacmapReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = PacmapReducer(
    ...     n_components=2,
    ...     n_neighbors=10,
    ...     nn_backend="faiss",
    ...     init="random",
    ... )
    >>> embedding = reducer.fit_transform(X)
    >>> embedding.shape
    (100, 2)
    >>> reducer.embedding_.shape
    (100, 2)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for PaCMAP.

        Returns
        -------
        dict
            Capability mapping describing PaCMAP as a nonlinear stochastic
            reducer without out-of-sample transform support.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_transform=False,
            supported_metadata=(),
            is_linear=False,
            is_stochastic=True,
        )

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        MN_ratio: float = 0.5,
        FP_ratio: float = 2.0,
        nn_backend: str = "faiss",
        init: str = "pca",
        **kwargs,
    ):
        """
        Initialize the PaCMAP reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of embedding dimensions.
        n_neighbors : int, default=10
            Number of neighbors used to form local pairs.
        MN_ratio : float, default=0.5
            Ratio of mid-near pairs.
        FP_ratio : float, default=2.0
            Ratio of far pairs.
        nn_backend : {"faiss", "annoy", "voyager"}, default="faiss"
            Nearest-neighbor backend used during pair construction when
            supported by the installed PaCMAP version.
        init : str, default="pca"
            Initialization strategy passed during fitting.
        **kwargs : dict
            Additional keyword arguments forwarded to `PaCMAP` after filtering.
        """
        super().__init__(n_components=n_components, **kwargs)
        self.n_neighbors = n_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.nn_backend = nn_backend
        self.init = init
        self.embedding_ = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "PacmapReducer":
        """
        Fit PaCMAP on the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        PacmapReducer
            Fitted reducer instance.

        Raises
        ------
        ImportError
            If `pacmap` is not installed.
        RuntimeError
            If `pacmap` is installed but fails during initialization.
        """
        pacmap_cls = import_optional_dependency(
            lambda: __import__("pacmap", fromlist=["PaCMAP"]).PaCMAP,
            feature="PacmapReducer",
            dependency="pacmap",
            install_hint="pip install coco-pipe[neighbor]",
        )

        self.model = self._build_estimator(
            pacmap_cls,
            params={
                **self.params,
                "n_neighbors": self.n_neighbors,
                "MN_ratio": self.MN_ratio,
                "FP_ratio": self.FP_ratio,
                "nn_backend": self.nn_backend,
            },
        )
        self.embedding_ = self.model.fit_transform(X, init=self.init)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Raise because PaCMAP does not support out-of-sample transformation.

        Parameters
        ----------
        X : ArrayLike
            Ignored input included for API compatibility.

        Raises
        ------
        NotImplementedError
            Always raised because PaCMAP does not support transforming new
            data without refitting.
        """
        raise NotImplementedError(
            "PacmapReducer cannot transform new data. Use fit_transform()."
        )

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Fit PaCMAP and return the embedding coordinates.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Embedded coordinates produced by PaCMAP.
        """
        self.fit(X, y=y)
        return self.embedding_


class TrimapReducer(BaseReducer):
    """
    TriMap reducer.

    TriMap uses triplet constraints to preserve relative similarities while
    emphasizing global layout preservation.

    Parameters
    ----------
    n_components : int, default=2
        Number of embedding dimensions.
    n_inliers : int, default=10
        Number of nearest-neighbor inlier triplets.
    n_outliers : int, default=5
        Number of outlier triplets.
    n_random : int, default=5
        Number of random triplets per sample.
    **kwargs : dict
        Additional keyword arguments forwarded to `trimap.TRIMAP` after
        signature filtering.

    Attributes
    ----------
    embedding_ : np.ndarray or None
        Learned training-set embedding after `fit` or `fit_transform`.
    model : trimap.TRIMAP or None
        Fitted TriMap estimator after `fit` or `fit_transform`.

    Notes
    -----
    `transform` is not supported because TriMap does not provide an
    out-of-sample projection API.

    See Also
    --------
    UMAPReducer : Nonlinear graph-based embedding with transform support.
    PacmapReducer : Nonlinear embedding balancing local and global structure.
    TSNEReducer : Nonlinear neighborhood-preserving visualization method.
    PHATEReducer : Diffusion-based embedding for continuous trajectories.
    IsomapReducer : Nonlinear geodesic-distance manifold embedding.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import TrimapReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = TrimapReducer(n_components=2)
    >>> reducer.fit_transform(X).shape
    (100, 2)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for TriMap.

        Returns
        -------
        dict
            Capability mapping describing TriMap as a nonlinear stochastic
            reducer without out-of-sample transform support.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_transform=False,
            supported_metadata=(),
            is_linear=False,
            is_stochastic=True,
        )

    def __init__(
        self,
        n_components: int = 2,
        n_inliers: int = 10,
        n_outliers: int = 5,
        n_random: int = 5,
        **kwargs,
    ):
        """
        Initialize the TriMap reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of embedding dimensions.
        n_inliers : int, default=10
            Number of inlier triplets.
        n_outliers : int, default=5
            Number of outlier triplets.
        n_random : int, default=5
            Number of random triplets.
        **kwargs : dict
            Additional keyword arguments forwarded to `TRIMAP` after filtering.
        """
        super().__init__(n_components=n_components, **kwargs)
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.embedding_ = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "TrimapReducer":
        """
        Fit TriMap on the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        TrimapReducer
            Fitted reducer instance.

        Raises
        ------
        ImportError
            If `trimap` is not installed.
        RuntimeError
            If `trimap` is installed but fails during initialization.
        """
        trimap_cls = import_optional_dependency(
            lambda: __import__("trimap", fromlist=["TRIMAP"]).TRIMAP,
            feature="TrimapReducer",
            dependency="trimap",
            install_hint="pip install coco-pipe[neighbor]",
        )

        self.model = self._build_estimator(
            trimap_cls,
            component_param="n_dims",
            n_inliers=self.n_inliers,
            n_outliers=self.n_outliers,
            n_random=self.n_random,
        )
        self.embedding_ = self.model.fit_transform(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Raise because TriMap does not support out-of-sample transformation.

        Parameters
        ----------
        X : ArrayLike
            Ignored input included for API compatibility.

        Raises
        ------
        NotImplementedError
            Always raised because TriMap does not support transforming new
            data without refitting.
        """
        raise NotImplementedError(
            "TrimapReducer cannot transform new data. Use fit_transform()."
        )

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Fit TriMap and return the embedding coordinates.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Embedded coordinates produced by TriMap.
        """
        self.fit(X, y=y)
        return self.embedding_


class PHATEReducer(BaseReducer):
    """
    PHATE reducer.

    Potential of Heat-diffusion for Affinity-based Transition Embedding (PHATE)
    is designed for data with continuous progression structure and uses
    diffusion-based distances to construct the embedding.

    Parameters
    ----------
    n_components : int, default=2
        Number of embedding dimensions.
    knn : int, default=5
        Number of nearest neighbors used in the kernel graph.
    decay : int, default=40
        Decay rate for the kernel.
    t : int or str, default="auto"
        Diffusion time.
    **kwargs : dict
        Additional keyword arguments forwarded to `phate.PHATE` after
        signature filtering.

    Attributes
    ----------
    model : phate.PHATE or None
        Fitted PHATE estimator after `fit`.

    See Also
    --------
    UMAPReducer : Nonlinear graph-based embedding with transform support.
    TSNEReducer : Nonlinear neighborhood-preserving visualization method.
    PacmapReducer : Nonlinear embedding balancing local and global structure.
    TrimapReducer : Nonlinear triplet-based embedding preserving global layout.
    ParametricUMAPReducer : Neural-network-backed UMAP approximation.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import PHATEReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = PHATEReducer(n_components=2, knn=5)
    >>> _ = reducer.fit(X)
    >>> reducer.transform(X[:10]).shape
    (10, 2)
    >>> reducer.get_diagnostics()["diff_potential"] is not None
    True
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for PHATE.

        Returns
        -------
        dict
            Capability mapping describing PHATE as a nonlinear reducer with
            transform support and a native plotting path.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_transform=True,
            has_native_plot=True,
            supported_diagnostics=("diff_potential",),
            supported_metadata=("n_features_in_",),
            is_linear=False,
            is_stochastic=True,
        )

    def __init__(
        self,
        n_components: int = 2,
        knn: int = 5,
        decay: int = 40,
        t: Any = "auto",
        **kwargs,
    ):
        """
        Initialize the PHATE reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of embedding dimensions.
        knn : int, default=5
            Number of nearest neighbors used in the kernel graph.
        decay : int, default=40
            Decay rate for the kernel.
        t : int or str, default="auto"
            Diffusion time.
        **kwargs : dict
            Additional keyword arguments forwarded to `PHATE` after filtering.
        """
        super().__init__(n_components=n_components, **kwargs)
        self.knn = knn
        self.decay = decay
        self.t = t

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "PHATEReducer":
        """
        Fit PHATE on the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        PHATEReducer
            Fitted reducer instance.

        Raises
        ------
        ImportError
            If `phate` is not installed.
        RuntimeError
            If `phate` is installed but fails during initialization.
        """
        phate_cls = import_optional_dependency(
            lambda: __import__("phate", fromlist=["PHATE"]).PHATE,
            feature="PHATEReducer",
            dependency="phate",
            install_hint="pip install coco-pipe[neighbor]",
        )

        self.model = self._build_estimator(
            phate_cls,
            knn=self.knn,
            decay=self.decay,
            t=self.t,
        )
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Project data using the fitted PHATE model.

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


class ParametricUMAPReducer(BaseReducer):
    """
    Parametric UMAP reducer.

    Parametric UMAP learns a neural network that approximates the UMAP
    embedding, enabling reusable out-of-sample projection through the trained
    network.

    Parameters
    ----------
    n_components : int, default=2
        Number of embedding dimensions.
    n_neighbors : int, default=15
        Size of the local neighborhood.
    min_dist : float, default=0.1
        Effective minimum distance between embedded points.
    metric : str, default="euclidean"
        Metric used for distance computation.
    n_epochs : int, optional
        Number of training epochs.
    batch_size : int, default=1000
        Batch size used during training.
    verbose : bool, default=False
        Whether to print backend training progress.
    **kwargs : dict
        Additional keyword arguments forwarded to
        `umap.parametric_umap.ParametricUMAP` after signature filtering.

    Attributes
    ----------
    model : umap.parametric_umap.ParametricUMAP or None
        Fitted Parametric UMAP estimator after `fit`.

    See Also
    --------
    UMAPReducer : Non-parametric UMAP with graph-based transform support.
    TSNEReducer : Nonlinear neighborhood-preserving visualization method.
    PHATEReducer : Diffusion-based embedding for continuous trajectories.
    IVISReducer : Neural metric-learning-based embedding.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import ParametricUMAPReducer
    >>> X = np.random.rand(50, 10).astype(np.float32)
    >>> reducer = ParametricUMAPReducer(n_components=2, n_epochs=5, verbose=False)
    >>> _ = reducer.fit(X)
    >>> reducer.transform(X[:10]).shape
    (10, 2)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for Parametric UMAP.

        Returns
        -------
        dict
            Capability mapping describing Parametric UMAP as a nonlinear
            stochastic reducer with transform support.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_transform=True,
            supported_diagnostics=("loss_history_",),
            supported_metadata=("n_features_in_",),
            is_linear=False,
            is_stochastic=True,
        )

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        n_epochs: Optional[int] = None,
        batch_size: int = 1000,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the Parametric UMAP reducer.

        Parameters
        ----------
        n_components : int, default=2
            Number of embedding dimensions.
        n_neighbors : int, default=15
            Size of the local neighborhood.
        min_dist : float, default=0.1
            Effective minimum distance between embedded points.
        metric : str, default="euclidean"
            Metric used for distance computation.
        n_epochs : int, optional
            Number of training epochs.
        batch_size : int, default=1000
            Batch size used during training.
        verbose : bool, default=False
            Whether to print backend training progress.
        **kwargs : dict
            Additional keyword arguments forwarded to Parametric UMAP after
            filtering.
        """
        super().__init__(n_components=n_components, **kwargs)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] = None
    ) -> "ParametricUMAPReducer":
        """
        Fit Parametric UMAP on the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Optional supervision supported by Parametric UMAP.

        Returns
        -------
        ParametricUMAPReducer
            Fitted reducer instance.

        Raises
        ------
        ImportError
            If `umap-learn` is not installed.
        RuntimeError
            If `umap-learn` is installed but fails during initialization.
        """
        parametric_umap_cls = import_optional_dependency(
            lambda: (
                __import__(
                    "umap.parametric_umap", fromlist=["ParametricUMAP"]
                ).ParametricUMAP
            ),
            feature="ParametricUMAPReducer",
            dependency="umap-learn",
            install_hint="pip install coco-pipe[parametric-umap]",
        )

        self.model = self._build_estimator(
            parametric_umap_cls,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        self.model.fit(X, y=y)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Project data using the fitted Parametric UMAP model.

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
    def loss_history_(self) -> list:
        """
        Training loss history for the parametric model.

        Returns
        -------
        list
            Recorded loss values across training epochs.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return list(self.model._history["loss"])

    def save(self, filepath: str) -> None:
        """
        Serialize the fitted reducer with joblib.

        Parameters
        ----------
        filepath : str
            Output path for the serialized reducer.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        super().save(filepath)
