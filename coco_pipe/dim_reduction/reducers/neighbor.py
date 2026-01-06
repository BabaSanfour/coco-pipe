"""
Neighbor Embedding Reducers
===========================

This module implements dimensionality reduction techniques based on neighbor
embeddings and graph layouts. It consolidates popular methods like t-SNE, UMAP,
PaCMAP, TriMap, and PHATE.

Classes
-------
TSNEReducer
    t-Distributed Stochastic Neighbor Embedding (sklearn wrapper).
UMAPReducer
    Uniform Manifold Approximation and Projection (umap-learn wrapper).
PacmapReducer
    Pairwise Controlled Manifold Approximation (pacmap wrapper).
TrimapReducer
    Large-scale dimensionality reduction using triplets (trimap wrapper).
PHATEReducer
    Potential of Heat-diffusion for Affinity-based Trajectory Embedding (phate wrapper).

References
----------
.. [1] Maaten, L. van der, & Hinton, G. (2008). Visualizing data using t-SNE. JMLR.
.. [2] McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold
       Approximation and Projection for Dimension Reduction. arXiv.
.. [3] Wang, Y., et al. (2021). PaCMAP: Pairwise Controlled Manifold Approximation. JMLR.
.. [4] Amid, E., & Warmuth, M. K. (2019). TriMap: Large-scale Dimensionality Reduction
       Using Triplets. arXiv.
.. [5] Moon, K. R., et al. (2019). Visualizing structure and transitions in high-dimensional
       biological data. Nature Biotechnology.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
        Sina Esmaeili (sina.esmaeili@umontreal.ca)
Date: 2026-01-06
"""

from typing import Optional, Any
import numpy as np

from sklearn.manifold import TSNE
import umap
import pacmap
import trimap
import phate

from .base import BaseReducer, ArrayLike


class TSNEReducer(BaseReducer):
    """
    t-SNE dimensionality reducer.

    t-Distributed Stochastic Neighbor Embedding (t-SNE) is a technique for dimensionality 
    reduction that is particularly well suited for the visualization of high-dimensional datasets.
    It converts similarities between data points to joint probabilities and tries to minimize 
    the Kullback-Leibler divergence between the joint probabilities of the low-dimensional 
    embedding and the high-dimensional data.

    Note: t-SNE does not support out-of-sample transformation (transform() raises error).

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions.
    **kwargs : dict
        Additional arguments passed to TSNE.
        Common arguments:
        - perplexity : float, default=30.0
        - learning_rate : float or 'auto', default='auto'
        - n_iter : int, default=1000

    Attributes
    ----------
    embedding_ : np.ndarray
        The learned embedding.
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.
    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.reducers.neighbor import TSNEReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = TSNEReducer(n_components=2, random_state=42)
    >>> X_reduced = reducer.fit_transform(X)
    >>> print(X_reduced.shape)
    (100, 2)
    >>> print(f"{reducer.kl_divergence_:.4f}")  # Access diagnostic property
    """

    def __init__(self, n_components: int = 2, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.embedding_ = None
        self.model = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "TSNEReducer":
        """
        Fit t-SNE with X.
        
        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : TSNEReducer
            Returns the instance itself.
        """
        self.model = TSNE(n_components=self.n_components, **self.params)
        self.model.fit(X)
        self.embedding_ = self.model.embedding_
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform X.
        
        Raises
        ------
        NotImplementedError
            t-SNE does not support transforming new data.
        """
        raise NotImplementedError("TSNEReducer cannot transform new data. Use fit_transform().")

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Fit and return embedding.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        embedding : np.ndarray of shape (n_samples, n_components)
            The learned embedding.
        """
        self.fit(X, y=y)
        return self.embedding_

    @property
    def kl_divergence_(self) -> float:
        """
        Kullback-Leibler divergence after optimization.

        Returns
        -------
        kl_divergence_ : float
        """
        if self.model is None or not hasattr(self.model, "kl_divergence_"):
             raise RuntimeError("Model is not fitted yet.")
        return self.model.kl_divergence_

    @property
    def n_iter_(self) -> int:
        """
        Number of iterations run.

        Returns
        -------
        n_iter_ : int
        """
        if self.model is None or not hasattr(self.model, "n_iter_"):
             raise RuntimeError("Model is not fitted yet.")
        return self.model.n_iter_


class UMAPReducer(BaseReducer):
    """
    UMAP dimensionality reducer.

    Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique 
    that can be used for visualization similarly to t-SNE, but also for general non-linear 
    dimension reduction. It constructs a high dimensional graph representation of the data 
    then optimizes a low-dimensional graph to be structurally similar.

    Unlike t-SNE, UMAP supports out-of-sample transformation.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions.
    **kwargs : dict
        Additional arguments passed to umap.UMAP.
        Common arguments:
        - n_neighbors : int, default=15
        - min_dist : float, default=0.1
        - metric : str, default='euclidean'

    Attributes
    ----------
    model : umap.UMAP
        The underlying UMAP estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.reducers.neighbor import UMAPReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = UMAPReducer(n_components=2, n_neighbors=5, random_state=42)
    >>> X_reduced = reducer.fit_transform(X)
    >>> print(X_reduced.shape)
    (100, 2)
    >>> # Transform new data
    >>> X_new = np.random.rand(10, 10)
    >>> X_new_reduced = reducer.transform(X_new)
    >>> print(X_new_reduced.shape)
    (10, 2)
    """

    def __init__(self, n_components: int = 2, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.model = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "UMAPReducer":
        """
        Fit UMAP with X.
        
        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : UMAPReducer
            Returns the instance itself.
        """
        self.model = umap.UMAP(n_components=self.n_components, **self.params)
        self.model.fit(X, y=y)
        # Store expected input dimension for validation
        self.model.n_features_in_ = np.array(X).shape[1]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform X using the fitted UMAP model.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_components)
            Projection of X in the reduced space.
        """
        if self.model is None:
            raise RuntimeError("UMAPReducer must be fitted before calling transform().")
        return self.model.transform(X)

    @property
    def graph_(self) -> Any:
        """
        The fuzzy simplicial set graph computed by UMAP.
        
        Returns
        -------
        graph_ : scipy.sparse.csr.csr_matrix
            The graph of the fuzzy simplicial set.
        """
        if self.model is None or not hasattr(self.model, "graph_"):
             raise RuntimeError("Model is not fitted yet.")
        return self.model.graph_


class PacmapReducer(BaseReducer):
    """
    PaCMAP dimensionality reducer.

    Pairwise Controlled Manifold Approximation (PaCMAP) is a dimensionality reduction 
    method that preserves both local and global structure of the data. It achieves this 
    using three kinds of pairs (near, mid-near, and far) to optimize the embedding.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions.
    n_neighbors : int, default=10
        Number of neighbors.
    MN_ratio : float, default=0.5
        Mid-near ratio, controlling the balance between local and global structure.
    FP_ratio : float, default=2.0
        Far-pair ratio, weighting the repulsion of far points.
    **kwargs : dict
        Additional arguments.

    Attributes
    ----------
    model : pacmap.PaCMAP
        The underlying PaCMAP estimator.
    
    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.reducers.neighbor import PacmapReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = PacmapReducer(n_components=2, n_neighbors=10)
    >>> X_reduced = reducer.fit_transform(X)
    >>> print(X_reduced.shape)
    (100, 2)
    """

    def __init__(self, n_components: int = 2, n_neighbors: int = 10,
                 MN_ratio: float = 0.5, FP_ratio: float = 2.0, **kwargs):        
        # We manually handle these specific args to pass them cleanly
        self.specific_args = {
            'n_neighbors': n_neighbors,
            'MN_ratio': MN_ratio,
            'FP_ratio': FP_ratio
        }
        super().__init__(n_components=n_components, **kwargs)
        self.embedding_ = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "PacmapReducer":
        """
        Fit PaCMAP using X.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : PacmapReducer
            Returns the instance itself.
        """
        self.model = pacmap.PaCMAP(
            n_components=self.n_components,
            **self.specific_args,
            **self.params
        )
        self.embedding_ = self.model.fit_transform(X, init="pca")
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform X.
        
        Raises
        ------
        NotImplementedError
            PaCMAP does not support transforming new data efficiently without refitting. 
            Use fit_transform() on the full dataset instead.
        """
        raise NotImplementedError("PaCMAPReducer cannot transform new data. Use fit_transform().")

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Fit and return embedding.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_components)
            Projection of X in the reduced space.
        """
        self.fit(X, y=y)
        return self.embedding_


class TrimapReducer(BaseReducer):
    """
    TriMap dimensionality reducer.

    TriMap (Large-scale Dimensionality Reduction Using Triplets) is a dimensionality 
    reduction technique that preserves global structure better than t-SNE and UMAP 
    while being efficient. It uses triplet constraints (i, j, k) to capture the 
    relative similarity between points.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions.
    n_inliers : int, default=10
        Number of nearest neighbors for forming inlier triplets.
    n_outliers : int, default=5
        Number of outliers for forming outlier triplets.
    n_random : int, default=5
        Number of random triplets per point.
    **kwargs : dict
        Additional arguments.

    Attributes
    ----------
    model : trimap.TRIMAP
        The underlying TriMap estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.reducers.neighbor import TrimapReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = TrimapReducer(n_components=2)
    >>> X_reduced = reducer.fit_transform(X)
    >>> print(X_reduced.shape)
    (100, 2)
    """

    def __init__(self, n_components: int = 2, n_inliers: int = 10,
                 n_outliers: int = 5, n_random: int = 5, **kwargs):
        self.specific_args = {
            'n_inliers': n_inliers,
            'n_outliers': n_outliers,
            'n_random': n_random
        }
        super().__init__(n_components=n_components, **kwargs)
        self.embedding_ = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "TrimapReducer":
        """
        Fit TriMap using X.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : TrimapReducer
            Returns the instance itself.
        """
        self.model = trimap.TRIMAP(
            n_dims=self.n_components,
            **self.specific_args,
            **self.params
        )
        self.embedding_ = self.model.fit_transform(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform X.
        
        TriMap is transductive but the implementation provides a mechanism to embed
        new points by optimizing their position fixing the existing embedding.
        
        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
             New data to transform.
             
        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_components)
             Projected data.
        """
        if self.model is None:
            raise RuntimeError("TrimapReducer must be fitted before calling transform().")
        
        return self.model.transform(X)

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Fit and return embedding.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_components)
            Projection of X.
        """
        self.fit(X, y=y)
        return self.embedding_


class PHATEReducer(BaseReducer):
    """
    PHATE dimensionality reducer.

    Potential of Heat-diffusion for Affinity-based Trajectory Embedding (PHATE) is 
    designed to visualize high-dimensional data, specifically biological data with 
    continuous progression structures (trajectories). It uses information-theoretic 
    distances based on diffusion probabilities.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions.
    knn : int, default=5
        Number of nearest neighbors for kernel construction.
    decay : int, default=40 
        Decay rate for kernel.
    **kwargs : dict
        Additional arguments.

    Attributes
    ----------
    model : phate.PHATE
        The underlying PHATE estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.reducers.neighbor import PHATEReducer
    >>> X = np.random.rand(100, 10)
    >>> reducer = PHATEReducer(n_components=2, knn=5)
    >>> X_reduced = reducer.fit_transform(X)
    >>> print(X_reduced.shape)
    (100, 2)
    """

    def __init__(self, n_components: int = 2, **kwargs):        
        super().__init__(n_components=n_components, **kwargs)
        self.model = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "PHATEReducer":
        """
        Fit PHATE using X.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : Ignored

        Returns
        -------
        self : PHATEReducer
            Returns the instance itself.
        """
        self.model = phate.PHATE(n_components=self.n_components, **self.params)
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform X using the fitted PHATE model.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_components)
            Projected data.
        """
        if self.model is None:
            raise RuntimeError("PHATEReducer must be fitted before calling transform().")
        return self.model.transform(X)

    @property
    def diff_potential(self) -> np.ndarray:
        """
        The diffusion potential of the data.
        
        Returns
        -------
        diff_potential : np.ndarray
        """
        if self.model is None or not hasattr(self.model, "diff_potential"):
             raise RuntimeError("Model is not fitted yet.")
        return self.model.diff_potential

    @property
    def diff_op(self) -> Any:
        """
        The diffusion operator.
        
        Returns
        -------
        diff_op : scipy.sparse.csr_matrix or np.ndarray
        """
        if self.model is None or not hasattr(self.model, "diff_op"):
             raise RuntimeError("Model is not fitted yet.")
        return self.model.diff_op

    @property
    def graph(self) -> Any:
        """
        The k-nearest neighbor graph.
        
        Returns
        -------
        graph : scipy.sparse.csr_matrix
        """
        if self.model is None or not hasattr(self.model, "graph"):
             raise RuntimeError("Model is not fitted yet.")
        return self.model.graph
