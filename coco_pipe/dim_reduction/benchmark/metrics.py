"""
Dimensionality Reduction Quality Metrics
========================================

This module provides quantitative metrics to evaluate the quality of
dimensionality reduction embeddings.

Functions
---------
trustworthiness
    Measures to what extent the local structure is retained (penalizes false neighbors).
continuity
    Measures to what extent the local structure is retained (penalizes missing neighbors).
lcmc
    Local Continuity Meta-Criterion.
shepard_diagram_data
    Computes pairwise distances for Shepard diagram plotting.

References
----------
.. [1] Venna, J., & Kaski, S. (2001). Neighborhood preservation in nonlinear
       projection methods: An experimental study.
.. [2] Chen, L., & Buja, A. (2009). Local multidimensional scaling for nonlinear
       dimension reduction, graph drawing, and proximity analysis.

Author: Hamza Abdelhedi
Date: 2026-01-06
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from scipy.spatial.distance import pdist


def trustworthiness(X: np.ndarray, X_embedded: np.ndarray, n_neighbors: int = 5) -> float:
    """
    Compute the trustworthiness score.

    Trustworthiness measures the preservation of local neighborhoods.
    It penalizes checking points that are neighbors in the embedded space
    but not in the original space (false positives).
    
    Range: [0, 1], higher is better.

    Parameters
    ----------
    X : np.ndarray
        Original high-dimensional data.
    X_embedded : np.ndarray
        Low-dimensional embedding.
    n_neighbors : int, default=5
        Number of neighbors.

    Returns
    -------
    score : float
        Trustworthiness score.
    """
    return sklearn_trustworthiness(X, X_embedded, n_neighbors=n_neighbors)


def continuity(X: np.ndarray, X_embedded: np.ndarray, n_neighbors: int = 5) -> float:
    """
    Compute the continuity score.

    Continuity is defined analogously to trustworthiness but penalizes missing neighbors
    (false negatives). It checks if points that are neighbors in the original space
    are preserved in the embedded space.
    
    It is equivalent to trustworthiness computed with X and X_embedded swapped.

    Parameters
    ----------
    X : np.ndarray
        Original high-dimensional data.
    X_embedded : np.ndarray
        Low-dimensional embedding.
    n_neighbors : int, default=5
         Number of neighbors.

    Returns
    -------
    score : float
        Continuity score.
    """
    return sklearn_trustworthiness(X_embedded, X, n_neighbors=n_neighbors)


def lcmc(X: np.ndarray, X_embedded: np.ndarray, n_neighbors: int = 5) -> float:
    """
    Compute the Local Continuity Meta-Criterion (LCMC).

    LCMC = Avg(Overlap) - n_neighbors / (N - 1)
    Normalized to [0, 1].

    Parameters
    ----------
    X : np.ndarray
        Original data.
    X_embedded : np.ndarray
        Embedded data.
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    score : float
        LCMC score.
    """
    n_samples = X.shape[0]
    
    # KNN in original space
    knn_high = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    ind_high = knn_high.kneighbors(X, return_distance=False)
    
    # KNN in embedded space
    knn_low = NearestNeighbors(n_neighbors=n_neighbors).fit(X_embedded)
    ind_low = knn_low.kneighbors(X_embedded, return_distance=False)
    
    overlap = 0
    for i in range(n_samples):
        set_high = set(ind_high[i])
        set_low = set(ind_low[i])
        overlap += len(set_high.intersection(set_low))
    
    avg_overlap = overlap / n_samples
    return avg_overlap / n_neighbors


def shepard_diagram_data(X: np.ndarray, X_embedded: np.ndarray, sample_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise distances for Shepard Diagram.

    Parameters
    ----------
    X : np.ndarray
        Original data.
    X_embedded : np.ndarray
        Embedded data.
    sample_size : int, default=1000
        Number of points to sample if N is large (to avoid N^2 complexity).
        If N <= sample_size, uses all points.

    Returns
    -------
    dist_orig : np.ndarray
        Pairwise distances in high-dimensional space.
    dist_emb : np.ndarray
        Pairwise distances in low-dimensional space.
    """
    n_samples = X.shape[0]
    
    if n_samples > sample_size:
        indices = np.random.choice(n_samples, sample_size, replace=False)
        X_sub = X[indices]
        Emb_sub = X_embedded[indices]
    else:
        X_sub = X
        Emb_sub = X_embedded
        
    dist_orig = pdist(X_sub)
    dist_emb = pdist(Emb_sub)
    
    return dist_orig, dist_emb
