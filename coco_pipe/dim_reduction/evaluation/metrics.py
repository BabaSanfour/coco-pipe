"""
Advanced Dimensionality Reduction Metrics
=========================================

This module implements rigorous quality metrics based on the Co-ranking Matrix framework.
It includes efficient implementations of Trustworthiness, Continuity, LCMC, and MRRE.

Functions
---------
compute_coranking_matrix
    Compute the co-ranking matrix Q.
compute_mrre
    Compute Mean Relative Rank Errors (Intrusion/Extrusion).
trustworthiness
    Compute Trustworthiness using Q matrix.
continuity
    Compute Continuity using Q matrix.
lcmc
    Compute Local Continuity Meta-Criterion.

References
----------
.. [1] Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality
       reduction: Rank-based criteria. Neurocomputing.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
Date: 2026-01-08
"""


from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors


def compute_coranking_matrix(X: np.ndarray, X_emb: np.ndarray) -> np.ndarray:
    """
    Compute the co-ranking matrix Q.

    The co-ranking matrix Q_kl counts how many points have rank k in high-dimensional
    space and rank l in low-dimensional space.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        High-dimensional data.
    X_emb : np.ndarray of shape (n_samples, n_components)
        Low-dimensional embedding.

    Returns
    -------
    Q : np.ndarray of shape (n_samples-1, n_samples-1)
        The co-ranking matrix. Q[k, l] corresponds to rank k+1 and l+1 (0-indexed).
    """
    n = X.shape[0]

    # Use 'auto' for efficiency, but we need exact neighbors for valid Q
    nbrs_high = NearestNeighbors(n_neighbors=n - 1, algorithm="auto", n_jobs=-1).fit(X)
    _, indices_high = nbrs_high.kneighbors(X)

    nbrs_low = NearestNeighbors(n_neighbors=n - 1, algorithm="auto", n_jobs=-1).fit(
        X_emb
    )
    _, indices_low = nbrs_low.kneighbors(X_emb)

    # Calculate ranks in Low-D
    # rank_low[i, j] = rank of j w.r.t i in Low-D
    rank_low = np.zeros((n, n), dtype=int)
    rows = np.arange(n)[:, None]
    rank_low[rows, indices_low] = np.arange(n - 1)

    # For each point i, take its high-D neighbors j = indices_high[i, :]
    # Their High-D rank is simply the column index (0, 1, ..., n-2)
    # Their Low-D rank is looked up in rank_low

    # High-D ranks for all pairs (i, j) where j is a neighbor of i
    # This is just tile(0..n-2) effectively, but we use histogram to aggregate

    # Get the Low-D ranks for the High-D neighbors
    # This is the "l" coordinate for each pair
    l_indices = rank_low[rows, indices_high].flatten()

    # The "k" coordinate (High-D rank) is just repeating 0..n-2 for each row
    k_indices = np.tile(np.arange(n - 1), n)

    # Compute 2D histogram
    # bins are 0, 1, ..., n-1 (edges) -> centers are 0..n-2
    Q, _, _ = np.histogram2d(
        k_indices, l_indices, bins=n - 1, range=[[0, n - 1], [0, n - 1]]
    )

    return Q.astype(int)


def trustworthiness(Q: np.ndarray, k: int) -> float:
    """
    Compute Trustworthiness from Co-ranking matrix Q.

    Trustworthiness penalizes intrusions (upper triangle of Q).
    It measures how well local neighbors in the original space remain neighbors
    in the embedding.

    Parameters
    ----------
    Q : np.ndarray
        Co-ranking matrix.
    k : int
        Size of the neighborhood.

    Returns
    -------
    score : float
        Trustworthiness score in [0, 1]. Higher is better.

    Examples
    --------
    >>> Q = compute_coranking_matrix(X, X_emb)
    >>> t = trustworthiness(Q, k=10)
    """
    n = Q.shape[0] + 1
    if k >= n - 1:
        return 1.0  # Trivial case

    # Intrusions: High-D rank (r) > k, Low-D rank (c) <= k
    # We sum Q[r, c] * (r + 1 - k)

    # Slice the relevant region: r from k to n-2, c from 0 to k-1
    Q_sub = Q[k:, :k]

    # Weights depend only on r (row index in Q_sub)
    # r_actual = r_sub + k
    # weight = r_actual + 1 - k = r_sub + 1

    row_indices = np.arange(Q_sub.shape[0])  # 0 to n-2-k
    weights = row_indices + 1

    # Sum over columns first (axis 1), then weighted sum over rows
    # int_sum = sum_{r} weight[r] * sum_{c} Q[r,c]
    int_sum = np.sum(Q_sub.sum(axis=1) * weights)

    # Normalization
    if n <= 5:
        # Robust handling for small N edge cases
        # For N=5, K=3 -> 2N-3K-1 = 10-9-1=0.
        denom = n * k * (2 * n - 3 * k - 1)
        if denom == 0:
            term = 1.0  # Fallback
        else:
            term = 2 / denom
    else:
        term = 2 / (n * k * (2 * n - 3 * k - 1))

    return 1 - term * int_sum


def continuity(Q: np.ndarray, k: int) -> float:
    """
    Compute Continuity from Co-ranking matrix Q.

    Continuity penalizes extrusions (lower triangle of Q).
    It measures how well points close in the embedding are also close in the
    original space (no false clusters).

    Parameters
    ----------
    Q : np.ndarray
        Co-ranking matrix.
    k : int
        Size of the neighborhood.

    Returns
    -------
    score : float
        Continuity score in [0, 1]. Higher is better.

    Examples
    --------
    >>> c = continuity(Q, k=10)
    """
    n = Q.shape[0] + 1
    if k >= n - 1:
        return 1.0

    # Extrusions: High-D rank (r) <= k, Low-D rank (c) > k
    # Slice: r from 0 to k-1, c from k to n-2
    Q_sub = Q[:k, k:]

    # Weights depend only on c (col index in Q_sub)
    # c_actual = c_sub + k
    # weight = c_actual + 1 - k = c_sub + 1

    col_indices = np.arange(Q_sub.shape[1])
    weights = col_indices + 1

    # Sum over rows first, then weighted sum over cols
    ext_sum = np.sum(Q_sub.sum(axis=0) * weights)

    if n <= 5:
        denom = n * k * (2 * n - 3 * k - 1)
        if denom == 0:
            term = 1.0
        else:
            term = 2 / denom
    else:
        term = 2 / (n * k * (2 * n - 3 * k - 1))

    return 1 - term * ext_sum


def lcmc(Q: np.ndarray, k: int) -> float:
    """
    Compute Local Continuity Meta-Criterion (LCMC).

    LCMC measures the overlap between the k-nearest neighbors in the original
    space and the low-dimensional space. The value is normalized to be roughly
    independent of k.

    Parameters
    ----------
    Q : np.ndarray
        Co-ranking matrix.
    k : int
        Size of the neighborhood.

    Returns
    -------
    score : float
        LCMC score. Higher is better (ranges typically [0, 1]).

    Examples
    --------
    >>> score = lcmc(Q, k=10)
    """
    n = Q.shape[0] + 1
    if k >= n - 1:
        return 0.0  # LCMC typically -> 0 as k->N? Actually overlap is max.

    overlap = np.sum(Q[:k, :k])

    term1 = k / (1 - n)
    term2 = (1 / (n * k)) * overlap

    return term1 + term2


def compute_mrre(Q: np.ndarray, k: int) -> Tuple[float, float]:
    """
    Compute Mean Relative Rank Errors (MRRE).

    Calculates both MRRE_X (Intrusion) and MRRE_Y (Extrusion). These metrics
    measure the preservation of neighborhood ranks, weighting errors by their
    rank magnitude.

    Parameters
    ----------
    Q : np.ndarray
        Co-ranking matrix.
    k : int
        Size of the neighborhood.

    Returns
    -------
    mrre_int : float
        MRRE of Intrusions ($M_{int}$). Closer to 1 is better (0 is perfect preservation).
        *Note*: Interpretation depends on formula variant; here we compute standard error.
    mrre_ext : float
        MRRE of Extrusions ($M_{ext}$).

    Examples
    --------
    >>> m_int, m_ext = compute_mrre(Q, k=20)
    """
    n = Q.shape[0] + 1

    # Normalizer
    # H_k = sum_{i=1}^k |n - 2i + 1| / i
    i_vals = np.arange(1, k + 1)
    H_k = np.sum(np.abs(n - 2 * i_vals + 1) / i_vals)

    # --- Intrusions ---
    # r > k, c < k
    # weight = |rank_low - rank_high| / rank_high
    #        = |(c+1) - (r+1)| / (r+1) = |c-r| / (r+1)

    # Slice for Intrusions: Q[k:, :k]
    # r from k to n-2 (indices), c from 0 to k-1 (indices)
    rows_int = np.arange(k, n - 1)
    cols_int = np.arange(k)

    # Meshgrid for this sub-block
    R_int, C_int = np.meshgrid(rows_int, cols_int, indexing="ij")
    # Rank values (1-based)
    Rho_int = R_int + 1
    Rnk_int = C_int + 1

    W_int = np.abs(Rnk_int - Rho_int) / Rho_int

    # Element-wise multiply with Q slice and sum
    int_sum = np.sum(Q[k:, :k] * W_int)
    mrre_int = int_sum / (n * H_k)

    # --- Extrusions ---
    # r <= k, c > k
    # Slice Q[:k, k:]
    rows_ext = np.arange(k)
    cols_ext = np.arange(k, n - 1)

    R_ext, C_ext = np.meshgrid(rows_ext, cols_ext, indexing="ij")
    Rho_ext = R_ext + 1
    Rnk_ext = C_ext + 1

    W_ext = np.abs(Rnk_ext - Rho_ext) / Rho_ext

    ext_sum = np.sum(Q[:k, k:] * W_ext)
    mrre_ext = ext_sum / (n * H_k)

    return mrre_int, mrre_ext


def shepard_diagram_data(
    X: np.ndarray, X_embedded: np.ndarray, sample_size: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
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
