"""
Rank-based dimensionality reduction quality metrics.

This module provides co-ranking-matrix metrics for comparing high-dimensional
data with a low-dimensional embedding. The implementations are reducer-agnostic
and operate directly on NumPy arrays.

Functions
---------
compute_coranking_matrix
    Compute the co-ranking matrix between the original and embedded spaces.
trustworthiness
    Measure how well original neighbors remain neighbors after embedding.
continuity
    Measure how well embedded neighbors are close in the original space.
lcmc
    Compute the local continuity meta-criterion.
compute_mrre
    Compute mean relative rank errors for intrusions and extrusions.
shepard_diagram_data
    Sample pairwise distances for Shepard-diagram visualization.

References
----------
.. [1] Lee, J. A., & Verleysen, M. (2009). Quality assessment of
       dimensionality reduction: Rank-based criteria. Neurocomputing.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "compute_coranking_matrix",
    "trustworthiness",
    "continuity",
    "lcmc",
    "compute_mrre",
    "shepard_diagram_data",
]


def _validate_embedding_pair(
    X: np.ndarray,
    X_emb: np.ndarray,
    *,
    func_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired original and embedded sample matrices."""
    X_arr = np.asarray(X, dtype=float)
    X_emb_arr = np.asarray(X_emb, dtype=float)

    if X_arr.ndim != 2:
        raise ValueError(f"`X` must be a 2D array in `{func_name}`.")
    if X_emb_arr.ndim != 2:
        raise ValueError(f"`X_emb` must be a 2D array in `{func_name}`.")
    if X_arr.shape[0] != X_emb_arr.shape[0]:
        raise ValueError("`X` and `X_emb` must contain the same number of samples.")
    if X_arr.shape[0] < 2:
        raise ValueError("At least 2 samples are required.")
    return X_arr, X_emb_arr


def _validate_k(
    Q: np.ndarray,
    k: int,
    metric_name: str,
    *,
    require_positive_normalizer: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Validate co-ranking neighborhood size for a metric."""
    Q_arr = np.asarray(Q)
    if Q_arr.ndim != 2 or Q_arr.shape[0] != Q_arr.shape[1]:
        raise ValueError("`Q` must be a square 2D co-ranking matrix.")

    n = Q_arr.shape[0] + 1

    if not isinstance(k, (int, np.integer)):
        raise ValueError(f"Neighborhood size k for `{metric_name}` must be an integer.")
    k_int = int(k)

    if k_int <= 0:
        raise ValueError("Neighborhood size k must be > 0.")
    if k_int >= n - 1:
        raise ValueError(
            f"Neighborhood size k ({k_int}) must be less than n_samples - 1 ({n - 1})."
        )

    if require_positive_normalizer and (2 * n - 3 * k_int - 1) <= 0:
        raise ValueError(
            f"Neighborhood size k ({k_int}) is too large for `{metric_name}` with "
            f"n_samples={n}; the normalization term must stay positive."
        )

    return Q_arr, n, k_int


def _trust_continuity_scale(n: int, k: int, metric_name: str) -> float:
    """Return the shared normalization scale for trustworthiness/continuity."""
    denom = n * k * (2 * n - 3 * k - 1)
    if denom <= 0:
        raise ValueError(
            f"Neighborhood size k ({k}) is too large for `{metric_name}` with "
            f"n_samples={n}; the normalization term must stay positive."
        )
    return 2.0 / float(denom)


def compute_coranking_matrix(X: np.ndarray, X_emb: np.ndarray) -> np.ndarray:
    """
    Compute the co-ranking matrix between two sample spaces.

    The co-ranking matrix ``Q`` counts how often each point pair appears with
    high-dimensional rank ``k`` and low-dimensional rank ``l``. Self-neighbors
    are excluded from the rank construction.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Original high-dimensional data.
    X_emb : np.ndarray of shape (n_samples, n_components)
        Low-dimensional embedding of the same samples.

    Returns
    -------
    np.ndarray of shape (n_samples - 1, n_samples - 1)
        Integer co-ranking matrix where ``Q[k, l]`` corresponds to ranks
        ``k + 1`` and ``l + 1`` in the original and embedded spaces.

    Raises
    ------
    ValueError
        If the inputs are not two-dimensional, do not share the same sample
        count, or contain fewer than two samples.

    See Also
    --------
    trustworthiness : Compute intrusion-based neighborhood preservation.
    continuity : Compute extrusion-based neighborhood preservation.
    lcmc : Compute the local continuity meta-criterion.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [1.0], [2.0]])
    >>> X_emb = np.array([[0.0], [2.0], [4.0]])
    >>> Q = compute_coranking_matrix(X, X_emb)
    >>> Q.shape
    (2, 2)
    """
    X_arr, X_emb_arr = _validate_embedding_pair(
        X,
        X_emb,
        func_name="compute_coranking_matrix",
    )
    n = X_arr.shape[0]

    nbrs_high = NearestNeighbors(n_neighbors=n, algorithm="auto", n_jobs=1).fit(X_arr)
    _, indices_high = nbrs_high.kneighbors(X_arr)
    indices_high = indices_high[:, 1:]

    nbrs_low = NearestNeighbors(n_neighbors=n, algorithm="auto", n_jobs=1).fit(
        X_emb_arr
    )
    _, indices_low = nbrs_low.kneighbors(X_emb_arr)
    indices_low = indices_low[:, 1:]

    rank_low = np.zeros((n, n), dtype=int)
    row_indices = np.arange(n)[:, None]
    rank_low[row_indices, indices_low] = np.arange(n - 1)

    low_rank_indices = rank_low[row_indices, indices_high]
    high_rank_indices = np.broadcast_to(np.arange(n - 1), (n, n - 1))

    Q = np.zeros((n - 1, n - 1), dtype=int)
    np.add.at(Q, (high_rank_indices.ravel(), low_rank_indices.ravel()), 1)
    return Q


def trustworthiness(Q: np.ndarray, k: int) -> float:
    """
    Compute trustworthiness from a co-ranking matrix.

    Trustworthiness penalizes intrusions, i.e. points that appear among the
    ``k`` nearest neighbors in the embedding but were farther away in the
    original space.

    Parameters
    ----------
    Q : np.ndarray of shape (n_samples - 1, n_samples - 1)
        Co-ranking matrix.
    k : int
        Neighborhood size. The normalization used by trustworthiness requires
        ``2 * n_samples - 3 * k - 1 > 0``.

    Returns
    -------
    float
        Trustworthiness score in ``[0, 1]``. Higher is better.

    Raises
    ------
    ValueError
        If ``Q`` is invalid or if ``k`` falls outside the valid domain.

    See Also
    --------
    continuity : Complementary extrusion-based metric.
    compute_coranking_matrix : Construct the required co-ranking matrix.

    Examples
    --------
    >>> import numpy as np
    >>> Q = np.diag([1, 1, 1, 1])
    >>> trustworthiness(Q, k=1)
    1.0
    """
    Q_arr, n, k_int = _validate_k(
        Q,
        k,
        "trustworthiness",
        require_positive_normalizer=True,
    )

    intrusions = Q_arr[k_int:, :k_int]
    row_weights = np.arange(1, intrusions.shape[0] + 1, dtype=float)
    intrusion_sum = float(np.sum(intrusions.sum(axis=1) * row_weights))
    scale = _trust_continuity_scale(n, k_int, "trustworthiness")
    return 1.0 - scale * intrusion_sum


def continuity(Q: np.ndarray, k: int) -> float:
    """
    Compute continuity from a co-ranking matrix.

    Continuity penalizes extrusions, i.e. points that are among the
    ``k`` nearest neighbors in the original space but are pushed farther away in
    the embedding.

    Parameters
    ----------
    Q : np.ndarray of shape (n_samples - 1, n_samples - 1)
        Co-ranking matrix.
    k : int
        Neighborhood size. The normalization used by continuity requires
        ``2 * n_samples - 3 * k - 1 > 0``.

    Returns
    -------
    float
        Continuity score in ``[0, 1]``. Higher is better.

    Raises
    ------
    ValueError
        If ``Q`` is invalid or if ``k`` falls outside the valid domain.

    See Also
    --------
    trustworthiness : Complementary intrusion-based metric.
    compute_coranking_matrix : Construct the required co-ranking matrix.

    Examples
    --------
    >>> import numpy as np
    >>> Q = np.diag([1, 1, 1, 1])
    >>> continuity(Q, k=1)
    1.0
    """
    Q_arr, n, k_int = _validate_k(
        Q,
        k,
        "continuity",
        require_positive_normalizer=True,
    )

    extrusions = Q_arr[:k_int, k_int:]
    col_weights = np.arange(1, extrusions.shape[1] + 1, dtype=float)
    extrusion_sum = float(np.sum(extrusions.sum(axis=0) * col_weights))
    scale = _trust_continuity_scale(n, k_int, "continuity")
    return 1.0 - scale * extrusion_sum


def lcmc(Q: np.ndarray, k: int) -> float:
    """
    Compute the local continuity meta-criterion (LCMC).

    Parameters
    ----------
    Q : np.ndarray of shape (n_samples - 1, n_samples - 1)
        Co-ranking matrix.
    k : int
        Neighborhood size.

    Returns
    -------
    float
        LCMC score. Higher is better.

    Raises
    ------
    ValueError
        If ``Q`` is invalid or if ``k`` falls outside the valid domain.

    See Also
    --------
    trustworthiness : Neighbor-preservation metric.
    continuity : Neighbor-consistency metric.

    Examples
    --------
    >>> import numpy as np
    >>> Q = np.diag([1, 1, 1, 1])
    >>> isinstance(lcmc(Q, k=1), float)
    True
    """
    Q_arr, n, k_int = _validate_k(Q, k, "lcmc")
    overlap = float(np.sum(Q_arr[:k_int, :k_int]))
    return (overlap / (n * k_int)) - (k_int / (n - 1))


def compute_mrre(Q: np.ndarray, k: int) -> Tuple[float, float]:
    """
    Compute mean relative rank errors (MRRE).

    Both intrusion and extrusion MRRE are returned. These are error metrics, so
    lower values are better and ``0`` indicates perfect rank preservation.

    Parameters
    ----------
    Q : np.ndarray of shape (n_samples - 1, n_samples - 1)
        Co-ranking matrix.
    k : int
        Neighborhood size.

    Returns
    -------
    tuple[float, float]
        ``(mrre_intrusion, mrre_extrusion)``.

    Raises
    ------
    ValueError
        If ``Q`` is invalid or if ``k`` falls outside the valid domain.

    See Also
    --------
    trustworthiness : Intrusion-sensitive preservation score.
    continuity : Extrusion-sensitive preservation score.

    Examples
    --------
    >>> import numpy as np
    >>> Q = np.diag([1, 1, 1, 1])
    >>> compute_mrre(Q, k=1)
    (0.0, 0.0)
    """
    Q_arr, n, k_int = _validate_k(Q, k, "compute_mrre")

    i_vals = np.arange(1, k_int + 1, dtype=float)
    harmonic_like = float(np.sum(np.abs(n - 2 * i_vals + 1) / i_vals))

    rows_int = np.arange(k_int, n - 1)
    cols_int = np.arange(k_int)
    high_int, low_int = np.meshgrid(rows_int, cols_int, indexing="ij")
    high_rank_int = high_int + 1
    low_rank_int = low_int + 1
    weights_int = np.abs(low_rank_int - high_rank_int) / high_rank_int
    intrusion_sum = float(np.sum(Q_arr[k_int:, :k_int] * weights_int))

    rows_ext = np.arange(k_int)
    cols_ext = np.arange(k_int, n - 1)
    high_ext, low_ext = np.meshgrid(rows_ext, cols_ext, indexing="ij")
    high_rank_ext = high_ext + 1
    low_rank_ext = low_ext + 1
    weights_ext = np.abs(low_rank_ext - high_rank_ext) / high_rank_ext
    extrusion_sum = float(np.sum(Q_arr[:k_int, k_int:] * weights_ext))

    normalizer = n * harmonic_like
    return intrusion_sum / normalizer, extrusion_sum / normalizer


def shepard_diagram_data(
    X: np.ndarray,
    X_embedded: np.ndarray,
    sample_size: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sampled pairwise distances for a Shepard diagram.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Original high-dimensional data.
    X_embedded : np.ndarray of shape (n_samples, n_components)
        Low-dimensional embedding of the same samples.
    sample_size : int, default=1000
        Number of samples to keep before computing pairwise distances. If
        ``sample_size`` is at least ``n_samples``, all samples are used.
    random_state : int, optional
        Random seed used when subsampling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Pairwise distances in the original and embedded spaces.

    Raises
    ------
    ValueError
        If the inputs are invalid or if ``sample_size <= 1``.

    See Also
    --------
    compute_coranking_matrix : Rank-based global quality summary.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.RandomState(0).rand(10, 3)
    >>> X_emb = X[:, :2]
    >>> d_orig, d_emb = shepard_diagram_data(X, X_emb, sample_size=5, random_state=0)
    >>> len(d_orig) == len(d_emb)
    True
    """
    from sklearn.utils import check_random_state

    X_arr, X_emb_arr = _validate_embedding_pair(
        X,
        X_embedded,
        func_name="shepard_diagram_data",
    )

    if not isinstance(sample_size, (int, np.integer)):
        raise ValueError("`sample_size` must be an integer.")
    sample_size_int = int(sample_size)
    if sample_size_int <= 1:
        raise ValueError("`sample_size` must be greater than 1.")

    n_samples = X_arr.shape[0]

    if n_samples > sample_size_int:
        rng = check_random_state(random_state)
        indices = rng.choice(n_samples, sample_size_int, replace=False)
        X_sub = X_arr[indices]
        X_emb_sub = X_emb_arr[indices]
    else:
        X_sub = X_arr
        X_emb_sub = X_emb_arr

    return pdist(X_sub), pdist(X_emb_sub)
