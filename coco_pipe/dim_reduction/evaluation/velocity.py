"""
Dynamics visualization utilities for velocity-like embedding fields.

This module provides evaluation-adjacent helpers for estimating low-dimensional
velocity vectors from ordered high-dimensional samples. The implementation is
inspired by scVelo-style transition weighting, but it operates on generic
sample-feature matrices rather than a specific single-cell pipeline.

Functions
---------
compute_velocity_fields
    Estimate low-dimensional velocity vectors from ordered samples and an
    aligned embedding.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _validate_inputs(
    X: np.ndarray,
    X_emb: np.ndarray,
    delta_t: int,
    n_neighbors: int,
    sigma: float,
    groups: Optional[np.ndarray],
    times: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Validate and normalize velocity inputs."""
    X = np.asarray(X)
    X_emb = np.asarray(X_emb)

    if X.ndim != 2:
        raise ValueError("`X` must be a 2D array of shape (n_samples, n_features).")
    if X_emb.ndim != 2:
        raise ValueError(
            "`X_emb` must be a 2D array of shape (n_samples, n_components)."
        )
    if X.shape[0] != X_emb.shape[0]:
        raise ValueError("`X` and `X_emb` must have the same number of samples.")

    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2 for velocity computation.")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be > 0.")
    if n_neighbors >= n_samples:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) must be less than n_samples ({n_samples}) "
            "so each sample has at least one non-self neighbor."
        )
    if delta_t <= 0:
        raise ValueError("delta_t must be > 0.")
    if delta_t >= n_samples:
        raise ValueError(
            f"delta_t ({delta_t}) must be less than n_samples ({n_samples})."
        )
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    groups_arr = None
    if groups is not None:
        groups_arr = np.asarray(groups)
        if groups_arr.ndim != 1:
            raise ValueError("`groups` must be a 1D array aligned with samples.")
        if groups_arr.shape[0] != n_samples:
            raise ValueError("`groups` must have the same number of samples as `X`.")

    times_arr = None
    if times is not None:
        times_arr = np.asarray(times)
        if times_arr.ndim != 1:
            raise ValueError("`times` must be a 1D array aligned with samples.")
        if times_arr.shape[0] != n_samples:
            raise ValueError("`times` must have the same number of samples as `X`.")

    return X, X_emb, groups_arr, times_arr


def _iter_velocity_sequences(
    n_samples: int,
    groups: Optional[np.ndarray],
    times: Optional[np.ndarray],
) -> Iterable[np.ndarray]:
    """Yield ordered sample indices for each independent sequence."""
    if groups is None:
        indices = np.arange(n_samples)
        if times is not None:
            indices = indices[np.argsort(times, kind="mergesort")]
        yield indices
        return

    unique_groups = np.unique(groups)
    for group in unique_groups:
        indices = np.flatnonzero(groups == group)
        if times is not None and indices.size > 1:
            indices = indices[np.argsort(times[indices], kind="mergesort")]
        yield indices


def compute_velocity_fields(
    X: np.ndarray,
    X_emb: np.ndarray,
    delta_t: int = 1,
    n_neighbors: int = 30,
    sigma: float = 0.1,
    groups: Optional[np.ndarray] = None,
    times: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute velocity-like vectors in embedding space.

    The function estimates a forward transition direction in the original
    feature space, then projects that direction into the embedding by weighting
    low-dimensional neighbor displacements with cosine-aligned transition
    probabilities.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        High-dimensional data ordered by sequence position.
    X_emb : np.ndarray of shape (n_samples, n_components)
        Low-dimensional embedding aligned with ``X`` row-wise.
    delta_t : int, default=1
        Forward lag in samples used to compute the high-dimensional transition
        vector. This is a sample lag, not a physical time unit. When ``times``
        is provided, the high-dimensional transition is additionally divided by
        the elapsed time between the lagged observations.
    n_neighbors : int, default=30
        Number of non-self neighbors used for local projection.
    sigma : float, default=0.1
        Kernel width controlling the sharpness of transition probabilities.
    groups : np.ndarray of shape (n_samples,), optional
        Group labels defining independent ordered sequences. Velocity vectors
        are computed separately within each group to avoid cross-group
        transitions and cross-group neighbor mixing.
    times : np.ndarray of shape (n_samples,), optional
        Optional ordering variable. When provided, samples are sorted within
        each group before computing forward transitions. The same ordering is
        also used to derive elapsed time scaling for the high-dimensional
        transition vector.

    Returns
    -------
    np.ndarray of shape (n_samples, n_components)
        Velocity vectors in embedding space.

    Raises
    ------
    ValueError
        If the inputs are misaligned, not two-dimensional, contain invalid
        parameter values, or define non-increasing ``times`` within a sequence.

    Notes
    -----
    Samples without a valid forward lagged observation keep a zero velocity
    vector in the output. This typically affects the final ``delta_t`` samples
    in each independent sequence.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.rand(100, 10)
    >>> X_emb = np.random.rand(100, 2)
    >>> V = compute_velocity_fields(X, X_emb, delta_t=1, n_neighbors=10)
    >>> V.shape
    (100, 2)
    """
    X, X_emb, groups_arr, times_arr = _validate_inputs(
        X=X,
        X_emb=X_emb,
        delta_t=delta_t,
        n_neighbors=n_neighbors,
        sigma=sigma,
        groups=groups,
        times=times,
    )

    V_emb = np.zeros_like(X_emb, dtype=float)

    for seq_indices in _iter_velocity_sequences(
        n_samples=X.shape[0],
        groups=groups_arr,
        times=times_arr,
    ):
        if seq_indices.size <= delta_t:
            continue

        X_seq = X[seq_indices]
        X_emb_seq = X_emb[seq_indices]
        seq_times = times_arr[seq_indices] if times_arr is not None else None
        n_seq_neighbors = min(n_neighbors, X_seq.shape[0] - 1)
        nbrs = NearestNeighbors(n_neighbors=n_seq_neighbors + 1)
        nbrs.fit(X_seq)
        raw_neighbor_rows = nbrs.kneighbors(X_seq, return_distance=False)

        for pos in range(seq_indices.size - delta_t):
            next_pos = pos + delta_t
            v_high = X_seq[next_pos] - X_seq[pos]

            if seq_times is not None:
                elapsed = float(seq_times[next_pos] - seq_times[pos])
                if elapsed <= 0:
                    raise ValueError(
                        "`times` must be strictly increasing within each group."
                    )
                v_high = v_high / elapsed

            norm_v = np.linalg.norm(v_high)
            if norm_v < 1e-12:
                continue

            neighbor_idx = raw_neighbor_rows[pos]
            neighbor_idx = neighbor_idx[neighbor_idx != pos][:n_seq_neighbors]
            if neighbor_idx.size == 0:
                continue

            d_high = X_seq[neighbor_idx] - X_seq[pos]
            norm_d = np.linalg.norm(d_high, axis=1)
            valid = norm_d > 1e-12
            if not np.any(valid):
                continue

            d_high = d_high[valid]
            neighbor_idx = neighbor_idx[valid]
            corr = (d_high @ v_high) / (norm_v * np.linalg.norm(d_high, axis=1))

            probs = np.exp(corr / sigma)
            probs_sum = probs.sum()
            if probs_sum <= 0:
                continue
            probs = probs / probs_sum

            d_low = X_emb_seq[neighbor_idx] - X_emb_seq[pos]
            V_emb[seq_indices[pos]] = probs @ d_low

    return V_emb
