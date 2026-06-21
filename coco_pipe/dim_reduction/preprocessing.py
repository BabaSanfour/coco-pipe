"""
PCA score preprocessing utilities.

Functions
---------
apply_pca_score_baseline
    Baseline-correct PCA scores using an N-point or time-window approach.
flip_pc_scores_for_consistency
    Flip PC sign so the mean within a reference window is positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "apply_pca_score_baseline",
    "flip_pc_scores_for_consistency",
]


def apply_pca_score_baseline(
    time: np.ndarray,
    scores: pd.DataFrame,
    n_points: int | None = None,
    baseline_min_ms: float = -200.0,
    baseline_max_ms: float = 0.0,
) -> pd.DataFrame:
    """
    Baseline-correct PCA scores.

    Two modes are available:

    * **N-point mode** (``n_points`` is not ``None``): subtract the mean of
      the first ``n_points`` time samples from every PC row.  Applied before
      any time cropping so that it replicates older script behaviour.
    * **Time-window mode** (default): subtract the mean within
      ``[baseline_min_ms, baseline_max_ms]``.

    Parameters
    ----------
    time : np.ndarray of shape (n_times,)
        Time stamps aligned with the columns of ``scores``.
    scores : pd.DataFrame
        PC rows x time columns.
    n_points : int, optional
        Number of leading samples to use as baseline.  When set, takes
        priority over the time-window mode.
    baseline_min_ms, baseline_max_ms : float
        Time-window baseline boundaries used when ``n_points`` is ``None``.

    Returns
    -------
    pd.DataFrame
        Baseline-corrected copy of ``scores``.
    """
    scores = scores.copy()

    if n_points is not None:
        n = int(n_points)
        if n > 0 and scores.shape[1] >= n:
            baseline = scores.iloc[:, :n].mean(axis=1)
            scores = scores.sub(baseline, axis=0)
    else:
        bmask = (time >= baseline_min_ms) & (time <= baseline_max_ms)
        if bmask.any():
            baseline = scores.iloc[:, bmask].mean(axis=1)
            scores = scores.sub(baseline, axis=0)

    return scores


def flip_pc_scores_for_consistency(
    scores: pd.DataFrame,
    time: np.ndarray,
    flip_window_ms: tuple[float, float] = (-200.0, 800.0),
) -> pd.DataFrame:
    """
    Flip each PC row so its mean within a reference window is non-negative.

    This makes the sign of PCA components visually consistent across subjects
    and conditions, removing the arbitrary sign indeterminacy of PCA.

    Parameters
    ----------
    scores : pd.DataFrame
        PC rows x time columns.
    time : np.ndarray of shape (n_times,)
        Time stamps aligned with the columns of ``scores``.
    flip_window_ms : tuple[float, float], default=(-200.0, 800.0)
        Half-open time window used to compute the reference mean.

    Returns
    -------
    pd.DataFrame
        Copy of ``scores`` with sign-flipped rows as needed.
    """
    tmin, tmax = flip_window_ms
    mask = (time >= tmin) & (time <= tmax)
    if not mask.any():
        return scores.copy()

    out = scores.copy()
    for pc in out.index:
        m = np.nanmean(out.loc[pc].to_numpy(dtype=float)[mask])
        if np.isfinite(m) and m < 0:
            out.loc[pc] *= -1
    return out
