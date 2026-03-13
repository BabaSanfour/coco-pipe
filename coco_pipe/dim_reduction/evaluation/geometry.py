"""
Trajectory geometry metrics and smoothing utilities.

This module provides small, generic helpers for analyzing ordered trajectories
in embedded spaces. The functions are reducer-agnostic and operate on standard
NumPy arrays rather than domain-specific container types.

Functions
---------
moving_average
    Smooth a one-dimensional timecourse with a valid-mode moving average.
trajectory_acceleration
    Compute instantaneous acceleration magnitude from second-order derivatives.
trajectory_speed
    Compute instantaneous speed from first-order trajectory differences.
trajectory_curvature
    Compute geometric curvature from first- and second-order derivatives.
trajectory_dispersion
    Compute within-group trajectory spread across time.
trajectory_displacement
    Compute displacement from the initial trajectory state across time.
trajectory_path_length
    Compute total or cumulative path length.
trajectory_separation
    Compute time-resolved group separation across time using a selected method.
trajectory_tortuosity
    Compute the ratio between path length and net displacement.
trajectory_turning_angle
    Compute local turning angles between consecutive trajectory segments.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import itertools
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "moving_average",
    "trajectory_acceleration",
    "trajectory_speed",
    "trajectory_curvature",
    "trajectory_path_length",
    "trajectory_displacement",
    "trajectory_tortuosity",
    "trajectory_turning_angle",
    "trajectory_dispersion",
    "trajectory_separation",
]


def _validate_trajectory_array(traj: np.ndarray, min_timepoints: int = 2) -> np.ndarray:
    """Validate generic trajectory inputs with time on axis ``-2``."""
    traj = np.asarray(traj, dtype=float)
    if traj.ndim < 2:
        raise ValueError("`traj` must be at least 2D with shape (..., time, dims).")
    if traj.shape[-2] < min_timepoints:
        msg = f"Trajectory must contain at least {min_timepoints} time points."
        raise ValueError(msg)
    return traj


def _validate_trial_trajectory_labels(
    traj: np.ndarray,
    labels: Optional[np.ndarray] = None,
    min_unique_labels: int = 0,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Validate trial-wise trajectories and optional trial labels."""
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 3:
        raise ValueError("`traj` must have shape (n_trials, n_times, n_dims).")

    if labels is None:
        return traj, None, np.array([])

    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        raise ValueError("`labels` must be a 1D array of trial labels.")
    if labels_arr.shape[0] != traj.shape[0]:
        raise ValueError("`labels` must have one entry per trial in `traj`.")

    unique_labels = np.unique(labels_arr)
    if unique_labels.size < min_unique_labels:
        raise ValueError(
            f"At least {min_unique_labels} unique labels are required for this metric."
        )
    return traj, labels_arr, unique_labels


def _centroid_dispersion(samples: np.ndarray) -> np.ndarray:
    """Return RMS distance to the centroid at each time point."""
    centroid = np.mean(samples, axis=0)
    sq_dist = np.sum((samples - centroid) ** 2, axis=-1)
    return np.sqrt(np.mean(sq_dist, axis=0))


def _pairwise_distances(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Compute Euclidean pairwise distances between two point clouds."""
    diffs = points_a[:, None, :] - points_b[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def _mean_self_pairwise_distance(points: np.ndarray) -> float:
    """Return the mean pairwise distance within one point cloud."""
    if points.shape[0] == 0:
        return 0.0
    distances = _pairwise_distances(points, points)
    return float(np.mean(distances))


def _nearest_within_distances(points: np.ndarray) -> np.ndarray:
    """Return nearest-neighbor distances within one label group."""
    if points.shape[0] <= 1:
        return np.zeros(points.shape[0], dtype=float)
    distances = _pairwise_distances(points, points)
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)


def _pairwise_label_timecourses(
    traj: np.ndarray,
    labels: np.ndarray,
    pair_reducer,
    **kwargs,
) -> Dict[Tuple[str, str], np.ndarray]:
    """Apply a pairwise reducer to each label pair across time."""
    traj, labels, unique_labels = _validate_trial_trajectory_labels(
        traj,
        labels,
        min_unique_labels=2,
    )

    outputs = {}
    for label_a, label_b in itertools.combinations(unique_labels.tolist(), 2):
        samples_a = traj[labels == label_a]
        samples_b = traj[labels == label_b]
        outputs[(label_a, label_b)] = np.asarray(
            pair_reducer(samples_a, samples_b, **kwargs),
            dtype=float,
        )
    return outputs


def _centroid_separation_timecourse(
    samples_a: np.ndarray, samples_b: np.ndarray
) -> np.ndarray:
    """Compute Euclidean centroid separation for one label pair."""
    centroid_a = np.mean(samples_a, axis=0)
    centroid_b = np.mean(samples_b, axis=0)
    return np.linalg.norm(centroid_a - centroid_b, axis=-1)


def _within_between_ratio_timecourse(
    samples_a: np.ndarray, samples_b: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Compute within/between normalized separation for one label pair."""
    between = _centroid_separation_timecourse(samples_a, samples_b)
    within_a = _centroid_dispersion(samples_a)
    within_b = _centroid_dispersion(samples_b)
    return between / (within_a + within_b + eps)


def _mahalanobis_separation_timecourse(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    regularization: float = 1e-6,
) -> np.ndarray:
    """Compute Mahalanobis separation for one label pair."""
    timecourse = np.zeros(samples_a.shape[1], dtype=float)
    for t_idx in range(samples_a.shape[1]):
        points_a = samples_a[:, t_idx, :]
        points_b = samples_b[:, t_idx, :]
        centroid_a = np.mean(points_a, axis=0)
        centroid_b = np.mean(points_b, axis=0)
        diff = centroid_a - centroid_b

        centered_a = points_a - centroid_a
        centered_b = points_b - centroid_b
        scatter = centered_a.T @ centered_a + centered_b.T @ centered_b
        dof = max(points_a.shape[0] + points_b.shape[0] - 2, 1)
        covariance = scatter / float(dof)
        covariance = covariance + regularization * np.eye(covariance.shape[0])

        inv_covariance = np.linalg.pinv(covariance)
        mahal_sq = float(diff.T @ inv_covariance @ diff)
        timecourse[t_idx] = np.sqrt(max(mahal_sq, 0.0))
    return timecourse


def _distributional_separation_timecourse(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    metric: str = "energy",
) -> np.ndarray:
    """Compute distribution-level separation for one label pair."""
    if metric != "energy":
        raise ValueError("Only metric='energy' is currently supported.")

    timecourse = np.zeros(samples_a.shape[1], dtype=float)
    for t_idx in range(samples_a.shape[1]):
        points_a = samples_a[:, t_idx, :]
        points_b = samples_b[:, t_idx, :]
        cross = np.mean(_pairwise_distances(points_a, points_b))
        within_a = _mean_self_pairwise_distance(points_a)
        within_b = _mean_self_pairwise_distance(points_b)
        timecourse[t_idx] = max(2.0 * cross - within_a - within_b, 0.0)
    return timecourse


def _margin_separation_timecourse(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    agg: str = "median",
) -> np.ndarray:
    """Compute nearest-neighbor margin separation for one label pair."""
    if agg not in {"median", "mean"}:
        raise ValueError("`agg` must be either 'median' or 'mean'.")

    aggregate = np.median if agg == "median" else np.mean
    timecourse = np.zeros(samples_a.shape[1], dtype=float)
    for t_idx in range(samples_a.shape[1]):
        points_a = samples_a[:, t_idx, :]
        points_b = samples_b[:, t_idx, :]

        cross = _pairwise_distances(points_a, points_b)
        nearest_cross_a = np.min(cross, axis=1)
        nearest_cross_b = np.min(cross, axis=0)
        nearest_within_a = _nearest_within_distances(points_a)
        nearest_within_b = _nearest_within_distances(points_b)

        point_margins = np.concatenate(
            [
                nearest_cross_a - nearest_within_a,
                nearest_cross_b - nearest_within_b,
            ]
        )
        timecourse[t_idx] = float(aggregate(point_margins))
    return timecourse


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Smooth a one-dimensional array with a valid-mode moving average.

    Parameters
    ----------
    arr : np.ndarray of shape (n_samples,)
        Input array to smooth.
    window : int
        Size of the smoothing window. Must be a positive integer no larger than
        the array length.

    Returns
    -------
    np.ndarray
        Smoothed array. The output length is ``n_samples - window + 1``.
        If ``window == 1``, a copy of the input is returned.

    Raises
    ------
    ValueError
        If ``arr`` is not one-dimensional, if ``window`` is not positive, or
        if ``window`` is larger than the input length.

    See Also
    --------
    trajectory_speed : First-order trajectory dynamics without smoothing.
    trajectory_turning_angle : Local directional changes along a trajectory.

    Examples
    --------
    >>> import numpy as np
    >>> moving_average(np.array([1, 2, 3, 4, 5]), window=3)
    array([2., 3., 4.])
    """
    arr = np.asarray(arr)

    if arr.ndim != 1:
        raise ValueError("`arr` must be a 1D array.")
    if window <= 0:
        raise ValueError("`window` must be a positive integer.")
    if window > arr.shape[0]:
        raise ValueError("`window` cannot be larger than the input length.")
    if window == 1:
        return arr.copy()

    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="valid")


def trajectory_acceleration(traj: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Calculate instantaneous acceleration magnitude.

    Parameters
    ----------
    traj : np.ndarray of shape (..., n_times, n_dims)
        Trajectory array. The second-to-last axis is interpreted as time and
        the last axis as coordinates.
    dt : float, default=1.0
        Uniform time step between consecutive samples.

    Returns
    -------
    np.ndarray of shape (..., n_times)
        Acceleration-magnitude timecourse aligned with the input time axis.

    Raises
    ------
    ValueError
        If ``traj`` has fewer than two dimensions, contains fewer than three
        time points, or if ``dt <= 0``.

    See Also
    --------
    trajectory_speed : First-order trajectory dynamics.
    trajectory_curvature : Geometric bending of a trajectory.
    trajectory_turning_angle : Local directional changes between segments.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0.0, 2.0, 3)
    >>> traj = np.stack([t**2, np.zeros_like(t)], axis=1)
    >>> trajectory_acceleration(traj, dt=1.0).shape
    (3,)
    """
    traj = _validate_trajectory_array(traj, min_timepoints=3)
    if dt <= 0:
        raise ValueError("`dt` must be > 0.")

    velocity = np.gradient(traj, dt, axis=-2)
    acceleration = np.gradient(velocity, dt, axis=-2)
    return np.linalg.norm(acceleration, axis=-1)


def trajectory_speed(traj: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Calculate instantaneous trajectory speed.

    Parameters
    ----------
    traj : np.ndarray of shape (..., n_times, n_dims)
        Trajectory array. The second-to-last axis is interpreted as time and
        the last axis as coordinates.
    dt : float, default=1.0
        Uniform time step between consecutive samples.

    Returns
    -------
    np.ndarray of shape (..., n_times)
        Instantaneous speed timecourse. The final value is padded with the last
        computed speed so that the output length matches the number of time
        points.

    Raises
    ------
    ValueError
        If ``traj`` has fewer than two dimensions, contains fewer than two time
        points, or if ``dt <= 0``.

    Notes
    -----
    This function computes the norm of the first difference along the time
    axis, divided by ``dt``.

    See Also
    --------
    trajectory_acceleration : Second-order trajectory dynamics.
    trajectory_path_length : Total or cumulative traveled distance.
    trajectory_displacement : Distance from the initial state across time.

    Examples
    --------
    >>> import numpy as np
    >>> traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    >>> trajectory_speed(traj)
    array([1., 1., 1.])
    """
    traj = _validate_trajectory_array(traj, min_timepoints=2)
    if dt <= 0:
        raise ValueError("`dt` must be > 0.")

    diffs = np.diff(traj, axis=-2)
    speed = np.linalg.norm(diffs, axis=-1) / dt
    padding = np.take(speed, [-1], axis=-1)
    return np.concatenate([speed, padding], axis=-1)


def trajectory_curvature(traj: np.ndarray) -> np.ndarray:
    """
    Calculate geometric curvature of a trajectory.

    Parameters
    ----------
    traj : np.ndarray of shape (..., n_times, n_dims)
        Trajectory array. The second-to-last axis is interpreted as time and
        the last axis as coordinates.

    Returns
    -------
    np.ndarray of shape (..., n_times)
        Curvature timecourse aligned with the input time axis.

    Raises
    ------
    ValueError
        If ``traj`` has fewer than two dimensions or fewer than two time
        points.

    Notes
    -----
    For vector-valued trajectories, curvature is computed from first and second
    derivatives using the generalized formula

    ``sqrt(||v||^2 ||a||^2 - (v . a)^2) / ||v||^3``.

    The implementation assumes uniformly spaced samples.

    See Also
    --------
    trajectory_turning_angle : Discrete local directional change.
    trajectory_tortuosity : Path inefficiency relative to net displacement.
    trajectory_speed : First-order trajectory dynamics.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 2 * np.pi, 100)
    >>> traj = np.stack([np.cos(t), np.sin(t)], axis=1)
    >>> k = trajectory_curvature(traj)
    >>> k.shape
    (100,)
    """
    traj = _validate_trajectory_array(traj, min_timepoints=2)

    vel = np.gradient(traj, axis=-2)
    acc = np.gradient(vel, axis=-2)

    v_norm_sq = np.sum(vel**2, axis=-1)
    a_norm_sq = np.sum(acc**2, axis=-1)
    v_dot_a = np.sum(vel * acc, axis=-1)

    numerator_sq = v_norm_sq * a_norm_sq - v_dot_a**2
    numerator_sq = np.maximum(numerator_sq, 0.0)
    numerator = np.sqrt(numerator_sq)

    denominator = v_norm_sq**1.5
    eps = 1e-8
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = numerator / (denominator + eps)
    curvature[denominator < eps] = 0.0

    return curvature


def trajectory_path_length(traj: np.ndarray, *, cumulative: bool = False) -> np.ndarray:
    """
    Calculate trajectory path length.

    Parameters
    ----------
    traj : np.ndarray of shape (..., n_times, n_dims)
        Trajectory array. The second-to-last axis is interpreted as time and
        the last axis as coordinates.
    cumulative : bool, default=False
        If ``True``, return cumulative path length aligned with the input time
        axis. Otherwise return total path length for each trajectory.

    Returns
    -------
    np.ndarray
        Total path length with shape ``(...)`` when ``cumulative=False``, or
        cumulative path length with shape ``(..., n_times)`` when
        ``cumulative=True``.

    See Also
    --------
    trajectory_displacement : Distance from the initial state across time.
    trajectory_tortuosity : Ratio of path length to net displacement.
    trajectory_speed : First-order local motion magnitude.

    Examples
    --------
    >>> import numpy as np
    >>> traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    >>> trajectory_path_length(traj)
    np.float64(2.0)
    """
    traj = _validate_trajectory_array(traj, min_timepoints=2)
    segment_lengths = np.linalg.norm(np.diff(traj, axis=-2), axis=-1)

    if cumulative:
        cumulative_lengths = np.cumsum(segment_lengths, axis=-1)
        zeros = np.zeros(cumulative_lengths.shape[:-1] + (1,), dtype=float)
        return np.concatenate([zeros, cumulative_lengths], axis=-1)
    return np.sum(segment_lengths, axis=-1)


def trajectory_displacement(traj: np.ndarray) -> np.ndarray:
    """
    Calculate displacement from the initial state across time.

    Parameters
    ----------
    traj : np.ndarray of shape (..., n_times, n_dims)
        Trajectory array. The second-to-last axis is interpreted as time and
        the last axis as coordinates.

    Returns
    -------
    np.ndarray of shape (..., n_times)
        Euclidean displacement from the first time point at each time index.

    See Also
    --------
    trajectory_path_length : Total or cumulative traveled distance.
    trajectory_tortuosity : Ratio of traveled distance to final displacement.

    Examples
    --------
    >>> import numpy as np
    >>> traj = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    >>> trajectory_displacement(traj)
    array([0.        , 1.        , 1.41421356])
    """
    traj = _validate_trajectory_array(traj, min_timepoints=1)
    origin = traj[..., :1, :]
    return np.linalg.norm(traj - origin, axis=-1)


def trajectory_tortuosity(traj: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Calculate trajectory tortuosity.

    Tortuosity is defined as total path length divided by net displacement from
    the initial to the final state.

    Parameters
    ----------
    traj : np.ndarray of shape (..., n_times, n_dims)
        Trajectory array. The second-to-last axis is interpreted as time and
        the last axis as coordinates.
    eps : float, default=1e-8
        Small constant used to identify near-zero displacement.

    Returns
    -------
    np.ndarray of shape (...)
        Tortuosity for each trajectory. Stationary trajectories return ``1.0``;
        trajectories with nonzero path length but near-zero net displacement
        return ``np.inf``.

    See Also
    --------
    trajectory_path_length : Total traveled distance along the path.
    trajectory_displacement : Net displacement from start to end.
    trajectory_curvature : Local geometric bending.

    Examples
    --------
    >>> import numpy as np
    >>> traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    >>> trajectory_tortuosity(traj)
    np.float64(1.0)
    """
    traj = _validate_trajectory_array(traj, min_timepoints=2)
    total_length = trajectory_path_length(traj, cumulative=False)
    net_displacement = np.linalg.norm(traj[..., -1, :] - traj[..., 0, :], axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        tortuosity = total_length / net_displacement
    stationary = net_displacement < eps
    tortuosity = np.where(stationary & (total_length < eps), 1.0, tortuosity)
    tortuosity = np.where(stationary & (total_length >= eps), np.inf, tortuosity)
    return tortuosity


def trajectory_turning_angle(traj: np.ndarray) -> np.ndarray:
    """
    Calculate local turning angles between consecutive trajectory segments.

    Parameters
    ----------
    traj : np.ndarray of shape (..., n_times, n_dims)
        Trajectory array. The second-to-last axis is interpreted as time and
        the last axis as coordinates.

    Returns
    -------
    np.ndarray of shape (..., n_times)
        Turning-angle timecourse in radians. The first and last time points are
        padded with the nearest interior angle to preserve length.

    See Also
    --------
    trajectory_curvature : Continuous geometric bending.
    trajectory_speed : Local motion magnitude.
    trajectory_path_length : Total or cumulative traveled distance.

    Examples
    --------
    >>> import numpy as np
    >>> traj = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    >>> trajectory_turning_angle(traj)
    array([1.57079633, 1.57079633, 1.57079633])
    """
    traj = _validate_trajectory_array(traj, min_timepoints=3)
    steps = np.diff(traj, axis=-2)
    step_prev = steps[..., :-1, :]
    step_next = steps[..., 1:, :]

    prev_norm = np.linalg.norm(step_prev, axis=-1)
    next_norm = np.linalg.norm(step_next, axis=-1)
    denom = prev_norm * next_norm

    with np.errstate(divide="ignore", invalid="ignore"):
        cos_angle = np.sum(step_prev * step_next, axis=-1) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles = np.arccos(cos_angle)
    angles = np.where(denom < 1e-12, 0.0, angles)

    pad_start = np.take(angles, [0], axis=-1)
    pad_end = np.take(angles, [-1], axis=-1)
    return np.concatenate([pad_start, angles, pad_end], axis=-1)


def trajectory_dispersion(
    traj: np.ndarray, labels: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray] | np.ndarray:
    """
    Calculate within-group trajectory dispersion across time.

    Parameters
    ----------
    traj : np.ndarray of shape (n_trials, n_times, n_dims)
        Trial trajectory tensor.
    labels : np.ndarray of shape (n_trials,), optional
        Optional group label for each trial. If omitted, a single global
        dispersion timecourse is returned.

    Returns
    -------
    np.ndarray or dict[str, np.ndarray]
        Global dispersion timecourse when ``labels`` is omitted, otherwise a
        mapping from label to dispersion timecourse.

    See Also
    --------
    trajectory_separation : Unified separation entrypoint.
    trajectory_separation : Use ``method="within_between_ratio"`` for
        normalized separation.

    Examples
    --------
    >>> import numpy as np
    >>> traj = np.zeros((2, 3, 2))
    >>> traj[1, :, 0] = 1.0
    >>> trajectory_dispersion(traj)
    array([0.5, 0.5, 0.5])
    """
    traj, labels_arr, unique_labels = _validate_trial_trajectory_labels(
        traj,
        labels,
        min_unique_labels=0,
    )

    if labels_arr is None:
        return _centroid_dispersion(traj)

    return {
        label: _centroid_dispersion(traj[labels_arr == label])
        for label in unique_labels.tolist()
    }


def trajectory_separation(
    traj: np.ndarray,
    labels: np.ndarray,
    method: str = "centroid",
    **kwargs,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Calculate time-resolved separation between labeled trajectory groups.

    Parameters
    ----------
    traj : np.ndarray of shape (n_trials, n_times, n_dims)
        Trajectory tensor containing one trajectory per trial.
    labels : np.ndarray of shape (n_trials,)
        Class label for each trial.
    method : {"centroid", "within_between_ratio", "mahalanobis",
        "distributional", "margin"}, default="centroid"
        Separation definition to compute.
    **kwargs : dict
        Additional keyword arguments forwarded to the selected separation
        method.

    Returns
    -------
    dict[tuple[str, str], np.ndarray]
        Mapping from label pairs to separation timecourses of shape ``(n_times,)``.

    Raises
    ------
    ValueError
        If the inputs are invalid or if an unsupported separation method is
        requested.

    Notes
    -----
    This is the high-level separation entrypoint for trajectory-group
    comparison. It dispatches to the more specific separation primitives in
    this module.

    Supported methods:

    - ``"centroid"``: Euclidean distance between label centroids.
    - ``"within_between_ratio"``: Between-centroid distance normalized by
      within-group dispersion.
    - ``"mahalanobis"``: Covariance-aware centroid separation.
    - ``"distributional"``: Energy-distance separation between trial clouds.
    - ``"margin"``: Nearest-cross minus nearest-within margin separation.

    See Also
    --------
    trajectory_dispersion : Within-group spread used by some separation methods.

    Examples
    --------
    >>> import numpy as np
    >>> traj = np.zeros((4, 5, 2))
    >>> labels = np.array(["A", "A", "B", "B"])
    >>> sep = trajectory_separation(traj, labels, method="centroid")
    >>> list(sep.keys())
    [('A', 'B')]
    """
    reducers = {
        "centroid": _centroid_separation_timecourse,
        "within_between_ratio": _within_between_ratio_timecourse,
        "mahalanobis": _mahalanobis_separation_timecourse,
        "distributional": _distributional_separation_timecourse,
        "margin": _margin_separation_timecourse,
    }
    if method not in reducers:
        supported = ", ".join(reducers)
        raise ValueError(
            f"Unsupported separation method '{method}'. Supported methods: {supported}."
        )
    return _pairwise_label_timecourses(traj, labels, reducers[method], **kwargs)
