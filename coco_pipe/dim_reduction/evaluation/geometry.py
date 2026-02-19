"""
Trajectory Geometry Metrics
===========================

Utilities for analyzing the geometric properties of neural state trajectories.
Includes metrics for speed, curvature, and conditions separation.

Functions
---------
moving_average
    Smooths a timecourse using a valid-mode moving average.
trajectory_speed
    Calculates instantaneous speed (scalar velocity) of an N-dimensional trajectory.
trajectory_curvature
    Calculates geometric curvature (1/Radius) of a trajectory.
trajectory_separation
    Calculates time-resolved Euclidean distance between class centroids.

Author: Hamza Abdelhedi
Date: 2026-02-19
"""

from typing import Dict, Tuple

import numpy as np


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Smooths input array using a valid-mode moving average.

    Parameters
    ----------
    arr : np.ndarray
        Input array (1D).
    window : int
        Size of the smoothing window.

    Returns
    -------
    np.ndarray
        Smoothed array. Length is len(arr) - window + 1.
        If window <= 1, returns original array.
    """
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def trajectory_speed(traj: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Calculate instantaneous speed (scalar velocity) of a trajectory.

    Generalized version of `compute_speed_from_xy`.

    Parameters
    ----------
    traj : np.ndarray
        Trajectory array of shape (..., n_samples, n_dims) or (n_samples, n_dims).
        The second-to-last dimension is treated as time.
    dt : float, default=1.0
        Time step between samples.

    Returns
    -------
    np.ndarray
        Speed array of shape (..., n_samples).
        The last value is padded with the second-to-last value to maintain length.
    """
    # Diff along time dimension (axis -2)
    # Support (N, T, D) or (T, D)
    if traj.ndim < 2:
        raise ValueError("Trajectory must be at least 2D (time, dimensions).")

    diffs = np.diff(traj, axis=-2)

    # Norm of displacement vector
    displacement = np.linalg.norm(diffs, axis=-1)
    speed = displacement / dt

    # Pad last value to match input length
    # We append the last calculated speed to the end
    padding = np.take(speed, [-1], axis=-1)
    full_speed = np.concatenate([speed, padding], axis=-1)

    return full_speed


def trajectory_curvature(traj: np.ndarray) -> np.ndarray:
    """
    Calculate geometric curvature (k = 1/R) of a trajectory.

    k = |x' x x''| / |x'|^3  (for 3D)
    Generalized: k = sqrt(|x'|^2 |x''|^2 - (x' . x'')^2) / |x'|^3

    Parameters
    ----------
    traj : np.ndarray
        Trajectory array of shape (T, D) or (N, T, D).

    Returns
    -------
    np.ndarray
        Curvature timecourse. Padded to match input length.
        First and last points are padded.
    """
    # First derivatives (Velocity)
    vel = np.gradient(traj, axis=-2)

    # Second derivatives (Acceleration)
    acc = np.gradient(vel, axis=-2)

    # Norms
    # shape (..., T)
    v_norm_sq = np.sum(vel**2, axis=-1)
    a_norm_sq = np.sum(acc**2, axis=-1)

    # Dot product v . a
    v_dot_a = np.sum(vel * acc, axis=-1)

    # Numerator: sqrt(|v|^2 |a|^2 - (v.a)^2)
    # This is equivalent to |v x a|
    nume_sq = v_norm_sq * a_norm_sq - v_dot_a**2
    # Floating point fix to avoid negative sqrt
    nume_sq[nume_sq < 0] = 0
    numerator = np.sqrt(nume_sq)

    # Denominator: |v|^3
    denominator = v_norm_sq**1.5

    # Avoid division by zero (stationary points)
    eps = 1e-8
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = numerator / (denominator + eps)

    curvature[denominator < eps] = 0

    return curvature


def trajectory_separation(
    traj: np.ndarray, labels: np.ndarray, metric: str = "euclidean"
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Calculate time-resolved distance between class centroids.

    Parameters
    ----------
    traj : np.ndarray
        Trajectory tensor of shape (n_trials, n_times, n_dims).
    labels : np.ndarray
        Labels for each trial of shape (n_trials,).
    metric : str
        Distance metric (currently only 'euclidean').

    Returns
    -------
    dict
        Dictionary mapping label pairs (label_A, label_B) to
        distance timecourse shape (n_times,).
    """
    unique_labels = np.unique(labels)
    centroids = {}

    # 1. Compute Centroids
    for lbl in unique_labels:
        mask = labels == lbl
        # Mean across trials -> (n_times, n_dims)
        centroids[lbl] = np.mean(traj[mask], axis=0)

    # 2. Compute Pairwise Distances
    distances = {}
    import itertools

    for l1, l2 in itertools.combinations(unique_labels, 2):
        c1 = centroids[l1]
        c2 = centroids[l2]

        # Dist shape (n_times,)
        dist = np.linalg.norm(c1 - c2, axis=-1)
        distances[(l1, l2)] = dist

    return distances
