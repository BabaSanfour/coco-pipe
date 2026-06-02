"""
Results Container.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from .geometry import (
    trajectory_acceleration,
    trajectory_auc_speed,
    trajectory_cohesion,
    trajectory_curvature,
    trajectory_dispersion,
    trajectory_displacement,
    trajectory_distance_from_center,
    trajectory_intra_spread,
    trajectory_jerk,
    trajectory_path_length,
    trajectory_separation,
    trajectory_speed,
    trajectory_tortuosity,
    trajectory_turning_angle,
)
from .metrics import (
    compute_coranking_matrix,
    compute_mrre,
    continuity,
    lcmc,
    shepard_diagram_data,
    trustworthiness,
)
from .velocity import compute_velocity_fields


class TrajectoryResult:
    """
    Unified Container for Trajectory Geometry Results.

    Provides tidy data views for easier analysis, visualization, and
    statistical assessment of trajectory dynamics across subjects and conditions.
    """

    _REDUCERS = [
        (
            "trajectory_length",
            lambda traj, t: float(np.nanmean(trajectory_path_length(traj))),
        ),
        (
            "straight_distance",
            lambda traj, t: float(
                np.nanmean(trajectory_displacement(traj, final=True))
            ),
        ),
        ("tortuosity", lambda traj, t: float(np.nanmean(trajectory_tortuosity(traj)))),
        (
            "mean_speed",
            lambda traj, t: float(np.nanmean(trajectory_speed(traj, time=t))),
        ),
        ("max_speed", lambda traj, t: float(np.nanmax(trajectory_speed(traj, time=t)))),
        (
            "auc_speed",
            lambda traj, t: float(np.nanmean(trajectory_auc_speed(traj, time=t))),
        ),
        (
            "mean_curvature",
            lambda traj, t: float(np.nanmean(trajectory_curvature(traj))),
        ),
        ("max_curvature", lambda traj, t: float(np.nanmax(trajectory_curvature(traj)))),
        (
            "mean_acceleration",
            lambda traj, t: float(np.nanmean(trajectory_acceleration(traj))),
        ),
        ("mean_jerk", lambda traj, t: float(np.nanmean(trajectory_jerk(traj)))),
        (
            "mean_turning_angle",
            lambda traj, t: float(np.nanmean(trajectory_turning_angle(traj))),
        ),
        (
            "mean_distance_from_center",
            lambda traj, t: float(np.nanmean(trajectory_distance_from_center(traj))),
        ),
        (
            "mean_dispersion",
            lambda traj, t: float(np.nanmean(trajectory_dispersion(traj))),
        ),
    ]

    def __init__(
        self,
        trajectories: np.ndarray,
        times: np.ndarray,
        subjects: np.ndarray,
        conditions: np.ndarray,
    ):
        """
        Initialize the TrajectoryResult container.

        Parameters
        ----------
        trajectories : np.ndarray
            Shape (n_trials, n_times, n_dims).
        times : np.ndarray
            Shape (n_times,). Time coordinates.
        subjects : np.ndarray
            Shape (n_trials,). Per-trial subject identifier (any hashable).
        conditions : np.ndarray
            Shape (n_trials,). Per-trial condition label (int).
        """
        if trajectories.ndim != 3:
            raise ValueError(
                "`trajectories` must be (n_trials, n_times, n_dims); "
                f"got {trajectories.shape}."
            )
        self.trajectories = trajectories
        self.times = times
        self.subjects = np.asarray(subjects)
        self.conditions = np.asarray(conditions).astype(int)

    def get_per_trial_scalars(self) -> pd.DataFrame:
        """Compute per-trial scalar metrics in long format.

        Returns
        -------
        pd.DataFrame
            Columns: ``subject``, ``condition``, ``trial``, ``metric``, ``value``.
        """
        rows: list[dict[str, Any]] = []
        for trial_idx in range(self.trajectories.shape[0]):
            traj = self.trajectories[trial_idx][None]  # add singleton trial axis
            sub = self.subjects[trial_idx]
            cond = self.conditions[trial_idx]
            for name, fn in self._REDUCERS:
                try:
                    val = fn(traj, self.times)
                except Exception:
                    val = float("nan")
                rows.append(
                    {
                        "subject": sub,
                        "condition": cond,
                        "trial": trial_idx,
                        "metric": name,
                        "value": val,
                    }
                )
        return pd.DataFrame(rows)

    def get_per_condition_scalars(self) -> pd.DataFrame:
        """Compute within-condition spread metrics (cohesion, intra_spread).

        Returns
        -------
        pd.DataFrame
            Columns: ``subject``, ``condition``, ``metric``, ``value``.
        """
        rows: list[dict[str, Any]] = []
        for sub in np.unique(self.subjects):
            for cond in np.unique(self.conditions):
                mask = (self.subjects == sub) & (self.conditions == cond)
                if not mask.any():
                    continue
                stack = self.trajectories[mask]
                try:
                    cohesion = float(np.nanmean(trajectory_cohesion(stack)))
                    intra = float(np.nanmean(trajectory_intra_spread(stack)))
                except Exception:
                    cohesion, intra = float("nan"), float("nan")
                rows.extend(
                    [
                        {
                            "subject": sub,
                            "condition": cond,
                            "metric": "mean_cohesion",
                            "value": cohesion,
                        },
                        {
                            "subject": sub,
                            "condition": cond,
                            "metric": "mean_intra_spread",
                            "value": intra,
                        },
                    ]
                )
        return pd.DataFrame(rows)

    def get_separation_pair_scalars(
        self, methods: Sequence[str] = ("centroid", "mahalanobis")
    ) -> pd.DataFrame:
        """Per-subject, per-condition-pair separation peak / peak-time / AUC.

        For each method in ``methods``, calls ``trajectory_separation`` on the
        trials for one subject and extracts scalar summaries per condition pair.

        Returns
        -------
        pd.DataFrame
            Columns: ``subject``, ``method``, ``pair`` (string ``"A_vs_B"``),
            ``label_a``, ``label_b``, ``metric``, ``value``.
            Metric values: ``peak_separation``, ``peak_separation_time``,
            ``auc_separation``.
        """
        rows: list[dict[str, Any]] = []
        for sub in np.unique(self.subjects):
            mask = self.subjects == sub
            traj_sub = self.trajectories[mask]
            cond_sub = self.conditions[mask]
            if np.unique(cond_sub).size < 2:
                continue
            for method in methods:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sep = trajectory_separation(traj_sub, cond_sub, method=method)
                for (a, b), curve in sep.items():
                    curve = np.asarray(curve, dtype=float)
                    if not np.any(np.isfinite(curve)):
                        continue
                    peak_idx = int(np.nanargmax(curve))
                    peak = float(curve[peak_idx])
                    peak_time = float(self.times[peak_idx])
                    auc = float(np.trapezoid(curve, self.times))
                    pair = f"{int(a)}_vs_{int(b)}"
                    for metric_name, val in [
                        ("peak_separation", peak),
                        ("peak_separation_time", peak_time),
                        ("auc_separation", auc),
                    ]:
                        rows.append(
                            {
                                "subject": sub,
                                "method": method,
                                "pair": pair,
                                "label_a": int(a),
                                "label_b": int(b),
                                "metric": metric_name,
                                "value": val,
                            }
                        )
        return pd.DataFrame(rows)

    def get_separation_timecourses(
        self, methods: Sequence[str] = ("centroid", "mahalanobis")
    ) -> dict[str, dict[Tuple[int, int], np.ndarray]]:
        """Pooled-across-subjects separation timecourses per condition pair.

        Returns
        -------
        dict
            ``{method: {(a, b): timecourse_array}}``. Use the result directly with
            ``coco_pipe.viz.interactive.plot_trajectory_separation``.
        """
        out: dict[str, dict[Tuple[int, int], np.ndarray]] = {}
        for method in methods:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sep = trajectory_separation(
                    self.trajectories, self.conditions, method=method
                )
            out[method] = {
                (int(a), int(b)): np.asarray(curve) for (a, b), curve in sep.items()
            }
        return out

    def slice_time(self, tmin: float, tmax: float) -> "TrajectoryResult":
        """Return a new TrajectoryResult restricted to a time window."""
        mask = (self.times >= tmin) & (self.times <= tmax)
        if not np.any(mask):
            raise ValueError(f"No timepoints found in window [{tmin}, {tmax}].")

        return type(self)(
            trajectories=self.trajectories[:, mask, :],
            times=self.times[mask],
            subjects=self.subjects,
            conditions=self.conditions,
        )

    def filter(
        self,
        subjects: Optional[Sequence] = None,
        conditions: Optional[Sequence[int]] = None,
    ) -> "TrajectoryResult":
        """Return a new TrajectoryResult containing only specified
        subjects/conditions."""
        mask = np.ones(self.trajectories.shape[0], dtype=bool)
        if subjects is not None:
            mask &= np.isin(self.subjects, subjects)
        if conditions is not None:
            mask &= np.isin(self.conditions, conditions)

        if not np.any(mask):
            raise ValueError("Filtering resulted in 0 trials.")

        return type(self)(
            trajectories=self.trajectories[mask],
            times=self.times,
            subjects=self.subjects[mask],
            conditions=self.conditions[mask],
        )

    def get_kinematic_timecourses(self, metrics: Sequence[str]) -> pd.DataFrame:
        """Compute and return continuous kinematic timecourses.

        Returns
        -------
        pd.DataFrame
            Columns: subject, condition, trial, time, metric, value.
        """
        funcs = {
            "speed": lambda tr, t: trajectory_speed(tr, time=t),
            "acceleration": lambda tr, t: trajectory_acceleration(tr),
            "jerk": lambda tr, t: trajectory_jerk(tr),
            "curvature": lambda tr, t: trajectory_curvature(tr),
            "turning_angle": lambda tr, t: trajectory_turning_angle(tr),
            "distance_from_center": lambda tr, t: trajectory_distance_from_center(tr),
            "dispersion": lambda tr, t: trajectory_dispersion(tr),
            "displacement": lambda tr, t: trajectory_displacement(tr, final=False),
            "path_length": lambda tr, t: trajectory_path_length(tr, cumulative=True),
        }
        rows = []
        n_trials, n_times, _ = self.trajectories.shape
        for metric in metrics:
            if metric not in funcs:
                raise ValueError(
                    f"Unknown metric {metric}. Available: {list(funcs.keys())}"
                )

            vals = funcs[metric](self.trajectories, self.times)
            n_vals = vals.shape[1]
            time_offset = n_times - n_vals

            for tr_idx in range(n_trials):
                sub = self.subjects[tr_idx]
                cond = self.conditions[tr_idx]
                for t_idx in range(n_vals):
                    rows.append(
                        {
                            "subject": sub,
                            "condition": cond,
                            "trial": tr_idx,
                            "time": self.times[t_idx + time_offset],
                            "metric": metric,
                            "value": float(vals[tr_idx, t_idx]),
                        }
                    )
        return pd.DataFrame(rows)

    def save(self, path: str | Path):
        """Save the TrajectoryResult object to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryResult":
        """Load a TrajectoryResult object from disk."""
        res = joblib.load(path)
        if not isinstance(res, cls):
            raise TypeError(
                f"Loaded object is not a TrajectoryResult. Got {type(res)}."
            )
        return res


class EmbeddingQualityResult:
    """
    Unified Container for Embedding Quality Metrics.

    Provides tidy data views for rank-based dimensionality reduction
    quality criteria (Trustworthiness, Continuity, LCMC, MRRE).
    """

    def __init__(self, X: np.ndarray, Z: np.ndarray):
        """
        Initialize the EmbeddingQualityResult.

        Parameters
        ----------
        X : np.ndarray
            High-dimensional original data matrix (n_samples, n_features).
        Z : np.ndarray
            Low-dimensional embedded data matrix (n_samples, n_components).
        """
        self.X = X
        self.Z = Z
        self._Q = None

    @property
    def Q(self) -> np.ndarray:
        """The co-ranking matrix, computed lazily."""
        if self._Q is None:
            self._Q = compute_coranking_matrix(self.X, self.Z)
        return self._Q

    def get_trustworthiness(self, k_values: Sequence[int]) -> pd.DataFrame:
        """Compute trustworthiness for various neighborhood sizes."""
        rows = []
        for k in k_values:
            val = trustworthiness(self.Q, k)
            rows.append({"metric": "trustworthiness", "k": k, "value": val})
        return pd.DataFrame(rows)

    def get_continuity(self, k_values: Sequence[int]) -> pd.DataFrame:
        """Compute continuity for various neighborhood sizes."""
        rows = []
        for k in k_values:
            val = continuity(self.Q, k)
            rows.append({"metric": "continuity", "k": k, "value": val})
        return pd.DataFrame(rows)

    def get_lcmc(self, k_values: Sequence[int]) -> pd.DataFrame:
        """Compute LCMC for various neighborhood sizes."""
        rows = []
        for k in k_values:
            val = lcmc(self.Q, k)
            rows.append({"metric": "lcmc", "k": k, "value": val})
        return pd.DataFrame(rows)

    def get_mrre(self, k_values: Sequence[int]) -> pd.DataFrame:
        """Compute MRRE (intrusion and extrusion) for various neighborhood sizes."""
        rows = []
        for k in k_values:
            m_int, m_ext = compute_mrre(self.Q, k)
            rows.append({"metric": "mrre_intrusion", "k": k, "value": m_int})
            rows.append({"metric": "mrre_extrusion", "k": k, "value": m_ext})
        return pd.DataFrame(rows)

    def summary(self, k_values: Sequence[int]) -> pd.DataFrame:
        """Compute all quality metrics across all provided k values."""
        dfs = [
            self.get_trustworthiness(k_values),
            self.get_continuity(k_values),
            self.get_lcmc(k_values),
            self.get_mrre(k_values),
        ]
        return pd.concat(dfs, ignore_index=True)

    def get_shepard_diagram_data(
        self, sample_size: int = 1000, random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (d_orig, d_emb) sampled pairwise distances."""
        return shepard_diagram_data(
            self.X, self.Z, sample_size=sample_size, random_state=random_state
        )

    def save(self, path: str | Path):
        """Save the EmbeddingQualityResult object to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingQualityResult":
        """Load an EmbeddingQualityResult object from disk."""
        res = joblib.load(path)
        if not isinstance(res, cls):
            raise TypeError(
                f"Loaded object is not an EmbeddingQualityResult. Got {type(res)}."
            )
        return res


class VelocityResult:
    """
    Unified Container for Velocity Dynamics Results.
    """

    def __init__(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        times: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """
        Initialize the VelocityResult.

        Parameters
        ----------
        X : np.ndarray
            High-dimensional original data matrix (n_samples, n_features).
        Z : np.ndarray
            Low-dimensional embedded data matrix (n_samples, n_components).
        times : np.ndarray, optional
            Time coordinates for samples.
        groups : np.ndarray, optional
            Group labels for independent sequences.
        """
        self.X = X
        self.Z = Z
        self.times = times
        self.groups = groups

    def get_velocity_fields(
        self,
        delta_t: int = 1,
        n_neighbors: int = 30,
        sigma: float = 0.1,
    ) -> np.ndarray:
        """Compute and return the velocity vectors in the embedding space."""
        return compute_velocity_fields(
            X=self.X,
            X_emb=self.Z,
            delta_t=delta_t,
            n_neighbors=n_neighbors,
            sigma=sigma,
            groups=self.groups,
            times=self.times,
        )

    def save(self, path: str | Path):
        """Save the VelocityResult object to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "VelocityResult":
        """Load a VelocityResult object from disk."""
        res = joblib.load(path)
        if not isinstance(res, cls):
            raise TypeError(f"Loaded object is not a VelocityResult. Got {type(res)}.")
        return res
