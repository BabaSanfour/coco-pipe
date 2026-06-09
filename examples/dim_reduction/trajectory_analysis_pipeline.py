"""
End-to-end trajectory analysis pipeline.

This script shows how to go from raw data files to publication-quality
trajectory figures and CSV exports in a single call.  It bundles:

* Data loading helpers for three common input formats (precomputed PCA
  score CSVs, long-format trajectory CSVs, wide-format channel tables
  with on-the-fly ROI PCA).
* A sliding-window metric aggregator.
* The full orchestration function :func:`run_trajectory_analysis`.

Adapt the CONFIG block near the bottom of the file to your dataset.

Usage
-----
    python trajectory_analysis_pipeline.py

Dependencies (beyond coco-pipe)
--------------------------------
    scikit-learn (for make_roi_pca_loader)
    statsmodels   (via coco_pipe.dim_reduction.evaluation.stats)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import sem

from coco_pipe.dim_reduction.evaluation.geometry import (
    trajectory_auc_speed,
    trajectory_curvature,
    trajectory_displacement,
    trajectory_distance_from_center,
    trajectory_path_length,
    trajectory_speed,
    trajectory_tortuosity,
    trajectory_turning_angle,
)
from coco_pipe.dim_reduction.evaluation.stats import (
    grouped_condition_stats,
    paired_condition_stats,
)
from coco_pipe.dim_reduction.preprocessing import (
    apply_pca_score_baseline,
    flip_pc_scores_for_consistency,
)

# ---------------------------------------------------------------------------
# Constants used by the wide-table loader and ROI PCA factory
# ---------------------------------------------------------------------------

_DEFAULT_META_EXCLUDE: frozenset[str] = frozenset(
    {
        "subject",
        "Subject",
        "SubjectID",
        "condition",
        "Condition",
        "channel",
        "Channel",
        "Channels",
        "Regions",
        "Electrode",
        "Age",
        "Gender",
        "Handedness",
        "Laterality",
        "Education",
        "Status",
    }
)

_DEFAULT_EXCLUDE_CHANNELS: frozenset[str] = frozenset(
    {"EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8", "Status"}
)


def _clean_name(x: object) -> str:
    return str(x).strip().replace(" ", "")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def detect_time_columns(
    df: pd.DataFrame,
    meta_exclude: Optional[set[str]] = None,
    min_numeric_ratio: float = 0.85,
) -> list[str]:
    """
    Auto-detect numeric time columns in a wide-format DataFrame.

    A column is considered a time column when its name can be parsed as a
    float and at least ``min_numeric_ratio`` of its values are numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format table containing one row per channel/region.
    meta_exclude : set[str], optional
        Column names to ignore regardless of content.
    min_numeric_ratio : float, default=0.85
        Minimum fraction of non-NaN numeric values required to keep a column.

    Returns
    -------
    list[str]
        Sorted list of column names identified as time columns.
    """
    exclude = _DEFAULT_META_EXCLUDE | (set(meta_exclude) if meta_exclude else set())
    time_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        try:
            float(col)
        except (TypeError, ValueError):
            continue
        if pd.to_numeric(df[col], errors="coerce").notna().mean() >= min_numeric_ratio:
            time_cols.append(col)

    time_cols = sorted(time_cols, key=float)
    if not time_cols:
        raise RuntimeError(
            "No numeric time columns were detected. Check meta_exclude "
            "or column naming."
        )
    return time_cols


def load_precomputed_pca_scores(
    folder: Path | str,
    subject: str,
    condition: str,
    roi: Optional[str] = None,
    file_template: str = "pca_scores_Subject_{subject}_Condition_{condition}.csv",
) -> tuple[np.ndarray, pd.DataFrame, Optional[pd.Series]]:
    """
    Load a single precomputed PCA score CSV file.

    The file is expected to have PC rows (e.g. ``PCA_1``, ``PCA_2``) and
    numeric time-point column headers.  An optional ``Explained_Variance``
    row is parsed if present.

    Parameters
    ----------
    folder : Path or str
        Directory that contains the PCA score files.
    subject, condition : str
        Identifiers used to format ``file_template``.
    roi : str, optional
        ROI identifier used when the template contains a ``{roi}`` placeholder.
    file_template : str
        File name template with ``{subject}``, ``{condition}``, ``{roi}``
        placeholders.

    Returns
    -------
    time : np.ndarray of shape (n_times,)
    scores : pd.DataFrame — PC rows × time columns
    explained : pd.Series or None — explained variance ratios, or None
    """
    folder = Path(folder)
    fname = file_template.format(subject=subject, condition=condition, roi=roi)
    path = folder / fname
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path, index_col=0)
    raw.index = raw.index.astype(str)

    pc_rows = [idx for idx in raw.index if idx.startswith("PCA_")]
    if not pc_rows:
        raise ValueError(
            f"No rows starting with 'PCA_' found in {path}. Check the file format."
        )

    time = pd.to_numeric(pd.Series(raw.columns), errors="coerce").to_numpy(dtype=float)
    scores = raw.loc[pc_rows].apply(pd.to_numeric, errors="coerce")

    explained: Optional[pd.Series] = None
    if "Explained_Variance" in raw.index:
        ev = pd.to_numeric(raw.loc["Explained_Variance"], errors="coerce")
        ev.index = [f"PCA_{i + 1}" for i in range(len(ev))]
        explained = ev

    return time, scores, explained


def load_trajectory_long(
    path: Path | str,
    subject_col: str = "subject",
    condition_col: str = "condition",
    time_col: str = "time_ms",
    x_col: str = "x",
    y_col: str = "y",
    time_min_ms: Optional[float] = None,
    time_max_ms: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load a long-format trajectory CSV.

    Parameters
    ----------
    path : Path or str
    subject_col, condition_col, time_col, x_col, y_col : str
        Column name overrides.
    time_min_ms, time_max_ms : float, optional
        Optional time crop boundaries applied after loading.

    Returns
    -------
    pd.DataFrame
        Loaded (and optionally cropped) trajectory table.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {subject_col, condition_col, time_col, x_col, y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trajectory file is missing required columns: {missing}")

    df = df.copy()
    df[subject_col] = df[subject_col].astype(str)
    df[condition_col] = df[condition_col].astype(str)

    t = df[time_col].to_numpy(dtype=float)
    mask = np.ones(len(t), dtype=bool)
    if time_min_ms is not None:
        mask &= t >= time_min_ms
    if time_max_ms is not None:
        mask &= t <= time_max_ms
    return df.loc[mask].copy()


def load_wide_table(
    path: Path | str,
    subject_col: str = "subject",
    condition_col: str = "condition",
    channel_col: str = "channel",
    exclude_channels: Optional[set[str]] = None,
    meta_exclude: Optional[set[str]] = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """
    Load a wide-format multivariate table and detect its time columns.

    Each row represents one channel/region for a given subject/condition.
    Columns are either metadata or numeric time points.

    Parameters
    ----------
    path : Path or str
    subject_col, condition_col, channel_col : str
    exclude_channels : set[str], optional
        Channel names to drop in addition to the built-in exclusion set.
    meta_exclude : set[str], optional
        Additional column names to exclude from time-column detection.

    Returns
    -------
    df : pd.DataFrame
    time_cols : list[str]
    time_vals : np.ndarray of shape (n_times,)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {subject_col, condition_col, channel_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Wide table is missing required columns: {missing}")

    df = df.copy()
    df[subject_col] = df[subject_col].astype(str)
    df[condition_col] = df[condition_col].astype(str)
    df[channel_col] = df[channel_col].astype(str).map(_clean_name)

    exclude = {
        _clean_name(ch)
        for ch in (_DEFAULT_EXCLUDE_CHANNELS | (exclude_channels or set()))
    }
    df = df[~df[channel_col].isin(exclude)].copy()

    extra_exclude = {subject_col, condition_col, channel_col}
    time_cols = detect_time_columns(
        df, meta_exclude=(meta_exclude or set()) | extra_exclude
    )
    time_vals = np.array([float(c) for c in time_cols], dtype=float)

    return df, time_cols, time_vals


def make_roi_pca_loader(
    wide_df: pd.DataFrame,
    time_cols: list[str],
    time_vals: np.ndarray,
    roi_name: str,
    roi_channels: list[str],
    n_components: int = 10,
    standardize: bool = True,
    flip_for_consistency: bool = True,
    flip_window_ms: tuple[float, float] = (-200.0, 800.0),
    subject_col: str = "subject",
    condition_col: str = "condition",
    channel_col: str = "channel",
) -> Callable:
    """
    Build a per-ROI callable that fits PCA on demand for one subject/condition.

    The returned callable has the signature
    ``loader(subject, condition) -> (time, scores_df, explained_series)``
    and can be passed directly to :func:`run_trajectory_analysis`.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Wide-format table as returned by :func:`load_wide_table`.
    time_cols : list[str]
        Time column names (detected by :func:`load_wide_table`).
    time_vals : np.ndarray
        Float time values corresponding to ``time_cols``.
    roi_name : str
        Human-readable ROI label used in error messages.
    roi_channels : list[str]
        Channel names belonging to the ROI.  Pass ``["__ALL__"]`` to use
        every available channel for each subject/condition.
    n_components : int, default=10
    standardize : bool, default=True
        Whether to z-score channels before PCA.
    flip_for_consistency : bool, default=True
        Whether to flip PC signs for visual consistency.
    flip_window_ms : tuple[float, float], default=(-200.0, 800.0)
    subject_col, condition_col, channel_col : str

    Returns
    -------
    Callable
        ``loader(subject, condition) -> (time, scores_df, explained_series)``
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    use_all_channels = any(str(ch).upper() == "__ALL__" for ch in roi_channels)
    roi_channels_clean = [_clean_name(ch) for ch in roi_channels]

    def _loader(
        subject: str, condition: str
    ) -> tuple[np.ndarray, pd.DataFrame, pd.Series]:
        sub = wide_df[
            (wide_df[subject_col] == str(subject))
            & (wide_df[condition_col] == str(condition))
        ].copy()

        if sub.empty:
            raise FileNotFoundError(
                f"No rows found for subject={subject}, condition={condition}"
            )

        if use_all_channels:
            sub_roi = sub.copy()
        else:
            sub_roi = sub[sub[channel_col].isin(roi_channels_clean)].copy()
            available = sorted(sub_roi[channel_col].unique().tolist())
            missing = [ch for ch in roi_channels_clean if ch not in available]
            if sub_roi.empty or len(available) < 2:
                raise ValueError(
                    f"{roi_name}: not enough channels for subject={subject}, "
                    f"condition={condition}. Available={available}; Missing={missing}"
                )

        sub_roi = sub_roi.groupby(channel_col, as_index=True)[time_cols].mean()
        channel_by_time = sub_roi.apply(pd.to_numeric, errors="coerce")

        X = channel_by_time.T.to_numpy(dtype=float)  # (n_times, n_channels)
        time = time_vals.copy()

        good_time = ~np.all(~np.isfinite(X), axis=1)
        X = X[good_time]
        time = time[good_time]

        good_channels = ~np.all(~np.isfinite(X), axis=0)
        X = X[:, good_channels]

        if X.shape[0] < 3 or X.shape[1] < 2:
            raise ValueError(
                f"{roi_name}: insufficient data shape after NaN filtering: {X.shape}"
            )

        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        nan_mask = ~np.isfinite(X)
        if nan_mask.any():
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        channel_sd = np.nanstd(X, axis=0)
        X = X[:, channel_sd > 0]

        if X.shape[0] < 3 or X.shape[1] < 2:
            raise ValueError(
                f"{roi_name}: insufficient variable channels after filtering: {X.shape}"
            )

        if standardize:
            X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

        k = min(n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=k)
        scores_arr = pca.fit_transform(X)

        pc_names = [f"PCA_{i + 1}" for i in range(k)]
        scores = pd.DataFrame(scores_arr.T, index=pc_names, columns=time)
        explained = pd.Series(
            pca.explained_variance_ratio_, index=pc_names, name="Explained_Variance"
        )

        if flip_for_consistency:
            scores = flip_pc_scores_for_consistency(scores, time, flip_window_ms)

        return time, scores, explained

    return _loader


# ---------------------------------------------------------------------------
# Sliding-window metric aggregator
# ---------------------------------------------------------------------------


def compute_window_metrics(
    time: np.ndarray,
    values: np.ndarray,
    metric_name: str,
    window_size_ms: float = 100.0,
    window_step_ms: float = 25.0,
) -> pd.DataFrame:
    """
    Compute sliding-window averages of a time-resolved metric series.

    Parameters
    ----------
    time : np.ndarray of shape (n_times,)
        Time stamps corresponding to ``values``.
    values : np.ndarray of shape (n_times,)
        Metric values at each time point.
    metric_name : str
        Name of the metric (stored in the ``"metric"`` column).
    window_size_ms : float, default=100.0
    window_step_ms : float, default=25.0

    Returns
    -------
    pd.DataFrame
        Columns: ``metric``, ``window_start_ms``, ``window_end_ms``, ``value``.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    if time.size == 0 or values.size == 0:
        return pd.DataFrame(
            columns=["metric", "window_start_ms", "window_end_ms", "value"]
        )

    start = float(np.nanmin(time))
    end = float(np.nanmax(time))
    rows = []
    current = start

    while current + window_size_ms <= end:
        mask = (time >= current) & (time < current + window_size_ms)
        rows.append(
            {
                "metric": metric_name,
                "window_start_ms": current,
                "window_end_ms": current + window_size_ms,
                "value": float(np.nanmean(values[mask]))
                if np.any(mask)
                else float("nan"),
            }
        )
        current += window_step_ms

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------


def _ensure_dirs(output_dir: Path, analysis_name: str) -> dict[str, Path]:
    base = output_dir / analysis_name
    paths = {
        "base": base,
        "csv": base / "csv",
        "figures": base / "figures",
        "trajectories": base / "figures" / "trajectories",
        "metrics": base / "figures" / "metrics",
        "qc": base / "qc",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _save_figure(fig, path: Path, save_pdf: bool, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scalar and continuous metric name lists
# ---------------------------------------------------------------------------

_SCALAR_METRICS = [
    "trajectory_length",
    "straight_distance",
    "tortuosity",
    "cohesion",
    "separation",
    "mean_speed",
    "mean_curvature",
    "max_speed",
    "max_curvature",
    "mean_angle",
    "auc_speed",
]

_CONTINUOUS_METRICS = [
    ("speed", "speed_time", "speed"),
    ("curvature", "curvature_time", "curvature"),
    ("angles", "angle_time", "angles"),
    ("distance_from_center", "distance_time", "distance_from_center"),
]


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_trajectory_analysis(
    loader: Callable,
    subjects: list[str],
    conditions: list[str],
    output_dir: Path | str,
    analysis_name: str = "analysis",
    pca_dims: tuple[str, str] = ("PCA_1", "PCA_2"),
    apply_baseline: bool = False,
    baseline_n_points: Optional[int] = None,
    baseline_min_ms: float = -200.0,
    baseline_max_ms: float = 0.0,
    time_min_ms: Optional[float] = None,
    time_max_ms: Optional[float] = None,
    window_size_ms: float = 100.0,
    window_step_ms: float = 25.0,
    condition_sets: Optional[dict[str, dict[str, list[str]]]] = None,
    condition_colors: Optional[dict[str, str]] = None,
    condition_cmaps: Optional[dict[str, str]] = None,
    save_pdf: bool = False,
    dpi: int = 300,
) -> dict:
    """
    End-to-end trajectory analysis: load → metrics → stats → plots → CSV.

    Parameters
    ----------
    loader : Callable
        ``loader(subject, condition) -> (time, scores_df, explained_series)``.
        Use :func:`make_roi_pca_loader` or :func:`load_precomputed_pca_scores`
        to build a compatible callable.
    subjects : list[str]
    conditions : list[str]
    output_dir : Path or str
        Root directory for all outputs.
    analysis_name : str, default="analysis"
        Sub-directory name under ``output_dir``.
    pca_dims : tuple[str, str], default=("PCA_1", "PCA_2")
        Row labels in ``scores_df`` to use as trajectory x/y coordinates.
    apply_baseline : bool, default=False
    baseline_n_points : int, optional
    baseline_min_ms, baseline_max_ms : float
    time_min_ms, time_max_ms : float, optional
    window_size_ms, window_step_ms : float
    condition_sets : dict, optional
    condition_colors : dict, optional
    condition_cmaps : dict, optional
    save_pdf : bool, default=False
    dpi : int, default=300

    Returns
    -------
    dict
        Keys: ``trajectory_df``, ``continuous_df``, ``scalar_df``,
        ``window_df``, ``explained_df``, ``missing_df``, ``stats_df``,
        ``grouped_stats_df``, ``paths``.
    """
    output_dir = Path(output_dir)
    paths = _ensure_dirs(output_dir, analysis_name)

    missing_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    continuous_rows: list[dict] = []
    scalar_rows: list[dict] = []
    window_rows: list[dict] = []
    explained_rows: list[dict] = []

    for subject in subjects:
        for condition in conditions:
            try:
                try:
                    time, scores, explained = loader(subject, condition)
                except TypeError:
                    time, scores, explained = loader(subject, condition, None)

                if apply_baseline:
                    scores = apply_pca_score_baseline(
                        time,
                        scores,
                        n_points=baseline_n_points,
                        baseline_min_ms=baseline_min_ms,
                        baseline_max_ms=baseline_max_ms,
                    )

                mask = np.ones(len(time), dtype=bool)
                if time_min_ms is not None:
                    mask &= time >= time_min_ms
                if time_max_ms is not None:
                    mask &= time <= time_max_ms
                time = time[mask]
                scores = scores.iloc[:, mask]

                dims = [d for d in pca_dims if d in scores.index]
                if len(dims) < 2:
                    raise ValueError(
                        f"Need at least 2 PCA dims, found {dims} in scores index "
                        f"{list(scores.index)}"
                    )

                coords = scores.loc[list(dims)].to_numpy(dtype=float).T

                speed = trajectory_speed(coords, time=time)
                angles = trajectory_turning_angle(coords)
                curvature = trajectory_curvature(coords)
                dist_from_center = trajectory_distance_from_center(coords)

                m = {
                    "coords": coords,
                    "time": time,
                    "speed_time": time[:-1],
                    "speed": speed,
                    "curvature_time": time[1:-1],
                    "curvature": curvature,
                    "angle_time": time[1:-1],
                    "angles": angles,
                    "distance_time": time,
                    "distance_from_center": dist_from_center,
                    "trajectory_length": float(trajectory_path_length(coords)),
                    "straight_distance": float(
                        trajectory_displacement(coords, final=True)
                    ),
                    "tortuosity": float(trajectory_tortuosity(coords)),
                    "cohesion": float(np.nanmean(dist_from_center)),
                    "separation": float(np.nanstd(dist_from_center)),
                    "mean_speed": float(np.nanmean(speed)),
                    "max_speed": float(np.nanmax(speed)),
                    "mean_curvature": float(np.nanmean(curvature)),
                    "max_curvature": float(np.nanmax(curvature)),
                    "mean_angle": float(np.nanmean(angles)),
                    "auc_speed": float(trajectory_auc_speed(coords, time=time)),
                }

            except FileNotFoundError as exc:
                missing_rows.append(
                    {
                        "analysis": analysis_name,
                        "subject": subject,
                        "condition": condition,
                        "reason": "missing_file",
                        "detail": str(exc),
                    }
                )
                continue
            except Exception as exc:
                missing_rows.append(
                    {
                        "analysis": analysis_name,
                        "subject": subject,
                        "condition": condition,
                        "reason": "failed",
                        "detail": repr(exc),
                    }
                )
                continue

            coords_out = m["coords"]
            for i, t in enumerate(m["time"]):
                trajectory_rows.append(
                    {
                        "analysis": analysis_name,
                        "subject": subject,
                        "condition": condition,
                        "time_ms": t,
                        "PC1": coords_out[i, 0],
                        "PC2": coords_out[i, 1],
                    }
                )

            for metric_name, time_key, value_key in _CONTINUOUS_METRICS:
                mt = m[time_key]
                mv = m[value_key]
                for t, v in zip(mt, mv):
                    continuous_rows.append(
                        {
                            "analysis": analysis_name,
                            "subject": subject,
                            "condition": condition,
                            "metric": metric_name,
                            "time_ms": float(t),
                            "value": float(v),
                        }
                    )
                wdf = compute_window_metrics(
                    mt, mv, metric_name, window_size_ms, window_step_ms
                )
                if not wdf.empty:
                    wdf.insert(0, "analysis", analysis_name)
                    wdf.insert(1, "subject", subject)
                    wdf.insert(2, "condition", condition)
                    window_rows.extend(wdf.to_dict("records"))

            for metric_name in _SCALAR_METRICS:
                scalar_rows.append(
                    {
                        "analysis": analysis_name,
                        "subject": subject,
                        "condition": condition,
                        "metric": metric_name,
                        "value": m[metric_name],
                    }
                )

            if explained is not None:
                for pc, value in explained.items():
                    explained_rows.append(
                        {
                            "analysis": analysis_name,
                            "subject": subject,
                            "condition": condition,
                            "pc": pc,
                            "value": float(value),
                        }
                    )

    trajectory_df = pd.DataFrame(trajectory_rows)
    continuous_df = pd.DataFrame(continuous_rows)
    scalar_df = pd.DataFrame(scalar_rows)
    window_df = pd.DataFrame(window_rows)
    explained_df = pd.DataFrame(explained_rows)
    missing_df = pd.DataFrame(missing_rows)

    trajectory_df.to_csv(paths["csv"] / "trajectory_coordinates_long.csv", index=False)
    continuous_df.to_csv(paths["csv"] / "continuous_metrics_long.csv", index=False)
    scalar_df.to_csv(paths["csv"] / "scalar_metrics_long.csv", index=False)
    window_df.to_csv(paths["csv"] / "window_metrics_long.csv", index=False)
    explained_df.to_csv(paths["csv"] / "explained_variance_long.csv", index=False)
    missing_df.to_csv(paths["qc"] / "missing_or_failed.csv", index=False)

    stats_df = pd.DataFrame()
    grouped_stats_df = pd.DataFrame()

    if not scalar_df.empty:
        scalar_summary = scalar_df.groupby(["metric", "condition"], as_index=False).agg(
            n=("value", "count"),
            mean=("value", "mean"),
            sd=("value", "std"),
            sem=("value", sem),
        )
        scalar_summary.to_csv(
            paths["csv"] / "scalar_metrics_summary_by_condition.csv", index=False
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stats_df = paired_condition_stats(scalar_df, conditions=conditions)
            stats_df.to_csv(
                paths["csv"] / "scalar_metrics_pairwise_stats.csv", index=False
            )

            if condition_sets:
                grouped_stats_df = grouped_condition_stats(scalar_df, condition_sets)
                grouped_stats_df.to_csv(
                    paths["csv"] / "scalar_metrics_grouped_stats.csv", index=False
                )

    if not trajectory_df.empty or not continuous_df.empty or not scalar_df.empty:
        from coco_pipe.viz.base import plot_distribution_groups
        from coco_pipe.viz.dim_reduction import (
            plot_eigenvalues,
            plot_trajectory,
            plot_trajectory_metric_series,
        )

        speed_df = (
            continuous_df[continuous_df["metric"] == "speed"].copy()
            if not continuous_df.empty
            else pd.DataFrame()
        )

        if not trajectory_df.empty:
            ordered_conditions = conditions
            traj_list, speed_list, traj_labels = [], [], []
            for condition in ordered_conditions:
                cond = trajectory_df[trajectory_df["condition"] == condition]
                if cond.empty:
                    continue
                mean_xy = cond.groupby("time_ms")[["PC1", "PC2"]].mean().sort_index()
                traj_list.append(mean_xy.to_numpy())
                traj_labels.append(condition)
                if not speed_df.empty:
                    spd = speed_df[speed_df["condition"] == condition]
                    mean_spd = spd.groupby("time_ms")["value"].mean().sort_index()
                    speed_list.append(mean_spd.to_numpy())

            if traj_list:
                X = np.stack(traj_list)
                traj_values = (
                    np.stack(speed_list) if len(speed_list) == len(traj_list) else None
                )
                fig, ax = plot_trajectory(
                    X,
                    values=traj_values,
                    labels=traj_labels,
                    add_start_end_markers=True,
                    title="Trajectory PC1 vs PC2",
                )
                _save_figure(
                    fig, paths["trajectories"] / "trajectory_PC1_PC2", save_pdf, dpi
                )

        if not continuous_df.empty:
            for metric_name in continuous_df["metric"].unique():
                dfm = continuous_df[continuous_df["metric"] == metric_name]
                times_sorted = np.array(sorted(dfm["time_ms"].unique()), dtype=float)
                rows, row_labels = [], []
                for condition in conditions:
                    for _, sub in dfm[dfm["condition"] == condition].groupby("subject"):
                        sub_vals = sub.sort_values("time_ms")["value"].to_numpy()
                        if len(sub_vals) == len(times_sorted):
                            rows.append(sub_vals)
                            row_labels.append(condition)
                if not rows:
                    continue
                fig, _ = plot_trajectory_metric_series(
                    np.stack(rows),
                    times=times_sorted,
                    labels=np.array(row_labels),
                    title=metric_name.replace("_", " ").title(),
                    ylabel=metric_name.replace("_", " "),
                )
                _save_figure(
                    fig, paths["metrics"] / f"timecourse_{metric_name}", save_pdf, dpi
                )

        if not scalar_df.empty:
            for metric_name in scalar_df["metric"].unique():
                dfm = scalar_df[scalar_df["metric"] == metric_name]
                cond_values = {
                    condition: dfm[dfm["condition"] == condition]["value"].to_numpy()
                    for condition in conditions
                    if not dfm[dfm["condition"] == condition].empty
                }
                if not cond_values:
                    continue
                fig, _ = plot_distribution_groups(
                    list(cond_values.values()),
                    labels=list(cond_values.keys()),
                    title=metric_name.replace("_", " ").title(),
                    ylabel=metric_name.replace("_", " "),
                )
                _save_figure(
                    fig, paths["metrics"] / f"box_{metric_name}", save_pdf, dpi
                )

        if not explained_df.empty:
            ev_dict: dict[str, np.ndarray] = {}
            for cond, grp in explained_df.groupby("condition"):
                pivot = grp.pivot(index="subject", columns="pc", values="value")
                pc_order = pivot.columns[
                    pivot.columns.str.extract(r"(\d+)")
                    .astype(int)
                    .squeeze(axis=1)
                    .argsort()
                ]
                ev_dict[cond] = pivot[pc_order].to_numpy(dtype=float)
            fig, _ = plot_eigenvalues(
                ev_dict,
                title="Explained Variance by Condition",
                ylabel="Explained variance ratio",
                condition_colors=condition_colors,
            )
            _save_figure(fig, paths["metrics"] / "explained_variance", save_pdf, dpi)

    return {
        "trajectory_df": trajectory_df,
        "continuous_df": continuous_df,
        "scalar_df": scalar_df,
        "window_df": window_df,
        "explained_df": explained_df,
        "missing_df": missing_df,
        "stats_df": stats_df,
        "grouped_stats_df": grouped_stats_df,
        "paths": paths,
    }


# ---------------------------------------------------------------------------
# CONFIG — adapt this section to your dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- dataset config ----
    DATA_DIR = Path("path/to/pca_scores")
    OUTPUT_DIR = Path("outputs/trajectory_analysis")
    SUBJECTS = ["S01", "S02", "S03"]
    CONDITIONS = ["CondA", "CondB", "CondC"]
    CONDITION_COLORS = {"CondA": "#4C72B0", "CondB": "#DD8452", "CondC": "#55A868"}

    # ---- build a loader (choose one approach) ----

    # Option A: precomputed PCA score CSVs
    def loader(subject, condition):
        return load_precomputed_pca_scores(DATA_DIR, subject, condition)

    # Option B: on-the-fly ROI PCA from a wide-format table
    # wide_df, time_cols, time_vals = load_wide_table("path/to/wide_table.csv")
    # loader = make_roi_pca_loader(
    #     wide_df, time_cols, time_vals,
    #     roi_name="Frontal", roi_channels=["Fp1", "Fp2", "F3", "F4"],
    # )

    results = run_trajectory_analysis(
        loader=loader,
        subjects=SUBJECTS,
        conditions=CONDITIONS,
        output_dir=OUTPUT_DIR,
        analysis_name="my_analysis",
        apply_baseline=True,
        condition_colors=CONDITION_COLORS,
    )

    print("Missing / failed:", results["missing_df"].shape[0])
    print("Outputs written to:", results["paths"]["base"])
