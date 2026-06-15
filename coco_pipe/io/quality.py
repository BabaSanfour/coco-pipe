"""Data quality measurement and QC gating for loaded data containers.

Four QC levels are defined:

* **Level 1** — NaN / Inf / extreme-value row drops at load time (handled by
  :func:`~coco_pipe.io.descriptors.load_descriptor_table`; counts surfaced via
  container meta).
* **Level 2** — Epoch-level MAD outlier rejection (:func:`drop_epoch_outliers`).
* **Level 3** — Subject-level outlier rejection (:func:`drop_subject_outliers`).
* **Level 4** — Compose levels 2 + 3 and record every decision in a
  :class:`QCResult` (:func:`run_qc`).

The module also provides lower-level primitives used by the levels above:
:func:`compute_row_outlier_scores` (MAD z-scores per row),
:func:`compute_subject_outlier_burden` (per-subject mean epoch burden), and
:func:`row_quality_score` (simple NaN/Inf/zero count per row, used for
quality-weighted sampling).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .structures import DataContainer

from ._constants import (
    _STATUS_ORDER,
    GROUP_BY_COLUMN,
    QCFlagLevel,
    QualityInput,
    QualityStatus,
)


@dataclass
class EpochDropRecord:
    """Record of one dropped observation."""

    obs_index: int
    obs_id: str
    outlier_fraction: float
    mad_z_max: float


@dataclass
class SubjectDropRecord:
    """Record of one dropped subject."""

    subject_id: str
    outlier_fraction: float
    n_outlier_features: float


@dataclass
class QCResult:
    """Structured log of QC decisions produced by :func:`run_qc`.

    Fields are populated by ``run_qc`` and, for the Level-1 counts
    (``n_rows_entering_qc``, ``n_dropped_nan_inf``, ``n_dropped_extreme``),
    read from the container's ``meta`` dict written by
    :func:`~coco_pipe.io.descriptors.load_descriptor_table`.

    ``family_qc`` is **not** populated by ``run_qc`` — the caller computes it
    via :func:`~coco_pipe.descriptors.qc.aggregate_family_qc` and attaches it
    afterwards (``result.family_qc = aggregate_family_qc(...)``), keeping the
    ``io`` layer free of ``descriptors`` imports.
    """

    n_rows_entering_qc: int | None = None
    n_dropped_nan_inf: int = 0
    n_dropped_extreme: int = 0
    n_obs_in: int = 0
    n_obs_out: int = 0
    n_subjects_in: int = 0
    n_subjects_out: int = 0
    epoch_drop_threshold: float | None = None
    epoch_outlier_fraction_threshold: float | None = None
    epochs_dropped: list[EpochDropRecord] = field(default_factory=list)
    subject_drop_threshold: float | None = None
    subject_outlier_fraction_threshold: float | None = None
    subjects_dropped: list[SubjectDropRecord] = field(default_factory=list)
    per_family_dropped: dict[str, list[EpochDropRecord | SubjectDropRecord]] = field(
        default_factory=dict
    )
    subject_outlier_burden: pd.DataFrame | None = None
    feature_missingness: pd.DataFrame | None = None
    feature_columns_dropped: pd.DataFrame | None = None
    family_qc: pd.DataFrame | None = None
    thresholds: dict[str, Any] = field(default_factory=dict)

    @property
    def n_epochs_dropped(self) -> int:
        return len(self.epochs_dropped)

    @property
    def n_subjects_dropped(self) -> int:
        return len(self.subjects_dropped)

    @property
    def retention_rate(self) -> float:
        """Return the fraction of input observations retained."""
        if self.n_obs_in == 0:
            return float("nan")
        return self.n_obs_out / self.n_obs_in

    @property
    def total_dropped(self) -> int:
        """Return total rows dropped across all QC levels."""
        return (
            self.n_dropped_nan_inf
            + self.n_dropped_extreme
            + (self.n_obs_in - self.n_obs_out)
        )

    def summary(self) -> dict[str, Any]:
        """Return a flat summary suitable for logs and report headers."""
        summary = {
            "n_rows_entering_qc": self.n_rows_entering_qc,
            "n_dropped_nan_inf": self.n_dropped_nan_inf,
            "n_dropped_extreme": self.n_dropped_extreme,
            "n_obs_in": self.n_obs_in,
            "n_obs_out": self.n_obs_out,
            "n_epochs_dropped": self.n_epochs_dropped,
            "n_subjects_in": self.n_subjects_in,
            "n_subjects_out": self.n_subjects_out,
            "n_subjects_dropped": self.n_subjects_dropped,
            "retention_rate": round(self.retention_rate, 4),
            "epoch_drop_threshold": self.epoch_drop_threshold,
            "subject_drop_threshold": self.subject_drop_threshold,
            "subject_outlier_fraction_threshold": (
                self.subject_outlier_fraction_threshold
            ),
        }
        if self.feature_columns_dropped is not None:
            summary["n_feature_columns_dropped"] = len(self.feature_columns_dropped)
        return summary


@dataclass
class CheckResult:
    """
    Result of a data quality check.

    Attributes
    ----------
    check_name : str
        Name of the check (e.g., "Missing Values").
    status : str
        "OK", "WARN", or "FAIL".
    message : str
        Human-readable description of the issue.
    severity : int
        0 (Info) to 10 (Critical).
    metric_name : str, optional
        Name of the metric evaluated (e.g., "missing_pct").
    metric_value : float, int, or str, optional
        Value of the metric.

    Examples
    --------
    >>> res = CheckResult("Missingness", "FAIL", "Too many NaNs", 9)
    >>> res.is_issue
    True
    """

    check_name: str
    status: QualityStatus
    message: str
    severity: int
    metric_name: str | None = None
    metric_value: float | int | str | None = None

    @property
    def is_issue(self) -> bool:
        """Return True if status is WARN or FAIL."""
        return self.status in {"WARN", "FAIL"}

    @classmethod
    def from_flag_dict(cls, flag: dict[str, Any]) -> "CheckResult":
        """Construct a CheckResult from a :func:`make_qc_flag` record."""
        level_map = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}
        status = level_map.get(str(flag.get("level", "pass")), "OK")
        return cls(
            check_name=str(flag.get("code", "")),
            status=status,
            message=str(flag.get("message", "")),
            severity={"OK": 0, "WARN": 5, "FAIL": 9}[status],
            metric_name=str(flag.get("scope") or ""),
            metric_value=flag.get("value"),
        )


def make_qc_flag(
    level: QCFlagLevel,
    code: str,
    message: str,
    value: float | int | str | None = None,
    threshold: float | int | str | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    """Create a structured QC flag record."""
    if level not in _STATUS_ORDER:
        raise ValueError("level must be one of: 'pass', 'warn', or 'fail'.")
    return {
        "level": level,
        "code": code,
        "message": message,
        "value": value,
        "threshold": threshold,
        "scope": scope or "",
    }


def resolve_qc_status(flags: list[dict[str, Any]]) -> str:
    """Return the worst status level from a list of QC flag dicts."""
    levels = [str(flag.get("level", "pass")) for flag in flags]
    valid_levels = [level if level in _STATUS_ORDER else "pass" for level in levels]
    return max(
        valid_levels,
        key=lambda status: _STATUS_ORDER.get(status, 0),
        default="pass",
    )


def _require_feature_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Return requested feature columns or raise with the missing names."""
    missing = [column for column in feature_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}.")
    return df.loc[:, feature_cols]


def compute_feature_missingness(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute per-column missingness and non-finite rates.

    NaN values contribute only to the missingness metrics. Positive and
    negative infinity contribute only to the non-finite metrics.
    """
    features = _require_feature_columns(df, feature_cols)
    n_rows = len(features)
    records = []
    for column in feature_cols:
        values = features[column]
        missing_mask = values.isna()
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        nonfinite_mask = np.isinf(numeric)
        missing_count = int(missing_mask.sum())
        nonfinite_count = int(nonfinite_mask.sum())
        records.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_rate": missing_count / n_rows if n_rows else 0.0,
                "nonfinite_count": nonfinite_count,
                "nonfinite_rate": nonfinite_count / n_rows if n_rows else 0.0,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "column",
            "missing_count",
            "missing_rate",
            "nonfinite_count",
            "nonfinite_rate",
        ],
    )


def compute_constant_feature_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    tol: float = 1e-12,
) -> pd.DataFrame:
    """Compute per-column variance and constant-feature indicators.

    Standard deviations use the population definition (``ddof=0``). Entirely
    NaN columns are identified separately and are not marked constant.
    """
    if tol < 0:
        raise ValueError("tol must be non-negative.")
    features = _require_feature_columns(df, feature_cols)
    records = []
    for column in feature_cols:
        values = features[column]
        is_all_nan = bool(values.isna().all())
        numeric = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        std = float(numeric.std(ddof=0)) if numeric.notna().any() else np.nan
        records.append(
            {
                "column": column,
                "std": std,
                "is_all_nan": is_all_nan,
                "is_constant": bool(not is_all_nan and np.isfinite(std) and std <= tol),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["column", "std", "is_all_nan", "is_constant"],
    )


def _validate_group_by(value: str) -> None:
    if value not in GROUP_BY_COLUMN:
        raise ValueError("group_by must be one of 'family', 'measure', or 'feature'.")


def _resolve_group_labels(descriptor_names: list[str], group_by: str) -> np.ndarray:
    """Map descriptor names to their grouping label at the chosen granularity."""
    _validate_group_by(group_by)
    from coco_pipe.descriptors.qc import classify_descriptor_columns

    classification = classify_descriptor_columns(
        [str(name) for name in descriptor_names]
    )
    return (
        classification[GROUP_BY_COLUMN[group_by]]
        .fillna("unknown")
        .astype(str)
        .to_numpy()
    )


def compute_row_outlier_scores(
    df: pd.DataFrame,
    feature_cols: list[str],
    z_threshold: float = 5.0,
    descriptor_names: list[str] | None = None,
    group_by: str | None = None,
) -> pd.DataFrame:
    """Compute per-row outlier fractions using MAD-based robust z-scores.

    When ``group_by`` is set (``"family"``, ``"measure"``, or ``"feature"``) the
    result also carries per-group ``outlier_fraction_<label>`` columns so a row
    can be judged within each descriptor group rather than across all features.
    """
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive.")

    features = (
        _require_feature_columns(df, feature_cols)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    n_rows = len(features)
    n_features = len(feature_cols)
    if n_features == 0:
        return pd.DataFrame(
            {
                "outlier_fraction": np.zeros(n_rows, dtype=float),
                "n_outlier_features": np.zeros(n_rows, dtype=int),
                "mad_z_max": np.zeros(n_rows, dtype=float),
            },
            index=df.index,
        )

    values = features.to_numpy(dtype=float)
    medians = features.median(axis=0, skipna=True).to_numpy(dtype=float)
    deviations = np.abs(values - medians)
    mad = pd.DataFrame(deviations).median(axis=0, skipna=True).to_numpy(dtype=float)
    scaled_mad = 1.4826 * mad
    with np.errstate(divide="ignore", invalid="ignore"):
        robust_z = np.where(
            scaled_mad == 0,
            np.where(deviations == 0, 0.0, np.inf),
            deviations / scaled_mad,
        )

    finite_values = np.isfinite(values)
    outlier_flags = finite_values & (robust_z > z_threshold)
    n_outliers = outlier_flags.sum(axis=1).astype(int)
    finite_z = np.where(finite_values, robust_z, np.nan)
    mad_z_max = pd.DataFrame(finite_z).max(axis=1, skipna=True).to_numpy()
    result = pd.DataFrame(
        {
            "outlier_fraction": n_outliers.astype(float) / n_features,
            "n_outlier_features": n_outliers,
            "mad_z_max": mad_z_max,
        },
        index=df.index,
    )
    if group_by is None:
        return result

    names = descriptor_names or feature_cols
    if len(names) != len(feature_cols):
        raise ValueError("descriptor_names must align with feature_cols.")
    labels = _resolve_group_labels(names, group_by)
    for label in dict.fromkeys(labels.tolist()):
        label_mask = labels == label
        label_count = outlier_flags[:, label_mask].sum(axis=1).astype(int)
        result[f"outlier_fraction_{label}"] = label_count.astype(float) / int(
            label_mask.sum()
        )
        result[f"n_outlier_features_{label}"] = label_count
    return result


def compute_subject_outlier_burden(
    df: pd.DataFrame,
    feature_cols: list[str],
    subject_col: str = "subject",
    z_threshold: float = 5.0,
) -> pd.DataFrame:
    """Aggregate row-level MAD outlier scores to one row per subject."""
    if subject_col not in df.columns:
        raise ValueError(f"Subject column '{subject_col}' not found.")
    scores = compute_row_outlier_scores(
        df,
        feature_cols,
        z_threshold=z_threshold,
    )
    scores = scores.copy()
    scores[subject_col] = df[subject_col].to_numpy()
    return (
        scores.groupby(subject_col, sort=False)
        .agg(
            outlier_fraction=("outlier_fraction", "mean"),
            n_outlier_features=("n_outlier_features", "mean"),
            n_epochs=("outlier_fraction", "count"),
        )
        .reset_index()
    )


def _numeric_values(data: QualityInput) -> np.ndarray:
    """Return finite-compatible numeric values for quality checks."""
    if isinstance(data, pd.DataFrame):
        return data.select_dtypes(include=np.number).to_numpy(dtype=float, copy=False)

    # Handle ndarrays safely
    arr = np.asarray(data)
    if arr.dtype.kind in "iufc":
        return arr.astype(float, copy=False)

    # Attempt to convert object arrays to float (handles mixed numerics in objects)
    try:
        return arr.astype(float)
    except (ValueError, TypeError):
        return np.array([], dtype=float)


def check_missingness(
    df: QualityInput,
    threshold_warn: float = 0.01,
    threshold_fail: float = 0.20,
) -> CheckResult:
    """
    Check for missing values (NaNs).

    Parameters
    ----------
    df : DataFrame or ndarray
        The data to check.
    threshold_warn : float
        Ratio of NaNs to trigger a warning. Default 0.01 (1%).
    threshold_fail : float
        Ratio of NaNs to trigger a failure. Default 0.20 (20%).

    Returns
    -------
    CheckResult
        Quality check result.

    Examples
    --------
    >>> data = np.array([1, 2, np.nan, 4])
    >>> check_missingness(data, threshold_warn=0.1)
    CheckResult(check_name='Missingness', status='FAIL', ...)
    """
    if getattr(df, "size", 0) == 0:
        return CheckResult("Missingness", "WARN", "Dataset is empty.", 5)

    if isinstance(df, np.ndarray):
        missing_count = int(pd.isna(df).sum())
        total = df.size
    else:
        missing_count = int(df.isna().sum().sum())
        total = df.size

    pct = missing_count / total

    if pct > threshold_fail:
        return CheckResult(
            "Missingness",
            "FAIL",
            f"Critical missing data ({pct:.1%}).",
            9,
            "missing_ratio",
            pct,
        )
    if pct > threshold_warn:
        return CheckResult(
            "Missingness",
            "WARN",
            f"High missing data ({pct:.1%}).",
            4,
            "missing_ratio",
            pct,
        )

    return CheckResult(
        "Missingness", "OK", "Missing data within limits.", 0, "missing_ratio", pct
    )


def check_constant_columns(df: QualityInput) -> list[CheckResult]:
    """
    Check for columns/features with zero variance.

    Parameters
    ----------
    df : DataFrame or ndarray
        The data to check.

    Returns
    -------
    List[CheckResult]
        List of findings. Empty if no constant columns found.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    >>> check_constant_columns(df)
    [CheckResult(check_name='Constant Features', ...)]
    """
    if getattr(df, "size", 0) == 0:
        return []

    if isinstance(df, np.ndarray):
        if df.ndim != 2:
            return []
        arr = _numeric_values(df)
        if arr.size == 0 or arr.shape[0] == 0:
            return []
        stds = np.nanstd(arr, axis=0)
        constant_indices = np.where(stds == 0)[0]
        if len(constant_indices) > 0:
            return [
                CheckResult(
                    "Constant Features",
                    "WARN",
                    f"Found {len(constant_indices)} constant features (zero variance).",
                    3,
                )
            ]
        return []

    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        return []

    stds = df[numeric_cols].std()
    constant_cols = stds[stds == 0].index.tolist()
    if not constant_cols:
        return []

    msg = (
        f"Found {len(constant_cols)} constant columns: "
        f"{', '.join(str(c) for c in constant_cols[:3])}"
        f"{'...' if len(constant_cols) > 3 else ''}."
    )
    return [CheckResult("Constant Features", "WARN", msg, 3)]


def check_outliers_zscore(df: QualityInput, sigma: float = 5.0) -> CheckResult | None:
    """
    Check for extreme values (> sigma).
    Uses a simple global Z-score approach.

    Parameters
    ----------
    df : DataFrame or ndarray
        Data to check.
    sigma : float
        Z-score threshold. Default 5.0.

    Returns
    -------
    Optional[CheckResult]
        CheckResult if outliers found, else None.
    """
    vals = _numeric_values(df)
    if vals.size == 0:
        return None

    valid_vals = vals[np.isfinite(vals)]
    if valid_vals.size == 0:
        return None

    mean = float(np.mean(valid_vals))
    std = float(np.std(valid_vals))

    if std == 0:
        return None

    z_scores = np.abs((valid_vals - mean) / std)
    max_z = float(np.max(z_scores))

    if max_z > sigma:
        return CheckResult(
            "Outliers",
            "WARN",
            f"Extreme values detected (Max Z-Score: {max_z:.1f}).",
            4,
            "max_z",
            max_z,
        )

    return None


def check_flatline(signal: QualityInput, threshold: float = 1e-10) -> CheckResult:
    """
    Check if signal is effectively dead (flatline).

    Parameters
    ----------
    signal : ndarray
        1D signal array or flattened data.
    threshold : float
        Standard deviation threshold. Default 1e-10.

    Returns
    -------
    CheckResult
        Result indicating if signal is flat.
    """
    if getattr(signal, "size", None) == 0:
        return CheckResult("Signal Quality", "WARN", "Signal is empty.", 5)

    values = _numeric_values(signal)
    if values.size == 0:
        return CheckResult("Signal Quality", "OK", "Signal is not numeric.", 0)

    valid_vals = values[np.isfinite(values)]
    if valid_vals.size == 0:
        return CheckResult(
            "Signal Quality", "WARN", "Signal contains no finite values.", 5
        )

    std = float(np.std(valid_vals))
    if std < threshold:
        return CheckResult(
            "Signal Quality",
            "FAIL",
            "Signal is flatlined (Zero or near-zero variance).",
            8,
            "std_dev",
            std,
        )
    return CheckResult("Signal Quality", "OK", "Signal variance OK.", 0)


def row_quality_score(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    count_zero: bool = True,
    normalize: bool = False,
) -> pd.Series:
    """Calculate per-row badness from NaN, Inf, and optionally zero counts.

    Higher values indicate worse quality. With ``normalize=True``, divide by
    the number of evaluated numeric columns so scores are in ``[0, 1]``.

    Parameters
    ----------
    df:
        Input rows to score.
    exclude_cols:
        Columns to exclude before selecting numeric values.
    count_zero:
        Whether zero values contribute to the badness score.
    normalize:
        Whether to divide counts by the number of evaluated numeric columns.

    Returns
    -------
    pandas.Series
        Row-aligned badness scores. Lower values indicate better quality.
    """
    use_df = df.drop(columns=exclude_cols, errors="ignore") if exclude_cols else df
    num = use_df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        dtype = float if normalize else int
        return pd.Series(np.zeros(len(df), dtype=dtype), index=df.index)

    nan_cnt = num.isna().sum(axis=1)
    arr = num.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        inf_mask = np.isinf(arr)
    inf_cnt = inf_mask.sum(axis=1)
    zero_cnt = num.eq(0).sum(axis=1) if count_zero else 0
    score = (nan_cnt + inf_cnt + zero_cnt).astype(int)
    if normalize:
        return score.astype(float) / num.shape[1]
    return score


def drop_epoch_outliers(
    container: "DataContainer",
    z_threshold: float = 5.0,
    outlier_fraction_threshold: float = 0.30,
    subject_col: str = "subject",
    feature_cols: list[str] | None = None,
    descriptor_names: list[str] | None = None,
    group_by: str | None = None,
    min_obs: int | None = None,
) -> tuple["DataContainer" | dict[str, np.ndarray], QCResult]:
    """Drop observations with a high fraction of MAD-based feature outliers.

    ``group_by=None`` makes one global drop decision across all features. When
    set (``"family"``, ``"measure"``, or ``"feature"``) the decision is made per
    descriptor group at that granularity and the returned dict is keyed by group
    label.
    """
    _validate_container(container)
    _validate_fraction_threshold(outlier_fraction_threshold)
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive.")
    if group_by is not None:
        _validate_group_by(group_by)
    if min_obs is not None and min_obs < 1:
        raise ValueError("min_obs must be a positive integer or None.")

    n_obs_in = container.X.shape[0]
    n_subjects_in = _count_unique_subjects(container, subject_col)
    feature_df = _container_to_feature_df(container, feature_cols)
    scores = compute_row_outlier_scores(
        feature_df,
        feature_df.columns.tolist(),
        z_threshold=z_threshold,
        descriptor_names=descriptor_names,
        group_by=group_by,
    )
    if group_by is not None:
        family_masks: dict[str, np.ndarray] = {}
        per_family_dropped: dict[str, list[EpochDropRecord]] = {}
        ids = (
            np.asarray(container.ids)
            if container.ids is not None
            else np.arange(n_obs_in).astype(str)
        )
        for column in scores.columns:
            prefix = "outlier_fraction_"
            if not column.startswith(prefix):
                continue
            family = column.removeprefix(prefix)
            fractions = scores[column].to_numpy(dtype=float)
            keep = fractions <= outlier_fraction_threshold
            if min_obs is not None and int(keep.sum()) < min_obs:
                raise RuntimeError(
                    f"Only {int(keep.sum())} observation(s) remain for family "
                    f"'{family}' after MAD rejection (minimum required: {min_obs})."
                )
            family_masks[family] = keep
            per_family_dropped[family] = [
                EpochDropRecord(
                    obs_index=index,
                    obs_id=str(ids[index]),
                    outlier_fraction=float(fractions[index]),
                    mad_z_max=float(scores.iloc[index]["mad_z_max"]),
                )
                for index in np.flatnonzero(~keep)
            ]
        combined_keep = np.logical_and.reduce(list(family_masks.values()))
        combined_dropped = _deduplicate_epoch_records(per_family_dropped)
        return family_masks, QCResult(
            n_obs_in=n_obs_in,
            n_obs_out=int(combined_keep.sum()),
            n_subjects_in=n_subjects_in,
            n_subjects_out=_count_unique_subjects(
                _filter_observations(container, combined_keep), subject_col
            ),
            epoch_drop_threshold=z_threshold,
            epoch_outlier_fraction_threshold=outlier_fraction_threshold,
            epochs_dropped=combined_dropped,
            per_family_dropped=per_family_dropped,
            thresholds={
                "epoch_z_threshold": z_threshold,
                "epoch_outlier_fraction_threshold": outlier_fraction_threshold,
                "group_by": group_by,
            },
        )
    outlier_fractions = scores["outlier_fraction"].to_numpy()
    keep_mask = outlier_fractions <= outlier_fraction_threshold
    if min_obs is not None and int(keep_mask.sum()) < min_obs:
        raise RuntimeError(
            f"Only {int(keep_mask.sum())} observation(s) remain after MAD rejection "
            f"(minimum required: {min_obs})."
        )
    ids = (
        np.asarray(container.ids)
        if container.ids is not None
        else np.arange(n_obs_in).astype(str)
    )
    dropped = [
        EpochDropRecord(
            obs_index=index,
            obs_id=str(ids[index]),
            outlier_fraction=float(outlier_fractions[index]),
            mad_z_max=float(scores.iloc[index]["mad_z_max"]),
        )
        for index in np.flatnonzero(~keep_mask)
    ]
    clean = _filter_observations(container, keep_mask)
    return clean, QCResult(
        n_obs_in=n_obs_in,
        n_obs_out=clean.X.shape[0],
        n_subjects_in=n_subjects_in,
        n_subjects_out=_count_unique_subjects(clean, subject_col),
        epoch_drop_threshold=z_threshold,
        epoch_outlier_fraction_threshold=outlier_fraction_threshold,
        epochs_dropped=dropped,
        thresholds={
            "epoch_z_threshold": z_threshold,
            "epoch_outlier_fraction_threshold": outlier_fraction_threshold,
        },
    )


def drop_subject_outliers(
    container: "DataContainer",
    z_threshold: float = 5.0,
    outlier_fraction_threshold: float = 0.20,
    subject_col: str = "subject",
    feature_cols: list[str] | None = None,
    descriptor_names: list[str] | None = None,
    group_by: str | None = None,
) -> tuple["DataContainer" | dict[str, np.ndarray], QCResult]:
    """Drop subjects with a high cohort-level feature outlier burden.

    ``group_by=None`` makes one global decision across all features. When set
    (``"family"``, ``"measure"``, or ``"feature"``) the burden is computed per
    descriptor group at that granularity and the returned dict is keyed by group
    label.
    """
    _validate_container(container)
    _validate_fraction_threshold(outlier_fraction_threshold)
    if group_by is not None:
        _validate_group_by(group_by)
    n_obs_in = container.X.shape[0]
    subject_ids = np.asarray(_get_subject_ids(container, subject_col), dtype=object)
    n_subjects_in = len(set(subject_ids.tolist()))
    feature_df = _container_to_feature_df(container, feature_cols).copy()
    feature_df[subject_col] = subject_ids
    score_columns = [column for column in feature_df.columns if column != subject_col]
    if group_by is not None:
        scores = compute_row_outlier_scores(
            feature_df,
            score_columns,
            z_threshold=z_threshold,
            descriptor_names=descriptor_names,
            group_by=group_by,
        ).copy()
        scores[subject_col] = subject_ids
        family_names = [
            column.removeprefix("outlier_fraction_")
            for column in scores.columns
            if column.startswith("outlier_fraction_")
        ]
        burden_parts = []
        family_masks: dict[str, np.ndarray] = {}
        per_family_dropped: dict[str, list[SubjectDropRecord]] = {}
        for family in family_names:
            fraction_col = f"outlier_fraction_{family}"
            count_col = f"n_outlier_features_{family}"
            family_burden = (
                scores.groupby(subject_col, sort=False)
                .agg(
                    outlier_fraction=(fraction_col, "mean"),
                    n_outlier_features=(count_col, "mean"),
                    n_epochs=(fraction_col, "count"),
                )
                .reset_index()
            )
            family_burden["family"] = family
            burden_parts.append(family_burden)
            bad = family_burden["outlier_fraction"] > outlier_fraction_threshold
            bad_subjects = set(family_burden.loc[bad, subject_col].astype(str))
            family_masks[family] = np.asarray(
                [str(subject) not in bad_subjects for subject in subject_ids],
                dtype=bool,
            )
            per_family_dropped[family] = [
                SubjectDropRecord(
                    subject_id=str(row[subject_col]),
                    outlier_fraction=float(row["outlier_fraction"]),
                    n_outlier_features=float(row["n_outlier_features"]),
                )
                for _, row in family_burden.loc[bad].iterrows()
            ]
        combined_keep = np.logical_and.reduce(list(family_masks.values()))
        combined_dropped = _deduplicate_subject_records(per_family_dropped)
        burden = pd.concat(burden_parts, ignore_index=True)
        return family_masks, QCResult(
            n_obs_in=n_obs_in,
            n_obs_out=int(combined_keep.sum()),
            n_subjects_in=n_subjects_in,
            n_subjects_out=_count_unique_subjects(
                _filter_observations(container, combined_keep), subject_col
            ),
            subject_drop_threshold=z_threshold,
            subject_outlier_fraction_threshold=outlier_fraction_threshold,
            subjects_dropped=combined_dropped,
            per_family_dropped=per_family_dropped,
            subject_outlier_burden=burden,
            thresholds={
                "subject_z_threshold": z_threshold,
                "subject_outlier_fraction_threshold": outlier_fraction_threshold,
                "group_by": group_by,
            },
        )
    burden = compute_subject_outlier_burden(
        feature_df,
        score_columns,
        subject_col=subject_col,
        z_threshold=z_threshold,
    )
    bad_rows = burden["outlier_fraction"] > outlier_fraction_threshold
    bad_subjects = set(burden.loc[bad_rows, subject_col].astype(str))
    dropped = [
        SubjectDropRecord(
            subject_id=str(row[subject_col]),
            outlier_fraction=float(row["outlier_fraction"]),
            n_outlier_features=float(row["n_outlier_features"]),
        )
        for _, row in burden.loc[bad_rows].iterrows()
    ]
    keep_mask = np.asarray(
        [str(subject) not in bad_subjects for subject in subject_ids],
        dtype=bool,
    )
    clean = _filter_observations(container, keep_mask)
    return clean, QCResult(
        n_obs_in=n_obs_in,
        n_obs_out=clean.X.shape[0],
        n_subjects_in=n_subjects_in,
        n_subjects_out=_count_unique_subjects(clean, subject_col),
        subject_drop_threshold=z_threshold,
        subject_outlier_fraction_threshold=outlier_fraction_threshold,
        subjects_dropped=dropped,
        subject_outlier_burden=burden,
        thresholds={
            "subject_z_threshold": z_threshold,
            "subject_outlier_fraction_threshold": outlier_fraction_threshold,
        },
    )


def run_qc(
    container: "DataContainer",
    epoch_z_threshold: float | None = 5.0,
    epoch_outlier_fraction_threshold: float = 0.30,
    subject_z_threshold: float | None = 5.0,
    subject_outlier_fraction_threshold: float = 0.20,
    subject_col: str = "subject",
    feature_cols: list[str] | None = None,
    compute_missingness: bool = True,
) -> tuple["DataContainer", QCResult]:
    """Run epoch QC followed by subject QC and return a merged result."""
    _validate_container(container)
    n_obs_in = container.X.shape[0]
    n_subjects_in = _count_unique_subjects(container, subject_col)
    meta = container.meta or {}
    current = container
    epochs_dropped: list[EpochDropRecord] = []
    subjects_dropped: list[SubjectDropRecord] = []
    subject_burden = None

    if epoch_z_threshold is not None:
        current, epoch_result = drop_epoch_outliers(
            current,
            z_threshold=epoch_z_threshold,
            outlier_fraction_threshold=epoch_outlier_fraction_threshold,
            subject_col=subject_col,
            feature_cols=feature_cols,
        )
        epochs_dropped = epoch_result.epochs_dropped

    if subject_z_threshold is not None:
        current, subject_result = drop_subject_outliers(
            current,
            z_threshold=subject_z_threshold,
            outlier_fraction_threshold=subject_outlier_fraction_threshold,
            subject_col=subject_col,
            feature_cols=feature_cols,
        )
        subjects_dropped = subject_result.subjects_dropped
        subject_burden = subject_result.subject_outlier_burden

    missingness = None
    if compute_missingness:
        clean_features = _container_to_feature_df(current, feature_cols)
        missingness = compute_feature_missingness(
            clean_features,
            clean_features.columns.tolist(),
        )

    return current, QCResult(
        n_rows_entering_qc=meta.get("n_rows_entering_qc"),
        n_dropped_nan_inf=int(meta.get("n_dropped_nan_inf", 0)),
        n_dropped_extreme=int(meta.get("dropped_extreme_rows", 0)),
        n_obs_in=n_obs_in,
        n_obs_out=current.X.shape[0],
        n_subjects_in=n_subjects_in,
        n_subjects_out=_count_unique_subjects(current, subject_col),
        epoch_drop_threshold=epoch_z_threshold,
        epoch_outlier_fraction_threshold=(
            epoch_outlier_fraction_threshold if epoch_z_threshold is not None else None
        ),
        epochs_dropped=epochs_dropped,
        subject_drop_threshold=subject_z_threshold,
        subject_outlier_fraction_threshold=(
            subject_outlier_fraction_threshold
            if subject_z_threshold is not None
            else None
        ),
        subjects_dropped=subjects_dropped,
        subject_outlier_burden=subject_burden,
        feature_missingness=missingness,
        feature_columns_dropped=meta.get("dropped_feature_columns"),
        thresholds={
            "epoch_z_threshold": epoch_z_threshold,
            "epoch_outlier_fraction_threshold": (epoch_outlier_fraction_threshold),
            "subject_z_threshold": subject_z_threshold,
            "subject_outlier_fraction_threshold": (subject_outlier_fraction_threshold),
        },
    )


def _validate_container(container: "DataContainer") -> None:
    if container.X.ndim != 2 or tuple(container.dims) != ("obs", "feature"):
        raise ValueError(
            "Quality functions require a flat 2D DataContainer with dims "
            f"('obs', 'feature'); got shape {container.X.shape} and "
            f"dims {container.dims}."
        )


def _validate_fraction_threshold(value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError("outlier_fraction_threshold must be between 0 and 1.")


def _deduplicate_epoch_records(
    records: dict[str, list[EpochDropRecord]],
) -> list[EpochDropRecord]:
    by_index: dict[int, EpochDropRecord] = {}
    for family_records in records.values():
        for record in family_records:
            previous = by_index.get(record.obs_index)
            if previous is None or record.outlier_fraction > previous.outlier_fraction:
                by_index[record.obs_index] = record
    return [by_index[index] for index in sorted(by_index)]


def _deduplicate_subject_records(
    records: dict[str, list[SubjectDropRecord]],
) -> list[SubjectDropRecord]:
    by_subject: dict[str, SubjectDropRecord] = {}
    for family_records in records.values():
        for record in family_records:
            previous = by_subject.get(record.subject_id)
            if previous is None or record.outlier_fraction > previous.outlier_fraction:
                by_subject[record.subject_id] = record
    return list(by_subject.values())


def _count_unique_subjects(
    container: "DataContainer",
    subject_col: str,
) -> int:
    return len(set(_get_subject_ids(container, subject_col)))


def _get_subject_ids(
    container: "DataContainer",
    subject_col: str,
) -> list[str]:
    if subject_col in (container.coords or {}):
        subjects = np.asarray(container.coords[subject_col])
        if subjects.ndim != 1 or len(subjects) != container.X.shape[0]:
            raise ValueError(
                f"Coordinate '{subject_col}' must align with observations."
            )
        return [str(subject) for subject in subjects]
    if container.ids is not None:
        ids = np.asarray(container.ids)
        if ids.ndim != 1 or len(ids) != container.X.shape[0]:
            raise ValueError("container.ids must align with observations.")
        return [str(identifier) for identifier in ids]
    return [str(index) for index in range(container.X.shape[0])]


def _container_to_feature_df(
    container: "DataContainer",
    feature_cols: list[str] | None,
) -> pd.DataFrame:
    _validate_container(container)
    names = list(container.coords.get("feature", []))
    if not names:
        names = [f"f{index}" for index in range(container.X.shape[1])]
    if len(names) != container.X.shape[1]:
        raise ValueError("Feature coordinates must align with the feature axis.")
    frame = pd.DataFrame(container.X, columns=names)
    if feature_cols is None:
        return frame
    missing = [column for column in feature_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"feature_cols not found in container: {missing}")
    return frame.loc[:, feature_cols]


def _clean_filter_meta(meta: dict) -> dict:
    """Remove stale per-load keys that must not bleed into filtered containers.

    ``dropped_extreme_rows`` is written by ``load_descriptor_table`` and
    reflects the *original* load, not the filtered slice.  Keeping it would
    cause ``run_qc`` to double-count those rows.
    """
    meta.pop("dropped_extreme_rows", None)
    return meta


def _filter_observations(
    container: "DataContainer",
    keep_mask: np.ndarray,
) -> "DataContainer":
    from .structures import DataContainer

    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.shape != (container.X.shape[0],):
        raise ValueError("keep_mask must align with observations.")
    coords = {}
    for name, values in (container.coords or {}).items():
        array = np.asarray(values)
        if array.ndim == 1 and len(array) == container.X.shape[0]:
            coords[name] = array[keep_mask]
        else:
            coords[name] = values
    y = None if container.y is None else np.asarray(container.y)[keep_mask]
    ids = None if container.ids is None else np.asarray(container.ids)[keep_mask]
    return DataContainer(
        X=container.X[keep_mask],
        dims=container.dims,
        coords=coords,
        y=y,
        ids=ids,
        meta=_clean_filter_meta(copy.deepcopy(container.meta)),
    )
