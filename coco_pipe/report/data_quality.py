"""
Small data-quality checks used by report builders.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

QualityStatus = Literal["OK", "WARN", "FAIL"]
QualityInput = pd.DataFrame | np.ndarray


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
    metric_value : float, optional
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
    metric_value: float | None = None

    @property
    def is_issue(self) -> bool:
        """Return True if status is WARN or FAIL."""
        return self.status in {"WARN", "FAIL"}


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
