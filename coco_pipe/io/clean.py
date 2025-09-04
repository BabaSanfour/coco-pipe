#!/usr/bin/env python3
"""
coco_pipe/io/clean.py
---------------------
Utilities to remove invalid feature columns (NaN/Inf) with sensor-wide option.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

__all__ = ["clean_features"]


def _split_column(name: str, sep: str, reverse: bool) -> Tuple[str, str]:
    """Split a column into (unit, feature) using `sep` and `reverse`.

    If `sep` not present, returns ("", name) so it behaves as a standalone feature.
    """
    if sep not in name:
        return "", name
    left, right = name.split(sep, 1)
    if reverse:
        return right, left
    else:
        return left, right


def clean_features(
    X: pd.DataFrame,
    mode: str = "any",  # "any" or "sensor_wide"
    sep: str = "_",
    reverse: bool = False,
    verbose: bool = False,
    min_abs_value: float | None = None,
    min_abs_fraction: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Remove invalid feature columns containing NaN, ±Inf, and optionally very small values.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    mode : {"any", "sensor_wide"}
        - "any": drop any individual column that has NaN/Inf.
        - "sensor_wide": if any column for feature `f` from any sensor/unit has
          NaN/Inf, drop all columns whose feature part equals `f` across sensors.
          Column naming is assumed to follow either "<unit><sep><feature>" or
          "<feature><sep><unit>" depending on `reverse`.
    sep : str, default="_"
        Separator used between unit and feature.
    reverse : bool, default=False
        If True, interpret columns as "<feature><sep><unit>".
    verbose : bool, default=False
        If True, include more details in the returned report.
    min_abs_value : float or None, default=None
        If set (e.g., 1e-12), values with absolute magnitude < min_abs_value are
        treated as invalid ("too small"). Only applies to numeric columns.
    min_abs_fraction : float, default=0.0
        Fraction threshold for tiny values when `min_abs_value` is set. If 0.0,
        a column is dropped if it contains any tiny value. If in (0,1], a column
        is dropped if the fraction of tiny values is >= this threshold.

    Returns
    -------
    X_clean : pd.DataFrame
        Cleaned feature matrix with offending columns removed.
    report : dict
        A small report with keys:
          - "dropped_columns": list of column names removed
          - "dropped_features": list of feature names removed (sensor_wide only)
          - "mode": the mode used
          - "n_before", "n_after": number of columns before/after cleaning
    """
    if X.shape[1] == 0:
        return X.copy(), {"dropped_columns": [], "dropped_features": [], "mode": mode, "n_before": 0, "n_after": 0}

    # Identify columns with NaN/Inf (and optional tiny values)
    num = X.select_dtypes(include=[np.number])
    # For non-numeric columns, treat string/object as valid unless they are entirely NaN
    other = X.drop(columns=num.columns, errors="ignore")

    bad_cols: List[str] = []
    if not num.empty:
        arr = num.to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            inf_mask = np.isinf(arr)
        bad_mask = num.isna().to_numpy() | inf_mask
        bad_any = bad_mask.any(axis=0)
        bad_cols.extend(num.columns[bad_any].tolist())

        # Tiny values handling
        if min_abs_value is not None:
            with np.errstate(invalid='ignore'):
                tiny_mask = np.abs(arr) < float(min_abs_value)
            if min_abs_fraction <= 0.0:
                tiny_cols = num.columns[tiny_mask.any(axis=0)].tolist()
            else:
                frac = tiny_mask.mean(axis=0)
                tiny_cols = num.columns[(frac >= min_abs_fraction)].tolist()
            bad_cols.extend(tiny_cols)
    # Consider completely NaN object columns as invalid too
    if not other.empty:
        obj_bad = other.isna().all(axis=0)
        bad_cols.extend(other.columns[obj_bad].tolist())

    dropped_columns: List[str] = []
    dropped_features: List[str] = []
    if mode == "any":
        dropped_columns = sorted(set(bad_cols))
        X_clean = X.drop(columns=dropped_columns, errors="ignore")
    elif mode == "sensor_wide":
        # Map features -> all columns carrying that feature across units
        feature_to_cols: Dict[str, List[str]] = {}
        for col in X.columns:
            unit, feat = _split_column(col, sep=sep, reverse=reverse)
            feature_to_cols.setdefault(feat, []).append(col)

        # Identify features to drop: any feature appearing in a bad column
        bad_features = set()
        for col in bad_cols:
            _, feat = _split_column(col, sep=sep, reverse=reverse)
            bad_features.add(feat)

        # Drop all columns for these features
        for feat in sorted(bad_features):
            dropped_columns.extend(feature_to_cols.get(feat, []))
        dropped_columns = sorted(set(dropped_columns))
        dropped_features = sorted(bad_features)
        X_clean = X.drop(columns=dropped_columns, errors="ignore")
    else:
        raise ValueError("mode must be one of {'any','sensor_wide'}")

    report = {
        "mode": mode,
        "dropped_columns": dropped_columns,
        "dropped_features": dropped_features if mode == "sensor_wide" else [],
        "n_before": X.shape[1],
        "n_after": X_clean.shape[1],
        "tiny_params": {"min_abs_value": min_abs_value, "min_abs_fraction": min_abs_fraction},
    }
    if verbose:
        # Add counts for convenience
        report["n_dropped_columns"] = len(dropped_columns)
        report["n_dropped_features"] = len(dropped_features)

    return X_clean, report
