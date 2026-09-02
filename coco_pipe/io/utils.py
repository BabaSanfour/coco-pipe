"""Miscellaneous IO helpers — BIDS loading, stratified sampling, and table utilities.

This module is intentionally thin: heavy quality logic lives in
:mod:`coco_pipe.io.quality`; data-structure definitions live in
:mod:`coco_pipe.io.structures`.  Everything here is either a small utility
(``read_table``, ``normalize_subject_value``) or a sampling helper
(``make_strata``, ``sample_indices``) with no dependency on the QC pipeline.
"""

import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from coco_pipe.utils import import_optional_dependency

logger = logging.getLogger(__name__)
mne = None
BIDSPath = None
read_raw_bids = None


def _get_mne():
    """Lazily import MNE so base IO structures stay lightweight."""
    global mne
    if mne is None:
        mne = import_optional_dependency(
            lambda: importlib.import_module("mne"),
            feature="IO workflows",
            dependency="mne",
        )
    return mne


def _get_bids_path():
    """Lazily import mne-bids' ``BIDSPath``."""
    global BIDSPath
    if BIDSPath is None:
        BIDSPath = import_optional_dependency(
            lambda: importlib.import_module("mne_bids").BIDSPath,
            feature="BIDS IO workflows",
            dependency="mne-bids",
        )
    return BIDSPath


def _get_read_raw_bids():
    """Lazily import the MNE-BIDS raw reader."""
    global read_raw_bids
    if read_raw_bids is None:
        read_raw_bids = import_optional_dependency(
            lambda: importlib.import_module("mne_bids").read_raw_bids,
            feature="BIDS IO workflows",
            dependency="mne-bids",
        )
    return read_raw_bids


def make_strata(
    df: "pd.DataFrame",
    covariates: list[str],
    n_bins: int = 5,
    binning: str = "quantile",
) -> "pd.Series":
    """
    Create a single stratification label from multiple covariates.
    Numeric covariates are binned.
    """
    labels = []
    for cov in covariates:
        s = df[cov]
        if is_numeric_dtype(s) or is_datetime64_any_dtype(s):
            if binning == "uniform":
                b = pd.cut(s, bins=n_bins)
            else:
                try:
                    b = pd.qcut(s, q=n_bins, duplicates="drop")
                except Exception:
                    b = pd.cut(s, bins=n_bins)
            labels.append(b.astype(str).fillna("NA"))
        else:
            labels.append(s.astype(str).fillna("NA"))

    if len(labels) == 1:
        return labels[0].astype("category")
    return (
        pd.concat(labels, axis=1).astype(str).agg("|".join, axis=1).astype("category")
    )


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


def sample_indices(
    df: "pd.DataFrame",
    target: str,
    size_map: dict[Any, int],
    rng,
    replace: bool,
    prefer_clean: bool,
    exclude: list[str],
) -> "pd.Index":
    """
    Sample indices for each class based on size_map.
    """
    indices = []
    for cls, n in size_map.items():
        sub = df[df[target] == cls]
        if n <= 0:
            continue

        if prefer_clean:
            q = row_quality_score(sub, exclude_cols=exclude)
            if not replace:
                sub_shuf = sub.sample(frac=1.0, random_state=rng.integers(0, 1 << 32))
                idx_top = q.loc[sub_shuf.index].sort_values(kind="mergesort").index[:n]
                indices.append(idx_top)
            else:
                w = (1.0 / (1.0 + q)).astype(float)
                sampled = sub.sample(
                    n=n, replace=True, weights=w, random_state=rng.integers(0, 1 << 32)
                )
                indices.append(sampled.index)
        else:
            if n <= len(sub) and not replace:
                sampled = sub.sample(
                    n=n, replace=False, random_state=rng.integers(0, 1 << 32)
                )
                indices.append(sampled.index)
            else:
                sampled = sub.sample(
                    n=n, replace=True, random_state=rng.integers(0, 1 << 32)
                )
                indices.append(sampled.index)

    if not indices:
        return pd.Index([])
    # Concat the indices, shuffle, and return the VALUES as the new Index
    combined = pd.concat([pd.Series(i) for i in indices]).sample(
        frac=1.0, random_state=rng
    )
    return pd.Index(combined.values)


def split_column(name: str, sep: str, reverse: bool) -> tuple[str, str]:
    """Split a column into (unit, feature) using `sep` and `reverse`."""
    if sep not in name:
        return "", name
    left, right = name.split(sep, 1)
    if reverse:
        return right, left
    return left, right


def read_bids_entry(
    bids_path: Any,
    is_pre_epoched: bool,
    is_evoked: bool,
    mode: str,
    window_length: float | None,
    stride: float | None,
    event_id: dict[str, int] | str | list[str] | None = None,
    tmin: float = -0.2,
    tmax: float = 0.5,
    baseline: tuple[float | None, float | None] | None = None,
    units: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], float, np.ndarray | None]:
    mne_mod = _get_mne()
    if is_pre_epoched:
        # Load existing Epochs
        fpath = bids_path.fpath
        if not fpath.exists():
            matches = bids_path.match()
            if matches:
                fpath = matches[0]

        epochs = mne_mod.read_epochs(fpath, verbose=False)
        if event_id is not None:
            filtered_epochs = None
            if isinstance(event_id, dict):
                event_names = list(event_id)
                event_codes = list(event_id.values())
            elif isinstance(event_id, str):
                event_names = [event_id]
                event_codes = None
            else:
                event_names = list(event_id)
                event_codes = None
            matching_epoch_names = [
                name for name in event_names if name in epochs.event_id
            ]
            if matching_epoch_names:
                filtered_epochs = epochs[matching_epoch_names]
            if event_codes is not None and (
                filtered_epochs is None or len(filtered_epochs) == 0
            ):
                matching_event_codes = np.isin(epochs.events[:, -1], event_codes)
                if np.any(matching_event_codes):
                    filtered_epochs = epochs[matching_event_codes]
            if filtered_epochs is None or len(filtered_epochs) == 0:
                raise ValueError(
                    "No epochs remain after filtering precomputed "
                    f"epochs with event_id={event_id}."
                )
            epochs = filtered_epochs
        data = epochs.get_data(copy=False, units=units)  # (N, C, T)
        return (
            data,
            epochs.times,
            epochs.ch_names,
            epochs.info["sfreq"],
            epochs.events[:, -1],
        )

    if is_evoked:
        # Load Evoked
        fpath = bids_path.fpath
        if not fpath.exists():
            matches = bids_path.match()
            if matches:
                fpath = matches[0]

        evokeds = mne_mod.read_evokeds(fpath, verbose=False)
        # Stack conditions (N_cond, C, T)
        data = np.stack([e.get_data(units=units) for e in evokeds], axis=0)
        labels = np.arange(len(evokeds))
        return (
            data,
            evokeds[0].times,
            evokeds[0].ch_names,
            evokeds[0].info["sfreq"],
            labels,
        )

    # Load Raw (default)
    raw = _get_read_raw_bids()(bids_path, verbose=False)
    raw.load_data()
    raw.pick_types(eeg=True, meg=True, eog=False)

    if mode == "continuous":
        data_raw = raw.get_data(units=units)  # (C, T)
        data = data_raw[np.newaxis, :, :]  # (1, C, T)
        times = raw.times
        labels = None
    elif event_id is not None:
        # Event-Based Epoching (Annotation aware)
        events, event_id_map = mne_mod.events_from_annotations(
            raw, event_id=event_id, verbose=False
        )
        epochs = mne_mod.Epochs(
            raw,
            events=events,
            event_id=event_id_map,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            verbose=False,
        )
        data = epochs.get_data(copy=False, units=units)
        times = epochs.times
        labels = epochs.events[:, -1]
    else:
        # Raw -> Fixed Length Epochs
        if window_length is None:
            data_raw = raw.get_data(units=units)
            data = data_raw[np.newaxis, :, :]
            times = raw.times
            labels = None
        else:
            dur_s = window_length
            stride_s = stride if stride else dur_s
            epochs = mne_mod.make_fixed_length_epochs(
                raw, duration=dur_s, overlap=dur_s - stride_s, verbose=False
            )
            data = epochs.get_data(copy=False, units=units)
            times = epochs.times
            labels = epochs.events[:, -1]

    return data, times, raw.ch_names, raw.info["sfreq"], labels


def load_participants_tsv(root: Path) -> dict[str, dict[str, Any]]:
    """
    Reads participants.tsv and returns dict: {sub_id: {col: val, ...}}.
    """
    tsv_path = root / "participants.tsv"
    if not tsv_path.exists():
        return {}

    try:
        df = pd.read_csv(tsv_path, sep="\t")
        # Normalize participant_id columns
        if "participant_id" in df.columns:
            # Standardize to just 'sub-XXX' or 'XXX'
            # We want keys to match 'sub' format used in _detect_subjects ('XXX')
            # But BIDS usually has 'sub-XXX'.

            # Map 'sub-XXX' -> 'XXX'
            df["participant_id"] = df["participant_id"].astype(str)
            # Create lookup
            lookup = {}
            for _, row in df.iterrows():
                pid = row["participant_id"].replace("sub-", "")
                # Store all other cols
                meta = row.drop("participant_id").to_dict()
                lookup[pid] = meta
            return lookup
        return {}
    except Exception as e:
        logger.warning(f"Failed to read participants.tsv: {e}")
        return {}


def detect_subjects(root: Path) -> list[str]:
    return [d.name.replace("sub-", "") for d in root.glob("sub-*") if d.is_dir()]


def detect_sessions(root: Path, subject: str) -> list[str]:
    sub_dir = root / f"sub-{subject}"
    if not sub_dir.exists():
        return []
    return sorted(
        [d.name.replace("ses-", "") for d in sub_dir.glob("ses-*") if d.is_dir()]
    )


def detect_runs(
    root: Path,
    subject: str,
    session: str | None = None,
    task: str | None = None,
    datatype: str = "eeg",
) -> list[str]:
    """
    Detect available runs for a given subject/session/task.
    """
    bp = _get_bids_path()(
        root=root,
        subject=subject,
        session=session,
        task=task,
        datatype=datatype,
        check=False,
    )
    matches = bp.match()
    runs = set()
    for m in matches:
        if m.run is not None:
            runs.add(m.run)
    return sorted(runs)


def normalize_subject_value(value: object) -> str:
    """Normalize a BIDS subject label to a zero-padded 4-digit string.

    The ``sub-`` prefix is stripped when present, while non-numeric labels are
    returned unchanged.

    Parameters
    ----------
    value : object
        Raw subject label from a metadata table or BIDS path component.

    Returns
    -------
    str
        Normalized subject string.
    """
    text = str(value).strip().replace("sub-", "")
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric):
        return f"{int(numeric):04d}"
    return text


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
