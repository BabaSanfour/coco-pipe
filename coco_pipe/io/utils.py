"""Miscellaneous IO helpers — stratified sampling and table utilities.

This module is intentionally thin: heavy quality logic lives in
:mod:`coco_pipe.io.quality`; data-structure definitions live in
:mod:`coco_pipe.io.structures`.  Everything here is either a small utility
(``load_participants_tsv``, ``normalize_subject_value``) or a sampling helper
(``make_strata``, ``sample_indices``) with no dependency on the QC pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

logger = logging.getLogger(__name__)


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


def _local_dataset(root: Path, uri: str, links: dict[str, str]) -> Path | None:
    """Directory a dataset URI points to, or ``None`` when it is not local.

    Handles the forms BIDS allows for a local dataset: a path relative to
    ``root``, a ``file:`` URI, and a BIDS URI ``bids:<name>:`` resolved through
    ``DatasetLinks``. Other schemes (``doi:``, ``https:``) are remote.
    """
    parsed = urlparse(uri)
    if parsed.scheme == "bids":
        name, _, relative = parsed.path.partition(":")
        target = links.get(name)
        if not name or not isinstance(target, str):
            return None
        linked = _local_dataset(root, target, {})
        return None if linked is None else linked / relative
    if parsed.scheme == "file":
        return Path(url2pathname(parsed.path))
    if len(parsed.scheme) > 1:
        return None
    # no scheme, or a one-letter "scheme" that is a Windows drive (C:\data)
    return root / uri


def load_participants_tsv(root: Path) -> dict[str, dict[str, Any]]:
    """Return the rows of ``participants.tsv``, by normalized subject label.

    A derivative dataset without one reads that of its first local source
    dataset that has one, from the ``URL`` of each ``SourceDatasets`` entry in
    ``dataset_description.json``: a relative path, a ``file:`` URI, or a BIDS URI
    ``bids:<name>:`` named in ``DatasetLinks``.
    """
    root = Path(root)
    path = root / "participants.tsv"
    description = root / "dataset_description.json"
    if not path.exists() and description.exists():
        info = json.loads(description.read_text(encoding="utf-8"))
        links = info.get("DatasetLinks") or {}
        for source in info.get("SourceDatasets", []):
            source_root = _local_dataset(root, str(source.get("URL", "")), links)
            if source_root is not None and (source_root / "participants.tsv").exists():
                path = source_root / "participants.tsv"
                break
    if not path.exists():
        return {}
    table = pd.read_csv(path, sep="\t")
    if "participant_id" not in table:
        return {}
    table.index = table.pop("participant_id").map(normalize_subject_value)
    return table.to_dict("index")


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

    Standard deviations use the population definition (``ddof=0``). A column is
    constant when its standard deviation is at most ``tol`` times its mean
    absolute value (``relative_std <= tol``), so the check does not depend on the
    unit: a power in V² varies by less than 1e-12 V² and is not constant. A
    column of zeros is constant. Entirely NaN columns are identified separately
    and are not marked constant.
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
        scale = float(numeric.abs().mean()) if numeric.notna().any() else np.nan
        relative_std = 0.0 if std == 0 else std / scale if scale > 0 else np.nan
        records.append(
            {
                "column": column,
                "std": std,
                "relative_std": relative_std,
                "is_all_nan": is_all_nan,
                "is_constant": bool(
                    not is_all_nan and np.isfinite(relative_std) and relative_std <= tol
                ),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["column", "std", "relative_std", "is_all_nan", "is_constant"],
    )
