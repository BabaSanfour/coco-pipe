"""Generic result-table helpers shared by decoding/dim-reduction reports.

Pure DataFrame utilities — status splitting, metric-ranked sorting, best-row
selection, display subsetting, and selector-column filtering — with no
study-specific column knowledge. Study reports layer their own column labels and
ordering on top of these.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from coco_pipe.decoding import resolve_primary_metric_name

from ._constants import (
    CV_SIGNATURE_COLUMNS,
    DEFAULT_DECODING_RESULT_COLUMN_LABELS,
    DEFAULT_DECODING_RESULT_COLUMN_ORDER,
    PRIMARY_TIE_BREAKERS,
)


def primary_metric_column(frame: pd.DataFrame) -> str | None:
    """Return the scalar ``<metric>_mean`` column that ranks results.

    Prefers ``balanced_accuracy_mean`` (the group-robust default), matching the
    metric contract stamped by :func:`coco_pipe.decoding.stamp_primary_metric`.
    Returns ``None`` when the frame carries no scalar metric column.
    """
    name = resolve_primary_metric_name(frame.columns)
    return f"{name}_mean" if name is not None else None


def ensure_primary_metric(frame: pd.DataFrame) -> pd.DataFrame:
    """Guarantee ``primary_metric``/``primary_metric_name`` columns exist.

    Records from the runner are already stamped by
    :func:`coco_pipe.decoding.stamp_primary_metric`; this is the frame-level
    fallback for externally-supplied tables (legacy CSVs, tests) so every
    leaderboard and comparison keys off one metric contract.
    """
    if frame.empty or "primary_metric" in frame.columns:
        return frame
    column = primary_metric_column(frame)
    if column is None:
        return frame
    frame = frame.copy()
    frame["primary_metric"] = pd.to_numeric(frame[column], errors="coerce")
    frame["primary_metric_name"] = column[: -len("_mean")]
    return frame


def make_cv_signature(frame: pd.DataFrame) -> pd.Series | None:
    """Return a readable ``cv_strategy | folds | seed`` Series, or None."""
    cv_columns = [
        column
        for column in ("cv_strategy", "effective_n_splits", "cv_random_state")
        if column in frame
    ]
    if cv_columns:
        return frame[cv_columns].astype(str).agg(" | ".join, axis=1)
    return None


def signature_compatibility(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    signature_columns: Sequence[str] = CV_SIGNATURE_COLUMNS,
) -> pd.DataFrame:
    """Per-group CV/cohort-signature compatibility for paired comparisons.

    Flags groups whose rows disagree on any signature field, so paired
    comparisons only combine rows sharing a grouped-CV and cohort design.
    """
    groups = [column for column in group_columns if column in frame]
    cv_cols = [column for column in signature_columns if column in frame]
    if not groups or not cv_cols:
        return pd.DataFrame()
    cv_counts = frame.groupby(groups, dropna=False)[cv_cols].nunique()
    mismatch_mask = cv_counts > 1
    compatibility = (~mismatch_mask.any(axis=1)).reset_index(name="paired_compatible")
    compatibility["mismatched_fields"] = mismatch_mask.apply(
        lambda row: ", ".join(mismatch_mask.columns[row]), axis=1
    ).values
    compatibility["n_rows"] = frame.groupby(groups, dropna=False).size().values
    return compatibility


def split_by_status(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into ``(successful_rows, failed_rows)``.

    Rows are successful when a ``status`` column equals ``"success"``; without a
    ``status`` column every row is treated as successful.
    """
    if frame.empty:
        return frame.copy(), frame.copy()
    if "status" not in frame:
        return frame.copy(), pd.DataFrame(columns=frame.columns)
    mask = frame["status"].astype(str) == "success"
    return frame[mask].copy(), frame[~mask].copy()


def sort_by_metrics(
    frame: pd.DataFrame,
    primary_metric: str = "primary_metric",
    tie_breakers: Sequence[tuple[str, bool]] = (),
) -> pd.DataFrame:
    """Sort descending by ``primary_metric``, then by ``tie_breakers``.

    ``tie_breakers`` is a sequence of ``(column_name, ascending)`` tuples.
    Strings/objects are coerced to numeric for safe sorting. When the primary
    metric column is absent the frame is returned unsorted rather than raising.
    """
    sortable = frame.copy()
    sort_columns: list[str] = []
    ascending: list[bool] = []

    if primary_metric in sortable:
        sortable[primary_metric] = pd.to_numeric(
            sortable[primary_metric], errors="coerce"
        )
        sort_columns.append(primary_metric)
        ascending.append(False)

    for col, is_ascending in tie_breakers:
        if col in sortable:
            temp_col = f"_sort_{col}"
            sortable[temp_col] = pd.to_numeric(sortable[col], errors="coerce")
            sort_columns.append(temp_col)
            ascending.append(is_ascending)

    if not sort_columns:
        return frame.copy()

    sorted_frame = sortable.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
        kind="stable",
    )
    temp_cols = [c for c in sort_columns if c.startswith("_sort_")]
    return sorted_frame.drop(columns=temp_cols, errors="ignore")


def best_rows(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    primary_metric: str | None = None,
    tie_breakers: Sequence[tuple[str, bool]] = (),
) -> tuple[pd.DataFrame, str | None]:
    """Return the best successful row per requested group.

    When *primary_metric* is ``None`` the ranking column is auto-detected via
    :func:`primary_metric_column` (falling back to ``"primary_metric"``); the
    resolved column name is returned alongside the rows.
    """
    if primary_metric is None:
        primary_metric = primary_metric_column(frame)
        if primary_metric is None:
            return pd.DataFrame(), None
    success = split_by_status(frame)[0]
    if success.empty:
        return pd.DataFrame(), primary_metric
    groups = [column for column in group_columns if column in success]
    ranked = sort_by_metrics(success, primary_metric, tie_breakers)
    if groups:
        ranked = ranked.groupby(groups, dropna=False, sort=False).head(1)
    return ranked.reset_index(drop=True), primary_metric


def selector_columns(frame: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    """Return interactive-table selector columns present in *frame*."""
    return [column for column in candidates if column in frame]


def relabel_columns(
    columns: Sequence[str],
    *,
    labels: Mapping[str, str] | None = None,
) -> list[str]:
    """Map raw column names to their display labels, leaving unmapped names as-is.

    Use this to translate selector/sort column names to match a frame that was
    already relabeled by :func:`display_frame`.
    """
    label_map = dict(DEFAULT_DECODING_RESULT_COLUMN_LABELS)
    if labels:
        label_map.update(labels)
    return [label_map.get(str(column), str(column)) for column in columns]


def display_frame(
    frame: pd.DataFrame,
    *,
    order: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    round_decimals: int | None = 4,
) -> pd.DataFrame:
    """Return a display-ready result frame: ordered, rounded, and relabeled.

    Selects the non-empty columns from *order* (defaulting to
    ``DEFAULT_DECODING_RESULT_COLUMN_ORDER``) in that order, rounds numeric
    columns to *round_decimals* (skipped when ``None``), and relabels the
    columns via ``DEFAULT_DECODING_RESULT_COLUMN_LABELS`` merged with *labels*.
    This is the single display path for decoding sweep-result tables.
    """
    label_map = dict(DEFAULT_DECODING_RESULT_COLUMN_LABELS)
    if labels:
        label_map.update(labels)
    column_order = (
        tuple(order) if order is not None else DEFAULT_DECODING_RESULT_COLUMN_ORDER
    )
    display_columns = [
        column
        for column in column_order
        if column in frame and frame[column].notna().any()
    ]
    display_columns = list(dict.fromkeys(display_columns))
    display = frame.loc[:, display_columns].copy()
    if round_decimals is not None:
        for column in display.select_dtypes(include="number").columns:
            display[column] = display[column].round(round_decimals)
    return display.rename(columns=label_map)


__all__ = [
    "CV_SIGNATURE_COLUMNS",
    "DEFAULT_DECODING_RESULT_COLUMN_LABELS",
    "DEFAULT_DECODING_RESULT_COLUMN_ORDER",
    "PRIMARY_TIE_BREAKERS",
    "best_rows",
    "display_frame",
    "ensure_primary_metric",
    "make_cv_signature",
    "primary_metric_column",
    "relabel_columns",
    "selector_columns",
    "signature_compatibility",
    "sort_by_metrics",
    "split_by_status",
]
