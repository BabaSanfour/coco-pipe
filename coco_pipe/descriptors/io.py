"""Descriptor-table file IO: save, load, merge, and feature-column consistency.

This is the descriptor-specific table IO layer. It builds on the generic
:func:`coco_pipe.io._serialization.read_table` primitive but owns the
descriptor concerns: the feature-column list each table carries in its
parquet metadata, loading a
descriptor table into a flat :class:`~coco_pipe.io.structures.DataContainer`,
and the cross-shard **merge** stage.

Author: Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from coco_pipe.io._serialization import read_table
from coco_pipe.io.structures import DataContainer
from coco_pipe.io.utils import normalize_subject_value

from .naming import parse_descriptor_feature_column

__all__ = [
    "check_feature_column_consistency",
    "feature_columns_of",
    "load_descriptor_table",
    "merge_descriptor_tables",
    "save_descriptor_table",
]

logger = logging.getLogger(__name__)


FEATURE_COLUMNS_ATTR = "feature_columns"
"""Key of the feature-column list in a descriptor table's ``DataFrame.attrs``."""


def save_descriptor_table(
    df: pd.DataFrame,
    base_path: Path | str,
    feature_columns: Sequence[str] | None = None,
) -> None:
    """Write a descriptor table as ``{base_path}.parquet``.

    The ordered feature-column list is kept in the table itself: in
    ``df.attrs["feature_columns"]``, which pandas stores in the parquet file's
    metadata and restores on read, so a table is one self-describing file.

    Parameters
    ----------
    df
        Table to write.
    base_path
        Output path without suffix, e.g. ``combined/sensor_recording_features``.
    feature_columns
        Ordered list of descriptor feature-column names; defaults to the
        table's own ``attrs``.
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_columns is not None:
        df = df.copy()
        df.attrs[FEATURE_COLUMNS_ATTR] = list(feature_columns)
    df.to_parquet(base_path.with_suffix(".parquet"), index=False)


def feature_columns_of(df: pd.DataFrame, source: Path | str) -> list[str]:
    """The feature-column list a descriptor table carries in its ``attrs``."""
    columns = df.attrs.get(FEATURE_COLUMNS_ATTR)
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        raise ValueError(
            f"{source} carries no feature-column list; write it with "
            "save_descriptor_table(..., feature_columns=...)."
        )
    return columns


def check_feature_column_consistency(
    feature_columns: Sequence[str],
    accumulated: dict[str, list[str] | None],
    col_key: str,
    source: Path | str,
) -> None:
    """Assert that a table's feature columns match the first table of its kind.

    Intended for merging per-shard descriptor outputs: on the first call for
    a given *col_key* the column list is stored in *accumulated*. On every
    subsequent call it is compared against the stored one and a
    :class:`ValueError` is raised on any mismatch, preventing a silent merge of
    shards produced with incompatible feature sets.

    Parameters
    ----------
    feature_columns
        Feature columns of the table being checked.
    accumulated
        Mapping of ``col_key -> feature column list``, mutated in place.
    col_key
        Key identifying which table kind the columns belong to
        (e.g. ``"sensor_epoch"``).
    source
        The table's path, for the error message.
    """
    feature_columns = list(feature_columns)
    if accumulated.get(col_key) is None:
        accumulated[col_key] = feature_columns
    elif feature_columns != accumulated[col_key]:
        raise ValueError(
            f"Feature column mismatch detected in {str(source)!r} "
            f"(key '{col_key}'): columns differ from the first shard.\n"
            "This usually means shards were produced with different configs. "
            "Clear the derivative root and re-run extraction with a single config."
        )


def merge_descriptor_tables(
    table_paths: Sequence[Path | str],
    *,
    out_base_path: Path | str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge per-shard tables of one *table kind* into a single table.

    The cross-shard **merge** stage. A "table kind" is one descriptor output
    table, e.g. ``sensor_epoch``, written once per shard; this row-concatenates
    that kind across shards. Each shard's feature columns are checked against
    the first via :func:`check_feature_column_consistency`, the rows are
    concatenated, and the combined table is optionally written via
    :func:`save_descriptor_table`. Discovery, manifests, and dataset-level QC are
    left to the caller, which calls this once per table kind.

    Parameters
    ----------
    table_paths
        Per-shard parquet tables for one table kind, in the desired row order.
    out_base_path
        Optional output path without suffix. When set, the combined table is
        written there via :func:`save_descriptor_table`.

    Returns
    -------
    tuple
        ``(combined_df, feature_columns)``.

    Raises
    ------
    ValueError
        If *table_paths* is empty, a table carries no feature-column list, or a
        shard's feature columns differ from the first shard.
    """
    table_paths = [Path(path) for path in table_paths]
    if not table_paths:
        raise ValueError("merge_descriptor_tables requires at least one table path.")

    accumulated: dict[str, list[str] | None] = {"features": None}
    frames: list[pd.DataFrame] = []
    for table_path in table_paths:
        # Preserve every column exactly so shards stay aligned on concat.
        frame = read_table(table_path, drop_all_empty=False)
        check_feature_column_consistency(
            feature_columns_of(frame, table_path), accumulated, "features", table_path
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    feature_columns = accumulated["features"]
    if out_base_path is not None:
        save_descriptor_table(combined, out_base_path, feature_columns=feature_columns)
    return combined, feature_columns


def load_descriptor_table(
    table_path: Path | str,
    known_families: tuple[str, ...] = ("band", "param", "complexity"),
    condition: str | None = None,
    target_col: str | None = None,
    subjects: Sequence[str] | None = None,
    subject_col: str = "subject",
    descriptor_families: Sequence[str] | None = None,
    descriptor_max_abs_value: float | None = None,
    drop_degenerate_columns: bool = False,
    max_missing_rate: float = 0.20,
    drop_constant_columns: bool = True,
    constant_tol: float = 1e-12,
    max_row_drop_rate: float | None = None,
    location_statistic: str | None = None,
    exclude_subfamilies: Sequence[str] | None = None,
) -> DataContainer:
    """Load a descriptor feature table into a :class:`~coco_pipe.io.DataContainer`."""
    table_path = Path(table_path)
    max_abs = None
    if descriptor_max_abs_value is not None:
        max_abs = float(descriptor_max_abs_value)
        if not np.isfinite(max_abs) or max_abs <= 0:
            raise ValueError("descriptor_max_abs_value must be positive.")

    df = read_table(
        table_path,
        drop_all_empty=not drop_degenerate_columns,
    )
    raw_cols = feature_columns_of(df, table_path)

    if condition is not None:
        if "condition" not in df.columns:
            raise ValueError(f"Condition column not found in {table_path}.")
        df = df[df["condition"].astype(str) == str(condition)].copy()

    if subjects:
        if subject_col not in df.columns:
            raise ValueError(
                f"Subject filter column '{subject_col}' not found in {table_path}."
            )
        wanted = {normalize_subject_value(subject) for subject in subjects}
        normalized = df[subject_col].map(normalize_subject_value)
        df = df[normalized.isin(wanted)].copy()

    if df.empty:
        raise RuntimeError(f"No rows survived filtering for condition={condition!r}.")
    n_rows_entering_qc = len(df)

    missing_columns = [column for column in raw_cols if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Descriptor columns not found in {table_path}: {missing_columns}."
        )

    table_items = [
        parse_descriptor_feature_column(column, known_families) for column in raw_cols
    ]
    parsed = table_items
    if descriptor_families:
        allowed = list(
            dict.fromkeys(str(value).strip() for value in descriptor_families)
        )
        present = list(dict.fromkeys(item["family"] for item in parsed))
        unknown = [family for family in allowed if family not in present]
        if unknown:
            raise ValueError(
                f"descriptor_families {unknown} not found in {table_path}; its "
                f"families are {present}."
            )
        parsed = [item for item in parsed if item["family"] in allowed]

    if location_statistic is not None:
        if location_statistic not in {"mean", "median"}:
            raise ValueError("location_statistic must be 'mean', 'median', or None.")
        unwanted = "median" if location_statistic == "mean" else "mean"
        parsed = [
            item for item in parsed if str(item["column"]).split("_", 1)[0] != unwanted
        ]
        if not parsed:
            raise RuntimeError(
                f"No features matched location_statistic={location_statistic!r}."
            )

    if exclude_subfamilies:
        from .naming import descriptor_subfamily

        excluded = {str(value).strip() for value in exclude_subfamilies}
        # checked against the whole table, so one exclude list can serve loads
        # of different families
        present = list(
            dict.fromkeys(
                descriptor_subfamily(item["family"], item["feature"])
                for item in table_items
            )
        )
        unknown = sorted(excluded.difference(present))
        if unknown:
            raise ValueError(
                f"exclude_subfamilies {unknown} not found in {table_path}; its "
                f"sub-families are {present}."
            )
        parsed = [
            item
            for item in parsed
            if descriptor_subfamily(item["family"], item["feature"]) not in excluded
        ]
        if not parsed:
            raise RuntimeError(
                f"No features remained after excluding sub-families {sorted(excluded)}."
            )

    feature_cols = [item["column"] for item in parsed]
    selected_feature_cols = list(feature_cols)
    feature_df = df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan)
    if drop_degenerate_columns:
        from .qc import select_viable_feature_columns

        feature_cols, dropped_feature_columns = select_viable_feature_columns(
            feature_df,
            feature_cols,
            max_missing_rate=max_missing_rate,
            drop_constant=drop_constant_columns,
            constant_tol=constant_tol,
            max_row_drop_rate=max_row_drop_rate,
            known_families=known_families,
        )
        if not feature_cols:
            raise RuntimeError("No descriptor feature columns survived column pruning.")
        dropped_names = set(dropped_feature_columns["column"].astype(str))
        parsed = [item for item in parsed if item["column"] not in dropped_names]
        feature_df = feature_df.loc[:, feature_cols]
        logger.warning(
            "Dropped %d degenerate descriptor column(s) from %s (condition=%r).",
            len(dropped_feature_columns),
            table_path,
            condition,
        )

    valid_mask = ~feature_df.isna().any(axis=1)
    n_dropped_nan_inf = int((~valid_mask).sum())
    if not valid_mask.all():
        logger.warning(
            "Dropping %d row(s) with NaN/Inf features from %s (condition=%r).",
            n_dropped_nan_inf,
            table_path,
            condition,
        )
        df = df.loc[valid_mask].copy()
        feature_df = feature_df.loc[valid_mask].copy()
    if df.empty:
        raise RuntimeError(
            f"No rows survived NaN/Inf filtering for condition={condition!r}."
        )

    dropped_extreme = 0
    if max_abs is not None:
        extreme_mask = feature_df.abs().gt(max_abs).any(axis=1)
        if extreme_mask.any():
            dropped_extreme = int(extreme_mask.sum())
            logger.warning(
                "Dropping %d row(s) with abs(feature) > %g from %s (condition=%r).",
                dropped_extreme,
                max_abs,
                table_path,
                condition,
            )
            df = df.loc[~extreme_mask].copy()
            feature_df = feature_df.loc[~extreme_mask].copy()
        if df.empty:
            raise RuntimeError(
                f"No rows survived extreme-value filtering for condition={condition!r}."
            )

    metadata_df = df.drop(columns=selected_feature_cols)
    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in {table_path}.")
        y = df[target_col].astype(str).to_numpy()
    else:
        y = None

    id_candidates = list(
        dict.fromkeys(("obs_id", "recording_id", "subject", subject_col))
    )
    for id_col in id_candidates:
        if id_col in df.columns:
            ids = df[id_col].astype(str).to_numpy()
            break
    else:
        raise ValueError(
            f"Cannot infer obs IDs from {table_path}. Expected one of: "
            "'obs_id', 'recording_id', 'subject', or subject_col."
        )

    coords = {column: metadata_df[column].to_numpy() for column in metadata_df.columns}
    meta_base = {
        "source": str(table_path),
        "n_rows_entering_qc": n_rows_entering_qc,
        "n_dropped_nan_inf": n_dropped_nan_inf,
        "descriptor_max_abs_value": descriptor_max_abs_value,
        "dropped_extreme_rows": dropped_extreme,
    }
    if drop_degenerate_columns:
        meta_base["dropped_feature_columns"] = dropped_feature_columns

    from .naming import descriptor_identity, descriptor_subfamily

    by_column = {item["column"]: item for item in parsed}
    ordered = [by_column[column] for column in feature_cols]
    coords["feature"] = np.asarray(feature_cols, dtype=object)
    coords["feature_family"] = np.asarray(
        [item["family"] for item in ordered], dtype=object
    )
    coords["feature_scope"] = np.asarray(
        [item["scope"] for item in ordered], dtype=object
    )
    coords["feature_channel"] = np.asarray(
        [item["sensor"] for item in ordered], dtype=object
    )
    coords["feature_measure"] = np.asarray(
        [item["feature"] for item in ordered], dtype=object
    )
    coords["feature_subfamily"] = np.asarray(
        [descriptor_subfamily(item["family"], item["feature"]) for item in ordered],
        dtype=object,
    )
    coords["feature_descriptor"] = np.asarray(
        [descriptor_identity(item["feature"]) for item in ordered], dtype=object
    )
    return DataContainer(
        X=feature_df.to_numpy(dtype=float),
        dims=("obs", "feature"),
        coords=coords,
        y=y,
        ids=ids,
        meta=meta_base,
    )
