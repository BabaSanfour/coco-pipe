"""Descriptor-table file IO: save, load, merge, and feature-column consistency.

This is the descriptor-specific table IO layer. It builds on the generic
:func:`coco_pipe.io._serialization.read_table` primitive but owns the
descriptor concerns: the ``_feature_columns.json`` sidecar contract, loading a
descriptor table into a :class:`~coco_pipe.io.structures.DataContainer` (flat or
sensor x feature), and the cross-shard **merge** stage.

Author: Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>
"""

from __future__ import annotations

import json
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
    "load_descriptor_table",
    "merge_descriptor_tables",
    "save_descriptor_table",
]

logger = logging.getLogger(__name__)


def save_descriptor_table(
    df: pd.DataFrame,
    base_path: Path | str,
    feature_columns: Sequence[str] | None = None,
    formats: Sequence[str] = ("parquet",),
) -> None:
    """Write a descriptor table, with an optional feature-column sidecar.

    :func:`load_descriptor_table` reads whichever single file it is pointed at,
    so by default only the canonical ``{base_path}.parquet`` is written.  Pass
    ``formats=("parquet", "csv")`` to additionally emit a human-readable
    ``{base_path}.csv`` (doubles the on-disk footprint).  A
    ``{base_path.name}_feature_columns.json`` sidecar is written when
    *feature_columns* is given.

    Parameters
    ----------
    df
        Table to write.
    base_path
        Output path without suffix, e.g. ``combined/sensor_subject_features``.
    feature_columns
        Optional ordered list of descriptor feature-column names written to
        a ``_feature_columns.json`` sidecar alongside the table.
    formats
        Table formats to write. Any subset of ``{"parquet", "csv"}``; defaults
        to parquet only.

    Raises
    ------
    ValueError
        If *formats* is empty or contains an unsupported format.
    """
    base_path = Path(base_path)
    requested = tuple(formats)
    unsupported = sorted(set(requested) - {"parquet", "csv"})
    if not requested or unsupported:
        raise ValueError(
            "formats must be a non-empty subset of {'parquet', 'csv'}; "
            f"got {list(requested)}."
        )
    base_path.parent.mkdir(parents=True, exist_ok=True)
    if "parquet" in requested:
        df.to_parquet(base_path.with_suffix(".parquet"), index=False)
    if "csv" in requested:
        df.to_csv(base_path.with_suffix(".csv"), index=False)
    if feature_columns is not None:
        (base_path.parent / f"{base_path.name}_feature_columns.json").write_text(
            json.dumps(list(feature_columns), indent=2),
            encoding="utf-8",
        )


def check_feature_column_consistency(
    shard_root: Path | str,
    json_name: str,
    accumulated: dict[str, list[str] | None],
    col_key: str,
) -> None:
    """Load a feature-column sidecar from *shard_root* and assert consistency.

    Intended for merging per-shard descriptor outputs: on the first call for
    a given *col_key* the loaded column list is stored in *accumulated*. On
    every subsequent call the loaded list is compared against the stored one
    and a :class:`ValueError` is raised on any mismatch, preventing a silent
    merge of shards produced with incompatible feature sets.

    Parameters
    ----------
    shard_root
        Directory containing the ``json_name`` feature-column sidecar.
    json_name
        Filename of the feature-column JSON sidecar within *shard_root*.
    accumulated
        Mapping of ``col_key -> feature column list``, mutated in place.
    col_key
        Key identifying which feature-column set this sidecar belongs to
        (e.g. ``"sensor_epoch"``).
    """
    shard_root = Path(shard_root)
    loaded: list[str] = json.loads((shard_root / json_name).read_text(encoding="utf-8"))
    if accumulated.get(col_key) is None:
        accumulated[col_key] = loaded
    elif loaded != accumulated[col_key]:
        raise ValueError(
            f"Feature column mismatch detected in shard {shard_root!r} "
            f"(key '{col_key}'): columns differ from the first shard.\n"
            "This usually means shards were produced with different configs. "
            "Clear the derivative root and re-run extraction with a single config."
        )


def merge_descriptor_tables(
    table_paths: Sequence[Path | str],
    feature_columns_paths: Sequence[Path | str] | None = None,
    *,
    out_base_path: Path | str | None = None,
    formats: Sequence[str] = ("parquet",),
) -> tuple[pd.DataFrame, list[str] | None]:
    """Merge per-shard tables of one *table kind* into a single table.

    The cross-shard **merge** stage. A "table kind" is one descriptor output
    table — e.g. ``sensor_epoch`` / ``sensor_subject`` / ``pooled_subject`` —
    written once per shard; this row-concatenates that kind across shards. (It
    is not about the band/param/complexity descriptor *family*.) Each shard is
    read, its feature-column sidecar is optionally checked against the first, the
    rows are concatenated, and the combined table (plus sidecar) is optionally
    written via :func:`save_descriptor_table`. Discovery, manifests, and
    dataset-level QC are deliberately left to the caller, which calls this once
    per table kind.

    Parameters
    ----------
    table_paths
        Per-shard table files (``.csv`` / ``.parquet``) for one table kind, in
        the desired row order.
    feature_columns_paths
        Optional per-shard feature-column JSON sidecars, aligned with
        *table_paths*. When given, cross-shard consistency is enforced via
        :func:`check_feature_column_consistency` and the agreed column list is
        used as the combined sidecar.
    out_base_path
        Optional output path without suffix. When set, the combined table is
        written there via :func:`save_descriptor_table`.
    formats
        Output formats forwarded to :func:`save_descriptor_table` (default
        parquet only).

    Returns
    -------
    tuple
        ``(combined_df, feature_columns)`` where ``feature_columns`` is the
        validated column list when *feature_columns_paths* was provided, else
        ``None``.

    Raises
    ------
    ValueError
        If *table_paths* is empty, the sidecar list is misaligned, or a shard's
        feature columns differ from the first shard.
    """
    table_paths = [Path(path) for path in table_paths]
    if not table_paths:
        raise ValueError("merge_descriptor_tables requires at least one table path.")

    sidecars: list[Path] | None = None
    if feature_columns_paths is not None:
        sidecars = [Path(path) for path in feature_columns_paths]
        if len(sidecars) != len(table_paths):
            raise ValueError(
                "feature_columns_paths must align with table_paths "
                f"({len(sidecars)} != {len(table_paths)})."
            )

    accumulated: dict[str, list[str] | None] = {"features": None}
    frames: list[pd.DataFrame] = []
    for index, table_path in enumerate(table_paths):
        # Preserve every column exactly so shards stay aligned on concat.
        frames.append(read_table(table_path, drop_all_empty=False))
        if sidecars is not None:
            sidecar = sidecars[index]
            check_feature_column_consistency(
                sidecar.parent, sidecar.name, accumulated, "features"
            )

    combined = pd.concat(frames, ignore_index=True)
    feature_columns = accumulated["features"]
    if out_base_path is not None:
        save_descriptor_table(
            combined,
            out_base_path,
            feature_columns=feature_columns,
            formats=formats,
        )
    return combined, feature_columns


def load_descriptor_table(
    table_path: Path | str,
    feature_columns_path: Path | str,
    known_families: tuple[str, ...] = ("band", "param", "complexity"),
    condition: str | None = None,
    target_col: str | None = None,
    subjects: Sequence[str] | None = None,
    subject_col: str = "subject",
    analysis_mode: str = "flat",
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
    feature_columns_path = Path(feature_columns_path)
    max_abs = None
    if descriptor_max_abs_value is not None:
        max_abs = float(descriptor_max_abs_value)
        if not np.isfinite(max_abs) or max_abs <= 0:
            raise ValueError("descriptor_max_abs_value must be positive.")

    df = read_table(
        table_path,
        drop_all_empty=not drop_degenerate_columns,
    )

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

    raw_cols = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cols, list) or not all(
        isinstance(column, str) for column in raw_cols
    ):
        raise ValueError(
            f"Expected {feature_columns_path} to contain a JSON list of columns."
        )

    missing_columns = [column for column in raw_cols if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Descriptor columns not found in {table_path}: {missing_columns}."
        )

    parsed = [
        parse_descriptor_feature_column(column, known_families) for column in raw_cols
    ]
    if descriptor_families:
        allowed = {str(value).strip() for value in descriptor_families}
        parsed = [item for item in parsed if item["family"] in allowed]
        if not parsed:
            raise RuntimeError(
                f"No features matched descriptor_families={list(descriptor_families)}."
            )

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
        from .qc import descriptor_subfamily

        excluded = {str(value).strip() for value in exclude_subfamilies}
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
    dropped_feature_columns = pd.DataFrame()
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

    if analysis_mode == "flat":
        coords["feature"] = np.asarray(feature_cols, dtype=object)
        return DataContainer(
            X=feature_df.to_numpy(dtype=float),
            dims=("obs", "feature"),
            coords=coords,
            y=y,
            ids=ids,
            meta=meta_base,
        )

    from .qc import descriptor_identity, descriptor_subfamily

    sensors = list(dict.fromkeys(item["sensor"] for item in parsed))
    features = list(dict.fromkeys(item["feature"] for item in parsed))
    sensor_index = {sensor: index for index, sensor in enumerate(sensors)}
    feature_index = {feature: index for index, feature in enumerate(features)}
    feature_family = {item["feature"]: item["family"] for item in parsed}
    feature_subfamily = {
        item["feature"]: descriptor_subfamily(item["family"], item["feature"])
        for item in parsed
    }
    feature_descriptor = {
        item["feature"]: descriptor_identity(item["feature"]) for item in parsed
    }

    X = np.full(
        (len(feature_df), len(sensors), len(features)),
        np.nan,
        dtype=float,
    )
    for item in parsed:
        X[
            :,
            sensor_index[item["sensor"]],
            feature_index[item["feature"]],
        ] = feature_df[item["column"]].to_numpy(dtype=float)

    coords["sensor"] = np.asarray(sensors, dtype=object)
    coords["feature"] = np.asarray(features, dtype=object)
    coords["feature_family"] = np.asarray(
        [feature_family[feature] for feature in features],
        dtype=object,
    )
    coords["feature_subfamily"] = np.asarray(
        [feature_subfamily[feature] for feature in features],
        dtype=object,
    )
    coords["feature_descriptor"] = np.asarray(
        [feature_descriptor[feature] for feature in features],
        dtype=object,
    )
    return DataContainer(
        X=X,
        dims=("obs", "sensor", "feature"),
        coords=coords,
        y=y,
        ids=ids,
        meta={
            **meta_base,
            "descriptor_families": list(
                dict.fromkeys(coords["feature_family"].tolist())
            ),
        },
    )
