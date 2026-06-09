"""Descriptor table loading utilities."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .structures import DataContainer
from .utils import normalize_subject_value, read_table

logger = logging.getLogger(__name__)


def save_descriptor_table(
    df: pd.DataFrame,
    base_path: Path | str,
    feature_columns: Sequence[str] | None = None,
) -> None:
    """Write a descriptor table as parquet + csv, with an optional feature-column
    sidecar.

    This is the canonical on-disk layout consumed by :func:`load_descriptor_table`:
    ``{base_path}.parquet``, ``{base_path}.csv``, and (if *feature_columns* is
    given) ``{base_path.name}_feature_columns.json`` listing the descriptor
    feature columns.

    Parameters
    ----------
    df
        Table to write.
    base_path
        Output path without suffix, e.g. ``combined/sensor_subject_features``.
    feature_columns
        Optional ordered list of descriptor feature-column names written to
        a ``_feature_columns.json`` sidecar alongside the table.
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(base_path.with_suffix(".parquet"), index=False)
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


def parse_descriptor_feature_column(
    column: str,
    known_families: tuple[str, ...],
) -> dict[str, str]:
    """Strictly parse one descriptor column into its constituent parts."""
    scope_re = re.compile(r"(?P<body>.+)_(?P<scope>chgrp|ch)-(?P<sensor>.+)$")
    match = scope_re.match(str(column))
    if match is None:
        raise ValueError(
            f"Could not parse descriptor column '{column}'. "
            "Expected format: '{family}_{feature}_{chgrp|ch}-{sensor}'."
        )

    body = match.group("body")
    family = feature = None
    for family_name in known_families:
        if body.startswith(f"{family_name}_"):
            family = family_name
            feature = body[len(f"{family_name}_") :]
            break
        token = f"_{family_name}_"
        if token in body:
            prefix, remainder = body.split(token, 1)
            family = family_name
            feature = f"{prefix}_{remainder}"
            break

    if family is None or feature is None:
        raise ValueError(
            f"Column '{column}' does not contain a known family token. "
            f"Known: {known_families}."
        )

    return {
        "column": str(column),
        "family": family,
        "feature": feature,
        "scope": "sensor_group" if match.group("scope") == "chgrp" else "sensor",
        "sensor": match.group("sensor"),
    }


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
) -> DataContainer:
    """Load a descriptor feature table into a :class:`DataContainer`."""
    table_path = Path(table_path)
    feature_columns_path = Path(feature_columns_path)
    max_abs = None
    if descriptor_max_abs_value is not None:
        max_abs = float(descriptor_max_abs_value)
        if not np.isfinite(max_abs) or max_abs <= 0:
            raise ValueError("descriptor_max_abs_value must be positive.")

    df = read_table(table_path)

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
                "No features matched "
                f"descriptor_families={list(descriptor_families)}."
            )

    feature_cols = [item["column"] for item in parsed]
    feature_df = df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan)

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
                "Dropping %d row(s) with abs(feature) > %g from %s " "(condition=%r).",
                dropped_extreme,
                max_abs,
                table_path,
                condition,
            )
            df = df.loc[~extreme_mask].copy()
            feature_df = feature_df.loc[~extreme_mask].copy()
        if df.empty:
            raise RuntimeError(
                "No rows survived extreme-value filtering for "
                f"condition={condition!r}."
            )

    metadata_df = df.drop(columns=feature_cols)
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

    sensors = list(dict.fromkeys(item["sensor"] for item in parsed))
    features = list(dict.fromkeys(item["feature"] for item in parsed))
    sensor_index = {sensor: index for index, sensor in enumerate(sensors)}
    feature_index = {feature: index for index, feature in enumerate(features)}
    feature_family = {item["feature"]: item["family"] for item in parsed}

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
