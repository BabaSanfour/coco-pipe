"""Family-aware QC for descriptor outputs.

Provides column classification and family-level quality aggregation on top of
the generic per-column helpers in :mod:`coco_pipe.io.quality`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from coco_pipe.descriptors._constants import (
    _CLASSIFICATION_COLUMNS,
    _CONSTANT_COLUMNS,
    _FAILURE_FAMILY_ALIASES,
    _FAMILY_QC_COLUMNS,
    _MISSINGNESS_COLUMNS,
    KNOWN_FAMILY_TOKENS,
)
from coco_pipe.io.descriptors import parse_descriptor_feature_column
from coco_pipe.io.quality import (
    compute_constant_feature_summary,
    compute_feature_missingness,
)


@lru_cache(maxsize=32)
def _classify_cached(
    descriptor_names: tuple[str, ...],
    known_families: tuple[str, ...],
) -> pd.DataFrame:
    """Classify one hashable descriptor-name set."""
    rows: list[dict[str, Any]] = []
    for column in descriptor_names:
        try:
            parsed = parse_descriptor_feature_column(
                column,
                known_families,
            )
        except ValueError:
            parsed = None

        if parsed is not None:
            family = parsed["family"]
            scope = parsed["scope"]
            channel = parsed["sensor"]
            measure = parsed["feature"]
        else:
            family = next(
                (
                    family_name
                    for family_name in known_families
                    if column.startswith(f"{family_name}_")
                    or f"_{family_name}_" in column
                ),
                None,
            )
            scope = ""
            channel = ""
            if family is None:
                measure = column
            elif column.startswith(f"{family}_"):
                measure = column[len(family) + 1 :]
            else:
                prefix, remainder = column.split(f"_{family}_", 1)
                measure = f"{prefix}_{remainder}"

        rows.append(
            {
                "column": column,
                "family": family,
                "scope": scope,
                "channel": channel,
                "measure": measure,
            }
        )

    return pd.DataFrame.from_records(rows, columns=_CLASSIFICATION_COLUMNS)


def classify_descriptor_columns(
    descriptor_names: list[str],
    known_families: tuple[str, ...] = KNOWN_FAMILY_TOKENS,
) -> pd.DataFrame:
    """Classify descriptor names into family, measure, channel, and scope.

    The last ``_ch-`` or ``_chgrp-`` marker is interpreted as the scope, so
    earlier channel markers remain part of cross-channel measure names.
    Unknown family prefixes are retained with ``family=None``. Each call
    returns a fresh, independently mutable DataFrame so caller changes cannot
    corrupt the cached canonical result.
    """
    return _classify_cached(tuple(descriptor_names), tuple(known_families)).copy()


def compute_family_missingness(
    df: pd.DataFrame,
    descriptor_names: list[str],
    known_families: tuple[str, ...] = KNOWN_FAMILY_TOKENS,
) -> pd.DataFrame:
    """Enrich per-column missingness with descriptor family metadata."""
    if not descriptor_names:
        return pd.DataFrame(columns=_MISSINGNESS_COLUMNS)
    missingness = compute_feature_missingness(df, descriptor_names)
    classification = classify_descriptor_columns(descriptor_names, known_families)
    return missingness.merge(classification, on="column", how="left")


def compute_family_constant_summary(
    df: pd.DataFrame,
    descriptor_names: list[str],
    tol: float = 1e-12,
    known_families: tuple[str, ...] = KNOWN_FAMILY_TOKENS,
) -> pd.DataFrame:
    """Enrich per-column constant-feature results with family metadata."""
    if not descriptor_names:
        return pd.DataFrame(columns=_CONSTANT_COLUMNS)
    constants = compute_constant_feature_summary(df, descriptor_names, tol)
    classification = classify_descriptor_columns(descriptor_names, known_families)
    return constants.merge(classification, on="column", how="left")


def aggregate_family_qc(
    df: pd.DataFrame,
    descriptor_names: list[str],
    failures_df: pd.DataFrame | None = None,
    known_families: tuple[str, ...] = KNOWN_FAMILY_TOKENS,
    tol: float = 1e-12,
) -> pd.DataFrame:
    """Aggregate descriptor health indicators to one row per family."""
    if not descriptor_names:
        return pd.DataFrame(columns=_FAMILY_QC_COLUMNS)

    missingness = compute_family_missingness(
        df,
        descriptor_names,
        known_families,
    )
    constants = compute_family_constant_summary(
        df,
        descriptor_names,
        tol,
        known_families,
    )
    present_families = set(missingness["family"].dropna().astype(str))
    n_observations = len(df)
    failure_families = pd.Series(dtype=object)
    if (
        failures_df is not None
        and not failures_df.empty
        and "family" in failures_df.columns
    ):
        failure_families = (
            failures_df["family"].astype(str).replace(_FAILURE_FAMILY_ALIASES)
        )

    rows: list[dict[str, Any]] = []
    for family in sorted(
        family_name for family_name in known_families if family_name in present_families
    ):
        family_missingness = missingness[missingness["family"] == family]
        family_constants = constants[constants["family"] == family]
        failure_count = int((failure_families == family).sum())

        rows.append(
            {
                "family": family,
                "n_features": int(len(family_missingness)),
                "missing_rate_mean": float(family_missingness["missing_rate"].mean()),
                "missing_rate_max": float(family_missingness["missing_rate"].max()),
                "nonfinite_rate_mean": float(
                    family_missingness["nonfinite_rate"].mean()
                ),
                "n_all_nan_features": int(family_constants["is_all_nan"].sum()),
                "n_constant_features": int(family_constants["is_constant"].sum()),
                "failure_count": failure_count,
                "failure_rate": (
                    failure_count / n_observations if n_observations > 0 else np.nan
                ),
            }
        )

    return pd.DataFrame.from_records(rows, columns=_FAMILY_QC_COLUMNS)
