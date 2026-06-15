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

# Aggregation-stat tokens that may prefix a subject-level measure
# (e.g. ``median_log_abs_alpha``); stripped before sub-family derivation.
_AGG_STAT_PREFIXES = frozenset(
    {"mean", "median", "iqr", "mad", "std", "var", "min", "max"}
)

# Band output-type patterns, longest/corrected first so e.g. ``corr_log_abs``
# matches before ``log_abs`` and ``abs``.
_BAND_SUBFAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("corr_log_abs", "corr_log_abs"),
    ("corr_rel", "corr_rel"),
    ("corr_ratio", "corr_ratio"),
    ("corr_abs", "corr_abs"),
    ("log_abs", "log_abs"),
    ("ratio", "ratio"),
    ("rel", "rel"),
    ("abs", "abs"),
)

_PARAM_SUBFAMILY: dict[str, str] = {
    "offset": "aperiodic",
    "exponent": "aperiodic",
    "knee": "aperiodic",
    "r_squared": "fit_quality",
    "fit_error": "fit_quality",
    "peak_count": "peaks",
    "peak_freq_dom": "peaks",
    "peak_power_dom": "peaks",
    "peak_bandwidth_dom": "peaks",
    "alpha_peak_freq": "peaks",
    "alpha_peak_power": "peaks",
}

_COMPLEXITY_SUBFAMILY: dict[str, str] = {
    "sample_entropy": "entropy",
    "perm_entropy": "entropy",
    "spectral_entropy": "entropy",
    "svd_entropy": "entropy",
    "fuzzy_entropy": "entropy",
    "dispersion_entropy": "entropy",
    "higuchi_fd": "fractal_complexity",
    "petrosian_fd": "fractal_complexity",
    "hurst_exponent": "fractal_complexity",
    "lziv_complexity": "fractal_complexity",
    "hjorth_mobility": "signal_dynamics",
    "hjorth_complexity": "signal_dynamics",
    "kurtosis": "signal_dynamics",
    "zero_crossings": "signal_dynamics",
}


def _strip_stat_prefix(measure: str) -> str:
    head, _, tail = str(measure).partition("_")
    return tail if head in _AGG_STAT_PREFIXES and tail else str(measure)


def descriptor_identity(measure: str) -> str:
    """Return a measure's descriptor identity (aggregation-stat prefix removed).

    Collapses the per-stat columns of one descriptor — e.g.
    ``mean_log_abs_alpha`` and ``iqr_log_abs_alpha`` both map to
    ``log_abs_alpha`` — so location and spread stay together as one unit.
    """
    return _strip_stat_prefix(measure)


def descriptor_subfamily(family: str | None, measure: str) -> str:
    """Map a ``(family, measure)`` pair to its descriptor sub-family.

    A sub-family is the *output type* within a family — finer than ``family``
    but coarser than ``measure``:

    - **band** → ``log_abs`` / ``rel`` / ``corr_log_abs`` / ``corr_rel`` /
      ``abs`` / ``corr_abs`` / ``ratio`` / ``corr_ratio`` (band name stripped)
    - **param** → ``aperiodic`` / ``peaks`` / ``fit_quality``
    - **complexity** → ``entropy`` / ``fractal_complexity`` / ``signal_dynamics``

    Robust to subject-level aggregation-stat prefixes (``median_…``). Unknown
    families/measures fall back to ``"<family>_other"`` (or ``"unknown"``).
    """
    if family is None:
        return "unknown"
    core = _strip_stat_prefix(measure)
    if family == "band":
        for pattern, label in _BAND_SUBFAMILY_PATTERNS:
            if pattern in core:
                return label
        return "band_other"
    if family == "param":
        return _PARAM_SUBFAMILY.get(core, "param_other")
    if family == "complexity":
        return _COMPLEXITY_SUBFAMILY.get(core, "complexity_other")
    return str(family)


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
                "subfamily": descriptor_subfamily(family, measure),
                "descriptor": descriptor_identity(measure),
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


def select_viable_feature_columns(
    feature_df: pd.DataFrame,
    descriptor_names: list[str],
    *,
    max_missing_rate: float = 0.20,
    drop_all_nan: bool = True,
    drop_constant: bool = True,
    constant_tol: float = 1e-12,
    max_row_drop_rate: float | None = None,
    known_families: tuple[str, ...] = KNOWN_FAMILY_TOKENS,
) -> tuple[list[str], pd.DataFrame]:
    """Select descriptor columns that pass missingness and degeneracy gates.

    After the missingness/constant gates, if ``max_row_drop_rate`` is set the
    surviving columns are further pruned (worst-NaN first) so the rows that the
    caller's any-NaN purge would still drop stay within ``rate * n_rows`` — i.e.
    prefer shedding a few NaN columns over losing observations.
    """
    if not 0 <= max_missing_rate <= 1:
        raise ValueError("max_missing_rate must be between 0 and 1.")
    missingness = compute_family_missingness(
        feature_df, descriptor_names, known_families
    )
    constants = compute_family_constant_summary(
        feature_df, descriptor_names, constant_tol, known_families
    )
    summary = missingness.merge(
        constants[["column", "std", "is_all_nan", "is_constant"]],
        on="column",
        how="left",
    )
    reasons = []
    for row in summary.itertuples(index=False):
        column_reasons = []
        if drop_all_nan and bool(row.is_all_nan):
            column_reasons.append("all_nan")
        elif float(row.missing_rate) > max_missing_rate:
            column_reasons.append("missing_rate")
        if drop_constant and bool(row.is_constant):
            column_reasons.append("constant")
        reasons.append(",".join(column_reasons))
    summary["drop_reason"] = reasons
    summary["dropped"] = summary["drop_reason"].astype(bool)
    surviving = summary.loc[~summary["dropped"], "column"].astype(str).tolist()
    drop_log = summary.loc[summary["dropped"]].reset_index(drop=True)

    if max_row_drop_rate is not None and len(feature_df) and surviving:
        sub = feature_df.loc[:, surviving]
        budget = int(np.floor(float(max_row_drop_rate) * len(sub)))
        row_preserving: list[str] = []
        while surviving:
            nan_cells = sub.isna()
            if int(nan_cells.any(axis=1).sum()) <= budget:
                break
            column_nan_counts = nan_cells.sum(axis=0)
            worst = str(column_nan_counts.idxmax())
            if int(column_nan_counts.max()) == 0:
                break
            surviving.remove(worst)
            sub = sub.drop(columns=[worst])
            row_preserving.append(worst)
        if row_preserving:
            drop_log = pd.concat(
                [
                    drop_log,
                    pd.DataFrame(
                        {
                            "column": row_preserving,
                            "drop_reason": "row_preserving",
                            "dropped": True,
                        }
                    ),
                ],
                ignore_index=True,
            )
    return surviving, drop_log


def summarize_failures(failure_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize an extraction failure log by family, channel, and exception.

    Parameters
    ----------
    failure_df
        Failure records as produced by
        :class:`~coco_pipe.descriptors.core.DescriptorPipeline`,
        with optional ``family``, ``channel_name``, ``exception_type``, and
        ``condition`` columns.

    Returns
    -------
    dict of DataFrame
        ``by_family``, ``by_channel``, ``by_exception_type``, ``by_condition``,
        ``by_family_channel`` (one row per (family, channel) pair), and
        ``combined`` (the four ``by_*`` group summaries stacked with a
        ``group`` column identifying their origin).
    """

    def _group_frame(column: str) -> pd.DataFrame:
        if failure_df.empty or column not in failure_df.columns:
            return pd.DataFrame(columns=["value", "count"])
        return (
            failure_df[column]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .rename_axis("value")
            .reset_index(name="count")
        )

    by_family = _group_frame("family")
    by_channel = _group_frame("channel_name")
    by_exception = _group_frame("exception_type")
    by_condition = _group_frame("condition")
    by_family_channel = (
        failure_df.fillna({"family": "unknown", "channel_name": "unknown"})
        .groupby(["family", "channel_name"], dropna=False)
        .size()
        .reset_index(name="count")
        if not failure_df.empty
        and {"family", "channel_name"}.issubset(failure_df.columns)
        else pd.DataFrame(columns=["family", "channel_name", "count"])
    )
    combined = (
        pd.concat(
            [
                by_family.assign(group="family"),
                by_channel.assign(group="channel"),
                by_exception.assign(group="exception_type"),
                by_condition.assign(group="condition"),
            ],
            ignore_index=True,
        )
        if any(
            not frame.empty
            for frame in (by_family, by_channel, by_exception, by_condition)
        )
        else pd.DataFrame(columns=["value", "count", "group"])
    )
    return {
        "by_family": by_family,
        "by_channel": by_channel,
        "by_exception_type": by_exception,
        "by_condition": by_condition,
        "by_family_channel": by_family_channel,
        "combined": combined,
    }


def add_family_diagnostics(
    family_summary_df: pd.DataFrame,
    feature_missingness_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add family-specific sanity diagnostics to a family-QC summary.

    Extends each row of *family_summary_df* (e.g. from :func:`aggregate_family_qc`)
    with diagnostics specific to the ``band``, ``param``, and ``complexity``
    descriptor families:

    - ``band``: rate of negative absolute-power values, out-of-range relative
      power values (outside ``[0, 1]``), and NaN ratio features.
    - ``param``: median/p05 of FOOOF ``r_squared``, median/p95 of
      ``fit_error``, and missingness of peak-related measures.
    - ``complexity``: median/max missingness across complexity measures and
      the non-finite rate.

    Parameters
    ----------
    family_summary_df
        One row per family, as produced by :func:`aggregate_family_qc`.
    feature_missingness_df
        Per-column missingness with family metadata, as produced by
        :func:`compute_family_missingness`.
    feature_df
        The underlying feature values (epoch- or subject-level) used to
        compute value-based diagnostics.

    Returns
    -------
    pd.DataFrame
        A copy of *family_summary_df* with additional family-specific columns.
    """
    if family_summary_df.empty:
        return family_summary_df
    rows: list[dict[str, Any]] = []
    for summary_row in family_summary_df.to_dict("records"):
        family = str(summary_row["family"])
        family_missingness = feature_missingness_df[
            feature_missingness_df["family"] == family
        ]
        family_cols = family_missingness["column"].tolist()
        row: dict[str, Any] = dict(summary_row)
        if family == "band":
            band_abs_cols = [
                column
                for column in family_cols
                if "_band_abs_" in f"_{column}_"
                or "band_abs_" in column
                or "band_corr_abs_" in column
            ]
            rel_cols = [column for column in family_cols if "band_rel_" in column]
            corr_rel_cols = [
                column for column in family_cols if "band_corr_rel_" in column
            ]
            ratio_cols = [column for column in family_cols if "ratio_" in column]
            row["band_abs_negative_rate"] = (
                float((feature_df[band_abs_cols] < 0).stack().mean())
                if band_abs_cols
                else 0.0
            )
            row["band_rel_out_of_range_rate"] = (
                float(
                    ((feature_df[rel_cols] < 0) | (feature_df[rel_cols] > 1))
                    .stack()
                    .mean()
                )
                if rel_cols
                else 0.0
            )
            row["band_corr_rel_out_of_range_rate"] = (
                float(
                    ((feature_df[corr_rel_cols] < 0) | (feature_df[corr_rel_cols] > 1))
                    .stack()
                    .mean()
                )
                if corr_rel_cols
                else 0.0
            )
            row["band_ratio_nan_rate"] = (
                float(feature_df[ratio_cols].isna().stack().mean())
                if ratio_cols
                else 0.0
            )
        elif family == "param":
            r2_cols = [column for column in family_cols if "param_r_squared_" in column]
            fit_error_cols = [
                column for column in family_cols if "param_fit_error_" in column
            ]
            peak_cols = [column for column in family_cols if "peak" in column]
            alpha_peak_cols = [
                column for column in family_cols if "alpha_peak_freq" in column
            ]
            row["param_r_squared_median"] = (
                float(
                    pd.to_numeric(feature_df[r2_cols].stack(), errors="coerce").median()
                )
                if r2_cols
                else np.nan
            )
            row["param_r_squared_p05"] = (
                float(
                    pd.to_numeric(
                        feature_df[r2_cols].stack(), errors="coerce"
                    ).quantile(0.05)
                )
                if r2_cols
                else np.nan
            )
            row["param_fit_error_median"] = (
                float(
                    pd.to_numeric(
                        feature_df[fit_error_cols].stack(), errors="coerce"
                    ).median()
                )
                if fit_error_cols
                else np.nan
            )
            row["param_fit_error_p95"] = (
                float(
                    pd.to_numeric(
                        feature_df[fit_error_cols].stack(), errors="coerce"
                    ).quantile(0.95)
                )
                if fit_error_cols
                else np.nan
            )
            row["param_peak_count_missing_rate"] = (
                float(feature_df[peak_cols].isna().stack().mean())
                if peak_cols
                else np.nan
            )
            row["param_alpha_peak_freq_missing_rate"] = (
                float(feature_df[alpha_peak_cols].isna().stack().mean())
                if alpha_peak_cols
                else np.nan
            )
        elif family == "complexity":
            row["complexity_measure_missingness_max"] = row["missing_rate_max"]
            row["complexity_measure_missingness_median"] = (
                float(family_missingness["missing_rate"].median())
                if not family_missingness.empty
                else 0.0
            )
            row["complexity_nonfinite_rate"] = row["nonfinite_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


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
