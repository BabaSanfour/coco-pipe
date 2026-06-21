"""Descriptor QC report builders.

Assembles the standard subject-level and dataset-level descriptor QC HTML
reports from the dataframes produced by
:mod:`coco_pipe.descriptors.qc` (family summaries, missingness, failure
summaries, etc.) and figure paths produced by descriptor-QC viz helpers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .core import Report, Section
from .elements import ImageElement, InteractiveTableElement, TableElement


def _add_optional_table(
    section: Section, data: pd.DataFrame | None, title: str
) -> None:
    if data is not None and not data.empty:
        section.add_element(TableElement(data, title=title))


def _add_optional_interactive_table(
    section: Section,
    data: pd.DataFrame | None,
    title: str,
    *,
    selector_columns: list[str] | None = None,
    default_sort: dict[str, str] | None = None,
    page_size: int = 5,
) -> None:
    if data is not None and not data.empty:
        section.add_element(
            InteractiveTableElement(
                data,
                title=title,
                selector_columns=selector_columns,
                default_sort=default_sort,
                page_size=page_size,
            )
        )


def _add_images(
    section: Section, figures: Mapping[str, Path], ordered_keys: Sequence[str]
) -> None:
    for key in ordered_keys:
        path = figures.get(key)
        if path and Path(path).exists():
            section.add_element(
                ImageElement(str(path), caption=key.replace("_", " ").title())
            )


# Family-specific diagnostic columns added by
# :func:`coco_pipe.descriptors.qc.add_family_diagnostics`, keyed by the
# descriptor family they apply to.
_FAMILY_DIAGNOSTIC_COLUMNS: dict[str, tuple[str, str, list[str]]] = {
    "band": (
        "Band Power Sanity",
        "🎚️",
        [
            "band_abs_negative_rate",
            "band_rel_out_of_range_rate",
            "band_corr_rel_out_of_range_rate",
            "band_ratio_nan_rate",
        ],
    ),
    "param": (
        "FOOOF Fit Quality",
        "📐",
        [
            "param_r_squared_median",
            "param_r_squared_p05",
            "param_fit_error_median",
            "param_fit_error_p95",
            "param_peak_count_missing_rate",
            "param_alpha_peak_freq_missing_rate",
        ],
    ),
    "complexity": (
        "Complexity Measure Sanity",
        "🌀",
        [
            "complexity_measure_missingness_max",
            "complexity_measure_missingness_median",
            "complexity_nonfinite_rate",
        ],
    ),
}


def _add_family_diagnostic_sections(
    report: Report, family_summary_df: pd.DataFrame | None
) -> None:
    """Add per-family sanity-check sections (band/FOOOF/complexity diagnostics).

    Each section shows the diagnostic columns added by
    :func:`coco_pipe.descriptors.qc.add_family_diagnostics` for the
    corresponding family, when present and non-empty in *family_summary_df*.
    """
    if family_summary_df is None or family_summary_df.empty:
        return
    for family, (title, icon, columns) in _FAMILY_DIAGNOSTIC_COLUMNS.items():
        if "family" not in family_summary_df.columns:
            continue
        present_columns = [
            column for column in columns if column in family_summary_df.columns
        ]
        if not present_columns:
            continue
        family_rows = family_summary_df[family_summary_df["family"] == family]
        if family_rows.empty:
            continue
        diag_df = family_rows[["family", *present_columns]].round(4)
        if diag_df[present_columns].isna().all(axis=None):
            continue
        section = Section(title, icon=icon)
        section.add_element(TableElement(diag_df, title=title))
        report.add_section(section)


def generate_descriptor_subject_report(
    output_path: Path,
    overview_df: pd.DataFrame,
    flags_df: pd.DataFrame,
    failure_summary_df: pd.DataFrame,
    feature_missingness_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    figure_paths: Mapping[str, Path],
) -> Path:
    """Build the per-shard (subject/session/condition) descriptor QC report.

    Parameters
    ----------
    output_path
        Destination ``.html`` path. Parent directories are created.
    overview_df
        Single-row dataframe with ``Subject``, ``Session``, ``Condition``,
        and summary metrics for the shard.
    flags_df
        QC flags raised for this shard (e.g. from
        :func:`coco_pipe.io.quality.make_qc_flag`).
    failure_summary_df
        Combined extraction-failure summary, e.g. the ``"combined"`` frame
        from :func:`coco_pipe.descriptors.qc.summarize_failures`.
    feature_missingness_df
        Per-feature missingness with family metadata, e.g. from
        :func:`coco_pipe.descriptors.qc.compute_family_missingness`.
    family_summary_df
        One row per descriptor family, e.g. from
        :func:`coco_pipe.descriptors.qc.aggregate_family_qc` (optionally
        extended via :func:`coco_pipe.descriptors.qc.add_family_diagnostics`).
    figure_paths
        Mapping of figure keys to image file paths, as produced by a
        descriptor-QC figure-saving helper.

    Returns
    -------
    Path
        *output_path*, after the report has been written.
    """
    report = Report(
        title=(
            "Descriptor QC Report - "
            f"{overview_df.iloc[0]['Subject']} "
            f"{overview_df.iloc[0]['Session']} "
            f"{overview_df.iloc[0]['Condition']}"
        )
    )

    overview = Section("Overview", icon="📋")
    _add_optional_table(overview, overview_df, "Shard Overview")
    report.add_section(overview)

    integrity = Section("Integrity Checks", icon="🧪")
    _add_optional_interactive_table(
        integrity,
        flags_df,
        "QC Flags",
        selector_columns=[
            column
            for column in ["level", "scope", "code"]
            if column in flags_df.columns
        ],
        default_sort={"column": "level", "direction": "desc"}
        if "level" in flags_df.columns
        else None,
        page_size=5,
    )
    report.add_section(integrity)

    failures = Section("Failure Summary", icon="⚠️")
    _add_optional_interactive_table(
        failures,
        failure_summary_df,
        "Failure Summary",
        selector_columns=[
            column
            for column in ["group", "value"]
            if column in failure_summary_df.columns
        ],
        default_sort={"column": "count", "direction": "desc"}
        if "count" in failure_summary_df.columns
        else None,
        page_size=5,
    )
    report.add_section(failures)

    missingness = Section("Missingness and Numerical Sanity", icon="📉")
    _add_optional_interactive_table(
        missingness,
        feature_missingness_df.round(4) if feature_missingness_df is not None else None,
        "Feature Missingness",
        selector_columns=[
            column
            for column in ["family", "scope", "sensor"]
            if column in feature_missingness_df.columns
        ],
        default_sort={"column": "missing_rate", "direction": "desc"}
        if "missing_rate" in feature_missingness_df.columns
        else None,
        page_size=5,
    )
    report.add_section(missingness)

    families = Section("Family-Specific Summary", icon="🧬")
    _add_optional_interactive_table(
        families,
        family_summary_df.round(4) if family_summary_df is not None else None,
        "Family Summary",
        selector_columns=[
            column
            for column in ["family"]
            if family_summary_df is not None and column in family_summary_df.columns
        ],
        default_sort={"column": "missing_rate", "direction": "desc"}
        if family_summary_df is not None and "missing_rate" in family_summary_df.columns
        else None,
        page_size=5,
    )
    report.add_section(families)

    _add_family_diagnostic_sections(report, family_summary_df)

    figures = Section("Figures", icon="📈")
    _add_images(
        figures,
        figure_paths,
        (
            "family_missingness",
            "failure_counts_by_family",
            "top_missing_features",
            "param_r_squared_hist",
            "param_fit_error_hist",
        ),
    )
    report.add_section(figures)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    return output_path


def generate_descriptor_dataset_report(
    output_path: Path,
    overview_df: pd.DataFrame,
    shard_summary_df: pd.DataFrame,
    flags_df: pd.DataFrame,
    failure_family_df: pd.DataFrame,
    failure_channel_df: pd.DataFrame,
    feature_missingness_df: pd.DataFrame,
    low_variance_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    figure_paths: Mapping[str, Path],
    manifest_df: pd.DataFrame | None = None,
    condition_breakdown_df: pd.DataFrame | None = None,
) -> Path:
    """Build the merged-dataset descriptor QC report.

    Parameters
    ----------
    output_path
        Destination ``.html`` path. Parent directories are created.
    overview_df
        Single-row dataframe with dataset-level summary metrics.
    shard_summary_df
        One row per merged shard with at least a ``qc_status`` column.
    flags_df
        Dataset-level QC flags.
    failure_family_df, failure_channel_df
        Failure counts grouped by family / channel, e.g. the ``"by_family"``
        and ``"by_channel"`` frames from
        :func:`coco_pipe.descriptors.qc.summarize_failures`.
    feature_missingness_df
        Per-feature missingness with family metadata.
    low_variance_df
        Features flagged as constant or near-zero variance.
    family_summary_df
        One row per descriptor family (optionally extended via
        :func:`coco_pipe.descriptors.qc.add_family_diagnostics`).
    figure_paths
        Mapping of figure keys to image file paths.
    manifest_df
        Optional single-row provenance table (e.g. derived from
        ``merge_manifest.json``) rendered in the Overview section.
    condition_breakdown_df
        Optional family x condition breakdown of failure counts and
        missingness, rendered in a dedicated section.

    Returns
    -------
    Path
        *output_path*, after the report has been written.
    """
    report = Report(title="Descriptor QC Dataset Report")

    overview = Section("Overview", icon="📋")
    _add_optional_table(overview, overview_df, "Dataset Overview")
    _add_optional_table(overview, manifest_df, "Merge Provenance")
    report.add_section(overview)

    shards = Section("Shard-Level QC Summary", icon="🗂️")
    _add_optional_interactive_table(
        shards,
        shard_summary_df,
        "Shard QC Summary",
        selector_columns=[
            column
            for column in ["session", "condition", "qc_status"]
            if column in shard_summary_df.columns
        ],
        default_sort={"column": "qc_status", "direction": "desc"}
        if "qc_status" in shard_summary_df.columns
        else None,
        page_size=10,
    )
    _add_optional_interactive_table(
        shards,
        flags_df,
        "Dataset QC Flags",
        selector_columns=[
            column
            for column in ["level", "scope", "code"]
            if column in flags_df.columns
        ],
        default_sort={"column": "level", "direction": "desc"}
        if "level" in flags_df.columns
        else None,
        page_size=5,
    )
    report.add_section(shards)

    failures = Section("Failures Summary", icon="⚠️")
    _add_optional_interactive_table(
        failures,
        failure_family_df,
        "Failures by Family",
        selector_columns=[
            column for column in ["value"] if column in failure_family_df.columns
        ],
        default_sort={"column": "count", "direction": "desc"}
        if "count" in failure_family_df.columns
        else None,
        page_size=10,
    )
    _add_optional_interactive_table(
        failures,
        failure_channel_df,
        "Failures by Channel",
        selector_columns=[
            column for column in ["value"] if column in failure_channel_df.columns
        ],
        default_sort={"column": "count", "direction": "desc"}
        if "count" in failure_channel_df.columns
        else None,
        page_size=10,
    )
    report.add_section(failures)

    missingness = Section("Missingness and Degeneracy", icon="📉")
    _add_optional_interactive_table(
        missingness,
        feature_missingness_df.round(4) if feature_missingness_df is not None else None,
        "Feature Missingness",
        selector_columns=[
            column
            for column in ["family", "scope", "sensor"]
            if column in feature_missingness_df.columns
        ],
        default_sort={"column": "missing_rate", "direction": "desc"}
        if "missing_rate" in feature_missingness_df.columns
        else None,
        page_size=10,
    )
    _add_optional_interactive_table(
        missingness,
        low_variance_df.round(6) if low_variance_df is not None else None,
        "Low-Variance Features",
        selector_columns=[
            column for column in ["family"] if column in low_variance_df.columns
        ],
        default_sort={"column": "std", "direction": "asc"}
        if "std" in low_variance_df.columns
        else None,
        page_size=10,
    )
    report.add_section(missingness)

    families = Section("Family-Specific Summary", icon="🧬")
    _add_optional_interactive_table(
        families,
        family_summary_df.round(4) if family_summary_df is not None else None,
        "Family Summary",
        selector_columns=[
            column
            for column in ["family"]
            if family_summary_df is not None and column in family_summary_df.columns
        ],
        default_sort={"column": "missing_rate", "direction": "desc"}
        if family_summary_df is not None and "missing_rate" in family_summary_df.columns
        else None,
        page_size=5,
    )
    report.add_section(families)

    _add_family_diagnostic_sections(report, family_summary_df)

    if condition_breakdown_df is not None and not condition_breakdown_df.empty:
        conditions = Section("Condition Breakdown", icon="🔀")
        _add_optional_interactive_table(
            conditions,
            condition_breakdown_df.round(4),
            "Family x Condition Breakdown",
            selector_columns=[
                column
                for column in ["family", "condition"]
                if column in condition_breakdown_df.columns
            ],
            default_sort={"column": "n_failures", "direction": "desc"}
            if "n_failures" in condition_breakdown_df.columns
            else None,
            page_size=10,
        )
        report.add_section(conditions)

    figures = Section("Figures", icon="📈")
    _add_images(
        figures,
        figure_paths,
        (
            "shard_status_counts",
            "failure_counts_by_family",
            "failure_counts_by_channel",
            "top_missing_features",
            "low_variance_by_family",
        ),
    )
    report.add_section(figures)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    return output_path
