"""Reusable scientific report section for subject-alignment diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from coco_pipe.viz.interactive import plot_grouped_bar, plot_scatter

from ._constants import (
    SUBJECT_ALIGNMENT_ASSESSMENT_COLUMNS,
    SUBJECT_ALIGNMENT_AUDIT_METRICS,
    SUBJECT_ALIGNMENT_IDENTITY_METRICS,
    SUBJECT_ALIGNMENT_LABEL_METRICS,
    SUBJECT_ALIGNMENT_METRIC_LABELS,
    SUBJECT_ALIGNMENT_PAIRING_COLUMNS,
    SUBJECT_ALIGNMENT_REQUIRED_COLUMNS,
)
from .core import PlotlyElement, Section
from .elements import (
    CalloutElement,
    ColumnsElement,
    InteractiveTableElement,
    StatCardElement,
)


def validate_subject_alignment_diagnostics(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and copy the generic subject-alignment diagnostics table."""
    if not isinstance(diagnostics, pd.DataFrame):
        raise TypeError("diagnostics must be a pandas DataFrame.")
    missing = sorted(SUBJECT_ALIGNMENT_REQUIRED_COLUMNS - set(diagnostics))
    if missing:
        raise ValueError(
            f"Subject-alignment diagnostics are missing required columns: {missing}."
        )
    return diagnostics.copy()


def select_subject_alignment_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    cohort_name: str,
    population: str,
) -> pd.DataFrame:
    """Validate and select one exact cohort/population diagnostic assessment."""
    if not cohort_name or not population:
        raise ValueError(
            "cohort_name and population must be explicit non-empty strings."
        )
    frame = validate_subject_alignment_diagnostics(diagnostics)
    selected = frame[
        (frame["cohort_name"].astype(str) == str(cohort_name))
        & (frame["population"].astype(str) == str(population))
    ].copy()
    if selected.empty:
        raise ValueError(
            "Requested subject-alignment diagnostics are absent: "
            f"cohort_name={cohort_name!r}, population={population!r}."
        )
    return selected.reset_index(drop=True)


def _present_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in frame]


def _assessment_label(frame: pd.DataFrame) -> pd.Series:
    columns = _present_columns(frame, SUBJECT_ALIGNMENT_ASSESSMENT_COLUMNS)
    if not columns:
        return pd.Series("All observations", index=frame.index)
    return frame.loc[:, columns].fillna("").astype(str).agg(" | ".join, axis=1)


def _add_probe_excess_above_chance(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Append chance-normalised subject-probe rows when chance is available."""
    if not {
        "metric",
        "value",
        "transform",
    }.issubset(diagnostics.columns):
        return diagnostics
    keys = _present_columns(
        diagnostics,
        (
            "model",
            "model_key",
            "embedding_model_key",
            "transform",
            "target",
            *SUBJECT_ALIGNMENT_PAIRING_COLUMNS,
        ),
    )
    probe = diagnostics[
        diagnostics["metric"] == "subject_probe_linear_balanced_accuracy"
    ].copy()
    chance = diagnostics[diagnostics["metric"] == "subject_probe_chance"].copy()
    if probe.empty or chance.empty:
        return diagnostics
    chance = chance.loc[:, [*keys, "value"]].rename(columns={"value": "chance"})
    probe = probe.merge(chance, on=keys, how="left")
    denominator = 1.0 - pd.to_numeric(probe["chance"], errors="coerce")
    probe_values = pd.to_numeric(probe["value"], errors="coerce")
    probe["value"] = np.where(
        denominator > 0,
        (probe_values - probe["chance"]) / denominator,
        np.nan,
    )
    probe["metric"] = "subject_probe_excess_above_chance"
    return pd.concat([diagnostics, probe.drop(columns="chance")], ignore_index=True)


def _add_raw_delta(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the raw (``transform='none'``) value and paired delta per cell."""
    if "transform" not in frame:
        frame["raw_value"] = np.nan
        frame["delta_vs_raw"] = np.nan
        return frame
    keys = [
        "metric",
        *_present_columns(
            frame,
            (
                "model",
                "model_key",
                "embedding_model_key",
                "target",
                *SUBJECT_ALIGNMENT_PAIRING_COLUMNS,
            ),
        ),
    ]
    raw = frame[frame["transform"].astype(str) == "none"].loc[:, [*keys, "value"]]
    raw = raw.drop_duplicates(keys).rename(columns={"value": "raw_value"})
    out = frame.merge(raw, on=keys, how="left")
    out["delta_vs_raw"] = pd.to_numeric(out["value"], errors="coerce") - pd.to_numeric(
        out["raw_value"], errors="coerce"
    )
    return out


def _metric_plot(
    frame: pd.DataFrame,
    metrics: frozenset[str],
    *,
    title: str,
    yaxis_title: str,
) -> Any | None:
    if "transform" not in frame:
        return None
    plot_frame = frame[frame["metric"].isin(metrics)].copy()
    plot_frame = plot_frame[pd.to_numeric(plot_frame["value"], errors="coerce").notna()]
    if plot_frame.empty:
        return None
    plot_frame["assessment"] = (
        _assessment_label(plot_frame)
        + " | "
        + plot_frame["metric"]
        .map(SUBJECT_ALIGNMENT_METRIC_LABELS)
        .fillna(plot_frame["metric"])
    )
    return plot_grouped_bar(
        plot_frame,
        x="assessment",
        y="value",
        group="transform",
        baseline=0.0,
        baseline_label="No excess / chance",
        title=title,
        yaxis_title=yaxis_title,
        legend_title="Transform",
        height=440,
    )


def _context_cards(frame: pd.DataFrame) -> ColumnsElement:
    def _range_text(column: str) -> str:
        values = (
            pd.to_numeric(frame[column], errors="coerce").dropna().unique()
            if column in frame
            else np.asarray([])
        )
        if not len(values):
            return "not recorded"
        low, high = int(values.min()), int(values.max())
        return str(low) if low == high else f"{low}-{high}"

    transforms = (
        frame.get("transform", pd.Series(dtype=object)).dropna().astype(str).unique()
    )
    populations = (
        frame.get("population", pd.Series(dtype=object)).dropna().astype(str).unique()
    )
    designs = frame.get("design", pd.Series(dtype=object)).dropna().astype(str).unique()
    cards = [
        StatCardElement("Transforms", len(transforms), color="purple"),
        StatCardElement(
            "Subjects per assessment", _range_text("n_subjects"), color="blue"
        ),
        StatCardElement(
            "Observations per assessment", _range_text("n_observations"), color="green"
        ),
        StatCardElement(
            "Assessment population",
            ", ".join(populations) if len(populations) else "not recorded",
            color="yellow",
        ),
        StatCardElement(
            "Variance design",
            ", ".join(designs) if len(designs) else "not recorded",
            color="yellow",
        ),
    ]
    return ColumnsElement(cards, cols=5)


def build_alignment_coverage_section(
    diagnostics: pd.DataFrame,
    *,
    title: str = "Cross-Model Diagnostic Coverage",
) -> Section | None:
    """Compare exact diagnostic populations across embedding models."""
    required = {
        "model",
        "transform",
        "scope",
        "target",
        "selection_fingerprint",
        "metric",
    }
    if diagnostics.empty:
        return None
    missing = sorted(required - set(diagnostics))
    if missing:
        raise ValueError(
            f"Alignment coverage diagnostics are missing columns: {missing}."
        )

    coverage = diagnostics[
        (diagnostics["transform"].astype(str) == "none")
        & (diagnostics["metric"].astype(str) == "between_subject_excess_over_null")
    ].copy()
    if coverage.empty:
        return None
    identity = ["model", "scope", "target"]
    if coverage.duplicated(identity).any():
        raise ValueError(
            "Alignment coverage requires one raw diagnostic assessment per "
            "model/scope/target."
        )
    fingerprints_per_cell = coverage.groupby(["scope", "target"], dropna=False)[
        "selection_fingerprint"
    ].transform("nunique")
    models_per_cell = coverage.groupby(["scope", "target"], dropna=False)[
        "model"
    ].transform("nunique")
    coverage["same_observation_selection"] = fingerprints_per_cell.eq(
        1
    ) & models_per_cell.gt(1)
    columns = _present_columns(
        coverage,
        (
            "model",
            "scope",
            "target",
            "n_subjects",
            "n_observations",
            "n_features",
            "n_constant_features",
            "selection_fingerprint",
            "same_observation_selection",
        ),
    )
    section = Section(title)
    section.add_element(
        CalloutElement(
            "Cross-model performance is directly comparable only when the exact "
            "observation-selection fingerprint matches. Counts describe the raw "
            "diagnostic population before alignment.",
            kind="info",
            title="Coverage interpretation",
        )
    )
    section.add_element(
        InteractiveTableElement(
            coverage.loc[:, columns],
            title="Coverage and selection compatibility",
            selector_columns=_present_columns(
                coverage,
                ("model", "scope", "target", "same_observation_selection"),
            ),
            default_sort={"column": "model", "direction": "asc"},
            page_size=30,
        )
    )
    return section


def build_alignment_tradeoff_section(
    performance: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    title: str = "Performance-Alignment Trade-off",
) -> Section | None:
    """Join paired decoding and leakage deltas and mark Pareto-optimal transforms."""
    performance_required = {
        "model",
        "decoder",
        "transform",
        "scope",
        "target",
        "performance",
    }
    diagnostics_required = {
        "model",
        "transform",
        "scope",
        "target",
        "metric",
        "value",
    }
    if performance.empty or diagnostics.empty:
        return None
    missing_performance = sorted(performance_required - set(performance))
    missing_diagnostics = sorted(diagnostics_required - set(diagnostics))
    if missing_performance or missing_diagnostics:
        raise ValueError(
            "Alignment trade-off inputs are missing columns: "
            f"performance={missing_performance}, diagnostics={missing_diagnostics}."
        )

    performance = performance.copy()
    performance["performance"] = pd.to_numeric(
        performance["performance"], errors="coerce"
    )
    performance_keys = ["model", "decoder", "scope", "target"]
    raw_performance = performance[performance["transform"].astype(str) == "none"].loc[
        :, [*performance_keys, "performance"]
    ]
    if raw_performance.duplicated(performance_keys).any():
        raise ValueError(
            "Alignment trade-off requires one raw decoding result per "
            "model/decoder/scope/target."
        )
    raw_performance = raw_performance.rename(columns={"performance": "raw_performance"})
    tradeoff = performance.merge(raw_performance, on=performance_keys, how="left")
    tradeoff["performance_delta"] = (
        tradeoff["performance"] - tradeoff["raw_performance"]
    )
    tradeoff = tradeoff[tradeoff["transform"].astype(str) != "none"]

    diagnostic_frame = _add_raw_delta(
        _add_probe_excess_above_chance(diagnostics.copy())
    )
    diagnostic_keys = ["model", "transform", "scope", "target"]
    leakage = diagnostic_frame[
        diagnostic_frame["metric"] == "subject_probe_excess_above_chance"
    ].loc[:, [*diagnostic_keys, "value", "raw_value"]]
    if leakage.duplicated(diagnostic_keys).any():
        raise ValueError(
            "Alignment trade-off requires one subject-probe assessment per "
            "model/transform/scope/target."
        )
    leakage["subject_leakage_reduction"] = pd.to_numeric(
        leakage["raw_value"], errors="coerce"
    ) - pd.to_numeric(leakage["value"], errors="coerce")
    tradeoff = tradeoff.merge(
        leakage.loc[:, [*diagnostic_keys, "subject_leakage_reduction"]],
        on=diagnostic_keys,
        how="inner",
    )
    tradeoff = tradeoff.dropna(
        subset=("performance_delta", "subject_leakage_reduction")
    )

    label = diagnostic_frame[
        diagnostic_frame["metric"] == "marginal_label_excess_over_null"
    ].loc[:, [*diagnostic_keys, "value", "raw_value"]]
    if not label.empty:
        if label.duplicated(diagnostic_keys).any():
            raise ValueError(
                "Alignment trade-off requires one label-variance assessment per "
                "model/transform/scope/target."
            )
        label["label_variance_retention"] = np.where(
            pd.to_numeric(label["raw_value"], errors="coerce") > 0,
            pd.to_numeric(label["value"], errors="coerce")
            / pd.to_numeric(label["raw_value"], errors="coerce"),
            np.nan,
        )
        tradeoff = tradeoff.merge(
            label.loc[:, [*diagnostic_keys, "label_variance_retention"]],
            on=diagnostic_keys,
            how="left",
        )
    if tradeoff.empty:
        return None

    tradeoff["pareto_optimal"] = False
    for _, group in tradeoff.groupby(performance_keys, dropna=False, sort=False):
        for index, row in group.iterrows():
            dominates = (
                (group["subject_leakage_reduction"] >= row["subject_leakage_reduction"])
                & (group["performance_delta"] >= row["performance_delta"])
                & (
                    (
                        group["subject_leakage_reduction"]
                        > row["subject_leakage_reduction"]
                    )
                    | (group["performance_delta"] > row["performance_delta"])
                )
            )
            tradeoff.loc[index, "pareto_optimal"] = not bool(dominates.any())

    tradeoff["point"] = (
        tradeoff["model"].astype(str)
        + " | "
        + tradeoff["decoder"].astype(str)
        + " | "
        + tradeoff["transform"].astype(str)
        + " | "
        + tradeoff["scope"].astype(str)
        + " | "
        + tradeoff["target"].astype(str)
    )
    figure = plot_scatter(
        tradeoff,
        x="subject_leakage_reduction",
        y="performance_delta",
        color="model",
        text="point",
        hovertemplate=(
            "%{text}<br>Leakage reduction=%{x:.3f}<br>Performance delta=%{y:.3f}"
        ),
        title="Clinical performance versus subject-leakage reduction",
        xaxis_title="Subject-probe excess reduction (higher is better)",
        yaxis_title="Decoding score delta vs paired raw embeddings",
        legend_title="Foundation model",
        height=520,
    )
    figure.add_vline(x=0.0, line_dash="dash", line_color="#888")
    figure.add_hline(y=0.0, line_dash="dash", line_color="#888")

    section = Section(title)
    section.add_element(
        CalloutElement(
            "The preferred region is upper-right: less subject-identifiable "
            "structure with preserved or improved clinical decoding. Pareto-optimal "
            "flags are computed separately within each model, decoder, scope, and "
            "target, using only exactly paired raw comparisons.",
            kind="info",
            title="Trade-off interpretation",
        )
    )
    section.add_element(PlotlyElement(figure))
    columns = _present_columns(
        tradeoff,
        (
            "model",
            "decoder",
            "transform",
            "scope",
            "target",
            "performance",
            "raw_performance",
            "performance_delta",
            "subject_leakage_reduction",
            "label_variance_retention",
            "pareto_optimal",
        ),
    )
    section.add_element(
        InteractiveTableElement(
            tradeoff.loc[:, columns],
            title="Paired transform ranking",
            selector_columns=_present_columns(
                tradeoff,
                ("model", "decoder", "transform", "scope", "target", "pareto_optimal"),
            ),
            default_sort={"column": "performance_delta", "direction": "desc"},
            page_size=30,
        )
    )
    return section


def build_subject_alignment_diagnostics_section(
    diagnostics: pd.DataFrame,
    *,
    scatter_figure: Any | None = None,
    title: str = "Subject Alignment Diagnostics",
) -> Section | None:
    """Build a scientific before/after report section for subject alignment.

    The section tests the intended trade-off: subject-identifiable structure
    should decrease while label-relevant structure is retained.  It supports
    the tidy output of :func:`coco_pipe.diagnostics.variance_decomposition_report`.
    The optional scatter is caller-provided because embedding loading and
    projection are dataset-specific and are qualitative only.
    """
    if diagnostics.empty or not {"metric", "value"}.issubset(diagnostics.columns):
        return None

    frame = _add_raw_delta(_add_probe_excess_above_chance(diagnostics.copy()))
    section = Section(title)
    section.add_element(
        CalloutElement(
            "Desired evidence is a decrease in **subject-probe excess above chance** "
            "and **subject variance excess over null**, without a material decrease "
            "in **label variance excess over null**. Marginal subject and label "
            "effects "
            "can overlap, so they are not an additive variance partition. Permutation "
            "p-values are diagnostic and are not multiplicity-corrected "
            "confirmatory tests. Raw deltas are reported only when population, "
            "target definition, and exact observation-selection fingerprint match; "
            "an unavailable delta therefore indicates an unpaired assessment, not "
            "a zero effect. Total sample variance and the variance participation "
            "ratio are representation-retention checks: compare them only with the "
            "paired raw representation from the same embedding model, not as "
            "absolute scales across models.",
            kind="info",
            title="Scientific interpretation",
        )
    )
    section.add_element(_context_cards(frame))

    identity_figure = _metric_plot(
        frame,
        SUBJECT_ALIGNMENT_IDENTITY_METRICS,
        title="Identity leakage before and after alignment",
        yaxis_title="Excess above chance or null (lower is better)",
    )
    if identity_figure is not None:
        section.add_element(PlotlyElement(identity_figure))
    label_figure = _metric_plot(
        frame,
        SUBJECT_ALIGNMENT_LABEL_METRICS,
        title="Label-relevant variance before and after alignment",
        yaxis_title="Excess over null (retention is desirable)",
    )
    if label_figure is not None:
        section.add_element(PlotlyElement(label_figure))

    variance_retention = frame[frame["metric"] == "total_sample_variance"].copy()
    variance_retention["variance_retained_vs_raw"] = np.where(
        pd.to_numeric(variance_retention["raw_value"], errors="coerce") > 0,
        pd.to_numeric(variance_retention["value"], errors="coerce")
        / pd.to_numeric(variance_retention["raw_value"], errors="coerce"),
        np.nan,
    )
    variance_retention = variance_retention[
        variance_retention["variance_retained_vs_raw"].notna()
    ]
    if not variance_retention.empty:
        variance_retention["assessment"] = _assessment_label(variance_retention)
        section.add_element(
            PlotlyElement(
                plot_grouped_bar(
                    variance_retention,
                    x="assessment",
                    y="variance_retained_vs_raw",
                    group="transform",
                    baseline=1.0,
                    baseline_label="Raw variance",
                    title="Representation variance retained after alignment",
                    yaxis_title="Fraction of paired raw variance",
                    legend_title="Transform",
                    height=440,
                )
            )
        )

    audit = frame[frame["metric"].isin(SUBJECT_ALIGNMENT_AUDIT_METRICS)].copy()
    if not audit.empty:
        audit["metric"] = (
            audit["metric"].map(SUBJECT_ALIGNMENT_METRIC_LABELS).fillna(audit["metric"])
        )
        columns = _present_columns(
            audit,
            (
                "transform",
                "cohort_name",
                "population",
                "scope",
                "eval_name",
                "target_col",
                "selection_fingerprint",
                "diagnostic_pair",
                "metric",
                "value",
                "raw_value",
                "delta_vs_raw",
                "design",
                "n_subjects",
                "n_observations",
                "n_features",
                "n_constant_features",
                "representation_variance_normalization",
                "representation_rank_metric",
                "representation_maximum_rank",
                "n_null_permutations",
                "probe_split_unit",
                "status",
                "reason",
            ),
        )
        selector_columns = _present_columns(
            audit,
            (
                "transform",
                "cohort_name",
                "population",
                "scope",
                "eval_name",
                "metric",
                "design",
                "status",
            ),
        )
        section.add_element(
            InteractiveTableElement(
                audit.loc[:, columns],
                title="Quantitative before/after audit",
                selector_columns=selector_columns,
                default_sort={"column": "transform", "direction": "asc"},
                page_size=30,
            )
        )
    if scatter_figure is not None:
        section.add_markdown(
            "### Qualitative projection check\n"
            "This view is exploratory only; interpret it alongside the quantitative "
            "audit above."
        )
        section.add_element(PlotlyElement(scatter_figure))
    return section
