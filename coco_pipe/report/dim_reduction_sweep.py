"""Records/artifact-based dimensionality-reduction sweep reporting.

The dim-reduction counterpart to :mod:`coco_pipe.report.decoding_sweep`: helpers
that operate on persisted fit/eval run artifacts (the
:mod:`coco_pipe.dim_reduction` schema) to merge and rank runs and to assemble a
cross-mode roll-up report. The per-reduction section builders live in
:mod:`coco_pipe.report.dim_reduction`; study-specific policy (cohort metadata,
scope ordering, topomaps, report links) stays in the consuming project.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coco_pipe.dim_reduction import (
    EVAL_METRIC_COLUMNS,
    SEPARATION_METRIC_KEY,
    SEPARATION_RF_METRIC_KEY,
    load_fit_artifact,
    load_fit_runs,
    occurrence_aligned_positions,
)
from coco_pipe.viz.topo import (
    feature_names_are_channels,
    plot_topomap_from_channel_values,
    plot_topomap_selector,
)

from .core import Report, Section
from .dim_reduction import reduction_embedding_element
from .elements import (
    AccordionElement,
    CalloutElement,
    ColumnsElement,
    ContainerElement,
    Element,
    ImageElement,
    InteractiveTableElement,
    PlotlyElement,
    StatCardElement,
    TableElement,
    TabsElement,
)
from .qc import build_qc_section
from .tables import best_rows, split_by_status

# Rank on the primary metric first; use LR separation before geometry as a
# tie-breaker when RF is the primary metric. ``(column, ascending)`` tuples.
DEFAULT_REDUCTION_TIE_BREAKERS: tuple[tuple[str, bool], ...] = (
    (SEPARATION_METRIC_KEY, False),
    ("trustworthiness", False),
    ("continuity", False),
)

_DEFAULT_ROLLUP_SELECTORS = (
    "analysis_mode",
    "scope",
    "condition",
    "reducer",
    "model",
    "eval_name",
)


def _available_eval_metrics(frame: pd.DataFrame) -> list[str]:
    return [column for column in EVAL_METRIC_COLUMNS if column in frame.columns]


def _eval_merge_frame(
    eval_frame: pd.DataFrame,
    *,
    include_target: bool = True,
) -> pd.DataFrame:
    base_columns = ["fit_id", "eval_name"]
    if include_target:
        base_columns.append("target_col")
    columns = [*base_columns, *EVAL_METRIC_COLUMNS]
    if eval_frame.empty:
        return pd.DataFrame(columns=columns)
    return eval_frame.loc[
        :, [column for column in columns if column in eval_frame.columns]
    ]


def _score_sort_columns(frame: pd.DataFrame, selection_metric: str) -> list[str]:
    columns = [selection_metric, *EVAL_METRIC_COLUMNS, "trustworthiness", "continuity"]
    return list(dict.fromkeys(column for column in columns if column in frame.columns))


def _sort_by_score(frame: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    sort_columns = _score_sort_columns(frame, selection_metric)
    if not sort_columns:
        return frame
    return frame.sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
        na_position="last",
    )


def _selection_or_fallback(
    frame: pd.DataFrame,
    selection_metric: str,
    fallback_metric: str,
) -> str:
    if selection_metric in frame.columns:
        return selection_metric
    for metric in EVAL_METRIC_COLUMNS:
        if metric in frame.columns:
            return metric
    return fallback_metric


def merge_fit_eval(
    fit_runs: Any,
    eval_runs: Any = None,
    *,
    eval_name: str | None = None,
    metrics: Sequence[str] = tuple(EVAL_METRIC_COLUMNS),
    key: str = "fit_id",
) -> pd.DataFrame:
    """Left-merge successful evaluation metrics onto fit runs by *key*.

    Returns the fit-run frame unchanged when there is no usable evaluation data.
    When *eval_name* is given, evaluations are filtered to it first so each fit
    run contributes at most one evaluation row (avoiding row multiplication).
    """
    fit = pd.DataFrame(fit_runs)
    eval_df = pd.DataFrame(eval_runs) if eval_runs is not None else pd.DataFrame()
    if (
        fit.empty
        or eval_df.empty
        or "status" not in eval_df.columns
        or key not in eval_df.columns
        or not any(metric in eval_df.columns for metric in metrics)
    ):
        return fit
    keep = [
        column
        for column in [key, "eval_name", "target_col", *metrics]
        if column in eval_df.columns
    ]
    eval_ok = eval_df.loc[eval_df["status"] == "success", keep].copy()
    if eval_name is not None and "eval_name" in eval_ok.columns:
        eval_ok = eval_ok[eval_ok["eval_name"] == eval_name].copy()
    if eval_ok.empty:
        return fit
    return fit.merge(eval_ok, on=key, how="left")


def rank_reduction_runs(
    fit_runs: Any,
    eval_runs: Any = None,
    *,
    selection_metric: str,
    group_by: Sequence[str] = ("scope", "condition"),
    eval_name: str | None = None,
    tie_breakers: Sequence[tuple[str, bool]] = DEFAULT_REDUCTION_TIE_BREAKERS,
    reducers: Sequence[str] | None = None,
    metrics: Sequence[str] = tuple(EVAL_METRIC_COLUMNS),
) -> pd.DataFrame:
    """Return the best successful fit run per group.

    Merges evaluation metrics onto the (successful) fit runs, optionally
    restricts to *reducers*, and takes the top run per *group_by* ranked by
    *selection_metric* then *tie_breakers*. Empty when no successful fit runs
    survive. The fit/eval frames follow the :mod:`coco_pipe.dim_reduction`
    artifact schema (``status``/``fit_id``/``reducer`` columns).
    """
    fit = pd.DataFrame(fit_runs)
    if fit.empty or "status" not in fit.columns:
        return pd.DataFrame()
    merged = merge_fit_eval(fit, eval_runs, eval_name=eval_name, metrics=metrics)
    if reducers and "reducer" in merged.columns:
        merged = merged[merged["reducer"].isin(list(reducers))].copy()
    if merged.empty:
        return pd.DataFrame()
    group_cols = [column for column in group_by if column in merged.columns]
    best, _ = best_rows(merged, group_cols, selection_metric, tie_breakers=tie_breakers)
    return best


_DEFAULT_BEST_RUN_METRICS: tuple[tuple[str, str, str], ...] = (
    (SEPARATION_RF_METRIC_KEY, "RF Balanced Accuracy", "blue"),
    (SEPARATION_METRIC_KEY, "LR Balanced Accuracy", "cyan"),
    ("trustworthiness", "Trustworthiness", "green"),
    ("continuity", "Continuity", "yellow"),
)


def add_reduction_best_run_cards(
    section: Section,
    runs: Any,
    *,
    group_col: str = "condition",
    selection_metric: str,
    fallback_metric: str = "trustworthiness",
    tie_breakers: Sequence[tuple[str, bool]] = DEFAULT_REDUCTION_TIE_BREAKERS,
    metrics: Sequence[tuple[str, str, str]] = _DEFAULT_BEST_RUN_METRICS,
) -> None:
    """Append a best-run callout + StatCards per group to *section*.

    For each group value (default ``condition``), renders the winning run's
    reducer / component count / optional unit plus its quality metrics. A no-op
    when there are no successful runs. Study code owns which section this lands
    in; the card layout is shared.
    """
    success = split_by_status(pd.DataFrame(runs))[0]
    if success.empty:
        return
    sort_metric = _selection_or_fallback(success, selection_metric, fallback_metric)
    best_per_group = best_rows(
        success, [group_col], sort_metric, tie_breakers=tie_breakers
    )[0]
    for _, best in best_per_group.iterrows():
        group_name = best.get(group_col, "Unknown")
        cards = [
            StatCardElement("Reducer", best.get("reducer", ""), color="blue"),
            StatCardElement(
                "Components",
                int(best["n_components"]) if pd.notna(best.get("n_components")) else "",
                color="purple",
            ),
        ]
        if "unit_name" in best and pd.notna(best["unit_name"]) and best["unit_name"]:
            cards.insert(
                0, StatCardElement("Best Unit", best["unit_name"], color="indigo")
            )
        for column, label, color in metrics:
            if column in best.index and pd.notna(best[column]):
                cards.append(
                    StatCardElement(label, round(float(best[column]), 4), color=color)
                )
        section.add_element(
            CalloutElement(
                f"Best run for {group_col}: **{group_name}**",
                kind="tip",
                title=f"Peak Performance ({sort_metric})",
            )
        )
        section.add_element(ColumnsElement(cards, cols=len(cards)))


def _eval_topn_unit_bar(
    frame: pd.DataFrame,
    *,
    metric: str,
    top_n: int,
    subtitle: str = "",
) -> PlotlyElement | None:
    """Horizontal top-*top_n* bar of the best run per unit, colored by unit.

    Each bar is one analysis unit (its best-scoring ``reducer`` and
    ``n_components`` config), ranked by *metric*. The bar color encodes the unit,
    and the on-bar
    text is the winning ``n_components``. Returns ``None`` when there is nothing
    plottable.
    """
    from coco_pipe.viz.interactive.base import plot_ranked_bar

    if frame.empty or metric not in frame.columns:
        return None
    rows = frame.dropna(subset=[metric]).copy()
    if rows.empty:
        return None
    # Collapse each unit's sweep to its single best config, so the ranking is over
    # distinct units rather than near-identical configs of one unit. (The sort
    # is only to make groupby.head(1) keep the best row; plot_ranked_bar does the
    # ranking and top-N selection.)
    rows = rows.sort_values(metric, ascending=False)
    group_cols = [c for c in ("family", "unit_name", "reducer") if c in rows.columns]
    if group_cols:
        rows = rows.groupby(group_cols, dropna=False, sort=False).head(1)
    if rows.empty:
        return None

    def _label(row: pd.Series) -> str:
        parts = []
        fam = str(row.get("family", "") or "").strip()
        unit = str(row.get("unit_name", "") or "").strip()
        reducer = str(row.get("reducer", "") or "").strip()
        if fam and fam != unit:
            parts.append(fam)
        if unit:
            parts.append(unit)
        if reducer:
            parts.append(reducer)
        return " · ".join(parts) or "run"

    # Disambiguate any repeated labels so bars keep distinct category slots.
    rows["_label"] = _unique_labels([_label(row) for _, row in rows.iterrows()])
    rows["_unit"] = rows["unit_name"].astype(str) if "unit_name" in rows.columns else ""
    rows["_n_label"] = (
        [f"n={int(v)}" if pd.notna(v) else "" for v in rows["n_components"]]
        if "n_components" in rows.columns
        else ""
    )

    n_bars = min(len(rows), top_n)
    title = f"Top {n_bars} units by {metric}"
    if subtitle:
        title += f" — {subtitle}"
    fig = plot_ranked_bar(
        rows,
        value=metric,
        category="_label",
        color="_unit",
        text="_n_label",
        top_n=top_n,
        orientation="horizontal",
        title=title,
        value_title=metric,
        category_title="unit · reducer",
        legend_title="Unit",
        height=max(320, 26 * n_bars + 140),
    )
    return PlotlyElement(fig)


def _build_condition_bar_tabs(
    frame: pd.DataFrame,
    *,
    metric: str,
    conditions: Sequence[str] | None,
    top_n: int,
) -> Element | None:
    """One top-N unit bar per condition, wrapped in condition sub-tabs."""
    if "condition" in frame.columns:
        present = list(dict.fromkeys(frame["condition"].astype(str)))
    else:
        present = [""]
    ordered = [c for c in (conditions or []) if c in present]
    ordered += [c for c in present if c not in ordered]

    cond_tabs: dict[str, Element] = {}
    for cond in ordered:
        cf = (
            frame[frame["condition"].astype(str) == cond]
            if "condition" in frame.columns
            else frame
        )
        element = _eval_topn_unit_bar(cf, metric=metric, top_n=top_n, subtitle=cond)
        if element is not None:
            cond_tabs[cond or "condition"] = element
    if not cond_tabs:
        return None
    if len(cond_tabs) == 1:
        return next(iter(cond_tabs.values()))
    return TabsElement(cond_tabs)


def _build_eval_topn_bars(
    frame: pd.DataFrame,
    *,
    metric: str,
    eval_name_order: Sequence[str] | None,
    conditions: Sequence[str] | None,
    top_n: int,
) -> Element | None:
    """Top-N unit bars organized as eval-name tabs → condition sub-tabs.

    The primary evaluation (first in *eval_name_order*) lands on the first tab.
    Collapses to the bare condition tabs when only one evaluation is present.
    """
    if metric not in frame.columns:
        return None
    if "eval_name" in frame.columns:
        present = list(dict.fromkeys(frame["eval_name"].astype(str)))
    else:
        present = [""]
    ordered = [e for e in (eval_name_order or []) if e in present]
    ordered += [e for e in present if e not in ordered]

    eval_tabs: dict[str, Element] = {}
    for ev in ordered:
        ev_frame = (
            frame[frame["eval_name"].astype(str) == ev]
            if "eval_name" in frame.columns
            else frame
        )
        element = _build_condition_bar_tabs(
            ev_frame, metric=metric, conditions=conditions, top_n=top_n
        )
        if element is not None:
            eval_tabs[ev or "evaluation"] = element
    if not eval_tabs:
        return None
    if len(eval_tabs) == 1:
        return next(iter(eval_tabs.values()))
    return TabsElement(eval_tabs)


def build_reduction_eval_results_section(
    eval_frame: Any,
    *,
    sort_col: str | None = None,
    selector_columns: Sequence[str] = (
        "scope",
        "condition",
        "family",
        "unit_name",
        "reducer",
        "eval_name",
    ),
    title: str = "Evaluation Results",
    accordion_label: str = "Show Post-hoc Evaluation Results",
    selection_metric: str | None = None,
    eval_name_order: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    top_n: int = 15,
) -> Section | None:
    """Build the post-hoc evaluation section, or None if empty.

    The section holds a collapsible results table plus, below it, top-*top_n*
    unit bar charts ranked by *selection_metric* (or the first available
    separation metric): one tab per evaluation (the primary evaluation first,
    per *eval_name_order*), each with a condition sub-tab whose bars are colored
    by unit and labeled with the winning ``n_components``.
    """
    frame = pd.DataFrame(eval_frame)
    if frame.empty:
        return None
    available_metrics = _available_eval_metrics(frame)
    resolved_sort = sort_col or (
        available_metrics[0]
        if available_metrics
        else str(frame.columns[-1])
        if len(frame.columns)
        else "eval_name"
    )
    section = Section(title, icon="🧪")
    accordion = AccordionElement(accordion_label, open=False)
    accordion.add_element(
        InteractiveTableElement(
            frame.round(4),
            title="Post-hoc evaluations",
            selector_columns=[
                column for column in selector_columns if column in frame.columns
            ],
            default_sort={"column": resolved_sort, "direction": "desc"},
            page_size=5,
        )
    )
    section.add_element(accordion)

    bar_metric = (
        selection_metric
        if selection_metric and selection_metric in frame.columns
        else (available_metrics[0] if available_metrics else None)
    )
    if bar_metric is not None:
        bar_tabs = _build_eval_topn_bars(
            frame,
            metric=bar_metric,
            eval_name_order=eval_name_order,
            conditions=conditions,
            top_n=top_n,
        )
        if bar_tabs is not None:
            section.add_element(bar_tabs)
    return section


def _reducer_radar(
    runs: pd.DataFrame,
    *,
    reducers: Sequence[str],
    radar_metrics: Sequence[str],
) -> PlotlyElement | None:
    """Radar of each reducer's best value across *radar_metrics*."""
    columns = [metric for metric in radar_metrics if metric in runs.columns]
    if len(reducers) <= 1 or len(columns) < 3:
        return None
    rows: dict[str, dict[str, float]] = {}
    for reducer in reducers:
        sub = runs[runs["reducer"] == reducer]
        if sub.empty:
            continue
        row = {
            metric: float(sub[metric].dropna().max())
            for metric in columns
            if not sub[metric].dropna().empty
        }
        if len(row) >= 3:
            rows[reducer] = row
    if len(rows) <= 1:
        return None
    from coco_pipe.viz.interactive.dim_reduction import plot_radar_comparison

    return PlotlyElement(
        plot_radar_comparison(
            pd.DataFrame(rows).T,
            normalize=False,
            title="Reducer comparison — best metric across conditions",
        )
    )


def build_reduction_condition_ranking_section(
    condition_runs: Any,
    *,
    conditions: Sequence[str],
    reducers: Sequence[str],
    selection_metric: str,
    fallback_metric: str = "trustworthiness",
    ranking_columns: Sequence[str] = (
        "condition",
        "family",
        "unit_name",
        "reducer",
        "n_components",
        "eval_name",
        *EVAL_METRIC_COLUMNS,
        "trustworthiness",
        "continuity",
    ),
    selector_columns: Sequence[str] = (
        "condition",
        "family",
        "unit_name",
        "reducer",
        "eval_name",
    ),
    radar_metrics: Sequence[str] = (
        "trustworthiness",
        "continuity",
        "shepard_correlation",
        *EVAL_METRIC_COLUMNS,
    ),
    default_sort: str | None = None,
    title: str = "Condition Ranking",
) -> Section | None:
    """Build the condition-ranking table + reducer-profile radars.

    *condition_runs* are the successful condition-scope runs (the caller owns the
    scope/status filter). The section shows a sortable ranking table plus, when
    there are multiple reducers, an overall and per-condition reducer-profile
    radar. Returns None when *condition_runs* is empty.
    """
    runs = pd.DataFrame(condition_runs)
    if runs.empty:
        return None
    columns = [column for column in ranking_columns if column in runs.columns]
    resolved_sort = default_sort or _selection_or_fallback(
        runs, selection_metric, fallback_metric
    )
    section = Section(title, icon="🏁")
    section.add_element(
        InteractiveTableElement(
            runs.loc[:, columns].round(4),
            title="Condition ranking",
            selector_columns=[
                column for column in selector_columns if column in columns
            ],
            default_sort={"column": resolved_sort, "direction": "desc"},
            page_size=5,
        )
    )
    tabs: dict[str, Any] = {}
    radar_overall = _reducer_radar(runs, reducers=reducers, radar_metrics=radar_metrics)
    if radar_overall is not None:
        tabs["Reducer Profile (Overall)"] = radar_overall

    for condition in conditions:
        sub_runs = runs[runs["condition"] == condition]
        if not sub_runs.empty:
            radar_cond = _reducer_radar(
                sub_runs, reducers=reducers, radar_metrics=radar_metrics
            )
            if radar_cond is not None:
                tabs[f"Reducer Profile ({condition})"] = radar_cond
    if tabs:
        section.add_element(TabsElement(tabs))
    return section


def build_reduction_rollup_report(
    leaderboard: Any,
    *,
    title: str,
    x_metric: str = "trustworthiness",
    y_metric: str = SEPARATION_RF_METRIC_KEY,
    sort_col: str | None = None,
    mode_label_map: Mapping[str, str] | None = None,
    strategy_note: str | None = None,
    link_rows: Sequence[Mapping[str, Any]] | None = None,
    task_failures: Sequence[Mapping[str, Any]] = (),
    selector_columns: Sequence[str] = _DEFAULT_ROLLUP_SELECTORS,
    asset_urls: dict[str, str] | str | None = None,
) -> Report:
    """Build the cross-mode dimensionality-reduction roll-up report.

    A sortable leaderboard table plus an *x_metric* vs *y_metric* scatter (e.g.
    faithful-vs-discriminative: trustworthiness x separation), optional per-mode
    report links and a failures table. The caller owns the metric choice, the
    ``analysis_mode`` label map, the strategy note and the report links; this
    function owns the generic scaffolding.
    """
    report = Report(title=title, asset_urls=asset_urls)
    frame = pd.DataFrame(leaderboard)

    overview = Section("Roll-up Overview", icon="🏆")
    overview.add_element(
        CalloutElement(
            strategy_note
            or (
                "Best run per analysis mode &times; condition. A strong "
                "representation is both geometrically faithful and clinically "
                "separating &mdash; aim for the top-right of the scatter below."
            ),
            kind="info",
            title="Roll-up Strategy",
        )
    )
    report.add_section(overview)

    if frame.empty:
        overview.add_markdown("*No successful runs available for the leaderboard.*")
        _append_failures(report, task_failures)
        return report

    resolved_sort = sort_col or (
        y_metric
        if y_metric in frame.columns
        else x_metric
        if x_metric in frame.columns
        else str(frame.columns[0])
    )
    display = frame.round(4)
    if mode_label_map and "analysis_mode" in display.columns:
        display["analysis_mode"] = display["analysis_mode"].map(
            lambda value: mode_label_map.get(value, value)
        )
    board = Section("Leaderboard", icon="📊")
    board.add_element(
        InteractiveTableElement(
            display,
            title="Best run per (mode, condition)",
            selector_columns=[
                column for column in selector_columns if column in display.columns
            ],
            default_sort={"column": resolved_sort, "direction": "desc"},
            page_size=15,
        )
    )
    scatter = _rollup_scatter(frame, x_metric=x_metric, y_metric=y_metric)
    if scatter is not None:
        board.add_element(scatter)
    report.add_section(board)

    if link_rows:
        links = Section("Per-mode reports", icon="🔗")
        links.add_element(
            TableElement(
                pd.DataFrame(list(link_rows)), title="Per-mode dataset summaries"
            )
        )
        report.add_section(links)

    _append_failures(report, task_failures)
    return report


def _rollup_scatter(
    frame: pd.DataFrame,
    *,
    x_metric: str,
    y_metric: str,
) -> PlotlyElement | None:
    """Faithful-vs-discriminative scatter, or None when the axes are absent."""
    if x_metric not in frame.columns or y_metric not in frame.columns:
        return None
    scatter_df = frame.dropna(subset=[x_metric, y_metric]).copy()
    if scatter_df.empty:
        return None
    from coco_pipe.viz.interactive.base import plot_scatter

    color_col = (
        "model"
        if "model" in scatter_df.columns
        else "analysis_mode"
        if "analysis_mode" in scatter_df.columns
        else None
    )
    modes = scatter_df.get("analysis_mode", pd.Series("", index=scatter_df.index))
    conds = scatter_df.get("condition", pd.Series("", index=scatter_df.index))
    scatter_df["hover_label"] = [
        f"{mode}/{condition}" for mode, condition in zip(modes, conds, strict=False)
    ]
    fig = plot_scatter(
        scatter_df,
        x=x_metric,
        y=y_metric,
        color=color_col,
        text="hover_label",
        hovertemplate=(
            "%{text}<br>"
            + x_metric
            + "=%{x:.3f}<br>"
            + y_metric
            + "=%{y:.3f}<extra></extra>"
        ),
        mode="markers",
        title="Faithful vs discriminative (top-right is best)",
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        legend_title=color_col,
    )
    return PlotlyElement(fig)


def _append_failures(
    report: Report,
    task_failures: Sequence[Mapping[str, Any]],
) -> None:
    if not task_failures:
        return
    section = Section("Task Failures", icon="⚠️")
    section.add_element(
        TableElement(
            pd.DataFrame(list(task_failures)),
            title="Modes that failed or were skipped",
        )
    )
    report.add_section(section)


@dataclass
class DimReductionReportContext:
    """Resolved configuration + injection seams for a dataset report.

    Constructed once by the consuming project and threaded through the section
    builders. Study-specific policy travels as **data** (metadata-exclusion
    sets, topomap channel vocabulary, unit labels, failure columns); the single
    behavioural seam is :attr:`container_builder`, which loads the dataset
    container for a condition, plus the optional :func:`build_dataset_report`
    ``overview_extras`` hook for cohort summaries.
    """

    analysis_mode: str
    selection_metric: str
    reducers: Sequence[str]
    conditions: Sequence[str]
    container_builder: Callable[[str], Any | None]
    output_root: Path
    eval_specs: Sequence[Mapping[str, Any]] = ()
    interactive: bool = False
    input_mode: str = ""
    representation: str = ""
    family_label: str = ""
    run_pooled: bool = False
    pooled_condition: str = ""
    dataset_name: str = ""
    report_title: str = "Dimensionality Reduction"
    # Metadata policy (data-driven).
    excluded_columns: frozenset[str] = frozenset()
    excluded_normalized: frozenset[str] = frozenset()
    excluded_normalized_substrings: tuple[str, ...] = ()
    excluded_suffixes: tuple[str, ...] = ()
    meta_extractors: Mapping[str, Callable[[dict[str, np.ndarray]], Any]] = field(
        default_factory=dict
    )
    # Topomap policy (data-driven): channel vocabulary for the sensor gate.
    topomap_channels: frozenset[str] | None = None
    # Unit-label policy.
    unit_labels: Mapping[str, str] = field(default_factory=dict)
    # Failure-table columns.
    fit_failure_columns: Sequence[str] = ()
    eval_failure_columns: Sequence[str] = ()

    def unit_label(self, default: str) -> str:
        """Return the human-readable unit label for the current mode."""
        return dict(self.unit_labels).get(self.analysis_mode, default)


def _feature_names(container: Any) -> list[str] | None:
    """Return the feature-axis names from a container, or None if unavailable."""
    if container is None:
        return None
    try:
        feat = (container.coords or {}).get("feature")
        if feat is not None:
            names = [str(f) for f in np.asarray(feat)]
            return names if names else None
    except (AttributeError, TypeError, KeyError, ValueError):
        # Container/coords may be absent or non-conforming; treat as unavailable.
        return None
    return None


def build_meta_dict(
    container: Any,
    ids: np.ndarray | None,
    ctx: DimReductionReportContext,
) -> dict[str, np.ndarray]:
    """Build a plotting-metadata dict from a container's observation frame.

    Filters excluded columns (per *ctx* policy), applies eval-spec label maps /
    filters, and runs any registered :attr:`~DimReductionReportContext.meta_extractors`
    (e.g. eye-state derivation). Rows are aligned to *ids* when provided.
    """
    # 1. Fetch and format metadata frame
    frame = container.observation_frame()
    frame = frame.drop(columns=["feature"], errors="ignore")
    if container.y is not None and "y" not in frame.columns:
        frame["y"] = np.asarray(container.y)
    frame = frame.rename(columns={"sample_id": "obs_id"})

    # 2. Align frame rows to match requested ids. Observation ids are not unique,
    # so use the same occurrence-disambiguated alignment as the eval path rather
    # than a naive obs_id lookup (which explodes or silently drops on duplicates).
    if ids is not None and "obs_id" in frame.columns:
        positions = occurrence_aligned_positions(frame["obs_id"].to_numpy(), ids)
        if positions is None:
            frame = pd.DataFrame()
        else:
            frame = frame.iloc[positions].reset_index(drop=True)

    if frame.empty:
        return {}

    meta: dict[str, np.ndarray] = {}

    # 3. Extract valid standard columns
    for col_name in frame.columns:
        col_str = str(col_name)
        normalized = "".join(ch for ch in col_str.lower() if ch.isalnum())

        is_excluded = (
            col_str in ctx.excluded_columns
            or normalized in ctx.excluded_normalized
            or any(sub in normalized for sub in ctx.excluded_normalized_substrings)
            or (
                bool(ctx.excluded_suffixes)
                and col_str.endswith(tuple(ctx.excluded_suffixes))
            )
        )
        if is_excluded:
            continue

        if 1 < frame[col_str].nunique(dropna=False) <= 200:
            meta[col_str] = frame[col_str].to_numpy()

    # 4. Extract columns defined in evaluation specs (YAML)
    for spec in ctx.eval_specs:
        target_col = spec.get("target_col")
        if target_col not in frame.columns:
            continue

        labels = frame[target_col].astype(str)
        if label_map := spec.get("label_map"):
            labels = labels.map(lambda v: label_map.get(v, v))

        labels = labels.replace({"nan": "unknown", "None": "unknown", "": "unknown"})

        if filters := spec.get("filters"):
            mask = pd.Series(True, index=frame.index)
            for f_spec in filters:
                col = f_spec["column"]
                if col in frame.columns:
                    valid_vals = {str(v) for v in f_spec["values"]}
                    mask &= frame[col].astype(str).isin(valid_vals)
                else:
                    mask[:] = False
                    break
            labels = labels.where(mask, "unknown")

        if 1 < labels.nunique(dropna=False) <= 200:
            meta[str(spec["name"])] = labels.to_numpy(dtype=object)

    # 5. Run registered post-hoc extractors (e.g. eye-state from condition)
    for name, extractor in ctx.meta_extractors.items():
        extracted = extractor(meta)
        if extracted is not None:
            meta[name] = extracted

    return meta


def build_best_fit_plots(
    title: str,
    artifact: dict[str, Any],
    meta_dict: dict[str, np.ndarray],
    ctx: DimReductionReportContext,
    feature_names: list[str] | None = None,
) -> Any:
    """Build the best-fit diagnostics element for a single reducer fit.

    Renders the native 2-D / first-3-dims embedding scatter and, when
    available, a scalp topomap of the component loadings (only when
    *feature_names* are montage channels per
    :attr:`~DimReductionReportContext.topomap_channels`) and a scree plot of
    the ``explained_variance_ratio`` diagnostic. Returns ``None`` when the
    embedding is not 2-D.
    """
    embedding = np.asarray(artifact["embedding"])
    if embedding.ndim != 2:
        return None

    plots = []

    if embedding.shape[1] == 2:
        plots.append(
            reduction_embedding_element(
                embedding,
                metadata=meta_dict,
                title=f"{title} - native 2D",
                dimensions=2,
                interactive=ctx.interactive,
            )
        )
    elif embedding.shape[1] >= 3:
        plots.append(
            reduction_embedding_element(
                embedding[:, :3],
                metadata=meta_dict,
                title=f"{title} - first 3 dims",
                dimensions=3,
                interactive=ctx.interactive,
            )
        )

    components = (artifact.get("diagnostics") or {}).get("components")
    if components is not None:
        loadings = np.asarray(components, dtype=float).T
        if loadings.ndim == 2:
            n_comp = min(loadings.shape[1], 10)
            topo_fig = None
            if feature_names_are_channels(feature_names, ctx.topomap_channels):
                topo_fig = plot_topomap_selector(
                    {
                        f"PC{i + 1}": (feature_names, loadings[:, i])
                        for i in range(n_comp)
                    },
                    title=f"{title} - component loadings (topomap, top {n_comp})",
                    unit="loading",
                )
            if topo_fig is not None:
                plots.append(PlotlyElement(topo_fig))

    explained_variance = (artifact.get("diagnostics") or {}).get(
        "explained_variance_ratio"
    )
    if explained_variance is not None:
        from coco_pipe.viz.interactive.dim_reduction import plot_scree

        scree_fig = plot_scree(np.asarray(explained_variance, dtype=float))
        scree_fig.update_layout(title=f"{title} - Scree Plot")
        plots.append(PlotlyElement(scree_fig))

    if plots:
        return ColumnsElement(plots, cols=len(plots))
    return None


def build_flat_condition_section(
    condition: str,
    condition_runs: pd.DataFrame,
    eval_frame: pd.DataFrame,
    ctx: DimReductionReportContext,
) -> Section:
    """Build the per-condition section for ``analysis_mode == "flat"``."""
    from coco_pipe.viz.interactive.base import plot_scatter

    container = ctx.container_builder(condition)
    artifacts = {
        str(row["fit_id"]): load_fit_artifact(ctx.output_root / row["artifact_path"])
        for _, row in condition_runs.iterrows()
    }
    fam_label = ctx.family_label
    section = Section(condition, icon="🧠")

    callout_text = (
        f"Input mode: **{ctx.input_mode}**<br/>Representation: **{ctx.representation}**"
    )
    if fam_label:
        callout_text += f"<br/>Descriptor families: **{fam_label}**"
    section.add_element(
        CalloutElement(callout_text, kind="info", title="Configuration Details")
    )

    section.add_element(
        ColumnsElement(
            [
                StatCardElement(
                    "Observations",
                    container.meta.get("loaded_obs", container.X.shape[0]),
                    color="blue",
                ),
                StatCardElement("Successful Fits", len(condition_runs), color="green"),
            ],
            cols=4,
        )
    )

    ranking_df = condition_runs.merge(
        _eval_merge_frame(eval_frame),
        on="fit_id",
        how="left",
    )
    ranking_sort = _selection_or_fallback(
        ranking_df, ctx.selection_metric, SEPARATION_METRIC_KEY
    )
    section.add_element(
        InteractiveTableElement(
            ranking_df.loc[
                :,
                [
                    col
                    for col in [
                        "reducer",
                        "n_components",
                        "eval_name",
                        *EVAL_METRIC_COLUMNS,
                        "trustworthiness",
                        "continuity",
                    ]
                    if col in ranking_df.columns
                ],
            ].round(4),
            title="Fit ranking",
            selector_columns=["reducer", "eval_name"],
            default_sort={"column": ranking_sort, "direction": "desc"},
            page_size=5,
        )
    )

    reducer_tabs = {}
    for reducer_name in ctx.reducers:
        reducer_runs = condition_runs[condition_runs["reducer"] == reducer_name].copy()
        if reducer_runs.empty:
            continue
        best_row = (
            reducer_runs.merge(
                _eval_merge_frame(eval_frame, include_target=False),
                on="fit_id",
                how="left",
            )
            .pipe(_sort_by_score, ctx.selection_metric)
            .iloc[0]
        )
        best_artifact = artifacts[str(best_row["fit_id"])]

        tab_elements = []
        meta_dict = build_meta_dict(container, best_artifact["ids"], ctx)

        plots_elem = build_best_fit_plots(
            f"{condition} - {reducer_name}",
            best_artifact,
            meta_dict,
            ctx,
            feature_names=_feature_names(container),
        )
        if plots_elem:
            tab_elements.append(plots_elem)

        sweep_df = reducer_runs.merge(
            _eval_merge_frame(eval_frame),
            on="fit_id",
            how="left",
        )
        sweep_metric = _selection_or_fallback(
            sweep_df, ctx.selection_metric, SEPARATION_METRIC_KEY
        )
        if not sweep_df.empty and sweep_metric in sweep_df.columns:
            sep_df = sweep_df.dropna(subset=[sweep_metric]).copy()
            sep_df["series"] = "separation: " + sep_df["eval_name"].astype(str)
            fig = plot_scatter(
                sep_df,
                x="n_components",
                y=sweep_metric,
                color="series",
                mode="lines+markers",
                title=f"{condition} - {reducer_name} separation vs n_components",
                xaxis_title="n_components",
                yaxis_title="score",
            )
            acc = AccordionElement("Show Hyperparameter Sweep Data", open=False)
            acc.add_element(PlotlyElement(fig))

            sweep_table = InteractiveTableElement(
                sweep_df.loc[
                    :,
                    [
                        col
                        for col in [
                            "n_components",
                            "eval_name",
                            *EVAL_METRIC_COLUMNS,
                            "trustworthiness",
                            "continuity",
                        ]
                        if col in sweep_df.columns
                    ],
                ].round(4),
                title=f"{condition} - {reducer_name} sweep summary",
                page_size=5,
            )
            acc.add_element(sweep_table)
            tab_elements.append(acc)

        if tab_elements:
            reducer_tabs[reducer_name] = (
                ColumnsElement(tab_elements, cols=1)
                if len(tab_elements) > 1
                else tab_elements[0]
            )

    if reducer_tabs:
        section.add_element(TabsElement(reducer_tabs))

    return section


def _unique_labels(labels: Sequence[str]) -> list[str]:
    """Disambiguate repeated labels (``lbl``, ``lbl (2)``, …) preserving order."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return out


def _value_axis_range(
    values: pd.Series, baseline: float | None
) -> tuple[float, float] | None:
    """Value-axis range anchored at *baseline* (e.g. chance), padded at the top."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    lo = float(numeric.min())
    hi = float(numeric.max())
    if baseline is not None:
        lo = min(lo, baseline)
    span = hi - lo
    pad = span * 0.12 if span > 0 else 0.02
    return (lo - 0.01, hi + pad)


def _unit_peak_bar(
    rows: pd.DataFrame,
    *,
    metric: str,
    unit_column: str,
    unit_label: str,
    title: str,
    baseline: float | None,
) -> PlotlyElement | None:
    """Ranked horizontal bar of *metric* across all units, colored by family.

    One bar per unit (already the best config per unit), sorted, labeled with the
    winning ``n_components``, with a chance reference line when *baseline* is set.
    Colors by ``family`` when families vary, else by the unit itself so bars stay
    distinguishable.
    """
    from coco_pipe.viz.interactive.base import plot_ranked_bar

    if rows.empty or metric not in rows.columns:
        return None
    data = rows.dropna(subset=[metric]).copy()
    if data.empty:
        return None

    fam_varies = "family" in data.columns and data["family"].nunique(dropna=True) > 1
    labels = []
    for _, row in data.iterrows():
        unit = str(row.get(unit_column, "") or "")
        fam = str(row.get("family", "") or "")
        prefixed = fam_varies and fam and fam != unit
        labels.append(f"{fam} / {unit}" if prefixed else (unit or "unit"))
    data["_label"] = _unique_labels(labels)
    color_col = "family" if fam_varies else unit_column
    data["_color"] = (
        data[color_col].astype(str) if color_col in data.columns else unit_label
    )
    data["_n_label"] = (
        [f"n={int(v)}" if pd.notna(v) else "" for v in data["n_components"]]
        if "n_components" in data.columns
        else ""
    )
    legend = "family" if fam_varies else unit_label
    fig = plot_ranked_bar(
        data,
        value=metric,
        category="_label",
        color="_color",
        text="_n_label",
        orientation="horizontal",
        title=title,
        value_title=metric,
        category_title=unit_label,
        legend_title=legend.capitalize(),
        baseline=baseline,
        baseline_label="chance" if baseline is not None else None,
        value_range=_value_axis_range(data[metric], baseline),
        height=max(300, 26 * len(data) + 130),
    )
    return PlotlyElement(fig)


def _unit_stability_box(
    rows: pd.DataFrame,
    *,
    metric: str,
    unit_column: str,
    unit_label: str,
    title: str,
    baseline: float | None,
) -> PlotlyElement | None:
    """Box distribution of *metric* across the sweep, one box per unit."""
    from coco_pipe.viz.interactive.base import plot_distribution_groups

    if rows.empty or metric not in rows.columns:
        return None
    data = rows.dropna(subset=[metric])
    if data.empty:
        return None
    units = list(data[unit_column].unique())
    groups = [data[data[unit_column] == unit][metric].to_numpy() for unit in units]
    if not any(len(group) for group in groups):
        return None
    fig = plot_distribution_groups(
        groups=groups,
        labels=[str(unit) for unit in units],
        kind="box",
        title=title,
        xaxis_title=unit_label,
        yaxis_title=metric,
        baseline=baseline,
        baseline_label="chance" if baseline is not None else None,
    )
    return PlotlyElement(fig)


def _unit_topomaps(
    rows: pd.DataFrame,
    *,
    unit_column: str,
    plot_metric: str,
    title_prefix: str,
    trace_label: str,
) -> Element | None:
    """Scalp topomaps of the best per-sensor value for the key metrics."""
    if rows.empty:
        return None
    topomaps: list[Element] = []
    for topo_metric in [plot_metric, "trustworthiness", "continuity"]:
        if topo_metric not in rows.columns:
            continue
        topo_df = rows.dropna(subset=[topo_metric])
        if topo_df.empty:
            continue
        topo_groups = {
            topo_metric: (
                topo_df[unit_column].astype(str).tolist(),
                topo_df[topo_metric].astype(float).to_numpy(),
            )
        }
        topo_plot = plot_topomap_selector(
            topo_groups, title=f"{title_prefix} best {topo_metric}", unit=topo_metric
        )
        if topo_plot is not None:
            topomaps.append(PlotlyElement(topo_plot))
            continue
        topo_label, (topo_names, topo_values) = next(iter(topo_groups.items()))
        try:
            topo_fig = plot_topomap_from_channel_values(
                channel_names=topo_names,
                values=topo_values,
                title=f"{title_prefix} best {topo_metric} - {topo_label}",
                unit=topo_metric,
            )
            if topo_fig is not None:
                topomaps.append(ImageElement(topo_fig, width="100%"))
        except Exception as exc:
            topomaps.append(
                CalloutElement(
                    f"Topomap rendering failed for '{topo_metric}' "
                    f"({trace_label}): {exc}",
                    kind="warning",
                )
            )
    if not topomaps:
        return None
    return ColumnsElement(topomaps, cols=len(topomaps))


def build_unit_summary(
    section: Section,
    unit_runs: pd.DataFrame,
    ctx: DimReductionReportContext,
    *,
    unit_label: str,
    title_prefix: str,
) -> None:
    """Append unit-level ranking tables, bar charts, stability boxes + topomaps.

    The peak/stability/topomap visuals are tabbed by ``eval_name`` (the
    scientific contrast) with every unit compared side by side inside each tab —
    rather than one tab per unit, which collapses to single-bar charts.
    """
    if unit_runs.empty:
        return

    selection_metric = ctx.selection_metric
    unit_column = (
        "unit_key" if ctx.analysis_mode == "descriptor_sensor" else "unit_name"
    )

    group_columns = [
        column
        for column in ["family", "subfamily", "eval_name", "target_col"]
        if column in unit_runs.columns and unit_runs[column].notna().any()
    ]
    sort_columns = _score_sort_columns(unit_runs, selection_metric)
    ranked_units = (
        unit_runs.sort_values(
            sort_columns,
            ascending=[False] * len(sort_columns),
            na_position="last",
        )
        if sort_columns
        else unit_runs
    )
    best_units = (
        ranked_units.groupby([*group_columns, unit_column], dropna=False).head(1).copy()
    )
    display_columns = [
        column
        for column in [
            *group_columns,
            unit_column,
            "reducer",
            "n_components",
            *sort_columns,
        ]
        if column in best_units.columns
    ]
    section.add_element(
        InteractiveTableElement(
            best_units.loc[:, display_columns].round(4),
            title=f"{title_prefix} {unit_label} ranking",
            selector_columns=[
                column
                for column in [*group_columns, "reducer"]
                if column in best_units.columns
            ],
            default_sort=(
                {"column": sort_columns[0], "direction": "desc"}
                if sort_columns
                else None
            ),
            page_size=5,
        )
    )

    sweep_df = unit_runs.loc[
        :,
        [
            column
            for column in [
                *group_columns,
                unit_column,
                "reducer",
                "n_components",
                *sort_columns,
            ]
            if column in unit_runs.columns
        ],
    ].copy()
    acc = AccordionElement("Show Full Unit Sweep Data", open=False)
    acc.add_element(
        InteractiveTableElement(
            sweep_df.round(4),
            title=f"{title_prefix} {unit_label} sweep",
            selector_columns=[
                column
                for column in [*group_columns, unit_column, "reducer"]
                if column in sweep_df.columns
            ],
            page_size=5,
        )
    )
    section.add_element(acc)

    plot_metric = sort_columns[0] if sort_columns else "trustworthiness"
    baseline = None

    if "eval_name" in best_units.columns and best_units["eval_name"].notna().any():
        eval_values: list[Any] = list(best_units["eval_name"].dropna().unique())
    else:
        eval_values = [None]

    perf_tabs: dict[str, Element] = {}
    stab_tabs: dict[str, Element] = {}
    topo_tabs: dict[str, Element] = {}

    for eval_name in eval_values:
        tab_label = str(eval_name) if eval_name is not None else "All"
        peak_rows = (
            best_units
            if eval_name is None
            else best_units[best_units["eval_name"] == eval_name]
        )
        sweep_rows = (
            sweep_df
            if eval_name is None or "eval_name" not in sweep_df.columns
            else sweep_df[sweep_df["eval_name"] == eval_name]
        )

        perf_elem = _unit_peak_bar(
            peak_rows,
            metric=plot_metric,
            unit_column=unit_column,
            unit_label=unit_label,
            title=f"{title_prefix} best {plot_metric} by {unit_label}",
            baseline=baseline,
        )
        if perf_elem is not None:
            perf_tabs[tab_label] = perf_elem

        stab_elem = _unit_stability_box(
            sweep_rows,
            metric=plot_metric,
            unit_column=unit_column,
            unit_label=unit_label,
            title=f"{title_prefix} stability by {unit_label}",
            baseline=baseline,
        )
        if stab_elem is not None:
            stab_tabs[tab_label] = stab_elem

        if unit_label == "sensor":
            topo_elem = _unit_topomaps(
                peak_rows,
                unit_column=unit_column,
                plot_metric=plot_metric,
                title_prefix=title_prefix,
                trace_label=tab_label,
            )
            if topo_elem is not None:
                topo_tabs[tab_label] = topo_elem

    viz_tabs = {}
    if perf_tabs:
        viz_tabs["Peak Performance"] = (
            next(iter(perf_tabs.values()))
            if len(perf_tabs) == 1
            else TabsElement(perf_tabs)
        )
    if stab_tabs:
        viz_tabs["Hyperparameter Stability"] = (
            next(iter(stab_tabs.values()))
            if len(stab_tabs) == 1
            else TabsElement(stab_tabs)
        )
    if topo_tabs:
        viz_tabs["Spatial Topomaps"] = (
            next(iter(topo_tabs.values()))
            if len(topo_tabs) == 1
            else TabsElement(topo_tabs)
        )

    if viz_tabs:
        section.add_element(TabsElement(viz_tabs))


def build_nonflat_condition_section(
    condition: str,
    condition_runs: pd.DataFrame,
    eval_frame: pd.DataFrame,
    ctx: DimReductionReportContext,
) -> Section:
    """Build the per-condition section for non-flat modes (family/sensor/etc.)."""
    from coco_pipe.viz.interactive.base import plot_scatter

    fam_label = ctx.family_label
    section = Section(condition, icon="📊")
    artifacts = {
        str(row["fit_id"]): load_fit_artifact(ctx.output_root / row["artifact_path"])
        for _, row in condition_runs.iterrows()
    }
    unit_label = ctx.unit_label(ctx.analysis_mode.replace("_", " "))
    intro = f"Primary analysis unit: **{unit_label}**"
    callout_text = f"Input mode: **{ctx.input_mode}**<br/>{intro}"
    if fam_label:
        callout_text += f"<br/>Descriptor families: **{fam_label}**"
    section.add_element(
        CalloutElement(callout_text, kind="info", title="Configuration Details")
    )
    section.add_element(
        StatCardElement("Successful Fits", len(condition_runs), color="green")
    )

    merged = condition_runs.merge(
        _eval_merge_frame(eval_frame),
        on="fit_id",
        how="left",
    )
    if ctx.analysis_mode == "family":
        build_unit_summary(
            section,
            merged,
            ctx,
            unit_label="family",
            title_prefix=condition,
        )
        family_container = ctx.container_builder(condition)

        reducer_tabs = {}
        for reducer_name in ctx.reducers:
            reducer_runs = merged[merged["reducer"] == reducer_name].copy()
            if reducer_runs.empty:
                continue

            tab_section = ContainerElement()

            comparison_metrics = _available_eval_metrics(reducer_runs)
            for m in ["trustworthiness", "continuity"]:
                if m in reducer_runs.columns:
                    comparison_metrics.append(m)

            curve_frames = []
            for family, family_runs in reducer_runs.groupby("family", dropna=False):
                family_best_by_n = _sort_by_score(family_runs, ctx.selection_metric)
                family_best_by_n = (
                    family_best_by_n.groupby("n_components", dropna=False)
                    .head(1)
                    .sort_values("n_components")
                )
                for metric_name in comparison_metrics:
                    metric_df = family_best_by_n.dropna(subset=[metric_name])
                    if metric_df.empty:
                        continue
                    curve_frames.append(
                        pd.DataFrame(
                            {
                                "n_components": metric_df["n_components"].to_numpy(),
                                "score": metric_df[metric_name].to_numpy(),
                                "series": f"{family}: {metric_name}",
                            }
                        )
                    )

            acc = AccordionElement(
                f"Show {reducer_name} Hyperparameter Sweep Details", open=False
            )
            if curve_frames:
                fig = plot_scatter(
                    pd.concat(curve_frames, ignore_index=True),
                    x="n_components",
                    y="score",
                    color="series",
                    mode="lines+markers",
                    title=(
                        f"{condition} - {reducer_name} family comparison "
                        "vs n_components"
                    ),
                    xaxis_title="n_components",
                    yaxis_title="score",
                )
                acc.add_element(PlotlyElement(fig))

            sweep_cols = ["family", "n_components"]
            if "eval_name" in reducer_runs.columns:
                sweep_cols.append("eval_name")
            for metric in EVAL_METRIC_COLUMNS:
                if metric in reducer_runs.columns:
                    sweep_cols.append(metric)
            for gm in ["trustworthiness", "continuity"]:
                if gm in reducer_runs.columns:
                    sweep_cols.append(gm)
            sweep_cols = list(dict.fromkeys(sweep_cols))

            acc.add_element(
                InteractiveTableElement(
                    reducer_runs.loc[:, sweep_cols].round(4),
                    title=f"{condition} - {reducer_name} family sweep",
                    selector_columns=["family", "eval_name"]
                    if "eval_name" in sweep_cols
                    else ["family"],
                    page_size=5,
                )
            )
            tab_section.add_element(acc)
            reducer_tabs[reducer_name] = tab_section

        if reducer_tabs:
            section.add_element(TabsElement(reducer_tabs))

        family_tabs = {}
        for family, family_runs in merged.groupby("family", dropna=False):
            best_row = _sort_by_score(family_runs, ctx.selection_metric).iloc[0]
            best_artifact = artifacts[str(best_row["fit_id"])]
            family_meta = build_meta_dict(family_container, best_artifact["ids"], ctx)

            fam_container = ContainerElement()
            fam_container.add_element(
                CalloutElement(
                    f"Best fit for family **{family}** uses **{best_row['reducer']}** "
                    f"with n={int(best_row['n_components'])}",
                    kind="tip",
                )
            )

            plots_elem = build_best_fit_plots(
                f"{condition} - {family}",
                best_artifact,
                family_meta,
                ctx,
                feature_names=_feature_names(family_container),
            )
            if plots_elem:
                fam_container.add_element(plots_elem)

            sweep_cols = ["reducer", "n_components"]
            if "eval_name" in family_runs.columns:
                sweep_cols.append("eval_name")
            for metric in EVAL_METRIC_COLUMNS:
                if metric in family_runs.columns:
                    sweep_cols.append(metric)
            for gm in ["trustworthiness", "continuity"]:
                if gm in family_runs.columns:
                    sweep_cols.append(gm)
            sweep_cols = list(dict.fromkeys(sweep_cols))

            acc = AccordionElement("Show Reducer Sweep Details", open=False)
            acc.add_element(
                InteractiveTableElement(
                    family_runs.loc[:, sweep_cols].round(4),
                    title=f"{condition} - {family} reducer/n sweep",
                    selector_columns=["reducer", "eval_name"]
                    if "eval_name" in sweep_cols
                    else ["reducer"],
                    page_size=5,
                )
            )
            fam_container.add_element(acc)
            family_tabs[str(family)] = fam_container

        if family_tabs:
            section.add_element(TabsElement(family_tabs))

        return section
    if ctx.analysis_mode in {"sensor_within_family", "sensor_within_subfamily"}:
        group_columns = ["family"]
        if (
            ctx.analysis_mode == "sensor_within_subfamily"
            and "subfamily" in merged.columns
        ):
            group_columns.append("subfamily")
        for group_key, family_runs in merged.groupby(group_columns, dropna=False):
            group_values = group_key if isinstance(group_key, tuple) else (group_key,)
            group_label = " / ".join(str(value) for value in group_values)
            section.add_markdown(f"### {group_label}")
            build_unit_summary(
                section,
                family_runs,
                ctx,
                unit_label="sensor",
                title_prefix=f"{condition} - {group_label}",
            )
        return section

    build_unit_summary(
        section,
        merged,
        ctx,
        unit_label=unit_label,
        title_prefix=condition,
    )
    if ctx.analysis_mode != "family":
        top_n = 2 if ctx.analysis_mode == "sensor" else 1
        sensor_container = ctx.container_builder(condition)

        reducer_tabs = {}
        for reducer_name in ctx.reducers:
            reducer_runs = merged[merged["reducer"] == reducer_name].copy()
            if reducer_runs.empty:
                continue

            tab_section = ContainerElement()

            sensor_group_columns = [
                column
                for column in ["eval_name", "target_col"]
                if column in reducer_runs.columns and reducer_runs[column].notna().any()
            ]
            sensor_groups = (
                list(reducer_runs.groupby(sensor_group_columns, dropna=False))
                if sensor_group_columns
                else [((), reducer_runs)]
            )
            for group_key, group_df in sensor_groups:
                best_rows_df = _sort_by_score(group_df, ctx.selection_metric).head(
                    top_n
                )

                for rank, (_, best_row) in enumerate(best_rows_df.iterrows(), 1):
                    best_artifact = artifacts[str(best_row["fit_id"])]
                    sensor_meta = build_meta_dict(
                        sensor_container, best_artifact["ids"], ctx
                    )
                    if not isinstance(group_key, tuple):
                        group_key = (group_key,)
                    label_parts = [
                        str(value)
                        for value in group_key
                        if pd.notna(value) and str(value) != ""
                    ]
                    label_suffix = (
                        f" [{' / '.join(label_parts)}]" if label_parts else ""
                    )

                    rank_prefix = f"#{rank} " if top_n > 1 else ""
                    n_comp = int(best_row["n_components"])
                    tab_section.add_element(
                        CalloutElement(
                            f"**{reducer_name}{label_suffix}** {rank_prefix}best unit: "
                            f"{best_row['unit_name']} (n={n_comp})",
                            kind="tip",
                        )
                    )

                    title = (
                        f"{condition} - {reducer_name}{label_suffix} - "
                        f"Rank {rank} ({best_row['unit_name']})"
                    )
                    plots_elem = build_best_fit_plots(
                        title,
                        best_artifact,
                        sensor_meta or {},
                        ctx,
                        feature_names=_feature_names(sensor_container),
                    )
                    if plots_elem:
                        tab_section.add_element(plots_elem)

            reducer_tabs[reducer_name] = tab_section

        if reducer_tabs:
            section.add_element(TabsElement(reducer_tabs))

    return section


def build_pooled_section(
    pooled_runs: pd.DataFrame,
    pooled_eval_runs: pd.DataFrame,
    ctx: DimReductionReportContext,
    pooled_container: Any | None = None,
) -> Section | None:
    """Build the pooled multi-condition section, or None when there are no runs."""
    from coco_pipe.viz.interactive.base import plot_scatter

    if pooled_runs.empty:
        return None
    fam_label = ctx.family_label
    section = Section("Pooled Multi-condition", icon="🌐")
    artifacts = {
        str(row["fit_id"]): load_fit_artifact(ctx.output_root / row["artifact_path"])
        for _, row in pooled_runs.iterrows()
    }
    callout_text = (
        "Shared fits across all requested conditions.<br/>Condition-separation scores "
        "show EO vs EC-style pooled separability when available."
    )
    if fam_label:
        callout_text += f"<br/>Descriptor families: **{fam_label}**"
    section.add_element(
        CalloutElement(callout_text, kind="info", title="Pooled Configuration")
    )
    section.add_element(
        StatCardElement("Pooled Fits", len(pooled_runs), color="purple")
    )
    merged = pooled_runs.merge(
        _eval_merge_frame(pooled_eval_runs),
        on="fit_id",
        how="left",
    )
    if pooled_container is None:
        with contextlib.suppress(Exception):
            from coco_pipe.io.structures import DataContainer

            source_containers = [ctx.container_builder(cond) for cond in ctx.conditions]
            valid_containers = [c for c in source_containers if c is not None]
            if valid_containers:
                pooled_container = DataContainer.concat(valid_containers)

    if ctx.analysis_mode == "family":
        build_unit_summary(
            section,
            merged,
            ctx,
            unit_label="family",
            title_prefix="Pooled",
        )
        reducer_tabs = {}
        for reducer_name in ctx.reducers:
            reducer_runs = merged[merged["reducer"] == reducer_name].copy()
            if reducer_runs.empty:
                continue

            tab_section = ContainerElement()
            comparison_metrics = [
                metric
                for metric in [
                    *EVAL_METRIC_COLUMNS,
                    "trustworthiness",
                    "continuity",
                    "shepard_correlation",
                ]
                if metric in reducer_runs.columns
            ]
            comparison_metrics = list(dict.fromkeys(comparison_metrics))
            curve_frames = []
            for family, family_runs in reducer_runs.groupby("family", dropna=False):
                family_best_by_n = (
                    _sort_by_score(family_runs, ctx.selection_metric)
                    .groupby("n_components", dropna=False)
                    .head(1)
                    .sort_values("n_components")
                )
                for metric_name in comparison_metrics:
                    metric_df = family_best_by_n.dropna(subset=[metric_name])
                    if metric_df.empty:
                        continue
                    curve_frames.append(
                        pd.DataFrame(
                            {
                                "n_components": metric_df["n_components"].to_numpy(),
                                "score": metric_df[metric_name].to_numpy(),
                                "series": f"{family}: {metric_name}",
                            }
                        )
                    )
            if curve_frames:
                fig = plot_scatter(
                    pd.concat(curve_frames, ignore_index=True),
                    x="n_components",
                    y="score",
                    color="series",
                    mode="lines+markers",
                    title=f"Pooled - {reducer_name} family comparison vs n_components",
                    xaxis_title="n_components",
                    yaxis_title="score",
                )
                acc = AccordionElement(
                    "Show Family Comparison Sweep Curves", open=False
                )
                acc.add_element(PlotlyElement(fig))

                sweep_cols = ["reducer", "n_components"]
                if "eval_name" in reducer_runs.columns:
                    sweep_cols.append("eval_name")
                for metric in EVAL_METRIC_COLUMNS:
                    if metric in reducer_runs.columns:
                        sweep_cols.append(metric)
                for gm in ["trustworthiness", "continuity"]:
                    if gm in reducer_runs.columns:
                        sweep_cols.append(gm)
                sweep_cols = list(dict.fromkeys(sweep_cols))
                acc.add_element(
                    InteractiveTableElement(
                        reducer_runs.loc[
                            :, [c for c in sweep_cols if c in reducer_runs.columns]
                        ].round(4),
                        title=f"Pooled - {reducer_name} sweep",
                        selector_columns=["reducer", "eval_name"]
                        if "eval_name" in sweep_cols
                        else ["reducer"],
                        page_size=5,
                    )
                )
                tab_section.add_element(acc)
            reducer_tabs[reducer_name] = tab_section

        if reducer_tabs:
            section.add_element(CalloutElement("Family Sweeps by Reducer", kind="info"))
            section.add_element(TabsElement(reducer_tabs))

        family_tabs = {}
        top_n = 2
        for family, family_runs in merged.groupby("family", dropna=False):
            family_best = _sort_by_score(family_runs, ctx.selection_metric).head(top_n)

            fam_container = ContainerElement()
            for rank, (_, best_row) in enumerate(family_best.iterrows(), 1):
                best_artifact = artifacts[str(best_row["fit_id"])]
                pool_meta = (
                    build_meta_dict(pooled_container, best_artifact["ids"], ctx)
                    if pooled_container is not None
                    else {}
                )

                rank_prefix = f"#{rank} " if top_n > 1 else ""
                n_comp = int(best_row["n_components"])
                fam_container.add_element(
                    CalloutElement(
                        f"**{family}** {rank_prefix}best fit uses "
                        f"**{best_row['reducer']}** with n={n_comp}",
                        kind="tip",
                    )
                )

                plots_elem = build_best_fit_plots(
                    f"Pooled - {family} - Rank {rank}",
                    best_artifact,
                    pool_meta,
                    ctx,
                    feature_names=_feature_names(pooled_container)
                    if pooled_container
                    else None,
                )
                if plots_elem:
                    fam_container.add_element(plots_elem)
            if fam_container.elements:
                family_tabs[str(family)] = fam_container

        if family_tabs:
            section.add_element(TabsElement(family_tabs))

        return section

    if ctx.analysis_mode == "flat":
        acc = AccordionElement("Show Hyperparameter Sweep Tables", open=False)
        sweep_cols = ["reducer", "n_components"]
        if "eval_name" in merged.columns:
            sweep_cols.append("eval_name")
        for metric in EVAL_METRIC_COLUMNS:
            if metric in merged.columns:
                sweep_cols.append(metric)
        for gm in ["trustworthiness", "continuity"]:
            if gm in merged.columns:
                sweep_cols.append(gm)
        sweep_cols = list(dict.fromkeys(sweep_cols))
        acc.add_element(
            InteractiveTableElement(
                merged.loc[:, [c for c in sweep_cols if c in merged.columns]].round(4),
                title="Pooled fit ranking",
                selector_columns=["reducer", "eval_name"]
                if "eval_name" in sweep_cols
                else ["reducer"],
                default_sort={
                    "column": _selection_or_fallback(
                        merged, ctx.selection_metric, SEPARATION_METRIC_KEY
                    ),
                    "direction": "desc",
                },
                page_size=5,
            )
        )
        section.add_element(acc)

        reducer_tabs = {}
        for reducer_name in ctx.reducers:
            reducer_runs = merged[merged["reducer"] == reducer_name].copy()
            if reducer_runs.empty:
                continue
            best_row = _sort_by_score(reducer_runs, ctx.selection_metric).iloc[0]
            best_artifact = artifacts[str(best_row["fit_id"])]
            pool_meta = (
                build_meta_dict(pooled_container, best_artifact["ids"], ctx)
                if pooled_container is not None
                else {}
            )

            plots_elem = build_best_fit_plots(
                f"Pooled - {reducer_name}",
                best_artifact,
                pool_meta,
                ctx,
                feature_names=_feature_names(pooled_container)
                if pooled_container
                else None,
            )
            if plots_elem:
                reducer_tabs[f"{reducer_name} (n={int(best_row['n_components'])})"] = (
                    plots_elem
                )
        if reducer_tabs:
            section.add_element(TabsElement(reducer_tabs))

        return section

    unit_label = ctx.unit_label("analysis unit")
    build_unit_summary(
        section,
        merged,
        ctx,
        unit_label=unit_label,
        title_prefix="Pooled",
    )

    top_n = 2 if ctx.analysis_mode == "sensor" else 1
    reducer_tabs = {}
    for reducer_name in ctx.reducers:
        reducer_runs = merged[merged["reducer"] == reducer_name].copy()
        if reducer_runs.empty:
            continue

        tab_section = ContainerElement()
        sensor_group_columns = [
            column
            for column in ["eval_name", "target_col"]
            if column in reducer_runs.columns and reducer_runs[column].notna().any()
        ]
        sensor_groups = (
            list(reducer_runs.groupby(sensor_group_columns, dropna=False))
            if sensor_group_columns
            else [((), reducer_runs)]
        )

        for group_key, group_df in sensor_groups:
            best_rows_df = _sort_by_score(group_df, ctx.selection_metric).head(top_n)

            for rank, (_, best_row) in enumerate(best_rows_df.iterrows(), 1):
                best_artifact = artifacts[str(best_row["fit_id"])]
                pool_meta = (
                    build_meta_dict(pooled_container, best_artifact["ids"], ctx)
                    if pooled_container is not None
                    else {}
                )

                label_parts = [
                    str(value)
                    for value in group_key
                    if pd.notna(value) and str(value) != ""
                ]
                label_suffix = f" [{' / '.join(label_parts)}]" if label_parts else ""

                rank_prefix = f"#{rank} " if top_n > 1 else ""
                unit_name = best_row.get("unit_name", "unit")
                n_comp = int(best_row["n_components"])
                tab_section.add_element(
                    CalloutElement(
                        f"**{reducer_name}{label_suffix}** {rank_prefix}best unit: "
                        f"{unit_name} (n={n_comp})",
                        kind="tip",
                    )
                )

                title = (
                    f"Pooled - {reducer_name}{label_suffix} - "
                    f"Rank {rank} ({best_row.get('unit_name', 'unit')})"
                )
                plots_elem = build_best_fit_plots(
                    title,
                    best_artifact,
                    pool_meta,
                    ctx,
                    feature_names=_feature_names(pooled_container)
                    if pooled_container
                    else None,
                )
                if plots_elem:
                    tab_section.add_element(plots_elem)

        reducer_tabs[reducer_name] = tab_section

    if reducer_tabs:
        section.add_element(TabsElement(reducer_tabs))

    return section


def build_data_availability_summary(
    overview_sec: Section,
    ctx: DimReductionReportContext,
    dataset_stats: list[dict[str, Any]] | None,
    fit_runs_df: pd.DataFrame,
) -> None:
    """Append a data-availability accordion to *overview_sec* (no-op if empty)."""
    scopes_conditions = [("condition", c) for c in ctx.conditions]
    if ctx.run_pooled:
        scopes_conditions.append(("pooled", ctx.pooled_condition))

    stats_map = {(s.get("scope"), s.get("condition")): s for s in (dataset_stats or [])}

    rows = []
    for scope, condition in scopes_conditions:
        stat_dict = stats_map.get((scope, condition), {})

        condition_runs = pd.DataFrame()
        if not fit_runs_df.empty:
            condition_runs = fit_runs_df[
                (fit_runs_df["scope"] == scope)
                & (fit_runs_df["condition"] == condition)
                & (fit_runs_df["status"] == "success")
            ]

        if not stat_dict and condition_runs.empty:
            continue

        reducers_str = ""
        n_comps_str = ""
        if not condition_runs.empty:
            if "reducer" in condition_runs.columns:
                reducers_str = ", ".join(
                    sorted(condition_runs["reducer"].dropna().astype(str).unique())
                )
            if "n_components" in condition_runs.columns:
                n_comps_str = ", ".join(
                    map(
                        str,
                        sorted(
                            condition_runs["n_components"].dropna().astype(int).unique()
                        ),
                    )
                )

        rows.append(
            {
                "scope": scope,
                "condition": condition,
                "loaded_observations": stat_dict.get("loaded_observations", ""),
                "samples_used": stat_dict.get("samples_used", ""),
                "unique_subjects": stat_dict.get("unique_subjects", ""),
                "unique_recordings": stat_dict.get("unique_recordings", ""),
                "successful_fits": len(condition_runs),
                "reducers": reducers_str,
                "valid_n_components": n_comps_str,
            }
        )

    if rows:
        acc = AccordionElement("Show Data Availability", open=False)
        acc.add_element(
            InteractiveTableElement(
                pd.DataFrame(rows), title="Data Availability", page_size=10
            )
        )
        overview_sec.add_element(acc)


def build_failure_sections(
    report: Report,
    fit_runs_df: pd.DataFrame,
    eval_runs_df: pd.DataFrame,
    ctx: DimReductionReportContext,
) -> None:
    """Append fit/eval failure sections to *report* when failures exist."""
    fit_failures = split_by_status(fit_runs_df)[1]
    if not fit_failures.empty:
        failures_sec = Section("Fit Failures", icon="⚠️")
        acc = AccordionElement("Show Fit Failures", open=False)
        acc.add_element(
            InteractiveTableElement(
                fit_failures.loc[
                    :,
                    [
                        column
                        for column in ctx.fit_failure_columns
                        if column in fit_failures.columns
                    ],
                ],
                title="Failed fits",
                selector_columns=[
                    column
                    for column in [
                        "scope",
                        "condition",
                        "family",
                        "unit_name",
                        "reducer",
                    ]
                    if column in fit_failures.columns
                ],
                default_sort={"column": "condition", "direction": "asc"}
                if "condition" in fit_failures.columns
                else None,
                page_size=5,
            )
        )
        failures_sec.add_element(acc)
        report.add_section(failures_sec)

    eval_failures = split_by_status(eval_runs_df)[1]
    if not eval_failures.empty:
        failures_sec = Section("Eval Failures", icon="⚠️")
        acc = AccordionElement("Show Eval Failures", open=False)
        acc.add_element(
            InteractiveTableElement(
                eval_failures.loc[
                    :,
                    [
                        column
                        for column in ctx.eval_failure_columns
                        if column in eval_failures.columns
                    ],
                ],
                title="Failed evals",
                selector_columns=[
                    column
                    for column in [
                        "scope",
                        "condition",
                        "family",
                        "unit_name",
                        "reducer",
                        "eval_name",
                    ]
                    if column in eval_failures.columns
                ],
                default_sort={"column": "condition", "direction": "asc"}
                if "condition" in eval_failures.columns
                else None,
                page_size=5,
            )
        )
        failures_sec.add_element(acc)
        report.add_section(failures_sec)


def build_dataset_report(
    ctx: DimReductionReportContext,
    *,
    fit_runs_path: Path,
    eval_runs_path: Path,
    containers_by_scope: dict[tuple[str, str], Any] | None = None,
    dataset_stats: list[dict[str, Any]] | None = None,
    overview_extras: Callable[[Section], None] | None = None,
) -> Report:
    """Assemble the full per-dataset dimensionality-reduction report.

    Builds the Overview (config + data availability + best-run cards, plus any
    *overview_extras* such as a cohort summary), QC sections, evaluation
    results, condition ranking, per-condition sections, the pooled section and
    failure tables. All study-specific policy is carried by *ctx*.
    """
    fit_runs_df = pd.DataFrame(load_fit_runs(fit_runs_path))

    if eval_runs_path.exists():
        eval_runs_df = pd.DataFrame(
            json.loads(eval_runs_path.read_text(encoding="utf-8"))
        )
    else:
        eval_runs_df = pd.DataFrame()

    if not eval_runs_df.empty and ctx.eval_specs:
        wanted_eval_names = {spec["name"] for spec in ctx.eval_specs}
        if "eval_name" in eval_runs_df.columns:
            eval_runs_df = eval_runs_df[
                eval_runs_df["eval_name"].isin(wanted_eval_names)
            ]

    available_eval_metrics = [
        col for col in EVAL_METRIC_COLUMNS if col in eval_runs_df.columns
    ]
    if eval_runs_df.empty or not available_eval_metrics:
        eval_frame = pd.DataFrame()
    else:
        eval_base_cols = [
            "fit_id",
            "scope",
            "condition",
            "analysis_mode",
            "family",
            "unit_name",
            "eval_name",
            "target_col",
            "reducer",
            "n_components",
        ]
        cols_to_keep = [
            c
            for c in [*eval_base_cols, *available_eval_metrics]
            if c in eval_runs_df.columns
        ]
        eval_frame = eval_runs_df.loc[
            eval_runs_df["status"] == "success", cols_to_keep
        ].copy()

    fit_success = split_by_status(fit_runs_df)[0]
    eval_merge_cols = ["fit_id", "eval_name", "target_col", *available_eval_metrics]

    if not eval_frame.empty:
        eval_subset = eval_frame.loc[
            :, [c for c in eval_merge_cols if c in eval_frame.columns]
        ]
        fit_eval_ranking = fit_success.merge(eval_subset, on="fit_id", how="left")
    else:
        empty_evals = pd.DataFrame(columns=eval_merge_cols)
        fit_eval_ranking = fit_success.merge(empty_evals, on="fit_id", how="left")

    fam_label = ctx.family_label
    report = Report(title=ctx.report_title)

    overview_sec = Section("Overview", icon="📋")
    config_cards = [
        StatCardElement("Dataset", ctx.dataset_name, color="blue"),
        StatCardElement("Input Mode", ctx.input_mode, color="purple"),
        StatCardElement("Analysis Mode", ctx.analysis_mode, color="indigo"),
    ]
    if ctx.representation:
        config_cards.append(
            StatCardElement("Representation", ctx.representation, color="cyan")
        )

    overview_sec.add_element(ColumnsElement(config_cards, cols=len(config_cards)))

    config_html = (
        f"**Conditions**: {', '.join(ctx.conditions)}<br/>"
        f"**Reducers**: {', '.join(ctx.reducers)}"
    )
    if fam_label:
        config_html += f"<br/>**Families**: {fam_label}"
    overview_sec.add_element(
        CalloutElement(config_html, kind="info", title="Run Configuration Details")
    )

    build_data_availability_summary(overview_sec, ctx, dataset_stats, fit_runs_df)

    # Best overall run per condition — shared coco-pipe card layout.
    add_reduction_best_run_cards(
        overview_sec,
        fit_eval_ranking,
        selection_metric=ctx.selection_metric,
    )

    if overview_extras is not None:
        overview_extras(overview_sec)
    report.add_section(overview_sec)

    if containers_by_scope:
        for (scope, condition), container in containers_by_scope.items():
            qc_result = (container.meta or {}).get("qc_result")
            if qc_result is None:
                continue
            qc_section = build_qc_section(qc_result)
            qc_section.title = f"Data Quality (QC): {scope} / {condition}"
            report.add_section(qc_section)

    eval_conditions = list(ctx.conditions)
    if ctx.run_pooled and ctx.pooled_condition:
        eval_conditions.append(ctx.pooled_condition)
    eval_sec = build_reduction_eval_results_section(
        eval_frame,
        selection_metric=ctx.selection_metric,
        eval_name_order=[
            str(spec["name"]) for spec in ctx.eval_specs if spec.get("name")
        ],
        conditions=eval_conditions,
    )
    if eval_sec is not None:
        report.add_section(eval_sec)

    # Condition ranking table + cross-condition bar + reducer radar.
    condition_runs = fit_eval_ranking[
        (fit_eval_ranking["scope"] == "condition")
        & (fit_eval_ranking["status"] == "success")
    ].copy()
    ranking_sec = build_reduction_condition_ranking_section(
        condition_runs,
        conditions=ctx.conditions,
        reducers=ctx.reducers,
        selection_metric=ctx.selection_metric,
    )
    if ranking_sec is not None:
        report.add_section(ranking_sec)

    # --- Per-condition sections ---
    section_builder = (
        build_flat_condition_section
        if ctx.analysis_mode == "flat"
        else build_nonflat_condition_section
    )
    for condition in ctx.conditions:
        condition_fit_runs = fit_runs_df[
            (fit_runs_df["scope"] == "condition")
            & (fit_runs_df["condition"] == condition)
            & (fit_runs_df["status"] == "success")
        ].copy()
        if condition_fit_runs.empty:
            continue
        report.add_section(
            section_builder(
                condition,
                condition_fit_runs,
                eval_frame[eval_frame["condition"] == condition].copy()
                if not eval_frame.empty
                else pd.DataFrame(),
                ctx,
            )
        )

    if ctx.run_pooled:
        pooled_runs = fit_runs_df[
            (fit_runs_df["scope"] == "pooled")
            & (fit_runs_df["condition"] == ctx.pooled_condition)
            & (fit_runs_df["status"] == "success")
        ].copy()
        pooled_eval = (
            eval_frame[
                (eval_frame["scope"] == "pooled")
                & (eval_frame["condition"] == ctx.pooled_condition)
            ].copy()
            if not eval_frame.empty
            else pd.DataFrame()
        )
        pooled_container = None
        if containers_by_scope is not None:
            pooled_container = containers_by_scope.get(("pooled", ctx.pooled_condition))
        pooled_section = build_pooled_section(
            pooled_runs, pooled_eval, ctx, pooled_container=pooled_container
        )
        if pooled_section is not None:
            report.add_section(pooled_section)

    build_failure_sections(report, fit_runs_df, eval_runs_df, ctx)

    return report


__all__ = [
    "DEFAULT_REDUCTION_TIE_BREAKERS",
    "DimReductionReportContext",
    "add_reduction_best_run_cards",
    "build_best_fit_plots",
    "build_data_availability_summary",
    "build_dataset_report",
    "build_failure_sections",
    "build_flat_condition_section",
    "build_meta_dict",
    "build_nonflat_condition_section",
    "build_pooled_section",
    "build_reduction_condition_ranking_section",
    "build_reduction_eval_results_section",
    "build_reduction_rollup_report",
    "build_unit_summary",
    "merge_fit_eval",
    "rank_reduction_runs",
]
