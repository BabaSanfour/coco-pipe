"""Records/artifact-based dimensionality-reduction sweep reporting.

The dim-reduction counterpart to :mod:`coco_pipe.report.decoding_sweep`: helpers
that operate on persisted fit/eval run artifacts (the
:mod:`coco_pipe.dim_reduction` schema) to merge and rank runs and to assemble a
cross-mode roll-up report. The per-reduction section builders live in
:mod:`coco_pipe.report.dim_reduction`; study-specific policy (cohort metadata,
scope ordering, topomaps, report links) stays in the consuming project.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from coco_pipe.dim_reduction import SEPARATION_METRIC_KEY

from .core import Report, Section
from .elements import (
    AccordionElement,
    CalloutElement,
    ColumnsElement,
    InteractiveTableElement,
    PlotlyElement,
    StatCardElement,
    TableElement,
    TabsElement,
)
from .tables import best_rows, split_by_status

# Rank on structure preservation (trustworthiness, continuity) then class
# separation. ``(column, ascending)`` tuples; all descending here.
DEFAULT_REDUCTION_TIE_BREAKERS: tuple[tuple[str, bool], ...] = (
    ("trustworthiness", False),
    ("continuity", False),
    (SEPARATION_METRIC_KEY, False),
)

_DEFAULT_ROLLUP_SELECTORS = (
    "analysis_mode",
    "scope",
    "condition",
    "reducer",
    "model",
    "eval_name",
)


def merge_fit_eval(
    fit_runs: Any,
    eval_runs: Any = None,
    *,
    eval_name: str | None = None,
    metrics: Sequence[str] = (SEPARATION_METRIC_KEY,),
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
    metrics: Sequence[str] = (SEPARATION_METRIC_KEY,),
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
    (SEPARATION_METRIC_KEY, "Separation", "blue"),
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
    sort_metric = (
        selection_metric if selection_metric in success.columns else fallback_metric
    )
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
) -> Section | None:
    """Build a collapsible post-hoc evaluation-results table, or None if empty."""
    frame = pd.DataFrame(eval_frame)
    if frame.empty:
        return None
    resolved_sort = sort_col or (
        SEPARATION_METRIC_KEY
        if SEPARATION_METRIC_KEY in frame.columns
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
    return section


def _condition_reducer_bar(
    runs: pd.DataFrame,
    *,
    conditions: Sequence[str],
    reducers: Sequence[str],
    metric: str,
    tie_breakers: Sequence[tuple[str, bool]],
) -> PlotlyElement | None:
    """Grouped bar of the best *metric* per (condition, reducer)."""
    if runs.empty or len(conditions) <= 1 or metric not in runs.columns:
        return None
    frames = []
    for reducer in reducers:
        sub = runs[runs["reducer"] == reducer].copy()
        if sub.empty:
            continue
        best = best_rows(sub, ["condition"], metric, tie_breakers=tie_breakers)[0]
        best = best.set_index("condition").reindex(list(conditions)).reset_index()
        best["reducer"] = reducer
        best["n_label"] = (
            best["n_components"].map(
                lambda value: f"n={int(value)}" if pd.notna(value) else ""
            )
            if "n_components" in best.columns
            else ""
        )
        frames.append(best)
    if not frames:
        return None
    from coco_pipe.viz.interactive.base import plot_grouped_bar

    fig = plot_grouped_bar(
        pd.concat(frames, ignore_index=True),
        x="condition",
        y=metric,
        group="reducer",
        text="n_label",
        x_order=list(conditions),
        title=f"Best {metric} per (condition, reducer)",
        xaxis_title="condition",
        yaxis_title=metric,
        legend_title="Reducer",
    )
    return PlotlyElement(fig)


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
    tie_breakers: Sequence[tuple[str, bool]] = DEFAULT_REDUCTION_TIE_BREAKERS,
    ranking_columns: Sequence[str] = (
        "condition",
        "family",
        "unit_name",
        "reducer",
        "n_components",
        "trustworthiness",
        "continuity",
        "eval_name",
        SEPARATION_METRIC_KEY,
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
        SEPARATION_METRIC_KEY,
    ),
    default_sort: str = "trustworthiness",
    title: str = "Condition Ranking",
) -> Section | None:
    """Build the condition-ranking table + cross-condition bar + reducer radar.

    *condition_runs* are the successful condition-scope runs (the caller owns the
    scope/status filter). The section shows a sortable ranking table plus, when
    there are multiple conditions/reducers, a best-per-(condition, reducer) bar
    and a reducer-profile radar. Returns None when *condition_runs* is empty.
    """
    runs = pd.DataFrame(condition_runs)
    if runs.empty:
        return None
    columns = [column for column in ranking_columns if column in runs.columns]
    section = Section(title, icon="🏁")
    section.add_element(
        InteractiveTableElement(
            runs.loc[:, columns].round(4),
            title="Condition ranking",
            selector_columns=[
                column for column in selector_columns if column in columns
            ],
            default_sort={"column": default_sort, "direction": "desc"},
            page_size=5,
        )
    )
    metric = selection_metric if selection_metric in runs.columns else fallback_metric
    tabs: dict[str, Any] = {}
    bar = _condition_reducer_bar(
        runs,
        conditions=conditions,
        reducers=reducers,
        metric=metric,
        tie_breakers=tie_breakers,
    )
    if bar is not None:
        tabs["Cross-Condition Summary"] = bar
    radar = _reducer_radar(runs, reducers=reducers, radar_metrics=radar_metrics)
    if radar is not None:
        tabs["Reducer Profile"] = radar
    if tabs:
        section.add_element(TabsElement(tabs))
    return section


def build_reduction_rollup_report(
    leaderboard: Any,
    *,
    title: str,
    x_metric: str = "trustworthiness",
    y_metric: str = SEPARATION_METRIC_KEY,
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


__all__ = [
    "DEFAULT_REDUCTION_TIE_BREAKERS",
    "add_reduction_best_run_cards",
    "build_reduction_condition_ranking_section",
    "build_reduction_eval_results_section",
    "build_reduction_rollup_report",
    "merge_fit_eval",
    "rank_reduction_runs",
]
