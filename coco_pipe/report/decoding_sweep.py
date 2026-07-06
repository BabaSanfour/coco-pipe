"""Decoding sweep report: cross-result comparison engine + sweep orchestration.

Owns both the comparison primitives (:class:`ResultCollection`, result loading,
comparison figures, :func:`build_comparison_section`, :func:`build_result_tabs`,
:func:`make_experiment_results_report`) and the sweep-report layer built on top
of them: the scientific overview, leaderboards, failures and feature-selection
sections, the taxonomy recipe builders (flat/sensor/grouped/descriptor), the
shared :func:`make_decoding_sweep_report` skeleton, and the head-to-head
comparison report.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from coco_pipe.decoding import ExperimentResult, stamp_primary_metric
from coco_pipe.io.units import split_unit_sensor
from coco_pipe.viz import (
    info_from_montage,
    plot_decoding_topomap,
    plot_distribution_groups,
    plot_head_to_head,
    plot_heatmap,
    plot_paired_delta,
)

from ._constants import (
    CV_SIGNATURE_COLUMNS,
    DEFAULT_BEST_RESULT_GROUP_COLUMNS,
    DEFAULT_MODE_RESULT_CONTEXT_COLUMNS,
    MAX_INLINE_DIAGNOSTIC_RESULTS,
    PRIMARY_TIE_BREAKERS,
)
from ._utils import _ensure_static_matplotlib_backend, _figure_element
from .core import Report, Section
from .decoding import build_decoding_sections
from .elements import (
    AccordionElement,
    CalloutElement,
    ColumnsElement,
    ImageElement,
    InteractiveTableElement,
    StatCardElement,
    TableElement,
    TabsElement,
)
from .qc import build_qc_section
from .tables import (
    best_rows,
    display_frame,
    ensure_primary_metric,
    make_cv_signature,
    relabel_columns,
    selector_columns,
    signature_compatibility,
    sort_by_metrics,
    split_by_status,
)

# ---------------------------------------------------------------------------
# Cross-result comparison engine (formerly decoding_comparison)
# ---------------------------------------------------------------------------


@dataclass
class ResultCollection:
    """A labelled collection of decoding results and their tidy summaries.

    Attributes
    ----------
    by
        Ordered axis names that label the collection. The order is significant:
        the **first** axis is treated as the primary comparison axis by helpers
        that need a default (e.g. it seeds the default comparison axis in
        :func:`make_experiment_results_report`). ``results``/``contexts`` are
        keyed by tuples aligned to this order.
    results
        Loaded result objects keyed by the axis-value tuple.
    contexts
        Per-result context metadata keyed by the same axis-value tuple.
    summary
        Concatenated tidy per-result summary rows.
    """

    by: tuple[str, ...]
    results: dict[tuple[Any, ...], Any]
    contexts: dict[tuple[Any, ...], dict[str, Any]]
    summary: pd.DataFrame

    def filter(self, **criteria: Any) -> ResultCollection:
        """Return a collection restricted to exact context matches."""
        keys = {
            key
            for key, context in self.contexts.items()
            if all(context.get(name) == value for name, value in criteria.items())
        }
        results = {key: value for key, value in self.results.items() if key in keys}
        contexts = {key: value for key, value in self.contexts.items() if key in keys}
        frame = self.summary.copy()
        for name, value in criteria.items():
            if name in frame:
                frame = frame[frame[name] == value]
        return ResultCollection(
            self.by, results, contexts, frame.reset_index(drop=True)
        )

    @property
    def successful_summary(self) -> pd.DataFrame:
        """Return only successfully loaded result summaries."""
        if "_status" not in self.summary:
            return self.summary.copy()
        return self.summary[self.summary["_status"] == "success"].copy()


def collect_results(
    items: Iterable[tuple[Mapping[str, Any], Any]],
    *,
    by: Sequence[str],
    metrics: Sequence[str] | None = None,
) -> ResultCollection:
    """Load labelled results and concatenate their scalar summaries."""
    axes = tuple(by)
    if not axes:
        raise ValueError("by must contain at least one context field.")

    results: dict[tuple[Any, ...], Any] = {}
    contexts: dict[tuple[Any, ...], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for raw_context, source in items:
        context = dict(raw_context)
        missing = [name for name in axes if name not in context]
        if missing:
            raise ValueError(f"Result context is missing axis fields: {missing}")
        key = tuple(context[name] for name in axes)
        if key in contexts:
            previous = contexts[key].get("_source")
            current = str(source)
            if previous == current:
                continue
            raise ValueError(f"Duplicate result key {key!r} has multiple sources.")
        context["_source"] = str(source)
        contexts[key] = context
        try:
            # Duck-type on ``summary()`` so already-loaded, in-memory result-like
            # objects are accepted, not only ExperimentResult instances or paths.
            if isinstance(source, ExperimentResult) or (
                hasattr(source, "summary") and callable(source.summary)
            ):
                result, source_path = source, None
            else:
                path = Path(source).expanduser()
                if path.is_dir():
                    path = path / "result.joblib"
                result, source_path = ExperimentResult.load(path), str(path)

            summary = pd.DataFrame(result.summary())
            if summary.index.name:
                summary = summary.reset_index()
            elif "Model" not in summary and len(getattr(result, "raw", {})) == len(
                summary
            ):
                summary = summary.copy()
                summary.insert(0, "Model", list(result.raw))
            if metrics is not None:
                prefixes = tuple(f"{metric}_" for metric in metrics)
                keep = [
                    column
                    for column in summary
                    if column == "Model" or column.startswith(prefixes)
                ]
                summary = summary.loc[:, keep]
            results[key] = result
            for record in summary.to_dict("records"):
                rows.append(
                    stamp_primary_metric(
                        {
                            **context,
                            **record,
                            "_status": "success",
                            "_error": None,
                            "_source": source_path,
                        },
                        metrics,
                    )
                )
            if summary.empty:
                rows.append(
                    {
                        **context,
                        "_status": "empty",
                        "_error": "result.summary() returned no rows",
                        "_source": source_path,
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    **context,
                    "_status": "failed",
                    "_error": f"{type(exc).__name__}: {exc}",
                    "_source": str(source),
                }
            )
    return ResultCollection(axes, results, contexts, pd.DataFrame(rows))


@dataclass(frozen=True)
class _ComparisonContext:
    """Resolved plotting parameters shared by per-kind figure builders."""

    kind: str
    axis: str | None
    row: str | None
    column: str
    value: str
    plot_center: float | None
    info: Any
    coords: Any
    title: str | None


def _group_title(label: str, title: str | None) -> str | None:
    """Return the per-group figure title (fall back to the section title)."""
    return label if label != "Comparison" else title


def _heatmap_figure(group: pd.DataFrame, label: str, ctx: _ComparisonContext) -> Any:
    """Axis/grid heatmap of *value* over an axis x model matrix (None if absent)."""
    index_name = ctx.row or ctx.axis
    if index_name is None or index_name not in group or ctx.column not in group:
        return None
    matrix = group.pivot_table(
        index=index_name,
        columns=ctx.column,
        values=ctx.value,
        aggfunc="mean",
    )
    if matrix.empty:
        return None
    return plot_heatmap(
        matrix,
        annotate=ctx.kind == "axis_heatmap",
        annotation_format=".2f",
        cmap="RdBu_r" if ctx.plot_center is not None else "viridis",
        center=ctx.plot_center,
        minimum_half_range=0.02 if ctx.plot_center is not None else 0.0,
        colorbar_label=ctx.value,
        title=_group_title(label, ctx.title),
        xlabel=ctx.column,
        ylabel=index_name,
        figsize=(
            max(6.0, 1.2 * matrix.shape[1] + 4.0),
            max(5.0, 0.4 * matrix.shape[0] + 2.0),
        ),
    )


def _sensor_topomap_figure(
    group: pd.DataFrame, label: str, ctx: _ComparisonContext
) -> Any:
    """Topomap of *value* across sensors (None without a sensor axis or montage)."""
    sensor_axis = ctx.axis or "unit_name"
    if sensor_axis not in group or (ctx.info is None and ctx.coords is None):
        return None
    sensor_frame = group[[sensor_axis, ctx.value]].rename(
        columns={sensor_axis: "FeatureName"}
    )
    return plot_decoding_topomap(
        sensor_frame,
        ctx.value,
        info=ctx.info,
        coords=ctx.coords,
        center=ctx.plot_center,
        title=_group_title(label, ctx.title),
    )


def _model_bars_figure(group: pd.DataFrame, label: str, ctx: _ComparisonContext) -> Any:
    """Mean +/- std bars of *value* per axis level (None if the axis is absent)."""
    label_axis = ctx.axis or ctx.column
    if label_axis not in group:
        return None
    summary = (
        group.groupby(label_axis, dropna=False)[ctx.value]
        .agg(["mean", "std"])
        .reset_index()
    )
    return plot_head_to_head(
        summary,
        label=label_axis,
        value="mean",
        error="std",
        reference=ctx.plot_center,
        title=_group_title(label, ctx.title) or "Comparison",
        ylabel=ctx.value,
    )


def _spread_figure(group: pd.DataFrame, label: str, ctx: _ComparisonContext) -> Any:
    """Distribution of *value* per axis level (None if the axis is absent)."""
    spread_axis = ctx.axis or ctx.column
    if spread_axis not in group:
        return None
    grouped = list(group.groupby(spread_axis, dropna=False, sort=False))
    return plot_distribution_groups(
        [values[ctx.value] for _, values in grouped],
        [name for name, _ in grouped],
        title=_group_title(label, ctx.title),
        ylabel=ctx.value,
    )


def _metric_matrix_figure(
    group: pd.DataFrame, label: str, ctx: _ComparisonContext
) -> Any:
    """Heatmap of all mean metrics per axis level (None without metric columns)."""
    index_name = ctx.axis or ctx.column
    metrics = [
        column
        for column in group
        if column.endswith("_mean")
        and pd.to_numeric(group[column], errors="coerce").notna().any()
    ]
    if index_name not in group or not metrics:
        return None
    matrix = group.groupby(index_name, dropna=False)[metrics].mean()
    matrix.columns = [name.removesuffix("_mean") for name in matrix.columns]
    return plot_heatmap(
        matrix,
        annotate=True,
        annotation_format=".2f",
        center=ctx.plot_center,
        minimum_half_range=0.02 if ctx.plot_center is not None else 0.0,
        cmap="RdBu_r" if ctx.plot_center is not None else "viridis",
        colorbar_label="score",
        title=_group_title(label, ctx.title),
        xlabel="metric",
        ylabel=index_name,
    )


def _paired_delta_figure(
    group: pd.DataFrame, label: str, ctx: _ComparisonContext
) -> Any:
    """Paired metric deltas with optional confidence intervals."""
    delta_axis = ctx.axis or "comparison"
    return plot_paired_delta(
        group,
        label=delta_axis,
        delta=ctx.value,
        lower="ci_lower" if "ci_lower" in group else None,
        upper="ci_upper" if "ci_upper" in group else None,
        title=_group_title(label, ctx.title) or "Paired Delta",
    )


_COMPARISON_FIGURE_BUILDERS: dict[
    str, Callable[[pd.DataFrame, str, _ComparisonContext], Any]
] = {
    "axis_heatmap": _heatmap_figure,
    "grid_heatmap": _heatmap_figure,
    "sensor_topomap": _sensor_topomap_figure,
    "model_bars": _model_bars_figure,
    "head_to_head": _model_bars_figure,
    "spread": _spread_figure,
    "metric_matrix": _metric_matrix_figure,
    "paired_delta": _paired_delta_figure,
}
"""Dispatch from comparison ``kind`` to its per-group figure builder."""


def build_comparison_section(
    collection: ResultCollection,
    *,
    kind: str,
    axis: str | None = None,
    row: str | None = None,
    column: str = "Model",
    value: str = "accuracy_mean",
    group_by: Sequence[str] = (),
    center: float | None = None,
    info: Any = None,
    coords: Any = None,
    title: str | None = None,
    include_table: bool = False,
    on_error: str = "warn",
) -> Section | None:
    """Build one reusable cross-result comparison section.

    Notes
    -----
    Unlike :func:`build_decoding_sections`, ``on_error`` here accepts only
    ``"raise"`` or ``"warn"`` (not ``"placeholder"``). A comparison renders many
    per-group figures into one section, so a single group's failure is warned and
    skipped rather than replaced with a placeholder card.
    """
    _ensure_static_matplotlib_backend()
    if on_error not in {"raise", "warn"}:
        raise ValueError(
            "on_error must be 'raise' or 'warn' (placeholder is not supported for "
            "comparison sections)."
        )
    frame = collection.successful_summary
    if frame.empty:
        return None
    if value in frame:
        frame[value] = pd.to_numeric(frame[value], errors="coerce")
        frame = frame[frame[value].notna()]
    if frame.empty:
        return None

    section = Section(title=title or kind.replace("_", " ").title())
    if include_table:
        section.add_element(
            InteractiveTableElement(
                frame.drop(columns=["_error"], errors="ignore"),
                title="Comparison Data",
                selector_columns=[
                    item
                    for item in (*collection.by, "Model", "_status")
                    if item in frame
                ],
            )
        )

    plot_center = (
        0.5
        if center is None and value.startswith(("accuracy", "balanced_accuracy"))
        else center
    )
    ctx = _ComparisonContext(
        kind=kind,
        axis=axis,
        row=row,
        column=column,
        value=value,
        plot_center=plot_center,
        info=info,
        coords=coords,
        title=title,
    )
    tabs: dict[str, ImageElement] = {}
    group_columns = [column for column in group_by if column in frame]

    if not group_columns:
        group_iterator = [("Comparison", frame)]
    else:
        group_iterator = (
            (
                " / ".join(
                    str(value)
                    for value in (keys if isinstance(keys, tuple) else (keys,))
                ),
                group,
            )
            for keys, group in frame.groupby(group_columns, dropna=False, sort=False)
        )

    for label, group in group_iterator:
        # Guard, not dispatch: paired deltas are only meaningful when every row
        # shares a CV/cohort design. We veto incompatible groups here and skip.
        if kind == "paired_delta":
            mismatches = [
                column
                for column in CV_SIGNATURE_COLUMNS
                if column in group and group[column].dropna().nunique() > 1
            ]
            if mismatches:
                message = (
                    f"Paired delta {label!r} was not plotted because "
                    f"{', '.join(mismatches)} differ across rows."
                )
                if on_error == "raise":
                    raise ValueError(message)
                section.add_element(
                    CalloutElement(
                        message,
                        kind="warning",
                        title="Incompatible comparison design",
                    )
                )
                continue
        try:
            builder = _COMPARISON_FIGURE_BUILDERS.get(kind)
            if builder is None:
                raise ValueError(f"Unknown decoding comparison kind: {kind!r}")
            figure = builder(group, label, ctx)
        except (ImportError, TypeError, ValueError) as exc:
            if on_error == "raise":
                raise
            warnings.warn(
                f"Decoding comparison {kind!r} for {label!r} failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if figure is None:
            continue
        tabs[label] = _figure_element(
            figure,
            f"{kind.replace('_', ' ')} comparison: {label}",
        )

    if len(tabs) == 1:
        section.add_element(next(iter(tabs.values())))
    elif tabs:
        section.add_element(TabsElement(tabs))
    if not section.children:
        return None
    return section


def build_result_tabs(
    collection: ResultCollection,
    *,
    sections: str | Sequence[str] = "compact",
    feature_metadata: pd.DataFrame | None = None,
    info: Any = None,
    coords: Any = None,
    interactive: bool = False,
    on_error: str = "warn",
) -> TabsElement | AccordionElement | None:
    """Build nested compact per-result sections from a collection."""
    tabs: dict[str, AccordionElement] = {}
    for key, result in collection.results.items():
        context = collection.contexts[key]
        label = " / ".join(str(context[name]) for name in collection.by)
        content = AccordionElement("Show result diagnostics", open=True)
        for section in build_decoding_sections(
            result,
            sections=sections,
            feature_metadata=feature_metadata,
            info=info,
            coords=coords,
            interactive=interactive,
            on_error=on_error,
        ):
            content.add_element(section)
        if content.children:
            tabs[label] = content
    if len(tabs) == 1:
        return next(iter(tabs.values()))
    if tabs:
        return TabsElement(tabs)
    return None


def _showcase_specs(by: Sequence[str]) -> list[dict[str, Any]]:
    """Return the multi-view comparison spec set for the ``"showcase"`` preset.

    Renders a model comparison plus per-axis heatmap / score-spread / metric-matrix
    views keyed on the first context axis, so a multi-experiment sweep (e.g. across
    cohorts, sensors, or feature spaces) is navigable in a single report.
    """
    specs: list[dict[str, Any]] = [
        {
            "kind": "model_bars",
            "axis": "Model",
            "title": "Model Comparison",
            "include_table": True,
        }
    ]
    axis = next(iter(by), None)
    if axis is not None:
        specs += [
            {
                "kind": "axis_heatmap",
                "row": axis,
                "column": "Model",
                "title": f"{axis} x Model",
            },
            {"kind": "spread", "axis": axis, "title": f"Score Spread by {axis}"},
            {
                "kind": "metric_matrix",
                "axis": axis,
                "title": f"Metric Matrix by {axis}",
            },
        ]
    return specs


def make_experiment_results_report(
    items: Iterable[tuple[Mapping[str, Any], Any]],
    *,
    by: Sequence[str],
    comparisons: str | Sequence[Mapping[str, Any]] = "default",
    per_result: str | Sequence[str] | None = "compact",
    nest: bool = True,
    feature_metadata: pd.DataFrame | None = None,
    info: Any = None,
    coords: Any = None,
    interactive: bool = False,
    title: str = "Decoding Comparison",
    config: Mapping[str, Any] | None = None,
    asset_urls: Mapping[str, str] | str | None = None,
    qc_result: Any = None,
    output_path: str | Path | None = None,
    on_error: str = "warn",
) -> Report:
    """Build a comparison report over many labelled decoding results.

    Parameters
    ----------
    comparisons
        ``"default"`` (a single model-comparison view), ``"showcase"`` (model
        comparison plus per-axis heatmap, score spread, and metric matrix keyed
        on the first ``by`` axis), a single comparison ``kind`` string, or an
        explicit sequence of comparison specs.
    interactive
        If True, per-result diagnostic sections render interactive Plotly figures
        (cross-result comparison figures remain static Matplotlib images).
    """
    collection = collect_results(items, by=by)
    report = Report(
        title=title,
        config=config,
        asset_urls=asset_urls,
    )
    if qc_result is not None:
        report.add_section(build_qc_section(qc_result))
    failed = collection.summary[
        collection.summary.get("_status", pd.Series(dtype=object)) != "success"
    ]
    if not failed.empty:
        section = Section(title="Collection Status")
        section.status = "WARN"
        section.add_element(
            CalloutElement(
                f"{len(failed)} result(s) could not be summarized.",
                kind="warning",
            )
        )
        section.add_element(InteractiveTableElement(failed, title="Load Failures"))
        report.add_section(section)

    if comparisons == "default":
        specs = _showcase_specs(by)[:1]
    elif comparisons == "showcase":
        specs = _showcase_specs(by)
    elif isinstance(comparisons, str):
        specs = [{"kind": comparisons}]
    else:
        specs = comparisons

    for spec in specs:
        section = build_comparison_section(
            collection,
            info=info,
            coords=coords,
            **dict(spec),
        )
        if section is not None:
            report.add_section(section)

    if nest and per_result is not None:
        nested = build_result_tabs(
            collection,
            sections=per_result,
            feature_metadata=feature_metadata,
            info=info,
            coords=coords,
            interactive=interactive,
            on_error=on_error,
        )
        if nested is not None:
            section = Section(title="Per-Result Diagnostics")
            section.add_element(nested)
            report.add_section(section)
    if output_path is not None:
        report.save(output_path)
    return report


# ---------------------------------------------------------------------------
# Sweep-report layer
# ---------------------------------------------------------------------------


def collect_mode_results(
    mode_frame: pd.DataFrame,
    *,
    context_columns: Sequence[str] = DEFAULT_MODE_RESULT_CONTEXT_COLUMNS,
) -> ResultCollection | None:
    """Collect unique successful result artifacts from a sweep result frame."""
    if "output_dir" not in mode_frame.columns:
        return None
    successful = split_by_status(mode_frame)[0]
    successful = successful[successful["output_dir"].notna()].copy()
    if successful.empty:
        return None
    context = [
        column
        for column in context_columns
        if column in successful and successful[column].notna().any()
    ]
    if not context:
        return None
    unique = successful[[*context, "output_dir"]].drop_duplicates()
    items = [
        (
            {column: row[column] for column in context},
            Path(str(row["output_dir"])) / "result.joblib",
        )
        for row in unique.to_dict("records")
    ]
    return collect_results(items, by=context)


def best_result_tabs(
    frame: pd.DataFrame,
    primary_metric: str,
    *,
    group_columns: Sequence[str] = DEFAULT_BEST_RESULT_GROUP_COLUMNS,
    context_columns: Sequence[str] = DEFAULT_MODE_RESULT_CONTEXT_COLUMNS,
    feature_metadata: pd.DataFrame | None = None,
    sections: str | Sequence[str] = "compact",
    limit: int = MAX_INLINE_DIAGNOSTIC_RESULTS,
    on_error: str = "placeholder",
) -> TabsElement | AccordionElement | None:
    """Render diagnostics for only the top-N ranked sweep rows."""
    ranked, _ = best_rows(
        frame,
        group_columns,
        primary_metric,
        tie_breakers=PRIMARY_TIE_BREAKERS,
    )
    if ranked.empty or "output_dir" not in ranked:
        return None
    top = ranked[ranked["output_dir"].notna()].head(limit)
    if top.empty:
        return None
    collection = collect_mode_results(top, context_columns=context_columns)
    if collection is None:
        return None
    return build_result_tabs(
        collection,
        sections=sections,
        feature_metadata=feature_metadata,
        on_error=on_error,
    )


def hp_tuning_section(
    frame: pd.DataFrame,
    *,
    primary_metric: str = "primary_metric",
    feature_metadata: pd.DataFrame | None = None,
    title: str = "Hyperparameter Tuning",
    limit: int = MAX_INLINE_DIAGNOSTIC_RESULTS,
) -> Section | None:
    """Surface hyperparameter-search diagnostics for the top-ranked runs.

    The classical and foundation reports both include this section so HP tuning
    is reported the same way regardless of model family. Loads the best runs'
    artifacts and renders their search / best-parameter diagnostics; returns
    None when no run carries a hyperparameter search (``on_error="warn"`` drops
    non-tuned runs instead of emitting placeholders).
    """
    diagnostics = best_result_tabs(
        frame,
        primary_metric,
        feature_metadata=feature_metadata,
        sections=("tuning",),
        limit=limit,
        on_error="warn",
    )
    if diagnostics is None:
        return None
    section = Section(title)
    section.add_markdown(
        "Hyperparameter search results and best parameters for the top-ranked "
        "runs. Runs without a hyperparameter search are omitted."
    )
    section.add_element(diagnostics)
    return section


def summary_collection(
    mode_frame: pd.DataFrame,
    primary_metric: str = "primary_metric",
    *,
    descriptor_family: Mapping[str, str] | None = None,
    include_sensor: bool = False,
    context_columns: Sequence[str] = DEFAULT_MODE_RESULT_CONTEXT_COLUMNS,
) -> ResultCollection | None:
    """Build a lightweight comparison collection from persisted sweep metrics."""
    summary = split_by_status(mode_frame)[0]
    if primary_metric not in summary:
        return None
    summary = summary[summary[primary_metric].notna()].copy()
    if summary.empty:
        return None

    if "model" in summary and "Model" not in summary:
        summary["Model"] = summary["model"]

    if descriptor_family is not None and "unit_name" not in summary:
        return None
    if descriptor_family is not None:
        summary["descriptor_family"] = (
            summary["unit_name"].map(dict(descriptor_family)).fillna("Other")
        )

    if include_sensor:
        required = {"unit_name", "unit_key"}
        if not required.issubset(summary.columns):
            return None
        summary["sensor"] = [
            split_unit_sensor(str(unit_name), str(unit_key))
            for unit_name, unit_key in zip(
                summary["unit_name"],
                summary["unit_key"],
                strict=False,
            )
        ]
        summary = summary[summary["sensor"].notna()]
        if summary.empty:
            return None

    by = tuple(
        column
        for column in context_columns
        if column in summary and summary[column].notna().any()
    )
    if not by:
        return None
    return ResultCollection(by=by, results={}, contexts={}, summary=summary)


def add_scientific_overview(
    report: Report,
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    kind: str,
    strategy_note: str,
    empty_message: str | None = None,
) -> None:
    """Add high-level decoding-sweep coverage cards and strategy text."""
    section = Section("Scientific Overview")
    if frame.empty:
        section.add_markdown(
            empty_message
            or (
                "No foundation decoding units were produced."
                if kind == "foundation"
                else "No decoding analysis units were produced."
            )
        )
        report.add_section(section)
        return

    success = split_by_status(frame)[0]
    best_score = (
        success["primary_metric"].max()
        if "primary_metric" in success and not success["primary_metric"].dropna().empty
        else None
    )
    targets = frame["target"].dropna().nunique() if "target" in frame else 0
    scopes = (
        frame["scope"].dropna().nunique()
        if "scope" in frame
        else frame["condition"].dropna().nunique()
        if "condition" in frame
        else 0
    )
    model_column = (
        "model_key" if kind == "foundation" and "model_key" in frame else "model"
    )
    models = frame[model_column].dropna().nunique() if model_column in frame else 0
    selection_or_mode = (
        frame["train_mode"].dropna().nunique()
        if kind == "foundation" and "train_mode" in frame
        else frame["selection_mode"].dropna().nunique()
        if "selection_mode" in frame
        else 0
    )

    cards = [
        StatCardElement("Dataset", dataset_name, color="blue"),
        StatCardElement("Targets", targets, color="purple"),
        StatCardElement("Scopes", scopes, color="green"),
        StatCardElement("Models", models, color="yellow"),
        StatCardElement(
            "Train Modes" if kind == "foundation" else "Feature Selection",
            selection_or_mode,
            color="purple",
        ),
    ]
    if best_score is not None:
        cards.append(StatCardElement("Best Score", float(best_score), color="green"))
    section.add_element(ColumnsElement(cards, cols=min(len(cards), 4)))

    status_counts = (
        frame["status"].value_counts() if "status" in frame else pd.Series(dtype=int)
    )
    section.add_element(
        TableElement(
            pd.DataFrame(
                {
                    "Metric": ["Analysis rows", "Successful", "Skipped", "Failed"],
                    "Value": [
                        len(frame),
                        int(status_counts.get("success", 0)),
                        int(status_counts.get("skipped", 0)),
                        int(status_counts.get("failed", 0)),
                    ],
                }
            ),
            title="Run Coverage",
        )
    )
    section.add_element(
        CalloutElement(strategy_note, kind="info", title="Report Strategy")
    )
    report.add_section(section)


def leaderboard_section(
    frame: pd.DataFrame,
    *,
    title: str,
    table_title: str,
    filters: Mapping[str, Any],
    comparison_axis: str,
    group_by: Sequence[str],
) -> Section | None:
    """Build a primary leaderboard plus a compact model/bar comparison."""
    if any(column not in frame.columns for column in filters):
        return None
    subset = frame.copy()
    for column, value in filters.items():
        subset = subset[subset[column] == value]
    ranked, metric = best_rows(
        subset,
        ("scope", "condition", "target", comparison_axis),
        primary_metric="primary_metric",
    )
    if ranked.empty or metric is None:
        return None

    section = Section(title)
    display = display_frame(
        sort_by_metrics(
            ranked,
            "primary_metric",
            tie_breakers=PRIMARY_TIE_BREAKERS,
        ),
        order=(
            "scope",
            "condition",
            "target",
            comparison_axis,
            "model",
            "model_key",
            "train_mode",
            "selection_mode",
            "primary_metric_name",
            "primary_metric",
            "p_value",
            "p_value_fdr",
            "significant_fdr",
            "cv_signature",
            "cohort_signature",
        ),
    )
    section.add_element(
        InteractiveTableElement(
            display,
            title=table_title,
            selector_columns=selector_columns(
                display,
                relabel_columns(
                    (
                        "scope",
                        "condition",
                        "target",
                        comparison_axis,
                        "model",
                        "model_key",
                        "train_mode",
                        "selection_mode",
                        "significant_fdr",
                    )
                ),
            ),
            default_sort={
                "column": relabel_columns(["primary_metric"])[0],
                "direction": "desc",
            },
            page_size=10,
        )
    )

    collection_summary = ranked.copy()
    if "model" in collection_summary and "Model" not in collection_summary:
        collection_summary["Model"] = collection_summary["model"]
    collection_summary["_status"] = "success"

    axes = tuple(c for c in (*group_by, comparison_axis) if c in collection_summary)
    if not axes:
        axes = ("Model",) if "Model" in collection_summary else ("primary_metric_name",)
    comparison = build_comparison_section(
        ResultCollection(by=axes, results={}, contexts={}, summary=collection_summary),
        kind="model_bars",
        axis=comparison_axis,
        group_by=tuple(column for column in group_by if column in ranked),
        value="primary_metric",
        center=0.5,
        title=f"{title}: score comparison",
    )
    if comparison is not None:
        section.add_element(comparison)
    return section


def failures_section(frame: pd.DataFrame) -> Section | None:
    """Build a collapsible skipped/failed unit table."""
    if frame.empty or "status" not in frame:
        return None
    failures = frame[frame["status"].astype(str) != "success"].copy()
    if failures.empty:
        return None
    section = Section("Skipped and Failed Units")
    acc = AccordionElement("Show skipped and failed decoding units", open=False)
    display = display_frame(
        failures,
        order=(
            "scope",
            "condition",
            "target",
            "analysis_mode",
            "unit_name",
            "model",
            "model_key",
            "train_mode",
            "selection_mode",
            "status",
            "reason",
        ),
    )
    acc.add_element(
        InteractiveTableElement(
            display,
            title="Skipped and failed units",
            selector_columns=selector_columns(
                display,
                relabel_columns(
                    (
                        "scope",
                        "condition",
                        "target",
                        "analysis_mode",
                        "model",
                        "model_key",
                        "train_mode",
                        "selection_mode",
                        "status",
                    )
                ),
            ),
            page_size=10,
        )
    )
    section.add_element(acc)
    return section


def feature_selection_section(
    frame: pd.DataFrame,
    *,
    feature_metadata: pd.DataFrame | None,
    baseline_label: str = "baseline",
    selection_title: str = "Feature-Selection Diagnostics",
) -> Section | None:
    """Render SFS / non-baseline feature-selection diagnostics."""
    if frame.empty or "selection_mode" not in frame:
        return None
    status = frame.get("status", pd.Series(index=frame.index, dtype=object))
    selected = frame[
        frame["selection_mode"].notna()
        & (frame["selection_mode"].astype(str) != baseline_label)
        & (status == "success")
    ].copy()
    if selected.empty:
        return None

    section = Section(selection_title)
    ranked = sort_by_metrics(
        selected,
        "primary_metric",
        tie_breakers=PRIMARY_TIE_BREAKERS,
    )
    display = display_frame(
        ranked,
        order=(
            "scope",
            "target",
            "analysis_mode",
            "unit_name",
            "family",
            "subfamily",
            "model",
            "selection_mode",
            "primary_metric_name",
            "primary_metric",
            "p_value_fdr",
            "significant_fdr",
        ),
    )
    section.add_element(
        InteractiveTableElement(
            display,
            title="SFS / feature-selection leaderboard",
            selector_columns=selector_columns(
                display,
                relabel_columns(
                    (
                        "scope",
                        "target",
                        "analysis_mode",
                        "unit_name",
                        "model",
                        "selection_mode",
                        "significant_fdr",
                    )
                ),
            ),
            default_sort={
                "column": relabel_columns(["primary_metric"])[0],
                "direction": "desc",
            },
            page_size=10,
        )
    )

    diagnostics = best_result_tabs(
        selected,
        "primary_metric",
        feature_metadata=feature_metadata,
        sections=("features",),
    )
    if diagnostics is not None:
        acc = AccordionElement("Show top selected-feature diagnostics", open=False)
        acc.add_element(diagnostics)
        section.add_element(acc)
    return section


def flat_section(
    mode_frame: pd.DataFrame,
    *,
    scope_label: str,
    feature_metadata: pd.DataFrame | None,
    kind: str,
    title_prefix: str,
    group_by: tuple[str, ...],
    **axis_kwargs: Any,
) -> Any:
    """Build comparison and diagnostics tabs for flat decoding units."""
    collection = collect_mode_results(mode_frame)
    if collection is None:
        return None

    tabs: dict[str, Any] = {}
    model_comparison = build_comparison_section(
        collection,
        kind=kind,
        group_by=group_by,
        value="primary_metric",
        center=0.5,
        title=f"{title_prefix} Comparison: {scope_label}",
        include_table=True,
        **axis_kwargs,
    )
    if model_comparison is not None:
        tabs["Model Comparison"] = model_comparison

    diagnostics = build_result_tabs(
        collection,
        sections="full",
        feature_metadata=feature_metadata,
        on_error="placeholder",
    )
    if diagnostics is not None:
        tabs["Per-Result Diagnostics"] = diagnostics

    return TabsElement(tabs) if len(tabs) > 1 else next(iter(tabs.values()), None)


def sensor_section(
    mode_frame: pd.DataFrame,
    *,
    scope_label: str,
    montage: str = "standard_1020",
    **kwargs: Any,
) -> Any:
    """Build sensor-level comparison views and optional topomap."""
    _ = kwargs
    collection = collect_mode_results(mode_frame)
    if collection is None:
        return None

    tabs: dict[str, Any] = {}
    heatmap = build_comparison_section(
        collection,
        kind="axis_heatmap",
        axis="unit_name",
        column="Model",
        group_by=("target", "selection_mode"),
        value="primary_metric",
        center=0.5,
        title=f"Sensor x Model Accuracy: {scope_label}",
        include_table=True,
    )
    if heatmap is not None:
        tabs["Model Heatmap"] = heatmap
    spread = build_comparison_section(
        collection,
        kind="spread",
        axis="unit_name",
        group_by=("target", "selection_mode", "Model"),
        value="primary_metric",
        center=0.5,
        title=f"Sensor Score Spread: {scope_label}",
    )
    if spread is not None:
        tabs["Score Spread"] = spread

    topomap_source = mode_frame.copy()
    topomap_group_cols = [
        column
        for column in ("target", "selection_mode", "model")
        if column in topomap_source
    ]
    if topomap_group_cols:
        topomap_source = topomap_source.groupby(
            topomap_group_cols,
            dropna=False,
            sort=False,
        ).filter(lambda group: group["unit_name"].dropna().nunique() >= 3)
    if not topomap_source.empty and topomap_source["unit_name"].dropna().nunique() >= 3:
        topomap_collection = collect_mode_results(topomap_source)
        if topomap_collection is not None:
            sensors = list(dict.fromkeys(str(v) for v in topomap_source["unit_name"]))
            topomap = build_comparison_section(
                topomap_collection,
                kind="sensor_topomap",
                value="primary_metric",
                center=0.5,
                info=info_from_montage(sensors, montage=montage),
                title=f"Sensor-wise Accuracy: {scope_label}",
            )
            if topomap is not None:
                tabs["Spatial Topomap"] = topomap
    return TabsElement(tabs) if len(tabs) > 1 else next(iter(tabs.values()), None)


def grouped_section(
    mode_frame: pd.DataFrame,
    *,
    scope_label: str,
    group_col: str,
    kind: str,
    title_prefix: str,
    group_by: tuple[str, ...],
    **axis_kwargs: Any,
) -> Any:
    """Build a grouped taxonomy comparison over an existing sweep frame."""
    axis_kwargs.pop("feature_metadata", None)
    collection = collect_mode_results(mode_frame)
    if collection is None or (group_col == "family" and "family" not in mode_frame):
        return None

    return build_comparison_section(
        collection,
        kind=kind,
        group_by=group_by,
        value="primary_metric",
        center=0.5,
        title=f"{title_prefix} Accuracy: {scope_label}",
        include_table=True,
        **axis_kwargs,
    )


def descriptor_section(
    mode_frame: pd.DataFrame,
    *,
    scope_label: str,
    feature_metadata: pd.DataFrame | None,
    kind: str,
    title_prefix: str,
    group_by: tuple[str, ...],
    descriptor_family: Mapping[str, str] | None = None,
    include_sensor: bool = False,
    **axis_kwargs: Any,
) -> Any:
    """Build descriptor-level summary comparisons plus top diagnostics."""
    collection = summary_collection(
        mode_frame,
        descriptor_family=descriptor_family,
        include_sensor=include_sensor,
    )
    if collection is None:
        return None

    tabs: dict[str, Any] = {}
    heatmap = build_comparison_section(
        collection,
        kind=kind,
        group_by=group_by,
        value="primary_metric",
        center=0.5,
        title=f"{title_prefix} Accuracy: {scope_label}",
        include_table=True,
        **axis_kwargs,
    )
    if heatmap is not None:
        tabs["Descriptor Heatmap"] = heatmap

    diagnostics = best_result_tabs(
        mode_frame,
        "primary_metric",
        feature_metadata=feature_metadata,
        sections="compact",
    )
    if diagnostics is not None:
        tabs["Top Diagnostics"] = diagnostics

    return TabsElement(tabs) if len(tabs) > 1 else next(iter(tabs.values()), None)


def paired_delta_vs_baseline(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    baseline_family: str,
    family_column: str = "comparison_family",
    label_column: str = "model_label",
    metric_column: str = "primary_metric",
    pair_column: str = "comparison_pair",
    delta_column: str = "comparison_delta",
    compatible_lookup: Mapping[tuple[Any, ...], bool] | None = None,
) -> pd.DataFrame:
    """Compute paired metric deltas against the best baseline row per group."""
    if frame.empty:
        return pd.DataFrame()
    required = [*group_columns, family_column, label_column, metric_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"paired_delta_vs_baseline missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    group_names = list(group_columns)
    for raw_keys, group in frame.groupby(group_names, dropna=False, sort=False):
        keys = raw_keys if isinstance(raw_keys, tuple) else (raw_keys,)
        key_lookup = tuple(keys)
        if compatible_lookup is not None and not compatible_lookup.get(
            key_lookup,
            True,
        ):
            continue
        baseline = group[group[family_column] == baseline_family].copy()
        if baseline.empty:
            continue
        baseline_row = sort_by_metrics(
            baseline,
            metric_column,
            tie_breakers=PRIMARY_TIE_BREAKERS,
        ).iloc[0]
        for _, row in group.iterrows():
            if row[family_column] == baseline_family:
                continue
            output = dict(zip(group_names, keys, strict=False))
            output.update(
                {
                    pair_column: (
                        f"{row[family_column]} | {row[label_column]} - "
                        f"{baseline_row[label_column]}"
                    ),
                    delta_column: float(row[metric_column])
                    - float(baseline_row[metric_column]),
                    metric_column: float(row[metric_column]),
                    "baseline_metric": float(baseline_row[metric_column]),
                    "_status": "success",
                }
            )
            rows.append(output)
    return pd.DataFrame(rows)


def prepare_sweep_frame(
    records: Iterable[Mapping[str, Any]],
    *,
    add_cv_signature: bool = True,
    scope_from: str | None = "condition",
    default_scope: str | None = "all",
) -> pd.DataFrame:
    """Build the tidy sweep frame shared by every decoding sweep report.

    Concatenates *records*, stamps the primary metric, attaches a readable
    ``cv_signature`` when derivable, and guarantees a ``scope`` column: taken
    from *scope_from* when present, else filled with *default_scope*.
    """
    frame = ensure_primary_metric(pd.DataFrame([dict(record) for record in records]))
    if add_cv_signature and (sig := make_cv_signature(frame)) is not None:
        frame["cv_signature"] = sig
    if not frame.empty and "scope" not in frame.columns:
        if scope_from and scope_from in frame.columns:
            frame["scope"] = frame[scope_from]
        elif default_scope is not None:
            frame["scope"] = default_scope
    return frame


def make_decoding_sweep_report(
    records: Iterable[Mapping[str, Any]],
    *,
    title: str,
    kind: str,
    strategy_note: str,
    dataset_name: str = "dataset",
    leaderboards: Sequence[Mapping[str, Any]] = (),
    body: Callable[[Report, pd.DataFrame], None] | None = None,
    pre_sections: Iterable[Section] = (),
    empty_message: str | None = None,
    frame: pd.DataFrame | None = None,
    scope_from: str | None = "condition",
    default_scope: str | None = "all",
    add_cv_signature: bool = True,
    config: Mapping[str, Any] | None = None,
    asset_urls: dict[str, str] | str | None = "inline",
    output_path: str | Path | None = None,
) -> Report:
    """Assemble the skeleton every decoding sweep report shares.

    Runs: frame prep -> scientific overview -> *pre_sections* (e.g. QC or a
    foundation capability matrix) -> each ``leaderboard_section(**spec)`` in
    *leaderboards* -> the domain *body* hook (only when the frame is non-empty)
    -> failures section -> optional save. The ``body`` callable owns the
    per-axis layout that differs between classical and foundation reports.
    """
    if frame is None:
        frame = prepare_sweep_frame(
            records,
            add_cv_signature=add_cv_signature,
            scope_from=scope_from,
            default_scope=default_scope,
        )
    report = Report(
        title=title,
        config=dict(config or {}),
        asset_urls=asset_urls,
    )
    add_scientific_overview(
        report,
        frame,
        dataset_name=dataset_name,
        kind=kind,
        strategy_note=strategy_note,
        empty_message=empty_message,
    )
    for section in pre_sections:
        if section is not None:
            report.add_section(section)
    for spec in leaderboards:
        board = leaderboard_section(frame, **spec)
        if board is not None:
            report.add_section(board)
    if body is not None and not frame.empty:
        body(report, frame)
    failures = failures_section(frame)
    if failures is not None:
        report.add_section(failures)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.save(output)
    return report


def build_capability_matrix_section(
    capability_records: Iterable[Mapping[str, Any]],
) -> Section | None:
    """Build the foundation preflight capability matrix, or None when empty."""
    capabilities = pd.DataFrame([dict(record) for record in capability_records])
    if capabilities.empty:
        return None
    section = Section("Foundation Capability Matrix")
    section.add_markdown(
        "Preflight decisions for every model and training mode. Unsupported "
        "combinations are reported explicitly and never silently downgraded."
    )
    columns = [
        column
        for column in (
            "condition",
            "target",
            "model_key",
            "train_mode",
            "status",
            "reason",
            "sfreq",
            "n_channels",
            "n_times",
        )
        if column in capabilities.columns
    ]
    display = capabilities[columns] if columns else capabilities
    section.add_element(
        InteractiveTableElement(
            display,
            title="Preflight Decisions",
            selector_columns=[
                column
                for column in ("model_key", "train_mode", "status")
                if column in display.columns
            ],
        )
    )
    return section


def build_foundation_comparison_sections(
    report: Report,
    frame: pd.DataFrame,
    *,
    group_by: tuple[str, ...] = ("condition", "target"),
    per_result_sections: str | Sequence[str] = "compact",
) -> None:
    """Add foundation model/train-mode comparison figures + per-result tabs.

    A `body` hook for :func:`make_decoding_sweep_report`: loads successful result
    artifacts, renders linear-probe bars, a training-mode heatmap, a metric
    matrix and a spread plot, then the coco-pipe per-result diagnostics tabs.
    """
    required = {"status", "output_dir", "model_key", "train_mode", "target"}
    if not required.issubset(frame.columns):
        gap = Section("Performance Figures")
        gap.add_markdown(
            "Foundation results are missing the context or output paths required "
            "for coco-pipe comparison figures."
        )
        report.add_section(gap)
        return

    success = frame[(frame["status"] == "success") & frame["output_dir"].notna()].copy()
    context_columns = [
        column
        for column in ("condition", "target", "model_key", "train_mode")
        if column in success and success[column].notna().any()
    ]
    success["unit"] = (
        success["model_key"].astype(str) + " · " + success["train_mode"].astype(str)
    )
    context_columns.append("unit")
    unique = success[[*context_columns, "output_dir"]].drop_duplicates()
    collection = collect_results(
        [
            (
                {column: row[column] for column in context_columns},
                Path(str(row["output_dir"])) / "result.joblib",
            )
            for row in unique.to_dict("records")
        ],
        by=context_columns,
    )

    comparison_specs = [
        (
            collection.filter(train_mode="linear_probe"),
            {
                "kind": "model_bars",
                "axis": "model_key",
                "group_by": group_by,
                "value": "primary_metric",
                "center": 0.5,
                "title": "Primary Result: Linear Probe Accuracy",
                "include_table": True,
            },
        ),
        (
            collection,
            {
                "kind": "axis_heatmap",
                "axis": "model_key",
                "column": "train_mode",
                "group_by": group_by,
                "value": "primary_metric",
                "center": 0.5,
                "title": "Training-Mode Comparison",
                "include_table": True,
            },
        ),
        (
            collection,
            {
                "kind": "metric_matrix",
                "axis": "unit",
                "group_by": group_by,
                "value": "primary_metric",
                "center": 0.5,
                "title": "Metric Matrix",
            },
        ),
        (
            collection,
            {
                "kind": "spread",
                "axis": "train_mode",
                "group_by": group_by,
                "value": "primary_metric",
                "center": 0.5,
                "title": "Accuracy Spread by Training Mode",
            },
        ),
    ]
    for comparison_collection, spec in comparison_specs:
        comparison = build_comparison_section(comparison_collection, **spec)
        if comparison is not None:
            report.add_section(comparison)

    per_result = build_result_tabs(
        collection,
        sections=per_result_sections,
        on_error="placeholder",
    )
    if per_result is not None:
        section = Section("Per-Result Diagnostics")
        section.add_element(
            CalloutElement(
                "Compact diagnostics are shown by default. Pass "
                "per_result_sections='full' to include the full coco-pipe "
                "per-result report payload.",
                kind="info",
                title="Diagnostics Scope",
            )
        )
        section.add_element(per_result)
        report.add_section(section)


# ---------------------------------------------------------------------------
# Classical taxonomy body (the classical counterpart to
# build_foundation_comparison_sections): renders the analysis-unit taxonomy
# — flat / sensor / family / descriptor — as the body of the classical report.
# ---------------------------------------------------------------------------

CLASSICAL_MODE_TITLES: dict[str, str] = {
    "flat": "Full Analysis: All Sensors x All Features",
    "sensor": "Sensor-wise Analyses",
    "family": "Family Analyses: All Sensors",
    "subfamily": "Subfamily Analyses: All Sensors",
    "sensor_within_family": "Sensor x Family Analyses",
    "sensor_within_subfamily": "Sensor x Subfamily Analyses",
    "descriptor": "Single Descriptor (all stats): All Sensors",
    "descriptor_sensor": "Single Descriptor (all stats) x Single Sensor",
}

# Maps each taxonomy analysis_mode to its comparison recipe. These are the
# generic coco-pipe recipes; a study can pass its own mapping to override.
DEFAULT_CLASSICAL_SECTION_BUILDERS: dict[str, Callable[..., Any]] = {
    "flat": partial(
        flat_section,
        kind="model_bars",
        title_prefix="Flat Model",
        group_by=("target", "selection_mode"),
        axis="Model",
    ),
    "sensor": partial(sensor_section, montage="standard_1020"),
    "family": partial(
        grouped_section,
        group_col="family",
        kind="axis_heatmap",
        title_prefix="Family",
        group_by=("target", "selection_mode"),
        axis="family",
        column="Model",
    ),
    "subfamily": partial(
        grouped_section,
        group_col="subfamily",
        kind="axis_heatmap",
        title_prefix="Subfamily",
        group_by=("target", "selection_mode"),
        axis="subfamily",
        column="Model",
    ),
    "sensor_within_family": partial(
        grouped_section,
        group_col="family",
        kind="grid_heatmap",
        title_prefix="Sensor x Family",
        group_by=("target", "selection_mode", "Model"),
        row="family",
        column="unit_name",
    ),
    "sensor_within_subfamily": partial(
        grouped_section,
        group_col="subfamily",
        kind="grid_heatmap",
        title_prefix="Sensor x Subfamily",
        group_by=("target", "selection_mode", "Model"),
        row="subfamily",
        column="unit_name",
    ),
    "descriptor": partial(
        descriptor_section,
        kind="axis_heatmap",
        title_prefix="Single-Descriptor",
        group_by=("target", "selection_mode", "descriptor_family"),
        axis="unit_name",
        column="model",
    ),
    "descriptor_sensor": partial(
        descriptor_section,
        kind="grid_heatmap",
        title_prefix="Single Descriptor x Sensor",
        group_by=("target", "selection_mode", "model", "descriptor_family"),
        include_sensor=True,
        row="unit_name",
        column="sensor",
    ),
}


def build_classical_mode_elements(
    mode_frame: pd.DataFrame,
    table_title: str,
    analysis_modes: tuple[str, ...],
    scope_label: str,
    feature_metadata: pd.DataFrame | None,
    *,
    section_builders: Mapping[str, Callable[..., Any]] | None = None,
) -> list[Any]:
    """Build the per-mode result table plus its taxonomy comparison element."""
    section_builders = section_builders or DEFAULT_CLASSICAL_SECTION_BUILDERS
    if "selection_mode" in mode_frame:
        mode_frame = mode_frame.assign(
            _is_not_baseline=(
                mode_frame["selection_mode"].astype(str) != "baseline"
            ).astype(int)
        )
    sort_columns = [
        column
        for column in (
            "target",
            "unit_name",
            "subfamily",
            "_is_not_baseline",
            "selection_mode",
            "model",
        )
        if column in mode_frame
    ]
    if sort_columns:
        mode_frame = mode_frame.sort_values(
            sort_columns, kind="stable", na_position="last"
        )
    mode_frame = mode_frame.drop(columns=["_is_not_baseline"], errors="ignore")

    result_display = display_frame(mode_frame)
    display_selectors = [
        column
        for column in (
            "Target",
            "Analysis Unit",
            "Subfamily",
            "Family",
            "Model",
            "Feature Selection",
            "Status",
        )
        if column in result_display
    ]
    elements: list[Any] = [
        InteractiveTableElement(
            result_display,
            title=table_title,
            selector_columns=display_selectors,
        )
    ]

    builder = section_builders.get(analysis_modes[0])
    if builder is not None:
        builder_kwargs: dict[str, Any] = {
            "scope_label": scope_label,
            "feature_metadata": feature_metadata,
        }
        if (
            analysis_modes[0] in {"descriptor", "descriptor_sensor"}
            and feature_metadata is not None
            and not feature_metadata.empty
            and {"Feature", "FeatureFamily"}.issubset(feature_metadata.columns)
        ):
            builder_kwargs["descriptor_family"] = dict(
                zip(
                    feature_metadata["Feature"],
                    feature_metadata["FeatureFamily"],
                    strict=False,
                )
            )
        element = builder(mode_frame, **builder_kwargs)
        if element is not None:
            elements.append(element)
    return elements


def build_classical_taxonomy_sections(
    report: Report,
    frame: pd.DataFrame,
    *,
    feature_metadata: pd.DataFrame | None = None,
    section_builders: Mapping[str, Callable[..., Any]] | None = None,
    mode_titles: Mapping[str, str] | None = None,
    scope_order: Sequence[str] | None = None,
    scope_intro: str | None = None,
) -> None:
    """Render one taxonomy section per scope: the classical report body.

    A ``body`` hook for :func:`make_decoding_sweep_report`, mirroring
    :func:`build_foundation_comparison_sections`. For each scope it lays out the
    analysis-unit taxonomy from the broadest flat analysis down to the narrowest
    sensor/descriptor views, flags a missing flat baseline, and appends
    feature-selection diagnostics. Pass *scope_order* to control scope sequencing
    (e.g. a study's configured conditions with a pooled scope last).
    """
    if "analysis_mode" not in frame.columns:
        return
    section_builders = section_builders or DEFAULT_CLASSICAL_SECTION_BUILDERS
    titles = mode_titles or CLASSICAL_MODE_TITLES
    analysis_plan = [((mode,), title) for mode, title in titles.items()]
    if scope_order is None:
        scope_order = (
            list(dict.fromkeys(frame["scope"].astype(str)))
            if "scope" in frame
            else ["all"]
        )
    intro = scope_intro or (
        "Results are ordered from the full descriptor-space analysis through "
        "progressively narrower sensor and feature analyses."
    )
    for scope in scope_order:
        scope_frame = (
            frame[frame["scope"].astype(str) == scope].copy()
            if "scope" in frame
            else frame.copy()
        )
        # An explicitly configured scope with no rows is kept and flagged below,
        # so a missing condition surfaces as a FAIL section rather than vanishing.
        scope_label = "POOLED" if scope == "pooled" else scope
        section = Section(scope_label)
        section.add_markdown(intro)
        flat = scope_frame[scope_frame["analysis_mode"] == "flat"]
        successful_flat = flat[
            flat.get("status", pd.Series(index=flat.index, dtype=object)) == "success"
        ]
        if successful_flat.empty:
            section.status = "FAIL"
            section.add_markdown("**The full analysis is missing or incomplete.**")
        for analysis_modes, table_title in analysis_plan:
            mode_frame = scope_frame[
                scope_frame["analysis_mode"].isin(analysis_modes)
            ].copy()
            if mode_frame.empty:
                continue
            for element in build_classical_mode_elements(
                mode_frame,
                table_title,
                analysis_modes,
                scope_label,
                feature_metadata,
                section_builders=section_builders,
            ):
                section.add_element(element)
        fs_section = feature_selection_section(
            scope_frame, feature_metadata=feature_metadata
        )
        if fs_section is not None:
            section.add_element(fs_section)
        report.add_section(section)


def collect_comparison_runs(
    sources: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Read, filter and tag per-run result CSVs into one comparison frame.

    Each entry in *sources* is a spec describing one family of runs to pull:

    ``paths``
        Iterable of CSV paths (the caller owns the globbing policy).
    ``filters``
        Mapping of ``column -> value`` equality filters applied to each CSV.
        The filter columns are also required: a CSV missing any of them is
        skipped rather than raising.
    ``family``
        The ``comparison_family`` tag: either a string, or a callable
        ``(frame, path) -> str`` evaluated per surviving CSV.
    ``space`` / ``source``
        Optional overrides for the ``comparison_space`` / ``source_run`` tags;
        callables of ``path``. Default to ``path.parent.name`` and
        ``str(path.parent)``.

    The read/filter/tag/concat boilerplate lives here; callers keep only their
    study-specific glob patterns, row filters and family policy. Empty or
    malformed CSVs are skipped. Returns an empty frame when nothing survives.
    """
    frames: list[pd.DataFrame] = []
    for source in sources:
        filters = dict(source.get("filters", {}))
        required = set(filters)
        family = source["family"]
        space = source.get("space", lambda path: Path(path).parent.name)
        run = source.get("source", lambda path: str(Path(path).parent))
        for raw_path in source.get("paths", ()):
            path = Path(raw_path)
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if not required.issubset(frame.columns):
                continue
            for column, value in filters.items():
                frame = frame[frame[column] == value]
            frame = frame.copy()
            if frame.empty:
                continue
            frame["comparison_family"] = (
                family(frame, path) if callable(family) else family
            )
            frame["comparison_space"] = space(path) if callable(space) else space
            frame["source_run"] = run(path) if callable(run) else run
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


HEAD_TO_HEAD_DISPLAY_COLUMNS = (
    "comparison_family",
    "comparison_space",
    "source_run",
    "scope",
    "condition",
    "target",
    "model_label",
    "model",
    "model_key",
    "train_mode",
    "metric_name",
    "primary_metric",
    "balanced_accuracy_mean",
    "accuracy_mean",
    "balanced_accuracy",
    "accuracy",
    "p_value",
    "p_value_fdr",
    "significant_fdr",
    *CV_SIGNATURE_COLUMNS,
    "output_dir",
)


def enrich_head_to_head_frame(comparison_frame: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns a head-to-head comparison frame needs.

    Expects a concatenated frame already tagged with ``comparison_family`` (the
    caller's globbing/tagging policy). Stamps the primary metric, attaches a
    ``cv_signature``, fills ``scope`` from ``condition``, and derives
    ``model_label``/``comparison_label``/``metric_name`` used by the table and
    the head-to-head plot.
    """
    frame = ensure_primary_metric(comparison_frame.copy())
    if frame.empty:
        return frame
    if (sig := make_cv_signature(frame)) is not None:
        frame["cv_signature"] = sig
    if "condition" in frame:
        if "scope" not in frame:
            frame["scope"] = frame["condition"]
        else:
            frame["scope"] = frame["scope"].combine_first(frame["condition"])
    model_label = pd.Series("model", index=frame.index, dtype=object)
    if "model" in frame:
        model_label = frame["model"].combine_first(model_label)
    if "model_key" in frame:
        model_label = frame["model_key"].combine_first(model_label)
    frame["model_label"] = model_label.astype(str)
    if "comparison_family" in frame:
        frame["comparison_label"] = (
            frame["comparison_family"].astype(str) + " | " + frame["model_label"]
        )
    frame["metric_name"] = "primary_metric"
    return frame


def normalize_head_to_head_frame(
    frame: pd.DataFrame,
    *,
    display_columns: Sequence[str] = HEAD_TO_HEAD_DISPLAY_COLUMNS,
) -> pd.DataFrame:
    """Sort an enriched comparison frame and project the display columns.

    The single source of truth for the head-to-head leaderboard layout: used
    both for the persisted comparison CSV and for the report table, so the two
    never drift. *frame* must already be enriched (see
    :func:`enrich_head_to_head_frame`).
    """
    if frame.empty:
        return frame
    present = [column for column in display_columns if column in frame.columns]
    return sort_by_metrics(frame, tie_breakers=PRIMARY_TIE_BREAKERS).loc[:, present]


def make_head_to_head_report(
    comparison_frame: pd.DataFrame,
    *,
    baseline_family: str,
    group_columns: Sequence[str] = ("scope", "target"),
    title: str,
    table_title: str = "Descriptors vs. Embeddings vs. Linear Probe",
    intro: str | None = None,
    paired_delta_title: str | None = None,
    display_columns: Sequence[str] = HEAD_TO_HEAD_DISPLAY_COLUMNS,
    config: Mapping[str, Any] | None = None,
    asset_urls: dict[str, str] | str | None = "inline",
    output_path: str | Path | None = None,
) -> Report:
    """Build the head-to-head comparison report from a tagged comparison frame.

    ``comparison_frame`` must carry ``comparison_family`` (baseline vs. each
    challenger) plus per-run metrics; the caller owns how it is assembled. The
    report shows a normalized leaderboard table, a cohort/CV compatibility panel,
    a head-to-head accuracy plot, and paired deltas versus *baseline_family* for
    groups whose CV/cohort signatures are compatible.
    """
    frame = enrich_head_to_head_frame(comparison_frame)
    report = Report(title=title, config=dict(config or {}), asset_urls=asset_urls)

    section = Section("Head-to-Head Comparison")
    section.add_markdown(
        intro
        or (
            "Primary baseline results and challenger models are shown together. "
            "Compare rows only when their grouped-CV and cohort signatures match."
        )
    )
    if not frame.empty:
        normalized = normalize_head_to_head_frame(
            frame, display_columns=display_columns
        )
        section.add_element(
            InteractiveTableElement(
                normalized,
                title=table_title,
                selector_columns=selector_columns(
                    normalized,
                    (
                        "comparison_family",
                        "comparison_space",
                        "scope",
                        "condition",
                        "target",
                        "model_label",
                        "cv_signature",
                        "cohort_signature",
                    ),
                ),
                default_sort={"column": "primary_metric", "direction": "desc"},
            )
        )
    report.add_section(section)

    compatibility = (
        signature_compatibility(frame, tuple(group_columns))
        if not frame.empty
        else pd.DataFrame()
    )
    compatible_lookup: dict[tuple[Any, ...], bool] = {}
    if not compatibility.empty:
        compatible_lookup = {
            tuple(row[column] for column in group_columns): bool(
                row["paired_compatible"]
            )
            for row in compatibility.to_dict("records")
        }
        compat_section = Section("Comparison Compatibility")
        incompatible = compatibility[~compatibility["paired_compatible"]]
        if incompatible.empty:
            compat_section.add_element(
                CalloutElement(
                    "All groups share compatible cohort/CV signatures.",
                    kind="tip",
                    title="Paired Comparisons Available",
                )
            )
        else:
            compat_section.add_element(
                CalloutElement(
                    "Some groups have mismatched cohort or CV signatures; paired "
                    "delta plots are omitted for those groups.",
                    kind="warning",
                    title="Paired Comparisons Limited",
                )
            )
        compat_section.add_element(
            InteractiveTableElement(
                compatibility,
                title="Compatibility by group",
                selector_columns=selector_columns(
                    compatibility,
                    (*group_columns, "paired_compatible"),
                ),
            )
        )
        report.add_section(compat_section)

    if not frame.empty and "comparison_label" in frame:
        plot_frame = frame.copy()
        plot_frame["_status"] = "success"
        plot_section = build_comparison_section(
            ResultCollection(
                by=("comparison_label",),
                results={},
                contexts={},
                summary=plot_frame,
            ),
            kind="head_to_head",
            axis="comparison_label",
            value="primary_metric",
            center=0.5,
            group_by=tuple(group_columns),
            title="Head-to-Head Accuracy",
        )
        if plot_section is not None:
            report.add_section(plot_section)

    if not frame.empty:
        has_signature = any(column in frame.columns for column in CV_SIGNATURE_COLUMNS)
        delta_frame = paired_delta_vs_baseline(
            frame,
            group_columns=tuple(group_columns),
            baseline_family=baseline_family,
            compatible_lookup=compatible_lookup if has_signature else None,
        )
        if not delta_frame.empty:
            delta_section = build_comparison_section(
                ResultCollection(
                    by=("comparison_pair",),
                    results={},
                    contexts={},
                    summary=delta_frame,
                ),
                kind="paired_delta",
                axis="comparison_pair",
                value="comparison_delta",
                group_by=tuple(group_columns),
                title=paired_delta_title or f"Paired Delta vs {baseline_family}",
                include_table=True,
            )
            if delta_section is not None:
                report.add_section(delta_section)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.save(output)
    return report


__all__ = [
    "CLASSICAL_MODE_TITLES",
    "DEFAULT_CLASSICAL_SECTION_BUILDERS",
    "HEAD_TO_HEAD_DISPLAY_COLUMNS",
    "ResultCollection",
    "add_scientific_overview",
    "best_result_tabs",
    "build_capability_matrix_section",
    "build_classical_mode_elements",
    "build_classical_taxonomy_sections",
    "build_comparison_section",
    "build_foundation_comparison_sections",
    "build_result_tabs",
    "collect_comparison_runs",
    "collect_mode_results",
    "collect_results",
    "descriptor_section",
    "enrich_head_to_head_frame",
    "failures_section",
    "feature_selection_section",
    "flat_section",
    "grouped_section",
    "hp_tuning_section",
    "leaderboard_section",
    "make_decoding_sweep_report",
    "make_experiment_results_report",
    "make_head_to_head_report",
    "normalize_head_to_head_frame",
    "paired_delta_vs_baseline",
    "prepare_sweep_frame",
    "sensor_section",
    "summary_collection",
]
