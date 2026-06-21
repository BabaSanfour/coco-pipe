"""Cross-result collection and comparison helpers for decoding reports."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from coco_pipe.decoding import ExperimentResult
from coco_pipe.viz import (
    plot_decoding_topomap,
    plot_distribution_groups,
    plot_head_to_head,
    plot_heatmap,
    plot_paired_delta,
)

from ._utils import _ensure_static_matplotlib_backend, _figure_element
from .core import Report, Section
from .decoding import build_decoding_sections
from .elements import (
    AccordionElement,
    CalloutElement,
    ImageElement,
    InteractiveTableElement,
    TabsElement,
)
from .qc import build_qc_section


@dataclass
class ResultCollection:
    """A labelled collection of decoding results and their tidy summaries."""

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


def _load_result(source: Any) -> tuple[Any, str | None]:
    if hasattr(source, "summary") and callable(source.summary):
        return source, None
    path = Path(source).expanduser()
    if path.is_dir():
        path = path / "result.joblib"
    return ExperimentResult.load(path), str(path)


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
            result, source_path = _load_result(source)
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
                    {
                        **context,
                        **record,
                        "_status": "success",
                        "_error": None,
                        "_source": source_path,
                    }
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


def _comparison_groups(
    frame: pd.DataFrame,
    group_by: Sequence[str],
) -> Iterable[tuple[str, pd.DataFrame]]:
    columns = [column for column in group_by if column in frame]
    if not columns:
        yield "Comparison", frame
        return
    for keys, group in frame.groupby(columns, dropna=False, sort=False):
        values = keys if isinstance(keys, tuple) else (keys,)
        yield " / ".join(str(value) for value in values), group


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame
        if column.endswith("_mean")
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]


def _comparison_signature_mismatches(frame: pd.DataFrame) -> list[str]:
    """Return scientific-design fields that vary within a paired comparison."""
    signatures = (
        "cohort_signature",
        "cv_signature",
        "cv_strategy",
        "effective_n_splits",
        "cv_random_state",
    )
    return [
        column
        for column in signatures
        if column in frame and frame[column].dropna().nunique() > 1
    ]


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
    tabs: dict[str, ImageElement] = {}
    for label, group in _comparison_groups(frame, group_by):
        try:
            # Guard, not dispatch: paired deltas are only meaningful when every
            # row shares a CV/cohort design. We veto incompatible groups here and
            # `continue`; the actual plot is dispatched by the ``elif`` chain below.
            if kind == "paired_delta":
                mismatches = _comparison_signature_mismatches(group)
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
            if kind in {"axis_heatmap", "grid_heatmap"}:
                index_name = row or axis
                if index_name is None or index_name not in group or column not in group:
                    continue
                matrix = group.pivot_table(
                    index=index_name,
                    columns=column,
                    values=value,
                    aggfunc="mean",
                )
                if matrix.empty:
                    continue
                figure = plot_heatmap(
                    matrix,
                    annotate=kind == "axis_heatmap",
                    annotation_format=".2f",
                    cmap="RdBu_r" if plot_center is not None else "viridis",
                    center=plot_center,
                    minimum_half_range=0.02 if plot_center is not None else 0.0,
                    colorbar_label=value,
                    title=label if label != "Comparison" else title,
                    xlabel=column,
                    ylabel=index_name,
                    figsize=(
                        max(6.0, 1.2 * matrix.shape[1] + 4.0),
                        max(5.0, 0.4 * matrix.shape[0] + 2.0),
                    ),
                )
            elif kind == "sensor_topomap":
                sensor_axis = axis or "unit_name"
                if sensor_axis not in group or (info is None and coords is None):
                    continue
                sensor_frame = group[[sensor_axis, value]].rename(
                    columns={sensor_axis: "FeatureName"}
                )
                figure = plot_decoding_topomap(
                    sensor_frame,
                    value,
                    info=info,
                    coords=coords,
                    center=plot_center,
                    title=label if label != "Comparison" else title,
                )
            elif kind in {"model_bars", "head_to_head"}:
                label_axis = axis or column
                if label_axis not in group:
                    continue
                summary = (
                    group.groupby(label_axis, dropna=False)[value]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                figure = plot_head_to_head(
                    summary,
                    label=label_axis,
                    value="mean",
                    error="std",
                    reference=plot_center,
                    title=label if label != "Comparison" else (title or "Comparison"),
                    ylabel=value,
                )
            elif kind == "spread":
                spread_axis = axis or column
                if spread_axis not in group:
                    continue
                grouped = list(group.groupby(spread_axis, dropna=False, sort=False))
                figure = plot_distribution_groups(
                    [values[value] for _, values in grouped],
                    [name for name, _ in grouped],
                    title=label if label != "Comparison" else title,
                    ylabel=value,
                )
            elif kind == "metric_matrix":
                index_name = axis or column
                metrics = _metric_columns(group)
                if index_name not in group or not metrics:
                    continue
                matrix = group.groupby(index_name, dropna=False)[metrics].mean()
                matrix.columns = [name.removesuffix("_mean") for name in matrix.columns]
                figure = plot_heatmap(
                    matrix,
                    annotate=True,
                    annotation_format=".2f",
                    center=plot_center,
                    minimum_half_range=0.02 if plot_center is not None else 0.0,
                    cmap="RdBu_r" if plot_center is not None else "viridis",
                    colorbar_label="score",
                    title=label if label != "Comparison" else title,
                    xlabel="metric",
                    ylabel=index_name,
                )
            elif kind == "paired_delta":
                delta_axis = axis or "comparison"
                figure = plot_paired_delta(
                    group,
                    label=delta_axis,
                    delta=value,
                    lower="ci_lower" if "ci_lower" in group else None,
                    upper="ci_upper" if "ci_upper" in group else None,
                    title=label if label != "Comparison" else (title or "Paired Delta"),
                )
            else:
                raise ValueError(f"Unknown decoding comparison kind: {kind!r}")
        except (ImportError, TypeError, ValueError) as exc:
            if on_error == "raise":
                raise
            warnings.warn(
                f"Decoding comparison {kind!r} for {label!r} failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
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


def _resolve_comparison_specs(
    comparisons: str | Sequence[Mapping[str, Any]],
    by: Sequence[str],
) -> Sequence[Mapping[str, Any]]:
    """Resolve the ``comparisons`` argument to a list of comparison specs."""
    if comparisons == "default":
        return _showcase_specs(by)[:1]
    if comparisons == "showcase":
        return _showcase_specs(by)
    if isinstance(comparisons, str):
        return [{"kind": comparisons}]
    return comparisons


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

    specs = _resolve_comparison_specs(comparisons, by)
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


__all__ = [
    "ResultCollection",
    "build_comparison_section",
    "build_result_tabs",
    "collect_results",
    "make_experiment_results_report",
]
