"""Section builders for dimensionality-reduction reports."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from coco_pipe.io.quality import QCResult

from ._utils import _coerce_kind, _resolve_sections
from .core import Report, Section
from .elements import (
    DownloadAssetElement,
    ImageElement,
    MetricsTableElement,
    PlotlyElement,
    TableElement,
    TabsElement,
)
from .qc import build_qc_section

logger = logging.getLogger(__name__)


def _get_reducer_summary(reducer: Any) -> dict[str, Any]:
    """Collect the strict summary payload from a reduction-like object."""
    if not hasattr(reducer, "get_summary"):
        raise TypeError(
            "Reduction objects passed to Report.add_reduction() must implement "
            "get_summary()."
        )

    summary = reducer.get_summary()
    if not isinstance(summary, dict):
        raise TypeError("Reducer get_summary() must return a dictionary.")

    return {
        "method": summary.get("method") or type(reducer).__name__,
        "metrics": _coerce_kind(summary.get("metrics"), dict),
        "metric_records": _coerce_kind(summary.get("metric_records"), list),
        "quality_metadata": _coerce_kind(summary.get("quality_metadata"), dict),
        "diagnostics": _coerce_kind(summary.get("diagnostics"), dict),
        "interpretation": _coerce_kind(summary.get("interpretation"), dict),
        "interpretation_records": _coerce_kind(
            summary.get("interpretation_records"), list
        ),
        "capabilities": summary.get("capabilities") or {},
    }


def _trajectory_times(
    diagnostics: dict[str, Any], times: np.ndarray | None
) -> np.ndarray | None:
    """Return the explicit trajectory time axis when it aligns with diagnostics."""
    if times is not None:
        time_values = np.asarray(times).reshape(-1)
        if time_values.size > 0:
            return time_values

    diagnostic_times = diagnostics.get("trajectory_times_")
    if diagnostic_times is None:
        return None

    time_values = np.asarray(diagnostic_times).reshape(-1)
    return time_values if time_values.size > 0 else None


DEFAULT_REDUCTION_SECTIONS: list[str] = [
    "overview",
    "embedding",
    "metrics",
    "diagnostics",
    "coranking",
    "interpretation",
    "components",
    "trajectory",
    "trajectory_separation",
]
VALID_REDUCTION_SECTIONS = set(DEFAULT_REDUCTION_SECTIONS)


def _metrics_summary_table(metrics: Any) -> pd.DataFrame:
    """Reduce metric observations to a method x metric summary table.

    Lives in this submodule (rather than ``core``) because it depends on
    :mod:`coco_pipe.viz._utils.prepare_metrics_frame` — keeping the viz
    dependency out of :mod:`coco_pipe.report.core`.
    """
    from coco_pipe.viz._utils import prepare_metrics_frame

    metrics_df = prepare_metrics_frame(metrics)
    if metrics_df.empty:
        return pd.DataFrame()

    return metrics_df.pivot_table(
        index="Method", columns="Metric", values="Value", aggfunc="mean"
    )


def _summary(reduction: Any) -> dict[str, Any]:
    if hasattr(reduction, "get_summary"):
        value = reduction.get_summary()
        return value if isinstance(value, dict) else {}
    return {}


def _method_name(reduction: Any, index: int = 0) -> str:
    summary = _summary(reduction)
    method = summary.get("method") or getattr(reduction, "method", None)
    return str(method) if method else f"Method {index + 1}"


def add_reduction_overview(
    self: Report,
    reduction: Any,
    *,
    name: str = "Overview",
) -> Report:
    """Add a reduction-summary table to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    reduction : Any
        Scored reduction object exposing ``get_summary() -> dict``.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if the summary is
        empty.

    See Also
    --------
    make_reduction_report : Factory that calls this and all other adders.

    Examples
    --------
    >>> report = Report(title="My Report")
    >>> report.add_reduction_overview(pca_result)
    """
    summary = _summary(reduction)
    if not summary:
        return self
    rows = [
        {"Key": key, "Value": value}
        for key, value in summary.items()
        if key
        not in {
            "metrics",
            "metric_records",
            "diagnostics",
            "interpretation",
            "interpretation_records",
        }
    ]
    sec = Section(title=name)
    sec.add_element(TableElement(pd.DataFrame(rows), title="Reduction Summary"))
    self.add_section(sec)
    return self


def add_reduction_embedding(
    self: Report,
    X_emb: Any,
    *,
    labels: Any = None,
    metadata: Any = None,
    name: str = "Embedding",
) -> Report:
    """Add a scatter plot of the low-dimensional embedding to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    X_emb : array-like
        2-D or 3-D embedding array of shape ``(n_samples, n_dims)``.
    labels : array-like, optional
        Class labels aligned with *X_emb* rows.
    metadata : dict, optional
        Column-oriented metadata for point colouring.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if *X_emb* is
        ``None``.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_embedding

    Examples
    --------
    >>> report.add_reduction_embedding(X_2d, labels=y)
    """
    if X_emb is None:
        return self
    from coco_pipe.viz.dim_reduction import plot_embedding

    sec = Section(title=name)
    try:
        plot_result = plot_embedding(X_emb, labels=labels, metadata=metadata)
        sec.add_element(
            ImageElement(
                plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                caption="Embedding",
            )
        )
    except ValueError as exc:
        logger.debug("Embedding section skipped: %s", exc)
        return self
    self.add_section(sec)
    return self


def add_reduction_metrics(
    self: Report,
    reduction: Any,
    *,
    metric: str | None = None,
    name: str = "Quality Metrics",
) -> Report:
    """Add a quality-metrics table and bar chart to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    reduction : Any
        Scored reduction object; checked for ``get_scores()`` and
        ``get_summary()["metric_records"]``.
    metric : str, optional
        Highlight a specific metric in the bar chart.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if no scores are
        available.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_metrics

    Examples
    --------
    >>> report.add_reduction_metrics(pca_result)
    """
    from coco_pipe.viz.dim_reduction import plot_metrics

    summary = _summary(reduction)
    scores = None
    if hasattr(reduction, "get_scores"):
        try:
            scores = reduction.get_scores()
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("Reduction scores unavailable: %s", exc)
            scores = None
    if scores is None or (hasattr(scores, "empty") and scores.empty):
        scores = summary.get("metric_records") or summary.get("metrics")
    if scores is None or (hasattr(scores, "empty") and scores.empty) or scores == []:
        return self
    sec = Section(title=name)
    try:
        scores_df = pd.DataFrame(scores if isinstance(scores, list) else [scores])
        csv_data = scores_df.to_csv(index=False)
        sec.add_element(
            DownloadAssetElement(
                csv_data,
                "reduction_metrics.csv",
                "text/csv",
                label="Download Metrics CSV",
                style="gray",
            )
        )
        sec.add_element(TableElement(scores_df, title="Metrics"))
        plot_result = plot_metrics(scores, metric=metric)
        sec.add_element(
            ImageElement(
                plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                caption="Quality metrics",
            )
        )
    except ValueError as exc:
        logger.debug("Metrics section skipped: %s", exc)
        return self
    self.add_section(sec)
    return self


def add_reduction_diagnostics(
    self: Report,
    X_orig: Any,
    X_emb: Any,
    *,
    name: str = "Diagnostics",
) -> Report:
    """Add a Shepard diagram comparing original and embedded distances to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    X_orig : array-like
        Original high-dimensional data of shape ``(n_samples, n_features)``.
    X_emb : array-like
        Low-dimensional embedding of shape ``(n_samples, n_dims)``.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if either array is
        ``None``.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_shepard_diagram

    Examples
    --------
    >>> report.add_reduction_diagnostics(X_orig, X_2d)
    """
    if X_orig is None or X_emb is None:
        return self
    from coco_pipe.viz.dim_reduction import plot_shepard_diagram

    sec = Section(title=name)
    try:
        plot_result = plot_shepard_diagram(X_orig, X_emb)
        sec.add_element(
            ImageElement(
                plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                caption="Shepard diagram",
            )
        )
    except ValueError as exc:
        logger.debug("Diagnostics section skipped: %s", exc)
        return self
    self.add_section(sec)
    return self


def add_reduction_interpretation(
    self: Report,
    interpretation: Any,
    *,
    analysis: str = "loadings",
    name: str = "Interpretation",
) -> Report:
    """Add a component-interpretation plot to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    interpretation : dict or list
        Interpretation records returned by the reduction object.
    analysis : str
        Which analysis to visualise (e.g. ``"loadings"``).
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if *interpretation*
        is falsy.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_feature_importance

    Examples
    --------
    >>> report.add_reduction_interpretation(result.get_summary()["interpretation"])
    """
    if not interpretation:
        return self
    from coco_pipe.viz.dim_reduction import plot_feature_importance

    sec = Section(title=name)
    try:
        plot_result = plot_feature_importance(
            interpretation, analysis=analysis, title=name
        )
        sec.add_element(
            ImageElement(
                plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                caption=analysis,
            )
        )
    except ValueError as exc:
        logger.debug("Interpretation section skipped: %s", exc)
        return self
    self.add_section(sec)
    return self


def add_reduction_coranking(
    self: Report,
    coranking_matrix: Any,
    *,
    max_k: int | None = None,
    name: str = "Co-Ranking Matrix",
) -> Report:
    """Add a co-ranking-matrix heatmap to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    coranking_matrix : array-like
        Square co-ranking matrix of shape ``(n_samples - 1, n_samples - 1)``.
    max_k : int, optional
        Maximum neighbourhood size to display.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if
        *coranking_matrix* is ``None``.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_coranking_matrix

    Examples
    --------
    >>> report.add_reduction_coranking(diagnostics["coranking_matrix_"])
    """
    if coranking_matrix is None:
        return self
    from coco_pipe.viz.dim_reduction import plot_coranking_matrix

    sec = Section(title=name)
    try:
        csv_data = pd.DataFrame(coranking_matrix).to_csv(index=False)
        sec.add_element(
            DownloadAssetElement(
                csv_data,
                "coranking_matrix.csv",
                "text/csv",
                label="Download Matrix CSV",
                style="gray",
            )
        )
        plot_result = plot_coranking_matrix(coranking_matrix, max_k=max_k)
        sec.add_element(
            ImageElement(
                plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                caption="Co-ranking matrix",
            )
        )
    except ValueError as exc:
        logger.debug("Co-ranking section skipped: %s", exc)
        return self
    self.add_section(sec)
    return self


def add_reduction_components(
    self: Report,
    components: Any,
    *,
    feature_names: list[str] | None = None,
    name: str = "Component Loadings",
) -> Report:
    """Add a component-loadings heatmap to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    components : array-like
        Component matrix of shape ``(n_dims, n_features)``.
    feature_names : list of str, optional
        Column labels for the feature axis.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if *components* is
        ``None``.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_component_loadings

    Examples
    --------
    >>> report.add_reduction_components(pca.components_, feature_names=ch_names)
    """
    if components is None:
        return self
    from coco_pipe.viz.dim_reduction import plot_component_loadings

    sec = Section(title=name)
    try:
        plot_result = plot_component_loadings(components, feature_names=feature_names)
        sec.add_element(
            ImageElement(
                plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                caption="Component loadings",
            )
        )
    except ValueError as exc:
        logger.debug("Component section skipped: %s", exc)
        return self
    self.add_section(sec)
    return self


def add_reduction_trajectory(
    self: Report,
    X: Any,
    *,
    times: Any = None,
    labels: Any = None,
    name: str = "Trajectory",
) -> Report:
    """Add a temporal trajectory plot to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    X : array-like
        3-D embedding of shape ``(n_times, n_samples, n_dims)``.
    times : array-like, optional
        Time axis values aligned with ``X[0]``.
    labels : array-like, optional
        Class labels aligned with the sample axis.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if *X* is ``None``.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_trajectory

    Examples
    --------
    >>> report.add_reduction_trajectory(X_3d, times=times, labels=y)
    """
    if X is None:
        return self
    from coco_pipe.viz.dim_reduction import plot_trajectory

    sec = Section(title=name)
    try:
        plot_result = plot_trajectory(X, times=times, labels=labels)
        sec.add_element(
            ImageElement(
                plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                caption="Trajectory",
            )
        )
    except ValueError as exc:
        logger.debug("Trajectory section skipped: %s", exc)
        return self
    self.add_section(sec)
    return self


def add_reduction_trajectory_separation(
    self: Report,
    separation: dict,
    *,
    times: Any = None,
    top_n: int | None = None,
    name: str = "Trajectory Separation",
) -> Report:
    """Add a trajectory-separation line chart to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    separation : dict
        Separation scores keyed by class-pair label.
    times : array-like, optional
        Time axis values.
    top_n : int, optional
        Limit the chart to the top-N most-separated pairs.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if *separation* is
        empty.

    See Also
    --------
    coco_pipe.viz.dim_reduction.plot_trajectory_separation

    Examples
    --------
    >>> report.add_reduction_trajectory_separation(sep, times=times, top_n=5)
    """
    if not separation:
        return self
    from coco_pipe.viz.dim_reduction import plot_trajectory_separation

    sec = Section(title=name)
    try:
        plot_result = plot_trajectory_separation(separation, times=times, top_n=top_n)
        sec.add_element(
            ImageElement(
                plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                caption="Trajectory separation",
            )
        )
    except ValueError as exc:
        logger.debug("Trajectory separation section skipped: %s", exc)
        return self
    self.add_section(sec)
    return self


def _diagnostic_payload(reduction: Any) -> dict[str, Any]:
    summary = _summary(reduction)
    diagnostics = (
        summary.get("diagnostics") or getattr(reduction, "diagnostics_", None) or {}
    )
    return diagnostics if isinstance(diagnostics, dict) else {}


def _components_payload(reduction: Any) -> Any:
    if hasattr(reduction, "get_components"):
        try:
            components = reduction.get_components()
            if isinstance(components, dict):
                if components.get("components_") is not None:
                    return components.get("components_")
                return components.get("components")
            return components
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("Reduction components unavailable: %s", exc)
            return None
    return getattr(reduction, "components_", None)


def _original_data_payload(diagnostics: dict[str, Any]) -> Any:
    """Return original-space data from known diagnostic payload keys."""
    value = diagnostics.get("X_orig")
    return diagnostics.get("X_orig_") if value is None else value


def make_reduction_report(
    reductions: list,
    *,
    embeddings: list | None = None,
    labels: Any = None,
    metadata: Any = None,
    times: Any = None,
    sections: list[str] | Literal["default"] = "default",
    interactive: bool = False,
    theme: Literal["paper", "notebook", "poster"] = "paper",
    title: str = "Dimensionality Reduction Report",
    config: dict | None = None,
    asset_urls: dict[str, str] | None = None,
    qc_result: QCResult | None = None,
    output_path: str | None = None,
) -> Report:
    """Build a dimensionality-reduction report for one or more reduction objects.

    Parameters
    ----------
    reductions : list
        Scored reduction objects implementing ``get_summary()``.
    embeddings : list, optional
        Explicit embedding arrays aligned with *reductions*. Required for
        embedding, trajectory, and Shepard-diagram sections.
    labels : array-like, optional
        Class labels aligned with each embedding's sample axis.
    metadata : dict, optional
        Column-oriented metadata for point colouring in embedding plots.
    times : array-like, optional
        Time axis aligned with 3-D trajectory embeddings.
    sections : list of str or ``"default"``
        Ordered list of section keys (see :data:`DEFAULT_REDUCTION_SECTIONS`).
    interactive : bool
        Reserved for future use. Currently ignored; pass ``False``.
    theme : ``"paper"`` | ``"notebook"`` | ``"poster"``
        Matplotlib theme preset.
    title : str
        Report title.
    config : dict, optional
        Extra configuration metadata stored in the report header.
    asset_urls : dict, optional
        Override JavaScript asset URLs used by the report shell.
    qc_result : QCResult, optional
        Structured QC drop log rendered before analysis sections.
    output_path : str, optional
        If given, save the rendered HTML to this path.

    Returns
    -------
    Report
        Fully populated report. Sections whose data is absent are skipped
        silently.

    Raises
    ------
    ValueError
        If *embeddings* is provided but its length differs from *reductions*.

    See Also
    --------
    coco_pipe.report.api.from_reductions : Thin public wrapper.
    make_decoding_report : Equivalent factory for decoding results.

    Examples
    --------
    >>> report = make_reduction_report([pca, tsne], embeddings=[X_pca, X_tsne])
    >>> report.save("reduction_report.html")
    """
    if interactive:
        warnings.warn(
            "interactive=True is not yet implemented in make_reduction_report. "
            "All plots are currently rendered as static images.",
            stacklevel=2,
        )

    if embeddings is not None and len(embeddings) != len(reductions):
        raise ValueError("`embeddings` must align with `reductions`.")
    selected = _resolve_sections(
        sections,
        default=DEFAULT_REDUCTION_SECTIONS,
        valid=VALID_REDUCTION_SECTIONS,
        context="reduction",
    )
    report = Report(title=title, config=config, theme=theme, asset_urls=asset_urls)
    if qc_result is not None:
        report.add_section(build_qc_section(qc_result))

    for idx, reduction in enumerate(reductions):
        prefix = _method_name(reduction, idx)
        emb = None if embeddings is None else embeddings[idx]
        diagnostics = _diagnostic_payload(reduction)
        summary = _summary(reduction)
        interpretation = summary.get("interpretation_records") or summary.get(
            "interpretation"
        )
        trajectory = emb if getattr(emb, "ndim", None) == 3 else None
        coranking = diagnostics.get("coranking_matrix_")
        if coranking is None:
            coranking = diagnostics.get("coranking_matrix")
        separation = diagnostics.get("trajectory_separation_")
        if separation is None:
            separation = diagnostics.get("trajectory_separation") or {}

        # Collect embedding and Shepard images for side-by-side layout
        emb_2d = emb if getattr(emb, "ndim", None) == 2 else None

        for section in selected:
            try:
                if section == "overview":
                    add_reduction_overview(report, reduction, name=f"{prefix} Overview")
                elif section == "embedding":
                    if emb_2d is not None:
                        _add_embedding_and_shepard(
                            report,
                            reduction,
                            emb_2d,
                            labels=labels,
                            metadata=metadata,
                            prefix=prefix,
                            diagnostics=diagnostics,
                        )
                    else:
                        add_reduction_embedding(
                            report,
                            emb,
                            labels=labels,
                            metadata=metadata,
                            name=f"{prefix} Embedding",
                        )
                elif section == "metrics":
                    add_reduction_metrics(
                        report, reduction, name=f"{prefix} Quality Metrics"
                    )
                elif section == "diagnostics":
                    if "embedding" not in selected and emb is not None:
                        add_reduction_diagnostics(
                            report,
                            _original_data_payload(diagnostics),
                            emb,
                            name=f"{prefix} Diagnostics",
                        )
                elif section == "coranking":
                    add_reduction_coranking(
                        report,
                        coranking,
                        name=f"{prefix} Co-Ranking Matrix",
                    )
                elif section == "interpretation":
                    add_reduction_interpretation(
                        report, interpretation, name=f"{prefix} Interpretation"
                    )
                elif section == "components":
                    add_reduction_components(
                        report,
                        _components_payload(reduction),
                        name=f"{prefix} Component Loadings",
                    )
                elif section == "trajectory":
                    add_reduction_trajectory(
                        report,
                        trajectory,
                        times=times,
                        labels=labels,
                        name=f"{prefix} Trajectory",
                    )
                elif section == "trajectory_separation":
                    add_reduction_trajectory_separation(
                        report,
                        separation,
                        times=times,
                        name=f"{prefix} Trajectory Separation",
                    )
            except Exception as exc:
                logger.debug("Reduction report section %s skipped: %s", section, exc)

    if output_path is not None:
        report.save(output_path)
    return report


def _add_embedding_and_shepard(
    report: Report,
    reduction: Any,
    X_emb: Any,
    *,
    labels: Any,
    metadata: Any,
    prefix: str,
    diagnostics: dict,
) -> None:
    """Add embedding scatter and Shepard diagram side by side when both
    are available.
    """
    from coco_pipe.viz.dim_reduction import plot_embedding, plot_shepard_diagram

    emb_img: ImageElement | None = None
    shepard_img: ImageElement | None = None

    try:
        plot_result = plot_embedding(X_emb, labels=labels, metadata=metadata)
        emb_img = ImageElement(
            plot_result[0] if isinstance(plot_result, tuple) else plot_result,
            caption="Embedding",
        )
    except (TypeError, ValueError) as exc:
        logger.debug("Embedding plot skipped: %s", exc)

    X_orig = getattr(reduction, "X_", None)
    if X_orig is None:
        X_orig = diagnostics.get("X_orig")
    if X_orig is not None:
        try:
            plot_result = plot_shepard_diagram(X_orig, X_emb)
            shepard_img = ImageElement(
                plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                caption="Shepard diagram",
            )
        except (TypeError, ValueError) as exc:
            logger.debug("Shepard diagram skipped: %s", exc)

    sec = Section(title=f"{prefix} Embedding")
    if emb_img is not None and shepard_img is not None:
        tabs = {"Embedding": emb_img, "Shepard Diagram": shepard_img}
        sec.add_element(TabsElement(tabs))
    elif emb_img is not None:
        sec.add_element(emb_img)
    elif shepard_img is not None:
        sec.add_element(shepard_img)
    else:
        return

    report.add_section(sec)


def add_reduction(
    self: Report,
    reducer: Any,
    name: str | None = None,
    *,
    X_emb: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    times: np.ndarray | None = None,
) -> Report:
    """
    Add one scored and optionally interpreted reduction result to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    reducer : Any
        Reduction object implementing ``get_summary()``.
    name : str, optional
        Section title. Defaults to the reduction method name.
    X_emb : np.ndarray, optional
        Explicit embedding to visualize. When omitted, the section renders
        scalar summaries, diagnostics, and interpretation outputs only.
    labels : np.ndarray, optional
        Optional labels aligned with ``X_emb`` for embedding or trajectory plots.
    metadata : dict, optional
        Optional column-oriented metadata aligned with the sample axis of a 2D
        embedding.
    times : np.ndarray, optional
        Optional explicit time axis aligned with the time dimension of a 3D
        trajectory tensor.

    Returns
    -------
    Report
        *self* with the new section appended.

    Raises
    ------
    ValueError
        If ``X_emb`` is provided with an unsupported number of dimensions.

    See Also
    --------
    coco_pipe.dim_reduction.core.DimReduction.get_summary
    coco_pipe.viz.interactive.dim_reduction.plot_embedding
    coco_pipe.viz.interactive.dim_reduction.plot_trajectory
    """
    from coco_pipe.viz.interactive.dim_reduction import (
        plot_embedding as plot_embedding_interactive,
    )
    from coco_pipe.viz.interactive.dim_reduction import (
        plot_loss_history as plot_loss_history_interactive,
    )
    from coco_pipe.viz.interactive.dim_reduction import (
        plot_metrics as plot_metric_details,
    )
    from coco_pipe.viz.interactive.dim_reduction import (
        plot_scree as plot_scree_interactive,
    )
    from coco_pipe.viz.interactive.dim_reduction import (
        plot_trajectory as plot_trajectory_interactive,
    )
    from coco_pipe.viz.interactive.dim_reduction import (
        plot_trajectory_metric_series,
    )

    summary = _get_reducer_summary(reducer)
    method_name = summary["method"]
    title = name or method_name
    sec = Section(title=title)

    if X_emb is not None:
        emb = np.asarray(X_emb)
        if emb.ndim == 2:
            fig = plot_embedding_interactive(
                emb,
                labels=labels,
                metadata=metadata,
                title=f"{title} Embedding",
                dimensions=min(emb.shape[1], 3),
            )
            sec.add_element(PlotlyElement(fig))
        elif emb.ndim == 3:
            time_values = _trajectory_times(summary["diagnostics"], times)
            fig = plot_trajectory_interactive(
                emb,
                times=time_values,
                labels=labels,
                title=f"{title} Trajectory",
                dimensions=min(emb.shape[-1], 3),
            )
            sec.add_element(PlotlyElement(fig))
        else:
            raise ValueError("`X_emb` must be a 2D embedding or 3D trajectory tensor.")

    metrics = summary["metrics"]
    quality_metadata = summary["quality_metadata"]
    scalar_table = {
        **{
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float, np.number))
            and not isinstance(value, bool)
        },
        **{
            key: value
            for key, value in quality_metadata.items()
            if isinstance(value, (int, float, np.number))
            and not isinstance(value, bool)
        },
    }
    if scalar_table:
        sec.add_element(TableElement(scalar_table, title="Quality Metrics"))

    metric_records = summary["metric_records"]
    if metric_records:
        sec.add_element(
            PlotlyElement(
                plot_metric_details(metric_records, title="Metric Details"),
                height="380px",
            )
        )

    diagnostics = summary["diagnostics"]

    loss_history = diagnostics.get("loss_history_")
    if loss_history is not None:
        sec.add_element(
            PlotlyElement(
                plot_loss_history_interactive(loss_history),
                height="350px",
            )
        )

    explained_variance = diagnostics.get("explained_variance_ratio_")
    if explained_variance is not None:
        sec.add_element(
            PlotlyElement(
                plot_scree_interactive(explained_variance),
                height="350px",
            )
        )

    coranking = diagnostics.get("coranking_matrix_")
    if coranking is not None:
        import plotly.graph_objects as go

        fig_coranking = go.Figure(
            data=[
                go.Heatmap(
                    z=np.asarray(coranking),
                    colorscale="Viridis",
                    colorbar={"title": "Count"},
                )
            ]
        )
        fig_coranking.update_layout(
            title="Co-Ranking Matrix",
            xaxis_title="Embedded Rank",
            yaxis_title="Original Rank",
            margin={"l": 40, "r": 40, "b": 40, "t": 40},
            template="plotly_white",
        )
        sec.add_element(PlotlyElement(fig_coranking, height="420px"))

    time_values = _trajectory_times(diagnostics, times)
    trajectory_series = (
        "trajectory_speed_",
        "trajectory_acceleration_",
        "trajectory_curvature_",
        "trajectory_turning_angle_",
        "trajectory_dispersion_",
        "trajectory_path_length_",
        "trajectory_displacement_",
    )
    for metric_key in trajectory_series:
        values = diagnostics.get(metric_key)
        if values is None:
            continue
        sec.add_element(
            PlotlyElement(
                plot_trajectory_metric_series(
                    values,
                    times=time_values,
                    labels=labels,
                    title=metric_key.rstrip("_").replace("_", " ").title(),
                ),
                height="360px",
            )
        )

    separation = diagnostics.get("trajectory_separation_")
    if separation is not None:
        sec.add_element(
            PlotlyElement(
                plot_trajectory_metric_series(
                    separation,
                    times=time_values,
                    title="Trajectory Separation",
                ),
                height="380px",
            )
        )

    self.add_section(sec)
    return self


def add_comparison(
    self: Report,
    metrics_df: Any,
    name: str = "Method Comparison",
) -> Report:
    """
    Add a comparison section for multiple reduction methods to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    metrics_df : DataFrame or MethodSelector-like
        Wide/tidy metric data or an object exposing ``to_frame()``.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended.

    Raises
    ------
    ValueError
        If no comparison metrics are available after normalization.

    See Also
    --------
    coco_pipe.viz.interactive.dim_reduction.plot_metrics
    coco_pipe.dim_reduction.evaluation.core.MethodSelector
    """
    from coco_pipe.viz._utils import prepare_metrics_frame
    from coco_pipe.viz.interactive.dim_reduction import (
        plot_metrics as plot_metric_details,
    )
    from coco_pipe.viz.interactive.dim_reduction import (
        plot_radar_comparison,
    )

    sec = Section(title=name)
    long_df = prepare_metrics_frame(metrics_df)
    summary_table = _metrics_summary_table(long_df)

    if summary_table.empty:
        raise ValueError("No comparison metrics available to add to the report.")

    # 1. Metrics Table (Best values highlighted)
    sec.add_element(MetricsTableElement(summary_table, title="Quality Metrics"))

    # 2. Primary visual summaries
    fig_heatmap = plot_metric_details(
        long_df, title="Metric Heatmap", plot_type="heatmap"
    )
    sec.add_element(PlotlyElement(fig_heatmap, height="400px"))

    fig_primary = plot_metric_details(long_df, title="Metric Details", plot_type="bar")
    sec.add_element(PlotlyElement(fig_primary, height="400px"))

    if (
        long_df["ScopeValue"].astype(str).nunique() == 1
        and summary_table.shape[1] >= 3
        and summary_table.shape[0] >= 2
    ):
        fig_radar = plot_radar_comparison(summary_table, normalize=True)
        sec.add_element(PlotlyElement(fig_radar, height="400px"))

    self.add_section(sec)
    return self


__all__ = [
    "DEFAULT_REDUCTION_SECTIONS",
    "add_comparison",
    "add_reduction",
    "add_reduction_components",
    "add_reduction_coranking",
    "add_reduction_diagnostics",
    "add_reduction_embedding",
    "add_reduction_interpretation",
    "add_reduction_metrics",
    "add_reduction_overview",
    "add_reduction_trajectory",
    "add_reduction_trajectory_separation",
    "make_reduction_report",
]

# Bind section adders as Report methods. Reversing the dependency this way
# keeps the fluent `report.add_reduction*()` / `report.add_comparison()` API
# without forcing core.py to import this module (which would be circular).
Report.add_reduction = add_reduction
Report.add_comparison = add_comparison
Report.add_reduction_overview = add_reduction_overview
Report.add_reduction_embedding = add_reduction_embedding
Report.add_reduction_metrics = add_reduction_metrics
Report.add_reduction_diagnostics = add_reduction_diagnostics
Report.add_reduction_interpretation = add_reduction_interpretation
Report.add_reduction_coranking = add_reduction_coranking
Report.add_reduction_components = add_reduction_components
Report.add_reduction_trajectory = add_reduction_trajectory
Report.add_reduction_trajectory_separation = add_reduction_trajectory_separation
