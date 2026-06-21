"""Composable section builders for decoding reports."""

from __future__ import annotations

import contextlib
import inspect
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import pandas as pd

from coco_pipe.io.quality import QCResult

from ._utils import (
    _add_tabs_or_single,
    _config_element,
    _csv_download,
    _ensure_static_matplotlib_backend,
    _plot_or_none,
    _table_from_mapping,
)
from .core import Report, Section
from .elements import (
    AccordionElement,
    BadgeElement,
    CalloutElement,
    ImageElement,
    TableElement,
    TabsElement,
)
from .qc import build_qc_section


class SectionDataUnavailable(RuntimeError):
    """Signal that a report section is inapplicable because its data is absent."""


DECODING_PRESETS: dict[str, list[str]] = {
    "compact": [
        "overview",
        "model_summary",
        "cv",
        "probability",
        "statistical",
        "features",
        "topomaps",
    ],
    "default": [
        "overview",
        "configuration",
        "provenance",
        "model_summary",
        "cv",
        "performance",
        "statistical",
        "probability",
        "temporal",
        "features",
        "fit_diagnostics",
        "tuning",
        "caveats",
        "export_inventory",
    ],
    "full": [
        "overview",
        "configuration",
        "provenance",
        "model_summary",
        "cv",
        "performance",
        "statistical",
        "probability",
        "temporal",
        "features",
        "topomaps",
        "fit_diagnostics",
        "tuning",
        "neural",
        "caveats",
        "export_inventory",
    ],
}
DEFAULT_SECTIONS = DECODING_PRESETS["default"]
_SECTION_ALIASES = {
    "cv_summary": "cv",
    "confusion_probability": "probability",
}


def _accepted_kwargs(method: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop kwargs the accessor does not accept, unless it takes ``**kwargs``."""
    if not kwargs:
        return {}
    signature = inspect.signature(method)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs)
    return {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }


def _result_frame(
    result: Any,
    accessor: str,
    *,
    required: bool = True,
    context: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fetch ``result.<accessor>(**kwargs)`` as a DataFrame.

    Every ``~coco_pipe.decoding.result.ExperimentResult`` accessor exists and returns an
    empty frame when it holds no data, so callers branch on emptiness rather than
    ``hasattr``. The two section policies share this one fetch:

    - ``required=True`` (a section's primary data): a missing accessor or empty
      result raises :class:`SectionDataUnavailable`, and accessor errors propagate.
    - ``required=False`` (one of several optional inputs): the same conditions, and
      any accessor error, yield an empty frame for the caller to test with ``.empty``.
    """
    label = context or accessor
    method = getattr(result, accessor, None)
    if not callable(method):
        if required:
            raise SectionDataUnavailable(f"{label} requires {accessor}().")
        return pd.DataFrame()
    try:
        frame = pd.DataFrame(method(**_accepted_kwargs(method, kwargs)))
    except (TypeError, ValueError, KeyError):
        if required:
            raise
        return pd.DataFrame()
    if frame.empty and required:
        raise SectionDataUnavailable(f"No {label.lower()} data are available.")
    return frame


def _resolve_decoding_plotter(
    name: str, interactive: bool
) -> tuple[Callable[..., Any], bool]:
    """Return ``(plotter, as_plotly)`` for *name*, preferring the interactive twin.

    When ``interactive`` is requested but no Plotly twin exists (topomap and
    sensor-profile plots are Matplotlib-only), the static plotter is returned and
    rendered as an image, so an interactive report degrades gracefully per plot.
    """
    if interactive:
        from coco_pipe.viz.interactive import decoding as interactive_viz

        plotter = getattr(interactive_viz, name, None)
        if plotter is not None:
            return plotter, True
    from coco_pipe.viz import decoding as static_viz

    return getattr(static_viz, name), False


def _decoding_plot(
    *args: Any,
    name: str,
    interactive: bool,
    caption: str,
    **kwargs: Any,
) -> Any:
    """Resolve and render a decoding plotter, returning ``None`` if it can't draw."""
    plotter, as_plotly = _resolve_decoding_plotter(name, interactive)
    return _plot_or_none(plotter, *args, caption=caption, as_plotly=as_plotly, **kwargs)


def build_decoding_overview_section(
    result: Any,
    *,
    name: str = "Overview",
) -> Section:
    """Build high-level decoding context."""
    meta = getattr(result, "meta", {}) or {}
    rows = [
        {
            "Task": meta.get("task"),
            "Samples": meta.get("n_samples"),
            "Features": meta.get("n_features"),
            "ObservationLevel": meta.get("observation_level"),
            "InferentialUnit": meta.get("inferential_unit"),
            "SchemaVersion": getattr(result, "schema_version", None),
        }
    ]
    section = Section(title=name)
    section.add_element(TableElement(pd.DataFrame(rows), title="Decoding Context"))
    return section


def build_decoding_summary_section(
    result: Any,
    *,
    name: str = "Model Performance",
) -> Section:
    """Build the scalar model-performance summary."""
    summary = _result_frame(result, "summary", context="Model performance")
    summary = summary.reset_index() if summary.index.name else summary
    section = Section(title=name)
    section.add_element(
        _csv_download(summary, "decoding_summary.csv", "Download Summary CSV")
    )
    section.add_element(TableElement(summary, title="Model Performance Summary"))
    return section


def build_cv_section(
    result: Any,
    *,
    metric: str | None = None,
    model: str | None = None,
    name: str = "Cross-Validation",
    include_tables: bool = False,
    interactive: bool = False,
) -> Section:
    """Build fold scores and score dispersion only."""
    scores = _result_frame(
        result,
        "get_detailed_scores",
        context="Cross-validation",
        model=model,
    )
    if metric is not None and "Metric" in scores:
        scores = scores[scores["Metric"] == metric]
    if scores.empty:
        raise SectionDataUnavailable("No matching cross-validation scores.")

    section = Section(
        title=name,
        description="Per-fold score spread — how stable the metric is across "
        "cross-validation folds.",
    )
    section.add_element(
        _csv_download(scores, "fold_scores.csv", "Download Fold Scores CSV")
    )
    if include_tables:
        section.add_element(TableElement(scores, title="Fold Scores"))
    image = _decoding_plot(
        scores,
        name="plot_fold_score_dispersion",
        interactive=interactive,
        metric=metric,
        model=model,
        caption="Fold score dispersion",
    )
    if image is not None:
        section.add_element(image)
    return section


def build_probability_section(
    result: Any,
    *,
    model: str | None = None,
    name: str = "Confusion and Probability",
    include_tables: bool = False,
    interactive: bool = False,
) -> Section:
    """Build confusion, ROC, precision-recall, and calibration diagnostics."""
    confusion = _result_frame(
        result, "get_confusion_matrices", required=False, model=model
    )

    models = [model] if model is not None else []
    if not models and "Model" in confusion:
        models = list(dict.fromkeys(confusion["Model"].dropna().astype(str)))
    if not models:
        models = list(getattr(result, "raw", {}) or {})
    if not models and confusion.empty:
        raise SectionDataUnavailable("No classification probability diagnostics.")

    section = Section(title=name)
    if not confusion.empty:
        section.add_element(
            _csv_download(
                confusion,
                "confusion_matrices.csv",
                "Download Confusion Matrix CSV",
            )
        )
        if include_tables:
            section.add_element(TableElement(confusion, title="Confusion Matrix Data"))

    model_blocks: dict[str, AccordionElement] = {}
    for model_name in models or [None]:
        label = str(model_name or "Model")
        block = AccordionElement(label, open=len(models) <= 1)
        images = {
            "Confusion Matrix": _decoding_plot(
                confusion,
                name="plot_confusion_matrix",
                interactive=interactive,
                model=model_name,
                caption=f"Confusion matrix for {label}",
            )
            if not confusion.empty
            else None,
            "ROC Curve": _decoding_plot(
                result,
                name="plot_roc_curve",
                interactive=interactive,
                model=model_name,
                mean_only=True,
                caption=f"ROC curve for {label}",
            ),
            "Precision-Recall": _decoding_plot(
                result,
                name="plot_pr_curve",
                interactive=interactive,
                model=model_name,
                mean_only=True,
                caption=f"Precision-recall curve for {label}",
            ),
            "Calibration": _decoding_plot(
                result,
                name="plot_calibration_curve",
                interactive=interactive,
                model=model_name,
                mean_only=True,
                caption=f"Calibration curve for {label}",
            ),
        }
        _add_tabs_or_single(block, images)
        if block.children:
            model_blocks[label] = block

    if not model_blocks and confusion.empty:
        raise SectionDataUnavailable("No classification probability diagnostics.")
    if len(model_blocks) == 1:
        section.add_element(next(iter(model_blocks.values())))
    elif model_blocks:
        section.add_element(TabsElement(model_blocks))
    return section


def build_decoding_diagnostics_section(
    result: Any,
    *,
    metric: str | None = None,
    model: str | None = None,
    name: str = "Decoding Diagnostics",
    include_tables: bool = False,
    interactive: bool = False,
) -> Section:
    """Combine the now-separated CV and probability blocks into one section.

    Legacy-only: this re-colocates diagnostics that the presets deliberately keep
    apart. It backs the deprecated :func:`add_decoding_diagnostics` method and is
    intentionally absent from every preset, so it cannot reintroduce the historical
    double-render. Prefer ``build_cv_section`` + ``build_probability_section``.
    """
    section = Section(title=name)
    builders = (
        (
            build_cv_section,
            {
                "metric": metric,
                "model": model,
                "include_tables": include_tables,
                "interactive": interactive,
            },
        ),
        (
            build_probability_section,
            {
                "model": model,
                "include_tables": include_tables,
                "interactive": interactive,
            },
        ),
    )
    for builder, kwargs in builders:
        try:
            child = builder(result, **kwargs)
        except SectionDataUnavailable:
            continue
        accordion = AccordionElement(child.title, open=True)
        for element in child.children:
            accordion.add_element(element)
        section.add_element(accordion)
    if not section.children:
        raise SectionDataUnavailable("No decoding diagnostics are available.")
    return section


def build_statistical_section(
    result: Any,
    *,
    metric: str | None = None,
    model: str | None = None,
    name: str = "Statistical Assessment",
    interactive: bool = False,
) -> Section:
    """Build finite-sample and temporal statistical assessment."""
    assessment = _result_frame(
        result,
        "get_statistical_assessment",
        context="Statistical assessment",
    )
    if metric is not None and "Metric" in assessment:
        assessment = assessment[assessment["Metric"] == metric]
    if model is not None and "Model" in assessment:
        assessment = assessment[assessment["Model"] == model]
    if assessment.empty:
        raise SectionDataUnavailable("No matching statistical assessment.")

    section = Section(title=name)
    section.add_element(
        _csv_download(
            assessment,
            "statistical_assessment.csv",
            "Download Statistical Assessment CSV",
        )
    )
    section.add_element(
        TableElement(assessment, title="Finite-Sample Statistical Assessment")
    )
    null_image = _decoding_plot(
        assessment,
        name="plot_null_interval_summary",
        interactive=interactive,
        caption="Null interval summary",
    )
    if null_image is not None:
        section.add_element(null_image)
    if "Time" in assessment and assessment["Time"].notna().any():
        temporal_image = _decoding_plot(
            assessment,
            name="plot_temporal_statistical_assessment",
            interactive=interactive,
            metric=metric,
            model=model,
            caption="Temporal statistical assessment",
        )
        if temporal_image is not None:
            section.add_element(temporal_image)
    return section


def build_temporal_section(
    result: Any,
    *,
    metric: str | None = None,
    model: str | None = None,
    name: str = "Temporal Decoding",
    include_tables: bool = False,
    interactive: bool = False,
) -> Section:
    """Build temporal score and generalization diagnostics when applicable."""
    summary = _result_frame(
        result,
        "get_temporal_score_summary",
        context="Temporal decoding",
        model=model,
    )
    if metric is not None and "Metric" in summary:
        summary = summary[summary["Metric"] == metric]
    if summary.empty:
        raise SectionDataUnavailable("No matching temporal decoding scores.")
    has_time = "Time" in summary and summary["Time"].notna().any()
    has_generalization = (
        {"TrainTime", "TestTime"}.issubset(summary.columns)
        and summary["TrainTime"].notna().any()
        and summary["TestTime"].notna().any()
    )
    if not has_time and not has_generalization:
        raise SectionDataUnavailable("The result is not temporally resolved.")

    section = Section(title=name)
    section.add_element(
        _csv_download(summary, "temporal_scores.csv", "Download Temporal Scores CSV")
    )
    if include_tables:
        section.add_element(TableElement(summary, title="Temporal Score Summary"))
    if metric:
        section.add_element(BadgeElement(f"Metric: {metric}", "purple"))
    if model:
        section.add_element(BadgeElement(f"Model: {model}", "blue"))
    if has_time:
        image = _decoding_plot(
            summary,
            name="plot_temporal_score_curve",
            interactive=interactive,
            metric=metric,
            model=model,
            caption="Temporal score curve",
        )
        if image is not None:
            section.add_element(image)
    if has_generalization:
        image = _decoding_plot(
            summary,
            name="plot_temporal_generalization_matrix",
            interactive=interactive,
            metric=metric,
            model=model,
            caption="Temporal generalization matrix",
        )
        if image is not None:
            section.add_element(image)
    return section


def build_performance_section(
    result: Any,
    *,
    metric: str | None = None,
    name: str = "Performance",
    interactive: bool = False,
) -> Section:
    """Build aggregate score and paired model-comparison figures."""
    scores = _result_frame(result, "get_detailed_scores", context="Performance")
    images = {
        "Score Distribution": _decoding_plot(
            scores,
            name="plot_decoding_scores",
            interactive=interactive,
            metric=metric,
            caption="Decoding scores",
        )
    }
    models = list(dict.fromkeys(scores["Model"])) if "Model" in scores else []
    if len(models) > 1:
        images["Model Comparison"] = _decoding_plot(
            result,
            name="plot_model_comparison",
            interactive=interactive,
            metric=metric or "accuracy",
            caption="Paired model comparison",
        )
    section = Section(
        title=name,
        description="Aggregate score distribution and between-model comparison — "
        "how the models rank, not fold-to-fold stability (see Cross-Validation).",
    )
    _add_tabs_or_single(section, images)
    if not section.children:
        raise SectionDataUnavailable("No performance plots are available.")
    return section


def build_features_section(
    result: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    model: str | None = None,
    top_n: int = 20,
    name: str = "Features",
    include_tables: bool = False,
    interactive: bool = False,
) -> Section:
    """Build feature importance, stability, and sensor-family summaries."""
    importances = _result_frame(
        result, "get_feature_importances", required=False, model=model
    )
    stability = _result_frame(
        result, "get_feature_stability", required=False, model=model
    )
    metadata = (
        pd.DataFrame(feature_metadata).drop_duplicates()
        if feature_metadata is not None
        else pd.DataFrame()
    )
    if importances.empty and stability.empty:
        raise SectionDataUnavailable("No feature diagnostics are available.")

    section = Section(title=name)
    if not importances.empty:
        section.add_element(
            _csv_download(
                importances,
                "feature_importances.csv",
                "Download Feature Importances CSV",
            )
        )
    if not stability.empty:
        section.add_element(
            _csv_download(
                stability,
                "feature_stability.csv",
                "Download Feature Stability CSV",
            )
        )

    models = [model] if model is not None else []
    if not models and "Model" in importances:
        models = list(dict.fromkeys(importances["Model"].dropna().astype(str)))
    if not models:
        models = [None]
    model_tabs: dict[str, AccordionElement] = {}
    for model_name in models:
        label = str(model_name or "Model")
        block = AccordionElement(label, open=len(models) <= 1)
        model_importances = (
            importances[importances["Model"].astype(str) == str(model_name)]
            if model_name is not None and "Model" in importances
            else importances
        )
        model_stability = (
            stability[stability["Model"].astype(str) == str(model_name)]
            if model_name is not None and "Model" in stability
            else stability
        )
        images: dict[str, Any] = {}
        if not model_importances.empty:
            images["Importance"] = _decoding_plot(
                model_importances,
                name="plot_feature_importance",
                interactive=interactive,
                model=model_name,
                top_n=top_n,
                caption=f"Top feature importances for {label}",
            )
            required = {"FeatureName", "Sensor", "FeatureFamily"}
            if required.issubset(metadata.columns):
                images["Sensor x Family"] = _decoding_plot(
                    model_importances,
                    name="plot_sensor_feature_heatmap",
                    interactive=interactive,
                    feature_metadata=metadata,
                    caption=f"Sensor-by-feature-family importance for {label}",
                )
        if not model_stability.empty:
            images["Stability"] = _decoding_plot(
                model_stability,
                name="plot_feature_stability",
                interactive=interactive,
                model=model_name,
                caption=f"Feature-selection stability for {label}",
            )
        _add_tabs_or_single(block, images)
        if block.children:
            model_tabs[label] = block
    if len(model_tabs) == 1:
        section.add_element(next(iter(model_tabs.values())))
    elif model_tabs:
        section.add_element(TabsElement(model_tabs))

    if not metadata.empty:
        section.add_element(
            _csv_download(
                metadata,
                "feature_metadata.csv",
                "Download Feature Metadata CSV",
            )
        )
        if include_tables:
            section.add_element(TableElement(metadata, title="Feature Metadata"))
    return section


def build_topomaps_section(
    result: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    info: Any = None,
    coords: Any = None,
    name: str = "Sensor Maps",
) -> Section:
    """Build feature-family sensor profiles when spatial metadata exists."""
    from coco_pipe.viz.decoding import plot_feature_sensor_profile

    metadata = pd.DataFrame(feature_metadata)
    if metadata.empty or "FeatureFamily" not in metadata:
        raise SectionDataUnavailable("Feature-family metadata is unavailable.")
    if info is None and coords is None:
        raise SectionDataUnavailable("Sensor positions are unavailable.")

    section = Section(title=name)
    tabs: dict[str, ImageElement] = {}
    for family in pd.unique(metadata["FeatureFamily"].dropna()):
        image = _plot_or_none(
            plot_feature_sensor_profile,
            result,
            feature_metadata=metadata,
            feature_family=str(family),
            info=info,
            coords=coords,
            caption=f"{family} sensor profile",
        )
        if image is not None:
            tabs[str(family)] = image
    _add_tabs_or_single(section, tabs)
    if not section.children:
        raise SectionDataUnavailable("No sensor maps could be rendered.")
    return section


def build_fit_diagnostics_section(
    result: Any,
    *,
    model: str | None = None,
    name: str = "Fit Diagnostics",
    include_tables: bool = False,
    supplementary: bool = True,
    interactive: bool = False,
) -> Section:
    """Build timing and warning diagnostics as supplementary content."""
    diagnostics = _result_frame(
        result,
        "get_fit_diagnostics",
        context="Fit diagnostics",
    )
    if model is not None and "Model" in diagnostics:
        diagnostics = diagnostics[diagnostics["Model"] == model]
    if diagnostics.empty:
        raise SectionDataUnavailable("No matching fit diagnostics.")

    section = Section(title=name)
    target: Section | AccordionElement = section
    if supplementary:
        target = AccordionElement("Show fit timing and warnings", open=False)
        section.add_element(target)
    target.add_element(
        _csv_download(
            diagnostics,
            "fit_diagnostics.csv",
            "Download Fit Diagnostics CSV",
        )
    )
    if include_tables:
        target.add_element(TableElement(diagnostics, title="Fit Diagnostics"))
    image = _decoding_plot(
        diagnostics,
        name="plot_fit_diagnostics",
        interactive=interactive,
        caption="Fit timing diagnostics",
    )
    if image is not None:
        target.add_element(image)
    warnings_frame = (
        diagnostics[diagnostics["WarningMessage"].notna()]
        if "WarningMessage" in diagnostics
        else pd.DataFrame()
    )
    if not warnings_frame.empty:
        target.add_element(
            CalloutElement(
                f"{len(warnings_frame)} training warning record(s) were captured.",
                kind="warning",
                title="Training warnings",
            )
        )
    return section


def build_tuning_section(
    result: Any,
    *,
    model: str | None = None,
    name: str = "Hyperparameter Tuning",
    include_tables: bool = False,
    supplementary: bool = True,
    interactive: bool = False,
) -> Section:
    """Build best-parameter and search-result diagnostics."""
    search = _result_frame(result, "get_search_results", context="Tuning")
    if model is not None and "Model" in search:
        search = search[search["Model"] == model]
    if search.empty:
        raise SectionDataUnavailable("No matching tuning results.")
    best = _result_frame(result, "get_best_params", required=False, model=model)

    section = Section(title=name)
    target: Section | AccordionElement = section
    if supplementary:
        target = AccordionElement("Show tuning details", open=False)
        section.add_element(target)
    target.add_element(
        _csv_download(search, "search_results.csv", "Download Search Results CSV")
    )
    if not best.empty:
        target.add_element(
            _csv_download(best, "best_params.csv", "Download Best Parameters CSV")
        )
    if include_tables:
        target.add_element(TableElement(search, title="Search Results"))
        if not best.empty:
            target.add_element(TableElement(best, title="Best Parameters"))
    image = _decoding_plot(
        search,
        name="plot_search_results",
        interactive=interactive,
        model=model,
        top_n=20,
        caption="Top hyperparameter-search candidates",
    )
    if image is not None:
        target.add_element(image)
    return section


def build_neural_section(
    result: Any,
    *,
    model: str | None = None,
    name: str = "Neural Artifacts",
    include_tables: bool = False,
    interactive: bool = False,
) -> Section:
    """Build neural training artifacts when available."""
    artifacts = _result_frame(result, "get_model_artifacts", context="Neural artifacts")
    if model is not None and "Model" in artifacts:
        artifacts = artifacts[artifacts["Model"] == model]
    if artifacts.empty:
        raise SectionDataUnavailable("No matching neural artifacts.")
    section = Section(title=name)
    section.add_element(
        _csv_download(artifacts, "model_artifacts.csv", "Download Model Artifacts CSV")
    )
    if include_tables:
        section.add_element(TableElement(artifacts, title="Model Artifacts"))
    image = _decoding_plot(
        artifacts,
        name="plot_training_history",
        interactive=interactive,
        model=model,
        caption="Training history",
    )
    if image is not None:
        section.add_element(image)
    return section


def build_configuration_section(
    result: Any,
    *,
    name: str = "Configuration",
) -> Section:
    """Build the stored run configuration."""
    config = getattr(result, "config", {}) or {}
    if not config:
        raise SectionDataUnavailable("Run configuration is unavailable.")
    section = Section(title=name)
    section.add_element(_config_element(config, title="Run Configuration"))
    return section


def build_provenance_section(
    result: Any,
    *,
    name: str = "Provenance",
) -> Section:
    """Build environment and provenance metadata."""
    metadata = getattr(result, "meta", {}) or {}
    if not metadata:
        raise SectionDataUnavailable("Provenance metadata are unavailable.")
    section = Section(title=name)
    section.add_element(_table_from_mapping(metadata, title="Environment"))
    return section


def build_caveats_section(
    result: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    name: str = "Caveats",
) -> Section:
    """Build explicit caveats for unavailable optional diagnostics."""
    caveats: list[str] = []
    if feature_metadata is None:
        caveats.append(
            "Feature metadata was not provided; sensor-wise feature plots were skipped."
        )
    if (
        callable(getattr(result, "get_probability_diagnostics", None))
        and _result_frame(result, "get_probability_diagnostics", required=False).empty
    ):
        caveats.append("Probability diagnostics were unavailable.")
    if not caveats:
        raise SectionDataUnavailable("No report caveats were identified.")
    section = Section(title=name)
    for caveat in caveats:
        section.add_element(CalloutElement(caveat, kind="warning"))
    return section


def build_export_inventory_section(
    result: Any,
    *,
    name: str = "Export Inventory",
) -> Section:
    """Build a compact inventory of available result accessors."""
    accessors = [
        "summary",
        "get_detailed_scores",
        "get_predictions",
        "get_fit_diagnostics",
        "get_statistical_assessment",
        "get_feature_importances",
        "get_search_results",
    ]
    rows = []
    for accessor in accessors:
        if not hasattr(result, accessor):
            continue
        try:
            value = getattr(result, accessor)()
            rows.append(
                {
                    "Accessor": accessor,
                    "Rows": len(value) if hasattr(value, "__len__") else None,
                }
            )
        except (TypeError, ValueError, KeyError) as exc:
            rows.append({"Accessor": accessor, "Rows": None, "Error": str(exc)})
    if not rows:
        raise SectionDataUnavailable("No result exports are available.")
    section = Section(title=name)
    section.add_element(TableElement(pd.DataFrame(rows), title="Available Tables"))
    return section


DECODING_SECTION_BUILDERS: dict[str, Callable[..., Section]] = {
    "overview": build_decoding_overview_section,
    "configuration": build_configuration_section,
    "provenance": build_provenance_section,
    "model_summary": build_decoding_summary_section,
    "cv": build_cv_section,
    "performance": build_performance_section,
    "statistical": build_statistical_section,
    "probability": build_probability_section,
    "temporal": build_temporal_section,
    "features": build_features_section,
    "topomaps": build_topomaps_section,
    "fit_diagnostics": build_fit_diagnostics_section,
    "tuning": build_tuning_section,
    "neural": build_neural_section,
    "caveats": build_caveats_section,
    "export_inventory": build_export_inventory_section,
}
VALID_SECTIONS = set(DECODING_SECTION_BUILDERS) | set(_SECTION_ALIASES)

_INTERACTIVE_AWARE_SECTIONS = {
    "cv",
    "probability",
    "statistical",
    "temporal",
    "performance",
    "features",
    "fit_diagnostics",
    "tuning",
    "neural",
}


def _resolve_decoding_sections(
    sections: str | Sequence[str],
) -> list[str]:
    if isinstance(sections, str):
        if sections not in DECODING_PRESETS:
            choices = ", ".join(sorted(DECODING_PRESETS))
            raise ValueError(
                f"Unknown decoding report preset {sections!r}. Choose from: {choices}."
            )
        return list(DECODING_PRESETS[sections])
    selected = [_SECTION_ALIASES.get(key, key) for key in sections]
    unknown = [key for key in selected if key not in DECODING_SECTION_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown decoding report section(s): {', '.join(unknown)}")
    return selected


def _builder_kwargs(
    key: str,
    *,
    feature_metadata: pd.DataFrame | None,
    info: Any,
    coords: Any,
    verbose: bool,
    interactive: bool,
    section_options: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    kwargs = dict(section_options.get(key, {}))
    if key in {"features", "topomaps", "caveats"}:
        kwargs.setdefault("feature_metadata", feature_metadata)
    if key == "topomaps":
        kwargs.setdefault("info", info)
        kwargs.setdefault("coords", coords)
    if key in {
        "cv",
        "probability",
        "temporal",
        "features",
        "fit_diagnostics",
        "tuning",
        "neural",
    }:
        kwargs.setdefault("include_tables", verbose)
    if key in _INTERACTIVE_AWARE_SECTIONS:
        kwargs.setdefault("interactive", interactive)
    return kwargs


def build_decoding_sections(
    result: Any,
    *,
    sections: str | Sequence[str] = "default",
    feature_metadata: pd.DataFrame | None = None,
    info: Any = None,
    coords: Any = None,
    verbose: bool | None = None,
    interactive: bool = False,
    on_error: Literal["raise", "warn", "placeholder"] = "warn",
    section_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Section]:
    """Build ordered decoding sections without creating or mutating a report."""
    # Even interactive reports embed Matplotlib images for static-only plots
    # (topomaps, sensor maps), so the headless backend is always required.
    _ensure_static_matplotlib_backend()
    if on_error not in {"raise", "warn", "placeholder"}:
        raise ValueError("on_error must be 'raise', 'warn', or 'placeholder'.")
    selected = _resolve_decoding_sections(sections)
    include_tables = sections == "full" if verbose is None else bool(verbose)
    options = section_options or {}
    built: list[Section] = []
    for key in selected:
        builder = DECODING_SECTION_BUILDERS[key]
        kwargs = _builder_kwargs(
            key,
            feature_metadata=feature_metadata,
            info=info,
            coords=coords,
            verbose=include_tables,
            interactive=interactive,
            section_options=options,
        )
        try:
            section = builder(result, **kwargs)
        except SectionDataUnavailable:
            continue
        except Exception as exc:
            if on_error == "raise":
                raise
            message = f"Decoding report section {key!r} failed: {exc}"
            if on_error == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            else:
                section = Section(title=key.replace("_", " ").title())
                section.status = "WARN"
                section.add_element(
                    CalloutElement(message, kind="error", title="Section failed")
                )
                built.append(section)
            continue
        built.append(section)
    return built


def make_decoding_report(
    result: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    info: Any = None,
    coords: Any = None,
    sections: str | Sequence[str] = "default",
    interactive: bool = False,
    theme: Literal["paper", "notebook", "poster"] = "paper",
    title: str = "Decoding Report",
    config: dict | None = None,
    asset_urls: dict[str, str] | str | None = None,
    qc_result: QCResult | None = None,
    output_path: str | None = None,
    verbose: bool | None = None,
    on_error: Literal["raise", "warn", "placeholder"] = "warn",
    section_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> Report:
    """Build a decoding report from one ``~coco_pipe.decoding.result.ExperimentResult``.

    With ``interactive=True``, chart-like sections render Plotly figures; topomap
    and sensor-map sections remain Matplotlib images (no Plotly twin exists).
    """
    run_config = {"theme": theme, **(config or {})}
    report = Report(title=title, config=run_config, theme=theme, asset_urls=asset_urls)
    if qc_result is not None:
        report.add_section(build_qc_section(qc_result))
    for section in build_decoding_sections(
        result,
        sections=sections,
        feature_metadata=feature_metadata,
        info=info,
        coords=coords,
        verbose=verbose,
        interactive=interactive,
        on_error=on_error,
        section_options=section_options,
    ):
        report.add_section(section)
    if output_path is not None:
        report.save(output_path)
    return report


def _append_section(
    report: Report,
    builder: Callable[..., Section],
    result: Any,
    **kwargs: Any,
) -> Report:
    with contextlib.suppress(SectionDataUnavailable):
        report.add_section(builder(result, **kwargs))
    return report


def add_decoding_overview(
    self: Report, result: Any, *, name: str = "Overview"
) -> Report:
    return _append_section(self, build_decoding_overview_section, result, name=name)


def add_decoding_summary(
    self: Report, result: Any, name: str = "Decoding Summary"
) -> Report:
    return _append_section(self, build_decoding_summary_section, result, name=name)


def add_decoding_diagnostics(
    self: Report,
    result: Any,
    metric: str | None = None,
    model: str | None = None,
    name: str = "Decoding Diagnostics",
) -> Report:
    return _append_section(
        self,
        build_decoding_diagnostics_section,
        result,
        metric=metric,
        model=model,
        name=name,
    )


def add_decoding_statistical_assessment(
    self: Report,
    result: Any,
    metric: str | None = None,
    model: str | None = None,
    name: str = "Statistical Assessment",
) -> Report:
    return _append_section(
        self,
        build_statistical_section,
        result,
        metric=metric,
        model=model,
        name=name,
    )


def add_decoding_temporal(
    self: Report,
    result: Any,
    metric: str | None = None,
    model: str | None = None,
    name: str = "Temporal Decoding",
) -> Report:
    return _append_section(
        self,
        build_temporal_section,
        result,
        metric=metric,
        model=model,
        name=name,
    )


def add_decoding_performance(
    self: Report,
    result: Any,
    *,
    metric: str | None = None,
    name: str = "Performance",
) -> Report:
    return _append_section(
        self, build_performance_section, result, metric=metric, name=name
    )


def add_decoding_features(
    self: Report,
    result: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    name: str = "Features",
) -> Report:
    return _append_section(
        self,
        build_features_section,
        result,
        feature_metadata=feature_metadata,
        name=name,
    )


def add_decoding_topomaps(
    self: Report,
    result: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    info: Any = None,
    coords: Any = None,
    name: str = "Sensor Maps",
) -> Report:
    return _append_section(
        self,
        build_topomaps_section,
        result,
        feature_metadata=feature_metadata,
        info=info,
        coords=coords,
        name=name,
    )


def add_decoding_neural_artifacts(
    self: Report,
    result: Any,
    model: str | None = None,
    name: str = "Neural Artifacts",
) -> Report:
    return _append_section(
        self,
        build_neural_section,
        result,
        model=model,
        name=name,
    )


__all__ = [
    "DECODING_PRESETS",
    "DECODING_SECTION_BUILDERS",
    "DEFAULT_SECTIONS",
    "VALID_SECTIONS",
    "SectionDataUnavailable",
    "add_decoding_diagnostics",
    "add_decoding_features",
    "add_decoding_neural_artifacts",
    "add_decoding_overview",
    "add_decoding_performance",
    "add_decoding_statistical_assessment",
    "add_decoding_summary",
    "add_decoding_temporal",
    "add_decoding_topomaps",
    "build_caveats_section",
    "build_configuration_section",
    "build_cv_section",
    "build_decoding_diagnostics_section",
    "build_decoding_overview_section",
    "build_decoding_sections",
    "build_decoding_summary_section",
    "build_export_inventory_section",
    "build_features_section",
    "build_fit_diagnostics_section",
    "build_neural_section",
    "build_performance_section",
    "build_probability_section",
    "build_provenance_section",
    "build_statistical_section",
    "build_temporal_section",
    "build_topomaps_section",
    "build_tuning_section",
    "make_decoding_report",
]

Report.add_decoding_overview = add_decoding_overview
Report.add_decoding_temporal = add_decoding_temporal
Report.add_decoding_summary = add_decoding_summary
Report.add_decoding_diagnostics = add_decoding_diagnostics
Report.add_decoding_statistical_assessment = add_decoding_statistical_assessment
Report.add_decoding_neural_artifacts = add_decoding_neural_artifacts
Report.add_decoding_performance = add_decoding_performance
Report.add_decoding_features = add_decoding_features
Report.add_decoding_topomaps = add_decoding_topomaps
