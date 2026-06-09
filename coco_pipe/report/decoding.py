"""Section builders for decoding (ExperimentResult) reports."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal

import pandas as pd

from coco_pipe.io.quality import QCResult

from ._utils import (
    _config_element,
    _resolve_sections,
    _table_from_mapping,
)
from .core import Report, Section
from .elements import (
    BadgeElement,
    ColumnsElement,
    DownloadAssetElement,
    ImageElement,
    TableElement,
    TabsElement,
)
from .qc import build_qc_section

logger = logging.getLogger(__name__)

DEFAULT_SECTIONS: list[str] = [
    "overview",
    "configuration",
    "provenance",
    "model_summary",
    "cv_summary",
    "performance",
    "statistical",
    "confusion_probability",
    "temporal",
    "features",
    "fit_diagnostics",
    "caveats",
    "export_inventory",
]
VALID_SECTIONS = set(DEFAULT_SECTIONS) | {"topomaps"}


def add_decoding_overview(
    self: Report,
    result: Any,
    *,
    name: str = "Overview",
) -> Report:
    """Add a summary table of the decoding context to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    result : Any
        Decoding result object (e.g. ``ExperimentResult``).
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended.

    See Also
    --------
    add_decoding_summary : Model-level performance summary table.
    make_decoding_report : Factory that calls this and all other adders.

    Examples
    --------
    >>> report = Report(title="My Report")
    >>> report.add_decoding_overview(result)
    >>> report.save("report.html")
    """
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
    sec = Section(title=name)
    sec.add_element(TableElement(pd.DataFrame(rows), title="Decoding Context"))
    self.add_section(sec)
    return self


def add_decoding_temporal(
    self: Report,
    result: Any,
    metric: str | None = None,
    model: str | None = None,
    name: str = "Temporal Decoding",
) -> Report:
    """Add temporal score curve and generalisation matrix plots to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    result : Any
        Must expose ``get_temporal_score_summary() -> pd.DataFrame``.
    metric : str, optional
        Filter to a single metric name.
    model : str, optional
        Filter to a single model name.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if data is absent.

    Raises
    ------
    TypeError
        If *result* does not provide ``get_temporal_score_summary``.

    See Also
    --------
    coco_pipe.viz.decoding.plot_temporal_score_curve
    coco_pipe.viz.decoding.plot_temporal_generalization_matrix

    Examples
    --------
    >>> report = Report(title="My Report")
    >>> report.add_decoding_temporal(result, metric="accuracy")
    """
    from coco_pipe.viz.decoding import (
        plot_temporal_generalization_matrix,
        plot_temporal_score_curve,
    )

    if not hasattr(result, "get_temporal_score_summary"):
        raise TypeError("result must provide get_temporal_score_summary().")
    summary = result.get_temporal_score_summary()
    if metric is not None and "Metric" in summary:
        summary = summary[summary["Metric"] == metric]
    if model is not None and "Model" in summary:
        summary = summary[summary["Model"] == model]
    if summary.empty:
        return self

    sec = Section(title=name)
    csv_data = summary.to_csv(index=False)
    if metric:
        sec.add_element(BadgeElement(f"Metric: {metric}", "purple"))
    if model:
        sec.add_element(BadgeElement(f"Model: {model}", "blue"))
    sec.add_element(
        DownloadAssetElement(
            csv_data,
            "temporal_scores.csv",
            "text/csv",
            label="Download Temporal CSV",
            style="gray",
        )
    )
    sec.add_element(TableElement(summary, title="Temporal Score Summary"))
    if "Time" in summary and summary["Time"].notna().any():
        try:
            plot_result = plot_temporal_score_curve(summary, metric=metric, model=model)
            sec.add_element(
                ImageElement(
                    plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                    caption="Temporal score curve",
                )
            )
        except ValueError as exc:
            logger.debug("Temporal curve skipped: %s", exc)
    if (
        {"TrainTime", "TestTime"}.issubset(summary.columns)
        and summary["TrainTime"].notna().any()
        and summary["TestTime"].notna().any()
    ):
        try:
            plot_result = plot_temporal_generalization_matrix(
                summary, metric=metric, model=model
            )
            sec.add_element(
                ImageElement(
                    plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                    caption="Temporal generalization matrix",
                )
            )
        except ValueError as exc:
            logger.debug("Temporal matrix skipped: %s", exc)
    self.add_section(sec)
    return self


def add_decoding_summary(
    self: Report,
    result: Any,
    name: str = "Decoding Summary",
) -> Report:
    """Add a model-performance summary table to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    result : Any
        Must expose ``summary() -> pd.DataFrame``.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if empty.

    Raises
    ------
    TypeError
        If *result* does not provide ``summary``.

    See Also
    --------
    add_decoding_overview : High-level context table.
    add_decoding_performance : Score distribution plots.

    Examples
    --------
    >>> report.add_decoding_summary(result, name="Model Performance")
    """
    if not hasattr(result, "summary"):
        raise TypeError("result must provide summary().")
    summary = result.summary()
    if summary.empty:
        return self
    sec = Section(title=name)
    csv_data = summary.to_csv(index=False)
    sec.add_element(
        DownloadAssetElement(
            csv_data,
            "decoding_summary.csv",
            "text/csv",
            label="Download Summary CSV",
            style="gray",
        )
    )
    sec.add_element(TableElement(summary, title="Model Performance Summary"))
    self.add_section(sec)
    return self


def add_decoding_diagnostics(
    self: Report,
    result: Any,
    metric: str | None = None,
    model: str | None = None,
    name: str = "Decoding Diagnostics",
) -> Report:
    """Add fold-level diagnostics, confusion matrix, ROC, PR, and calibration plots.

    Related plots are displayed side by side: confusion matrix next to ROC curve,
    and precision-recall curve next to calibration curve.

    Parameters
    ----------
    self : Report
        Target report.
    result : Any
        Decoding result object.
    metric : str, optional
        Filter scores to a single metric.
    model : str, optional
        Filter to a single model.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the diagnostics section appended, or unchanged if no data
        is available.

    See Also
    --------
    coco_pipe.viz.decoding.plot_confusion_matrix
    coco_pipe.viz.decoding.plot_roc_curve
    coco_pipe.viz.decoding.plot_pr_curve
    coco_pipe.viz.decoding.plot_calibration_curve

    Examples
    --------
    >>> report.add_decoding_diagnostics(result, model="SVM")
    """
    from coco_pipe.viz.decoding import (
        plot_calibration_curve,
        plot_confusion_matrix,
        plot_fold_score_dispersion,
        plot_pr_curve,
        plot_roc_curve,
    )

    sec = Section(title=name)
    added = False

    if hasattr(result, "get_detailed_scores"):
        scores = result.get_detailed_scores()
        if metric is not None and "Metric" in scores:
            scores = scores[scores["Metric"] == metric]
        if model is not None and "Model" in scores:
            scores = scores[scores["Model"] == model]
        if not scores.empty:
            csv_data = scores.to_csv(index=False)
            sec.add_element(
                DownloadAssetElement(
                    csv_data,
                    "fold_scores.csv",
                    "text/csv",
                    label="Download Fold Scores CSV",
                    style="gray",
                )
            )
            sec.add_element(TableElement(scores, title="Fold Scores"))
            added = True
            try:
                plot_result = plot_fold_score_dispersion(
                    scores, metric=metric, model=model
                )
                sec.add_element(
                    ImageElement(
                        plot_result[0]
                        if isinstance(plot_result, tuple)
                        else plot_result,
                        caption="Fold score dispersion",
                    )
                )
            except ValueError as exc:
                logger.debug("Fold score plot skipped: %s", exc)

    if hasattr(result, "get_fit_diagnostics"):
        diagnostics = result.get_fit_diagnostics()
        if model is not None and "Model" in diagnostics:
            diagnostics = diagnostics[diagnostics["Model"] == model]
        if not diagnostics.empty:
            cols = [
                c
                for c in ["Model", "Fold", "FitTime", "PredictTime", "TotalTime"]
                if c in diagnostics
            ]
            sec.add_element(
                TableElement(
                    diagnostics[cols].drop_duplicates(), title="Fit Diagnostics"
                )
            )
            added = True
            if "WarningMessage" in diagnostics:
                warns = diagnostics[diagnostics["WarningMessage"].notna()]
                if not warns.empty:
                    warn_cols = [
                        c
                        for c in [
                            "Model",
                            "Fold",
                            "Stage",
                            "WarningCategory",
                            "WarningMessage",
                        ]
                        if c in warns
                    ]
                    sec.add_element(
                        TableElement(warns[warn_cols], title="Training Warnings")
                    )

    # Confusion matrix + ROC curve — side by side
    confusion_img: ImageElement | None = None
    roc_img: ImageElement | None = None

    if hasattr(result, "get_confusion_matrices"):
        confusion = result.get_confusion_matrices(model=model)
        if not confusion.empty:
            sec.add_element(TableElement(confusion, title="Confusion Matrix Data"))
            added = True
            try:
                plot_result = plot_confusion_matrix(confusion, model=model)
                confusion_img = ImageElement(
                    plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                    caption="Confusion matrix",
                )
            except ValueError as exc:
                logger.debug("Confusion matrix skipped: %s", exc)

    try:
        plot_result = plot_roc_curve(result, model=model)
        roc_img = ImageElement(
            plot_result[0] if isinstance(plot_result, tuple) else plot_result,
            caption="ROC Curve",
        )
        added = True
    except (TypeError, ValueError) as exc:
        logger.debug("ROC Curve skipped: %s", exc)

    if confusion_img or roc_img:
        tabs = {}
        if confusion_img:
            tabs["Confusion Matrix"] = confusion_img
        if roc_img:
            tabs["ROC Curve"] = roc_img
        sec.add_element(TabsElement(tabs))

    # Precision-recall + Calibration — side by side
    pr_img: ImageElement | None = None
    cal_img: ImageElement | None = None

    try:
        plot_result = plot_pr_curve(result, model=model)
        pr_img = ImageElement(
            plot_result[0] if isinstance(plot_result, tuple) else plot_result,
            caption="Precision-Recall Curve",
        )
        added = True
    except (TypeError, ValueError) as exc:
        logger.debug("PR Curve skipped: %s", exc)

    try:
        plot_result = plot_calibration_curve(result, model=model)
        cal_img = ImageElement(
            plot_result[0] if isinstance(plot_result, tuple) else plot_result,
            caption="Calibration Curve",
        )
        added = True
    except (TypeError, ValueError) as exc:
        logger.debug("Calibration Curve skipped: %s", exc)

    if pr_img or cal_img:
        tabs = {}
        if pr_img:
            tabs["Precision-Recall"] = pr_img
        if cal_img:
            tabs["Calibration"] = cal_img
        sec.add_element(TabsElement(tabs))

    if added:
        self.add_section(sec)
    return self


def _add_pair(section: Section, a: ImageElement | None, b: ImageElement | None) -> None:
    """Add two images side by side if both exist, otherwise add whichever is present."""
    pair = [img for img in (a, b) if img is not None]
    if len(pair) == 2:
        section.add_element(ColumnsElement(pair, cols=2))
    elif len(pair) == 1:
        section.add_element(pair[0])


def add_decoding_statistical_assessment(
    self: Report,
    result: Any,
    metric: str | None = None,
    model: str | None = None,
    name: str = "Statistical Assessment",
) -> Report:
    """Add null-interval summary and temporal statistical assessment plots.

    Parameters
    ----------
    self : Report
        Target report.
    result : Any
        Must expose ``get_statistical_assessment() -> pd.DataFrame``.
    metric : str, optional
        Filter to a single metric name.
    model : str, optional
        Filter to a single model name.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if data is absent.

    Raises
    ------
    TypeError
        If *result* does not provide ``get_statistical_assessment``.

    See Also
    --------
    coco_pipe.viz.decoding.plot_null_interval_summary
    coco_pipe.viz.decoding.plot_temporal_statistical_assessment

    Examples
    --------
    >>> report.add_decoding_statistical_assessment(result, metric="accuracy")
    """
    from coco_pipe.viz.decoding import (
        plot_null_interval_summary,
        plot_temporal_statistical_assessment,
    )

    if not hasattr(result, "get_statistical_assessment"):
        raise TypeError("result must provide get_statistical_assessment().")
    assessment = result.get_statistical_assessment()
    if metric is not None and "Metric" in assessment:
        assessment = assessment[assessment["Metric"] == metric]
    if model is not None and "Model" in assessment:
        assessment = assessment[assessment["Model"] == model]
    if assessment.empty:
        return self

    sec = Section(title=name)
    sec.add_element(
        TableElement(assessment, title="Finite-Sample Statistical Assessment")
    )
    try:
        plot_result = plot_null_interval_summary(assessment)
        sec.add_element(
            ImageElement(
                plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                caption="Null interval summary",
            )
        )
    except ValueError as exc:
        logger.debug("Null interval summary skipped: %s", exc)
    if "Time" in assessment and assessment["Time"].notna().any():
        try:
            plot_result = plot_temporal_statistical_assessment(
                assessment, metric=metric, model=model
            )
            sec.add_element(
                ImageElement(
                    plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                    caption="Temporal statistical assessment",
                )
            )
        except ValueError as exc:
            logger.debug("Temporal statistics skipped: %s", exc)
    self.add_section(sec)
    return self


def add_decoding_neural_artifacts(
    self: Report,
    result: Any,
    model: str | None = None,
    name: str = "Neural Artifacts",
) -> Report:
    """Add model artifact table and training-history plot to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    result : Any
        Must expose ``get_model_artifacts() -> pd.DataFrame``.
    model : str, optional
        Filter to a single model name.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if data is absent.

    Raises
    ------
    TypeError
        If *result* does not provide ``get_model_artifacts``.

    See Also
    --------
    coco_pipe.viz.decoding.plot_training_history

    Examples
    --------
    >>> report.add_decoding_neural_artifacts(result)
    """
    from coco_pipe.viz.decoding import plot_training_history

    if not hasattr(result, "get_model_artifacts"):
        raise TypeError("result must provide get_model_artifacts().")
    artifacts = result.get_model_artifacts()
    if model is not None and "Model" in artifacts:
        artifacts = artifacts[artifacts["Model"] == model]
    if artifacts.empty:
        return self

    sec = Section(title=name)
    sec.add_element(TableElement(artifacts, title="Model Artifacts"))
    try:
        plot_result = plot_training_history(artifacts, model=model)
        sec.add_element(
            ImageElement(
                plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                caption="Training history",
            )
        )
    except ValueError:
        pass
    self.add_section(sec)
    return self


def add_decoding_performance(
    self: Report,
    result: Any,
    *,
    metric: str | None = None,
    name: str = "Performance",
) -> Report:
    """Add decoding-score distribution and model-comparison plots to *self*.

    The two plots are rendered side by side when both are available.

    Parameters
    ----------
    self : Report
        Target report.
    result : Any
        Decoding result object.
    metric : str, optional
        Metric to visualise (e.g. ``"accuracy"``).
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if no plots succeed.

    See Also
    --------
    coco_pipe.viz.decoding.plot_decoding_scores
    coco_pipe.viz.decoding.plot_model_comparison

    Examples
    --------
    >>> report.add_decoding_performance(result, metric="roc_auc")
    """
    from coco_pipe.viz.decoding import plot_decoding_scores, plot_model_comparison

    sec = Section(title=name)
    scores_img: ImageElement | None = None
    compare_img: ImageElement | None = None

    try:
        plot_result = plot_decoding_scores(result, metric=metric)
        scores_img = ImageElement(
            plot_result[0] if isinstance(plot_result, tuple) else plot_result,
            caption="Decoding scores",
        )
    except (TypeError, ValueError) as exc:
        logger.debug("Decoding score plot skipped: %s", exc)

    try:
        plot_result = plot_model_comparison(result, metric=metric or "accuracy")
        compare_img = ImageElement(
            plot_result[0] if isinstance(plot_result, tuple) else plot_result,
            caption="Model comparison",
        )
    except (TypeError, ValueError) as exc:
        logger.debug("Model comparison plot skipped: %s", exc)

    if scores_img or compare_img:
        tabs = {}
        if scores_img:
            tabs["Score Distribution"] = scores_img
        if compare_img:
            tabs["Model Comparison"] = compare_img
        sec.add_element(TabsElement(tabs))

    if scores_img is not None or compare_img is not None:
        self.add_section(sec)
    return self


def add_decoding_features(
    self: Report,
    result: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    name: str = "Features",
) -> Report:
    """Add feature-importance and feature-stability plots to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    result : Any
        Decoding result object.
    feature_metadata : pd.DataFrame, optional
        Additional feature metadata rendered as a table.
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if nothing renders.

    See Also
    --------
    coco_pipe.viz.decoding.plot_feature_importance
    coco_pipe.viz.decoding.plot_feature_stability

    Examples
    --------
    >>> report.add_decoding_features(result, feature_metadata=meta_df)
    """
    from coco_pipe.viz.decoding import plot_feature_importance, plot_feature_stability

    sec = Section(title=name)
    importance_img: ImageElement | None = None
    stability_img: ImageElement | None = None

    try:
        plot_result = plot_feature_importance(result)
        importance_img = ImageElement(
            plot_result[0] if isinstance(plot_result, tuple) else plot_result,
            caption="Feature importance",
        )
    except (TypeError, ValueError) as exc:
        logger.debug("Feature importance skipped: %s", exc)

    try:
        plot_result = plot_feature_stability(result)
        stability_img = ImageElement(
            plot_result[0] if isinstance(plot_result, tuple) else plot_result,
            caption="Feature stability",
        )
    except (TypeError, ValueError) as exc:
        logger.debug("Feature stability skipped: %s", exc)

    if importance_img or stability_img:
        tabs = {}
        if importance_img:
            tabs["Importance"] = importance_img
        if stability_img:
            tabs["Stability"] = stability_img
        sec.add_element(TabsElement(tabs))
    added = importance_img is not None or stability_img is not None

    if feature_metadata is not None:
        sec.add_element(
            TableElement(pd.DataFrame(feature_metadata), title="Feature Metadata")
        )
        added = True
    if added:
        self.add_section(sec)
    return self


def add_decoding_topomaps(
    self: Report,
    result: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    info: Any = None,
    coords: Any = None,
    name: str = "Sensor Maps",
) -> Report:
    """Add per-family sensor-profile topomaps to *self*.

    Parameters
    ----------
    self : Report
        Target report.
    result : Any
        Decoding result object.
    feature_metadata : pd.DataFrame, optional
        Must contain a ``FeatureFamily`` column; skipped otherwise.
    info : mne.Info, optional
        MNE Info object used for topomap rendering.
    coords : array-like, optional
        Explicit 2-D sensor coordinates (alternative to *info*).
    name : str
        Section title.

    Returns
    -------
    Report
        *self* with the new section appended, or unchanged if data is absent.

    See Also
    --------
    coco_pipe.viz.decoding.plot_feature_sensor_profile

    Examples
    --------
    >>> report.add_decoding_topomaps(result, feature_metadata=meta_df, info=raw.info)
    """
    from coco_pipe.viz.decoding import plot_feature_sensor_profile

    if feature_metadata is None:
        return self
    meta = pd.DataFrame(feature_metadata)
    if "FeatureFamily" not in meta or meta.empty:
        return self
    sec = Section(title=name)
    added = False
    for family in pd.unique(meta["FeatureFamily"].dropna())[:4]:
        try:
            plot_result = plot_feature_sensor_profile(
                result,
                feature_metadata=meta,
                feature_family=str(family),
                info=info,
                coords=coords,
            )
            sec.add_element(
                ImageElement(
                    plot_result[0] if isinstance(plot_result, tuple) else plot_result,
                    caption=f"{family} sensor profile",
                )
            )
            added = True
        except (TypeError, ValueError, ImportError) as exc:
            logger.debug("Topomap skipped: %s", exc)
    if added:
        self.add_section(sec)
    return self


def _add_configuration(report: Report, result: Any) -> None:
    config = getattr(result, "config", {}) or {}
    if not config:
        return
    sec = Section(title="Configuration")
    sec.add_element(_config_element(config, title="Run Configuration"))
    report.add_section(sec)


def _add_provenance(report: Report) -> None:
    sec = Section(title="Provenance")
    metadata = getattr(report, "metadata", {}) or {}
    sec.add_element(_table_from_mapping(metadata, title="Environment"))
    report.add_section(sec)


def _add_confusion_probability(report: Report, result: Any) -> None:
    temp = Report(title="tmp")
    add_decoding_diagnostics(temp, result, name="Confusion and Probability")
    for child in temp.children:
        report.add_section(child)


def _add_fit_diagnostics(report: Report, result: Any) -> None:
    from coco_pipe.viz.decoding import plot_fit_diagnostics

    sec = Section(title="Fit Diagnostics")
    added = False
    if hasattr(result, "get_fit_diagnostics"):
        data = result.get_fit_diagnostics()
        if not data.empty:
            sec.add_element(TableElement(data, title="Fit Diagnostics"))
            added = True
            try:
                plot_result = plot_fit_diagnostics(data)
                sec.add_element(
                    ImageElement(
                        plot_result[0]
                        if isinstance(plot_result, tuple)
                        else plot_result,
                        caption="Fit diagnostics",
                    )
                )
            except ValueError as exc:
                logger.debug("Fit diagnostic plot skipped: %s", exc)
    if added:
        report.add_section(sec)


def _add_caveats(report: Report, result: Any, feature_metadata: Any = None) -> None:
    caveats = []
    if feature_metadata is None:
        caveats.append(
            "Feature metadata was not provided; sensor-wise feature plots were skipped."
        )
    if hasattr(result, "get_probability_diagnostics"):
        try:
            if result.get_probability_diagnostics().empty:
                caveats.append("Probability diagnostics were unavailable.")
        except Exception:
            caveats.append("Probability diagnostics could not be computed.")
    if caveats:
        sec = Section(title="Caveats")
        sec.add_element(
            TableElement(pd.DataFrame({"Caveat": caveats}), title="Caveats")
        )
        report.add_section(sec)


def _add_export_inventory(report: Report, result: Any) -> None:
    accessors = [
        "summary",
        "get_detailed_scores",
        "get_predictions",
        "get_fit_diagnostics",
        "get_statistical_assessment",
        "get_feature_importances",
    ]
    rows = []
    for accessor in accessors:
        if not hasattr(result, accessor):
            continue
        try:
            raw_value = getattr(result, accessor)
            value = raw_value() if callable(raw_value) else raw_value
            rows.append(
                {
                    "Accessor": accessor,
                    "Rows": len(value) if hasattr(value, "__len__") else None,
                }
            )
        except Exception as exc:
            rows.append({"Accessor": accessor, "Rows": None, "Error": str(exc)})
    if rows:
        sec = Section(title="Export Inventory")
        sec.add_element(TableElement(pd.DataFrame(rows), title="Available Tables"))
        report.add_section(sec)


def make_decoding_report(
    result: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    info: Any = None,
    coords: Any = None,
    sections: list[str] | Literal["default"] = "default",
    interactive: bool = False,
    theme: Literal["paper", "notebook", "poster"] = "paper",
    title: str = "Decoding Report",
    config: dict | None = None,
    asset_urls: dict[str, str] | None = None,
    qc_result: QCResult | None = None,
    output_path: str | None = None,
) -> Report:
    """Build a static decoding report from an ``ExperimentResult``.

    Parameters
    ----------
    result : Any
        Decoding result object (e.g. ``ExperimentResult``).
    feature_metadata : pd.DataFrame, optional
        Feature-level metadata for sensor map sections.
    info : mne.Info, optional
        MNE Info for topomap rendering.
    coords : array-like, optional
        Explicit sensor coordinates (alternative to *info*).
    sections : list of str or ``"default"``
        Ordered list of section keys to include. Use ``"default"`` for all
        standard sections (see :data:`DEFAULT_SECTIONS`).
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
        Fully populated report. Sections whose source data is absent are
        skipped without raising.

    See Also
    --------
    coco_pipe.report.api.from_experiment_result : Thin public wrapper.
    make_reduction_report : Equivalent factory for dimensionality reduction.

    Examples
    --------
    >>> report = make_decoding_report(result)
    >>> report.save("decoding_report.html")
    """
    if interactive:
        warnings.warn(
            "interactive=True is not yet implemented in make_decoding_report. "
            "All plots are currently rendered as static images.",
            stacklevel=2,
        )

    selected = _resolve_sections(
        sections,
        default=DEFAULT_SECTIONS,
        valid=VALID_SECTIONS,
        context="decoding",
    )
    run_config = {"theme": theme, **(config or {})}
    report = Report(title=title, config=run_config, theme=theme, asset_urls=asset_urls)
    if qc_result is not None:
        report.add_section(build_qc_section(qc_result))

    for section in selected:
        try:
            if section == "overview":
                add_decoding_overview(report, result)
            elif section == "configuration":
                _add_configuration(report, result)
            elif section == "provenance":
                _add_provenance(report)
            elif section == "model_summary":
                add_decoding_summary(report, result)
            elif section == "cv_summary":
                add_decoding_diagnostics(report, result)
            elif section == "performance":
                add_decoding_performance(report, result)
            elif section == "statistical":
                add_decoding_statistical_assessment(report, result)
            elif section == "confusion_probability":
                _add_confusion_probability(report, result)
            elif section == "temporal":
                add_decoding_temporal(report, result)
            elif section == "features":
                add_decoding_features(report, result, feature_metadata=feature_metadata)
            elif section == "fit_diagnostics":
                _add_fit_diagnostics(report, result)
            elif section == "caveats":
                _add_caveats(report, result, feature_metadata=feature_metadata)
            elif section == "export_inventory":
                _add_export_inventory(report, result)
            elif section == "topomaps":
                add_decoding_topomaps(
                    report,
                    result,
                    feature_metadata=feature_metadata,
                    info=info,
                    coords=coords,
                )
        except Exception as exc:
            logger.debug("Decoding report section %s skipped: %s", section, exc)

    if output_path is not None:
        report.save(output_path)
    return report


__all__ = [
    "DEFAULT_SECTIONS",
    "add_decoding_overview",
    "add_decoding_temporal",
    "add_decoding_summary",
    "add_decoding_diagnostics",
    "add_decoding_statistical_assessment",
    "add_decoding_neural_artifacts",
    "add_decoding_performance",
    "add_decoding_features",
    "add_decoding_topomaps",
    "make_decoding_report",
]

# Bind section adders as Report methods. Reversing the dependency this way
# keeps the fluent `report.add_decoding_*()` API without forcing core.py to
# import this module (which would be circular).
Report.add_decoding_overview = add_decoding_overview
Report.add_decoding_temporal = add_decoding_temporal
Report.add_decoding_summary = add_decoding_summary
Report.add_decoding_diagnostics = add_decoding_diagnostics
Report.add_decoding_statistical_assessment = add_decoding_statistical_assessment
Report.add_decoding_neural_artifacts = add_decoding_neural_artifacts
Report.add_decoding_performance = add_decoding_performance
Report.add_decoding_features = add_decoding_features
Report.add_decoding_topomaps = add_decoding_topomaps
