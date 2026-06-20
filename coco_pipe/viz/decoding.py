"""Static visualization helpers for decoding result tables."""

from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._utils import (
    _importance_with_metadata,
    _is_raw_importance_sequence,
    coerce_decoding_frame,
    finalize_axes,
    get_figure,
    prepare_confusion_matrix,
    prepare_curve_group_data,
    prepare_decoding_curve_frame,
    prepare_decoding_score_data,
    prepare_feature_importance_series,
    prepare_feature_stability_series,
    prepare_fit_diagnostics_frame,
    prepare_fold_score_data,
    prepare_model_comparison_frame,
    prepare_null_interval_frame,
    prepare_prediction_accuracy_scores,
    prepare_probability_diagnostics_summary,
    prepare_regression_prediction_data,
    prepare_search_results_frame,
    prepare_temporal_generalization_matrix,
    prepare_temporal_score_curve_frame,
    prepare_temporal_statistical_frame,
    prepare_training_history_artifacts,
    require_columns,
    require_non_empty,
    select_rows,
)
from .base import (
    plot_bar,
    plot_distribution_groups,
    plot_error_points,
    plot_heatmap,
    plot_line,
    plot_scatter2d,
    plot_topomap,
)
from .theme import DIVERGING, QUALITATIVE, SEQUENTIAL, _palette_for, coco_theme


def plot_confusion_matrix(
    result_or_matrix: Any,
    model: str | None = None,
    fold: int | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot an aggregated confusion matrix from decoding diagnostics.

    Parameters
    ----------
    result_or_matrix
        Experiment result with ``get_confusion_matrices()`` or a tidy confusion
        matrix DataFrame containing ``TrueLabel``, ``PredictedLabel``, and
        ``Value``.
    model
        Optional model name used to filter rows before aggregation.
    fold
        Optional fold index used to filter rows before aggregation.
    title
        Optional axes title. Defaults to ``"Confusion Matrix"``.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when ``ax`` is not provided.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_confusion_matrix :
        Interactive Plotly version.
    plot_roc_curve : Receiver-operating-characteristic curve.
    plot_probability_diagnostics : Probability-calibration quality metrics.
    plot_decoding_scores : Aggregate scalar score summary.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {"TrueLabel": list("AABB"),
    ...                           "PredictedLabel": list("ABAB"),
    ...                           "Value": [5, 1, 2, 4]}
    ... )
    >>> fig, ax = viz.plot_confusion_matrix(df)
    """
    with coco_theme():
        matrix = prepare_confusion_matrix(result_or_matrix, model=model, fold=fold)
        fig, ax = plot_heatmap(
            matrix,
            cmap=SEQUENTIAL,
            annotate=True,
            colorbar_label="Count",
            xlabel="Predicted",
            ylabel="True",
            title=title or "Confusion Matrix",
            xtick_ha="center",
            ax=ax,
            figsize=figsize or (6, 5),
        )
        return fig, ax


def _plot_curve_groups(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    mean_only: bool,
    ax: plt.Axes,
) -> None:
    """Draw per-fold or mean curve groups for ROC, PR, and calibration plots."""
    for curve in prepare_curve_group_data(frame, x_col, y_col, mean_only):
        if curve["kind"] == "mean":
            ax.plot(curve["x"], curve["y"], label=curve["label"], linewidth=2)
            ax.fill_between(
                curve["x"],
                curve["y"] - curve["yerr"],
                curve["y"] + curve["yerr"],
                alpha=0.15,
            )
        else:
            ax.plot(curve["x"], curve["y"], label=curve["label"], alpha=0.55)


def plot_roc_curve(
    result_or_curve: Any,
    model: str | None = None,
    fold: int | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    mean_only: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot receiver-operating-characteristic curves.

    Parameters
    ----------
    result_or_curve
        Experiment result with ``get_roc_curve()`` or a DataFrame containing
        ``Model``, ``FPR``, and ``TPR``. Optional ``Fold`` and ``Class`` columns
        are used for grouping.
    model
        Optional model name to display.
    fold
        Optional fold index to display.
    title
        Optional axes title. Defaults to ``"ROC Curve"``.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.
    mean_only
        If True, interpolate fold curves onto a common x-grid and draw the
        mean curve with a standard-deviation band.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_roc_curve : Interactive Plotly version.
    plot_pr_curve : Precision-recall curve.
    plot_calibration_curve : Probability calibration reliability curve.
    plot_confusion_matrix : Predicted-vs-true class heatmap.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {"Model": "SVM",
    ...                           "FPR": np.linspace(0, 1, 10),
    ...                           "TPR": np.linspace(0, 1, 10) ** 0.5}
    ... )
    >>> fig, ax = viz.plot_roc_curve(df)
    """
    with coco_theme():
        frame = prepare_decoding_curve_frame(
            result_or_curve,
            accessor="get_roc_curve",
            required_columns=["Model", "FPR", "TPR"],
            context="ROC curve",
            model=model,
            fold=fold,
        )
        fig, ax = get_figure(ax, figsize, (6, 5))
        _plot_curve_groups(frame, x_col="FPR", y_col="TPR", mean_only=mean_only, ax=ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
        finalize_axes(
            ax,
            title=title or "ROC Curve",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            legend=True,
        )
        return fig, ax


def plot_pr_curve(
    result_or_curve: Any,
    model: str | None = None,
    fold: int | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    mean_only: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot precision-recall curves from decoding diagnostics.

    Parameters
    ----------
    result_or_curve
        Experiment result with ``get_pr_curve()`` or a DataFrame containing
        ``Model``, ``Recall``, and ``Precision``. Optional ``Fold`` and
        ``Class`` columns are used for grouping.
    model
        Optional model name to display.
    fold
        Optional fold index to display.
    title
        Optional axes title. Defaults to ``"Precision-Recall Curve"``.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.
    mean_only
        If True, interpolate fold curves onto a common recall grid and draw the
        mean curve with a standard-deviation band.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_pr_curve : Interactive Plotly version.
    plot_roc_curve : Receiver-operating-characteristic curve.
    plot_calibration_curve : Probability calibration reliability curve.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": "SVM",
    ...         "Recall": np.linspace(0, 1, 10),
    ...         "Precision": np.linspace(1, 0.5, 10),
    ...     }
    ... )
    >>> fig, ax = viz.plot_pr_curve(df)
    """
    with coco_theme():
        frame = prepare_decoding_curve_frame(
            result_or_curve,
            accessor="get_pr_curve",
            required_columns=["Model", "Recall", "Precision"],
            context="precision-recall curve",
            model=model,
            fold=fold,
        )
        fig, ax = get_figure(ax, figsize, (6, 5))
        _plot_curve_groups(
            frame,
            x_col="Recall",
            y_col="Precision",
            mean_only=mean_only,
            ax=ax,
        )
        finalize_axes(
            ax,
            title=title or "Precision-Recall Curve",
            xlabel="Recall",
            ylabel="Precision",
            legend=True,
        )
        return fig, ax


def plot_calibration_curve(
    result_or_curve: Any,
    model: str | None = None,
    fold: int | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    mean_only: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot calibration reliability curves.

    Parameters
    ----------
    result_or_curve
        Experiment result with ``get_calibration_curve()`` or a DataFrame
        containing ``Model``, ``MeanPredictedProbability``, and
        ``FractionPositive``.
    model
        Optional model name to display.
    fold
        Optional fold index to display.
    title
        Optional axes title. Defaults to ``"Calibration Curve"``.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.
    mean_only
        If True, interpolate fold curves onto a common probability grid and
        draw the mean curve with a standard-deviation band.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_calibration_curve :
        Interactive Plotly version.
    plot_roc_curve : Receiver-operating-characteristic curve.
    plot_pr_curve : Precision-recall curve.
    plot_probability_diagnostics : Scalar probability-quality metrics.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": "SVM",
    ...         "MeanPredictedProbability": np.linspace(0, 1, 10),
    ...         "FractionPositive": np.linspace(0.05, 0.95, 10),
    ...     }
    ... )
    >>> fig, ax = viz.plot_calibration_curve(df)
    """
    with coco_theme():
        frame = prepare_decoding_curve_frame(
            result_or_curve,
            accessor="get_calibration_curve",
            required_columns=["Model", "MeanPredictedProbability", "FractionPositive"],
            context="calibration curve",
            model=model,
            fold=fold,
        )
        fig, ax = get_figure(ax, figsize, (6, 5))
        _plot_curve_groups(
            frame,
            x_col="MeanPredictedProbability",
            y_col="FractionPositive",
            mean_only=mean_only,
            ax=ax,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
        finalize_axes(
            ax,
            title=title or "Calibration Curve",
            xlabel="Mean Predicted Probability",
            ylabel="Fraction Positive",
            legend=True,
        )
        return fig, ax


def plot_fold_score_dispersion(
    result_or_scores: Any,
    metric: str | None = None,
    model: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot fold-level scalar score distributions by model and metric.

    Parameters
    ----------
    result_or_scores
        Experiment result with ``get_detailed_scores()`` or a detailed score
        DataFrame containing scalar ``Value`` rows.
    metric
        Optional metric name used to filter scores.
    model
        Optional model name used to filter scores.
    title
        Optional axes title. Defaults to ``"Fold Score Dispersion"``.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_fold_score_dispersion :
        Interactive Plotly version.
    plot_decoding_scores : Aggregate score summary with error bars.
    plot_model_comparison : Pairwise score-difference plot.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": "SVM",
    ...         "Fold": [0, 1, 2, 3, 4],
    ...         "Metric": "accuracy",
    ...         "Value": [0.82, 0.79, 0.85, 0.81, 0.83],
    ...     }
    ... )
    >>> fig, ax = viz.plot_fold_score_dispersion(df)
    """
    with coco_theme():
        data = prepare_fold_score_data(
            result_or_scores,
            model=model,
            metric=metric,
        )
        labels = []
        values = []
        for (model_name, metric_name), group in data.groupby(["Model", "Metric"]):
            labels.append(f"{model_name}\n{metric_name}")
            values.append(group["Value"].astype(float).to_numpy())
        fig, ax = plot_distribution_groups(
            values,
            labels,
            kind="box",
            ylabel="Value",
            title=title or "Fold Score Dispersion",
            ax=ax,
            figsize=figsize or (max(6, len(labels) * 1.5), 5),
        )
        return fig, ax


def plot_temporal_score_curve(
    result_or_scores: Any,
    metric: str | None = None,
    model: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    smooth_window: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot mean temporal decoding score curves.

    Parameters
    ----------
    result_or_scores
        Experiment result with ``get_temporal_score_summary()`` or a DataFrame
        containing ``Model``, ``Metric``, ``Time``, and ``Mean``.
    metric
        Optional metric name used to filter temporal scores.
    model
        Optional model name used to filter temporal scores.
    title
        Optional axes title. Defaults to ``"Temporal Decoding Value"``.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    Notes
    -----
    Non-numeric time labels are plotted at integer positions and displayed as
    rotated tick labels.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_temporal_score_curve :
        Interactive Plotly version.
    plot_temporal_generalization_matrix : Train-time × test-time heatmap.
    plot_temporal_statistical_assessment : Temporal curve with permutation null band.
    plot_null_interval_summary : Scalar null-interval summary per model.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> t = np.linspace(0, 0.5, 10)
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": "SVM",
    ...         "Metric": "accuracy",
    ...         "Time": t,
    ...         "Mean": 0.5 + 0.1 * np.arange(10) / 10,
    ...     }
    ... )
    >>> fig, ax = viz.plot_temporal_score_curve(df)
    """
    with coco_theme():
        curve_data = prepare_temporal_score_curve_frame(
            result_or_scores,
            model=model,
            metric=metric,
        )
        fig, ax = get_figure(ax, figsize, (10, 5))
        for (model_name, metric_name), group in curve_data.groupby(["Model", "Metric"]):
            numeric = pd.to_numeric(group["Time"], errors="coerce")
            use_numeric = numeric.notna().all()
            x_vals = numeric.to_numpy() if use_numeric else np.arange(len(group))

            y_vals_s = group["Mean"].astype(float)
            yerr_s = group["Std"].fillna(0).astype(float) if "Std" in group else None

            if smooth_window is not None and smooth_window > 1:
                y_vals_s = y_vals_s.rolling(
                    smooth_window, center=True, min_periods=1
                ).mean()
                if yerr_s is not None:
                    yerr_s = yerr_s.rolling(
                        smooth_window, center=True, min_periods=1
                    ).mean()

            y_vals = y_vals_s.to_numpy()
            yerr = yerr_s.to_numpy() if yerr_s is not None else None

            fig, ax = plot_line(
                x_vals,
                y_vals,
                yerr=yerr,
                error_style="band",
                marker="o",
                linewidth=2,
                label=f"{model_name} / {metric_name}",
                ax=ax,
            )
            if not use_numeric:
                ax.set_xticks(x_vals)
                ax.set_xticklabels([str(value) for value in group["Time"]], rotation=45)
        finalize_axes(
            ax,
            title=title or "Temporal Decoding Value",
            xlabel="Time",
            ylabel="Value",
            legend=True,
        )
        return fig, ax


def plot_temporal_generalization_matrix(
    result_or_scores: Any,
    metric: str | None = None,
    model: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a train-time by test-time temporal generalization matrix.

    Parameters
    ----------
    result_or_scores
        Experiment result with ``get_temporal_score_summary()`` or a DataFrame
        containing ``Model``, ``Metric``, ``TrainTime``, ``TestTime``, and
        ``Mean``.
    metric
        Optional metric name. A single model/metric pair must remain after
        filtering.
    model
        Optional model name. A single model/metric pair must remain after
        filtering.
    title
        Optional axes title. Defaults to ``"<model> / <metric>"``.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_temporal_generalization_matrix :
        Interactive Plotly version.
    plot_temporal_score_curve : Mean temporal score curve over time.
    plot_temporal_statistical_assessment : Temporal curve with permutation null band.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> times = np.linspace(0, 0.5, 5)
    >>> rows = [
    ...     {"Model": "SVM",
    ...                           "Metric": "accuracy",
    ...                           "TrainTime": t1,
    ...                           "TestTime": t2,
    ...                           "Mean": 0.7}
    ...     for t1 in times
    ...     for t2 in times
    ... ]
    >>> fig, ax = viz.plot_temporal_generalization_matrix(pd.DataFrame(rows))
    """
    with coco_theme():
        matrix, first = prepare_temporal_generalization_matrix(
            result_or_scores,
            model=model,
            metric=metric,
        )
        fig, ax = plot_heatmap(
            matrix,
            cmap=SEQUENTIAL,
            aspect="auto",
            origin="lower",
            colorbar_label="Value",
            xlabel="Test Time",
            ylabel="Train Time",
            title=title or f"{first['Model']} / {first['Metric']}",
            ax=ax,
            figsize=figsize or (7, 6),
        )
        return fig, ax


def plot_temporal_statistical_assessment(
    result_or_assessment: Any,
    metric: str | None = None,
    model: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot temporal statistical assessment results.

    Parameters
    ----------
    result_or_assessment
        Experiment result with ``get_statistical_assessment()`` or an
        assessment DataFrame containing ``Model``, ``Metric``, ``Observed``,
        and ``Time``.
    metric
        Optional metric name. A single model/metric pair must remain after
        filtering.
    model
        Optional model name. A single model/metric pair must remain after
        filtering.
    title
        Optional axes title. Defaults to a model/metric statistical-assessment
        title.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    Notes
    -----
    ``NullLower`` and ``NullUpper`` are rendered as a null band when present.
    ``Significant`` rows are overlaid as square markers.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_temporal_statistical_assessment :
        Interactive Plotly version.
    plot_temporal_score_curve : Mean temporal score curve over time.
    plot_null_interval_summary : Scalar null-interval summary per model.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> rng = np.random.default_rng(42)
    >>> t = np.linspace(0, 0.5, 10)
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": "SVM",
    ...         "Metric": "accuracy",
    ...         "Time": t,
    ...         "Observed": 0.5 + 0.1 * rng.normal(size=10),
    ...     }
    ... )
    >>> fig, ax = viz.plot_temporal_statistical_assessment(df)
    """
    with coco_theme():
        frame = prepare_temporal_statistical_frame(
            result_or_assessment,
            model=model,
            metric=metric,
        )
        first = frame.iloc[0]
        numeric = pd.to_numeric(frame["Time"], errors="coerce")
        use_numeric = numeric.notna().all()
        x_vals = numeric.to_numpy() if use_numeric else np.arange(len(frame))
        fig, ax = get_figure(ax, figsize, (10, 5))
        observed = frame["Observed"].astype(float).to_numpy()
        ax.plot(x_vals, observed, marker="o", linewidth=2, label="Observed")
        if {"NullLower", "NullUpper"}.issubset(frame.columns):
            lower = pd.to_numeric(frame["NullLower"], errors="coerce")
            upper = pd.to_numeric(frame["NullUpper"], errors="coerce")
            if lower.notna().any() and upper.notna().any():
                ax.fill_between(
                    x_vals, lower, upper, alpha=0.2, label="Permutation null band"
                )
        if "Significant" in frame and frame["Significant"].fillna(False).any():
            sig = frame["Significant"].fillna(False).to_numpy(dtype=bool)
            ax.scatter(
                x_vals[sig],
                observed[sig],
                marker="s",
                color="black",
                label="Significant",
                zorder=3,
            )
        if not use_numeric:
            ax.set_xticks(x_vals)
            ax.set_xticklabels([str(value) for value in frame["Time"]], rotation=45)
        finalize_axes(
            ax,
            title=title
            or f"{first['Model']} / {first['Metric']} statistical assessment",
            xlabel="Time",
            ylabel="Value",
            legend=True,
        )
        return fig, ax


def plot_null_interval_summary(
    result_or_assessment: Any,
    metric: str | None = None,
    model: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot observed scalar scores against null interval summaries.

    Parameters
    ----------
    result_or_assessment
        Experiment result with ``get_statistical_assessment()`` or an
        assessment DataFrame containing ``Model``, ``Metric``, and
        ``Observed``.
    metric
        Optional metric name used to filter assessment rows.
    model
        Optional model name used to filter assessment rows.
    title
        Optional axes title. Defaults to ``"Null Interval Summary"``.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    Notes
    -----
    ``NullLower``/``NullUpper`` are drawn as error intervals around their
    midpoint. ``NullMedian`` is shown as an optional marker series.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_null_interval_summary :
        Interactive Plotly version.
    plot_temporal_statistical_assessment : Significance-annotated temporal heatmap.
    plot_decoding_scores : Aggregate scalar score summary.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM", "SVM"],
    ...         "Metric": ["accuracy", "accuracy"],
    ...         "Observed": [0.72, 0.68],
    ...         "NullLower": [0.45, 0.44],
    ...         "NullUpper": [0.55, 0.54],
    ...     }
    ... )
    >>> fig, ax = viz.plot_null_interval_summary(df)
    """
    with coco_theme():
        frame = prepare_null_interval_frame(
            result_or_assessment,
            model=model,
            metric=metric,
        )
        labels = [f"{row.Model}\n{row.Metric}" for row in frame.itertuples()]
        x_vals = np.arange(len(frame))
        fig, ax = plot_error_points(
            x_vals,
            frame["Observed"].astype(float),
            labels=labels,
            label="Observed",
            capsize=0,
            ax=ax,
            figsize=figsize or (max(6, len(frame) * 1.2), 5),
        )
        if {"NullLower", "NullUpper"}.issubset(frame.columns):
            lower = pd.to_numeric(frame["NullLower"], errors="coerce")
            upper = pd.to_numeric(frame["NullUpper"], errors="coerce")
            center = (lower + upper) / 2
            yerr = np.vstack([center - lower, upper - center])
            fig, ax = plot_error_points(
                x_vals,
                center,
                yerr=yerr,
                label="Null band",
                ax=ax,
            )
        if "NullMedian" in frame and frame["NullMedian"].notna().any():
            fig, ax = plot_error_points(
                x_vals,
                frame["NullMedian"].astype(float),
                marker="x",
                label="Null median",
                capsize=0,
                ax=ax,
            )
        finalize_axes(
            ax,
            title=title or "Null Interval Summary",
            ylabel="Value",
            legend=True,
        )
        return fig, ax


def plot_training_history(
    result_or_artifacts: Any,
    model: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot neural training-history artifacts.

    Parameters
    ----------
    result_or_artifacts
        Experiment result with ``get_model_artifacts()`` or an artifact
        DataFrame containing ``Model``, ``Key``, ``ArtifactType``, and
        ``Value``. History values are expected to be records keyed by epoch and
        metric names.
    model
        Optional model name used to filter artifacts.
    title
        Optional axes title. Defaults to ``"Training History"``.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_training_history :
        Interactive Plotly version.
    plot_fit_diagnostics : Fit-time diagnostics by model or fold.
    plot_decoding_scores : Aggregate scalar score summary.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> history = {"epoch": [1, 2, 3],
    ...                           "loss": [0.9, 0.7, 0.5],
    ...                           "val_loss": [1.0, 0.8, 0.65]}
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["DNN"],
    ...         "Key": ["history"],
    ...         "ArtifactType": ["training_history"],
    ...         "Value": [history],
    ...     }
    ... )
    >>> fig, ax = viz.plot_training_history(df)
    """
    with coco_theme():
        rows = prepare_training_history_artifacts(result_or_artifacts, model=model)
        fig, ax = get_figure(ax, figsize, (8, 5))
        plotted = False
        for row in rows.itertuples():
            history = row.Value if row.Value is not None else []
            frame = pd.DataFrame(history)
            if frame.empty:
                continue
            if "epoch" not in frame:
                frame = frame.reset_index(names="epoch")
            for column in [col for col in frame.columns if col != "epoch"]:
                fig, ax = plot_line(
                    frame["epoch"],
                    frame[column],
                    marker="o",
                    label=f"{row.Model} {column}",
                    ax=ax,
                )
                plotted = True
        if not plotted:
            raise ValueError("No plottable training history artifacts available.")
        finalize_axes(
            ax,
            title=title or "Training History",
            xlabel="Epoch",
            ylabel="Value",
            legend=True,
        )
        return fig, ax


def plot_decoding_scores(
    result: Any,
    metric: str | None = None,
    model: str | None = None,
    kind: Literal["point", "bar", "box"] = "point",
    aggregate: Literal["mean", "median"] = "mean",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot aggregate scalar decoding scores by model and metric.

    Parameters
    ----------
    result
        Experiment result with ``get_detailed_scores()`` or a detailed score
        DataFrame containing scalar ``Value`` rows.
    metric
        Optional metric name used to filter scores.
    model
        Optional model name used to filter scores.
    kind
        Plot type: ``"point"`` for aggregate +/- SEM, ``"bar"`` for bar
        summaries, or ``"box"`` for fold-level distributions.
    aggregate
        Summary statistic used by ``"point"`` and ``"bar"`` plots.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_decoding_scores :
        Interactive Plotly version.
    plot_model_comparison : Score differences between model pairs.
    plot_fold_score_dispersion : Per-fold score spread.
    plot_null_interval_summary : Observed scores versus null bands.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM"] * 5,
    ...         "Metric": ["accuracy"] * 5,
    ...         "Type": ["detailed_score"] * 5,
    ...         "Value": [0.71, 0.73, 0.69, 0.74, 0.70],
    ...     }
    ... )
    >>> fig, ax = viz.plot_decoding_scores(df)
    """
    with coco_theme():
        data = prepare_decoding_score_data(result, model=model, metric=metric)
        grouped = data.groupby(["Model", "Metric"], dropna=False)["Value"]
        summary = grouped.agg(["mean", "median", "std", "count"]).reset_index()
        summary["center"] = summary[aggregate]
        summary["sem"] = summary["std"].fillna(0) / np.sqrt(
            summary["count"].clip(lower=1)
        )
        labels = summary.apply(lambda row: f"{row['Model']}\n{row['Metric']}", axis=1)
        if kind == "box":
            values = [g.astype(float).to_numpy() for _, g in grouped]
            fig, ax = plot_distribution_groups(
                values,
                labels,
                kind="box",
                ylabel="Value",
                title="Decoding Scores",
                ax=ax,
                figsize=figsize or (max(6, len(summary) * 1.3), 5),
            )
        elif kind == "bar":
            values = pd.Series(summary["center"].astype(float).to_numpy(), index=labels)
            errors = pd.Series(summary["sem"].astype(float).to_numpy(), index=labels)
            fig, ax = plot_bar(
                values,
                errors=errors,
                sort=False,
                color=plt.get_cmap(QUALITATIVE)(0),
                ax=ax,
                figsize=figsize or (max(6, len(summary) * 1.3), 5),
                ylabel="Value",
                title="Decoding Scores",
            )
        elif kind == "point":
            x_vals = np.arange(len(summary))
            fig, ax = plot_error_points(
                x_vals,
                summary["center"].astype(float),
                yerr=summary["sem"].astype(float),
                labels=labels,
                ax=ax,
                figsize=figsize or (max(6, len(summary) * 1.3), 5),
                ylabel="Value",
                title="Decoding Scores",
            )
        else:
            raise ValueError("kind must be 'point', 'bar', or 'box'.")
        return fig, ax


def plot_model_comparison(
    result: Any,
    metric: str = "accuracy",
    reference: str | None = None,
    paired: bool = True,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot model-comparison score differences.

    Parameters
    ----------
    result
        Experiment result or a comparison DataFrame. DataFrames must contain a
        ``Difference`` column and may include ``ModelA``, ``ModelB``,
        ``CILower``, and ``CIUpper``.
    metric
        Metric used when computing comparisons from an experiment result.
    reference
        Optional reference model. When provided with ``paired=True``, all other
        models are compared against this model.
    paired
        If True and ``reference`` is provided, use ``compare_models_paired``.
        Otherwise use ``compare_models`` when available. Results with only
        detailed scores fall back to mean-score differences.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_model_comparison :
        Interactive Plotly version.
    plot_decoding_scores : Aggregate scalar score summary.
    plot_fold_score_dispersion : Per-fold score spread.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "ModelA": ["SVM", "SVM"],
    ...         "ModelB": ["LDA", "RF"],
    ...         "Difference": [0.05, -0.02],
    ...         "CILower": [0.01, -0.06],
    ...         "CIUpper": [0.09, 0.02],
    ...     }
    ... )
    >>> fig, ax = viz.plot_model_comparison(df)
    """
    with coco_theme():
        comp = prepare_model_comparison_frame(
            result,
            metric=metric,
            reference=reference,
            paired=paired,
        )
        labels = []
        for row in comp.itertuples():
            if hasattr(row, "ModelA") and hasattr(row, "ModelB"):
                labels.append(f"{row.ModelA} - {row.ModelB}")
            else:
                labels.append(str(getattr(row, "Model", len(labels))))
        order = np.argsort(comp["Difference"].astype(float).to_numpy())
        diffs = comp["Difference"].astype(float).to_numpy()[order]
        labels = [labels[i] for i in order]
        y_vals = np.arange(len(labels))
        xerr = None
        if {"CILower", "CIUpper"}.issubset(comp.columns):
            lower = comp["CILower"].astype(float).to_numpy()[order]
            upper = comp["CIUpper"].astype(float).to_numpy()[order]
            xerr = np.vstack([diffs - lower, upper - diffs])
        fig, ax = plot_scatter2d(
            diffs,
            y_vals,
            color="black",
            xerr=xerr,
            reference_x=0,
            reference_style={"color": "0.4", "linestyle": "--", "linewidth": 1},
            error_color="0.35",
            ax=ax,
            figsize=figsize or (7, max(4, len(labels) * 0.55)),
            xlabel=f"{metric} difference",
            title=f"Model Comparison ({metric})",
        )
        ax.set_yticks(y_vals)
        ax.set_yticklabels(labels)
        return fig, ax


def plot_fit_diagnostics(
    result: Any,
    *,
    by: Literal["Model", "Fold"] = "Model",
    show_warnings: bool = True,
    kind: Literal["bar", "box"] = "bar",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot fit-time diagnostics by model or fold.

    Parameters
    ----------
    result
        Experiment result with ``get_fit_diagnostics()`` or a diagnostics
        DataFrame containing ``TotalTime`` plus the selected ``by`` column.
    by
        Column used for grouping, usually ``"Model"`` or ``"Fold"``.
    show_warnings
        If True, annotate the plot with the number of non-null
        ``WarningMessage`` entries when present.
    kind
        Plot type: ``"bar"`` for mean total time or ``"box"`` for grouped
        total-time distributions.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_fit_diagnostics :
        Interactive Plotly version.
    plot_training_history : Neural training-history artifacts.
    plot_decoding_scores : Aggregate scalar score summary.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM", "SVM", "LDA", "LDA"],
    ...         "Fold": [0, 1, 0, 1],
    ...         "TotalTime": [1.2, 1.3, 0.5, 0.6],
    ...     }
    ... )
    >>> fig, ax = viz.plot_fit_diagnostics(df)
    """
    with coco_theme():
        frame, data = prepare_fit_diagnostics_frame(result, by=by)
        if kind == "box":
            groups = [
                group["TotalTime"].astype(float).to_numpy()
                for _, group in data.groupby(by)
            ]
            labels = [str(name) for name, _ in data.groupby(by)]
            fig, ax = plot_distribution_groups(
                groups,
                labels,
                kind="box",
                xlabel=by,
                ylabel="Seconds",
                title="Fit Diagnostics",
                ax=ax,
                figsize=figsize or (7, 4),
            )
        elif kind == "bar":
            summary = data.groupby(by)["TotalTime"].mean().sort_values()
            fig, ax = plot_bar(
                summary,
                orientation="vertical",
                sort=False,
                ax=ax,
                figsize=figsize or (7, 4),
                xlabel=by,
                ylabel="Seconds",
                title="Fit Diagnostics",
            )
        else:
            raise ValueError("kind must be 'bar' or 'box'.")
        if show_warnings and "WarningMessage" in frame:
            n_warn = int(frame["WarningMessage"].notna().sum())
            if n_warn:
                ax.text(
                    0.99,
                    0.95,
                    f"Warnings: {n_warn}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                )
        return fig, ax


def plot_probability_diagnostics(
    result: Any,
    model: str | None = None,
    metric: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot probability-quality diagnostics.

    Parameters
    ----------
    result
        Experiment result with ``get_probability_diagnostics()`` or a DataFrame
        containing ``Model``, ``Metric``, and ``Value``.
    model
        Optional model name used to filter diagnostics.
    metric
        Optional diagnostic metric used to filter rows.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_probability_diagnostics :
        Interactive Plotly version.
    plot_calibration_curve : Calibration curve for probability estimates.
    plot_confusion_matrix : Aggregated confusion matrix.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM", "SVM"],
    ...         "Metric": ["brier_score", "log_loss"],
    ...         "Value": [0.18, 0.42],
    ...     }
    ... )
    >>> fig, ax = viz.plot_probability_diagnostics(df)
    """
    with coco_theme():
        summary = prepare_probability_diagnostics_summary(
            result,
            model=model,
            metric=metric,
        )
        labels = summary.apply(lambda row: f"{row['Model']}\n{row['Metric']}", axis=1)
        values = pd.Series(summary["Value"].astype(float).to_numpy(), index=labels)
        fig, ax = plot_bar(
            values,
            color=plt.get_cmap(QUALITATIVE)(1),
            sort=False,
            ax=ax,
            figsize=figsize or (max(6, len(summary) * 1.2), 4),
            ylabel="Mean diagnostic value",
            title="Probability Diagnostics",
        )
        return fig, ax


def plot_subject_diagnostics(
    result: Any,
    unit: str = "Subject",
    metric: str = "accuracy",
    model: str | None = None,
    kind: Literal["strip", "dot", "caterpillar"] = "strip",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot per-unit prediction accuracy diagnostics.

    Parameters
    ----------
    result
        Experiment result with ``get_predictions()`` or a prediction DataFrame
        containing ``Model``, ``y_true``, ``y_pred``, and the selected ``unit``
        column.
    unit
        Metadata column used as the unit of aggregation, such as ``"Subject"``.
    metric
        Metric to compute. Currently only ``"accuracy"`` is supported.
    model
        Optional model name used to filter predictions.
    kind
        Plot type. ``"strip"`` and ``"dot"`` show one point per unit grouped
        by model. ``"caterpillar"`` shows sorted unit scores with horizontal
        stems.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_subject_diagnostics :
        Interactive Plotly version.
    plot_group_summary : Group-level accuracy summaries.
    plot_decoding_scores : Aggregate scalar score summary.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM"] * 6,
    ...         "Subject": ["S1", "S1", "S2", "S2", "S3", "S3"],
    ...         "Fold": [0, 1, 0, 1, 0, 1],
    ...         "y_true": [0, 1, 1, 0, 0, 1],
    ...         "y_pred": [0, 1, 0, 0, 0, 1],
    ...     }
    ... )
    >>> fig, ax = viz.plot_subject_diagnostics(df)
    """
    if metric != "accuracy":
        raise ValueError(
            "plot_subject_diagnostics currently supports metric='accuracy'."
        )
    with coco_theme():
        scores = prepare_prediction_accuracy_scores(
            result,
            [unit],
            model=model,
            context="subject diagnostics",
        )
        if kind == "caterpillar":
            fig, ax = get_figure(
                ax, figsize, (8, max(4, scores[unit].nunique() * 0.35))
            )
            scores = scores.sort_values("Value")
            ax.hlines(np.arange(len(scores)), 0, scores["Value"], color="0.8")
            ax.scatter(scores["Value"], np.arange(len(scores)), color="black")
            ax.set_yticks(np.arange(len(scores)))
            ax.set_yticklabels(scores[unit].astype(str))
            finalize_axes(ax, title=f"{unit} Diagnostics", xlabel=metric)
        elif kind in {"strip", "dot"}:
            model_order = list(pd.unique(scores["Model"]))
            y_lookup = {name: idx for idx, name in enumerate(model_order)}
            y_vals = scores["Model"].map(y_lookup).to_numpy(dtype=float)
            fig, ax = plot_scatter2d(
                scores["Value"].astype(float),
                y_vals,
                labels=scores["Model"],
                alpha=0.75,
                ax=ax,
                figsize=figsize or (8, max(4, len(model_order) * 0.6)),
                xlabel=metric,
                title=f"{unit} Diagnostics",
            )
            ax.set_yticks(np.arange(len(model_order)))
            ax.set_yticklabels([str(name) for name in model_order])
        else:
            raise ValueError("kind must be 'strip', 'dot', or 'caterpillar'.")
        return fig, ax


def plot_group_summary(
    result: Any,
    group: str = "Group",
    metric: str = "accuracy",
    model: str | None = None,
    kind: Literal["point", "violin"] = "point",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot group-level prediction accuracy summaries.

    Parameters
    ----------
    result
        Experiment result with ``get_predictions()`` or a prediction DataFrame
        containing ``Model``, ``Fold``, ``y_true``, ``y_pred``, and the selected
        grouping column.
    group
        Prediction metadata column used to define groups.
    metric
        Metric to compute. Currently only ``"accuracy"`` is supported.
    model
        Optional model name used to filter predictions.
    kind
        Plot type: ``"point"`` for mean +/- SEM or ``"violin"`` for fold-level
        distributions.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_group_summary : Interactive Plotly version.
    plot_subject_diagnostics : Per-subject accuracy diagnostics.
    plot_decoding_scores : Aggregate scalar score summary.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM"] * 8,
    ...         "Group": ["A"] * 4 + ["B"] * 4,
    ...         "Fold": [0, 1, 2, 3] * 2,
    ...         "y_true": [0, 1, 1, 0, 0, 1, 0, 1],
    ...         "y_pred": [0, 1, 0, 0, 0, 1, 0, 1],
    ...     }
    ... )
    >>> fig, ax = viz.plot_group_summary(df)
    """
    if metric != "accuracy":
        raise ValueError("plot_group_summary currently supports metric='accuracy'.")
    with coco_theme():
        scores = prepare_prediction_accuracy_scores(
            result,
            [group, "Fold"],
            model=model,
            context="group summary",
        )
        labels = []
        values = []
        for keys, grp in scores.groupby(["Model", group], dropna=False):
            labels.append(f"{keys[0]}\n{keys[1]}")
            values.append(grp["Value"].to_numpy(dtype=float))
        if kind == "violin" and all(len(v) > 1 for v in values):
            fig, ax = plot_distribution_groups(
                values,
                labels,
                kind="violin",
                show_points=False,
                ylabel=metric,
                title="Group Summary",
                ax=ax,
                figsize=figsize or (8, 5),
            )
        elif kind == "point":
            centers = np.array([np.mean(v) for v in values])
            sem = np.array(
                [
                    np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
                    for v in values
                ]
            )
            x_vals = np.arange(len(values))
            fig, ax = plot_error_points(
                x_vals,
                centers,
                yerr=sem,
                labels=labels,
                ax=ax,
                figsize=figsize or (8, 5),
                ylabel=metric,
                title="Group Summary",
            )
        elif kind == "violin":
            raise ValueError(
                "kind='violin' requires at least two fold values per group."
            )
        else:
            raise ValueError("kind must be 'point' or 'violin'.")
        return fig, ax


def plot_regression_diagnostics(
    result: Any,
    model: str | None = None,
    fold: int | None = None,
    kind: Literal["scatter", "residual", "bin"] = "scatter",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot regression prediction diagnostics.

    Parameters
    ----------
    result
        Experiment result with ``get_predictions()`` or a prediction DataFrame
        containing numeric ``y_true`` and ``y_pred`` columns.
    model
        Optional model name used to filter predictions.
    fold
        Optional fold index used to filter predictions.
    kind
        Plot type: ``"scatter"`` for observed vs predicted, ``"residual"`` for
        residuals vs predicted, or ``"bin"`` for binned mean residuals.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_regression_diagnostics :
        Interactive Plotly version.
    plot_decoding_scores : Aggregate scalar score summary.
    plot_confusion_matrix : Classification confusion matrix.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> rng = np.random.default_rng(42)
    >>> y = rng.standard_normal(40)
    >>> df = pd.DataFrame(
    ...     {"Model": "Ridge", "y_true": y, "y_pred": y + 0.1 * rng.standard_normal(40)}
    ... )
    >>> fig, ax = viz.plot_regression_diagnostics(df)
    """
    with coco_theme():
        y_true, y_pred = prepare_regression_prediction_data(
            result,
            model=model,
            fold=fold,
        )
        if kind == "scatter":
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            fig, ax = plot_scatter2d(
                y_true,
                y_pred,
                alpha=0.7,
                ax=ax,
                figsize=figsize or (6, 5),
                xlabel="Observed",
                ylabel="Predicted",
                title="Regression Diagnostics",
            )
            ax.plot(lims, lims, linestyle="--", color="0.5")
        elif kind == "residual":
            fig, ax = plot_scatter2d(
                y_pred,
                y_true - y_pred,
                alpha=0.7,
                reference_y=0,
                reference_style={"linestyle": "--", "color": "0.5"},
                ax=ax,
                figsize=figsize or (6, 5),
                xlabel="Predicted",
                ylabel="Residual",
                title="Regression Diagnostics",
            )
        elif kind == "bin":
            bins = pd.qcut(
                y_pred.rank(method="first"), q=min(10, len(y_pred)), duplicates="drop"
            )
            summary = (
                pd.DataFrame({"pred": y_pred, "resid": y_true - y_pred, "bin": bins})
                .groupby("bin", observed=False)
                .mean()
            )
            require_non_empty(summary, "binned regression diagnostics")
            fig, ax = plot_line(
                summary["pred"],
                summary["resid"],
                marker="o",
                ax=ax,
                figsize=figsize or (6, 5),
                xlabel="Predicted",
                ylabel="Mean residual",
                title="Regression Diagnostics",
            )
            ax.axhline(0, linestyle="--", color="0.5")
        else:
            raise ValueError("kind must be 'scatter', 'residual', or 'bin'.")
        return fig, ax


def plot_search_results(
    result: Any,
    model: str | None = None,
    top_n: int | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot compact hyperparameter-search results.

    Parameters
    ----------
    result
        Experiment result with ``get_search_results()`` or a search-results
        DataFrame containing ``Model``, ``Rank``, and ``MeanTestScore``.
    model
        Optional model name used to filter search rows.
    top_n
        Optional positive number of top-ranked candidates to keep per model.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_search_results : Interactive Plotly version.
    plot_model_comparison : Score differences between model pairs.
    plot_decoding_scores : Aggregate scalar score summary.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {"Model": ["SVM"] * 3,
    ...                           "Rank": [1, 2, 3],
    ...                           "MeanTestScore": [0.80, 0.78, 0.75]}
    ... )
    >>> fig, ax = viz.plot_search_results(df)
    """
    with coco_theme():
        frame = prepare_search_results_frame(result, model=model, top_n=top_n)

        def _search_label(row: pd.Series) -> str:
            parts = [str(row["Model"])]
            if "Fold" in row and pd.notna(row["Fold"]):
                parts.append(f"fold {row['Fold']}")
            rank = row["Rank"]
            if pd.notna(rank):
                rank_value = int(rank) if float(rank).is_integer() else rank
                parts.append(f"rank {rank_value}")
            if "Candidate" in row and pd.notna(row["Candidate"]):
                parts.append(f"candidate {row['Candidate']}")
            return "\n".join(parts)

        labels = frame.apply(_search_label, axis=1)
        values = pd.Series(
            frame["MeanTestScore"].astype(float).to_numpy(), index=labels
        )
        fig, ax = plot_bar(
            values,
            color=plt.get_cmap(QUALITATIVE)(2),
            sort=False,
            ax=ax,
            figsize=figsize or (max(6, len(frame) * 0.8), 4),
            ylabel="Mean Test Score",
            title="Search Results",
        )
        return fig, ax


def plot_feature_importance(
    result: Any,
    model: str | None = None,
    top_n: int | None = 25,
    signed: bool = False,
    absolute: bool = False,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot ranked feature importances.

    Parameters
    ----------
    result
        Experiment result with ``get_feature_importances()``, a feature
        importance DataFrame, or a mapping/sequence coercible to a numeric
        Series. Non-numeric sequences are delegated to the dimensionality
        reduction feature-importance plot.
    model
        Optional model name used to filter importances.
    top_n
        Optional number of highest-magnitude features to display.
    signed
        If True, use a diverging palette when plotting signed values.
    absolute
        If True, rank and display absolute importance values.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.
    title
        Optional axes title. Defaults to ``"Feature Importance"``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_feature_importance :
        Interactive Plotly version.
    plot_feature_stability : Feature-selection stability across folds.
    plot_feature_scores : Univariate feature-selector scores.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM"] * 4,
    ...         "FeatureName": ["f1", "f2", "f3", "f4"],
    ...         "Mean": [0.5, 0.3, 0.15, 0.05],
    ...     }
    ... )
    >>> fig, ax = viz.plot_feature_importance(df)
    """
    with coco_theme():
        try:
            plot_values = prepare_feature_importance_series(
                result, model=model, top_n=top_n, absolute=absolute
            )
        except (TypeError, ValueError):
            if _is_raw_importance_sequence(result):
                from coco_pipe.viz.dim_reduction import (
                    plot_feature_importance as plot_reduction_feature_importance,
                )

                return plot_reduction_feature_importance(
                    result,
                    title=title or "Feature Importance",
                    top_n=top_n or 20,
                    figsize=figsize,
                    ax=ax,
                )
            raise
        fig, ax = plot_bar(
            plot_values.sort_values(),
            cmap=_palette_for(signed and not absolute),
            orientation="horizontal",
            ax=ax,
            figsize=figsize or (8, max(4, len(plot_values) * 0.28)),
            xlabel="Importance",
            title=title or "Feature Importance",
        )
        return fig, ax


def plot_feature_stability(
    result: Any,
    model: str | None = None,
    top_n: int | None = 25,
    kind: Literal["bar", "heatmap"] = "bar",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot feature-selection stability.

    Parameters
    ----------
    result
        Experiment result or DataFrame. ``kind="bar"`` expects
        ``get_feature_stability()`` data with ``FeatureName`` and
        ``SelectionFrequency``. ``kind="heatmap"`` first tries
        ``get_selected_features()`` data with ``Fold``, ``FeatureName``, and
        ``Selected``.
    model
        Optional model name used to filter feature rows.
    top_n
        Optional number of most stable features to display.
    kind
        Plot type: ``"bar"`` for mean selection frequency or ``"heatmap"`` for
        fold-by-feature selected masks.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_feature_stability :
        Interactive Plotly version.
    plot_feature_importance : Ranked feature importances.
    plot_feature_scores : Univariate feature-selector scores.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM"] * 3,
    ...         "FeatureName": ["f1", "f2", "f3"],
    ...         "SelectionFrequency": [0.9, 0.6, 0.4],
    ...     }
    ... )
    >>> fig, ax = viz.plot_feature_stability(df)
    """
    with coco_theme():
        if kind == "heatmap":
            try:
                selected = coerce_decoding_frame(
                    result, accessor="get_selected_features"
                )
            except TypeError:
                selected = None
            if selected is not None:
                selected = select_rows(selected, model=model)
                require_columns(
                    selected,
                    ["Fold", "FeatureName", "Selected"],
                    context="get_selected_features",
                )
                require_non_empty(selected, "feature stability heatmap")
                top = (
                    selected.groupby("FeatureName")["Selected"]
                    .mean()
                    .sort_values(ascending=False)
                )
                if top_n is not None:
                    top = top.head(top_n)
                selected = selected[selected["FeatureName"].isin(top.index)]
                matrix = selected.pivot_table(
                    index="Fold",
                    columns="FeatureName",
                    values="Selected",
                    aggfunc="mean",
                )
                fig, ax = plot_heatmap(
                    matrix,
                    cmap=SEQUENTIAL,
                    aspect="auto",
                    colorbar_label="Selected",
                    xlabel="Feature",
                    ylabel="Fold",
                    title="Feature Stability",
                    ax=ax,
                    figsize=figsize or (max(6, len(matrix.columns) * 0.4), 4),
                )
                return fig, ax
        elif kind != "bar":
            raise ValueError("kind must be 'bar' or 'heatmap'.")

        series = prepare_feature_stability_series(result, model=model, top_n=top_n)
        fig, ax = plot_bar(
            series.sort_values(),
            cmap=_palette_for(False),
            orientation="horizontal",
            ax=ax,
            figsize=figsize or (8, max(4, len(series) * 0.28)),
            xlabel="Selection Frequency",
            title="Feature Stability",
        )
        return fig, ax


def plot_feature_scores(
    result: Any,
    model: str | None = None,
    top_n: int | None = 25,
    show_pvalues: bool = True,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot univariate feature-selector scores.

    Parameters
    ----------
    result
        Experiment result with ``get_feature_scores()`` or a feature-score
        DataFrame containing ``FeatureName`` and ``Score``.
    model
        Optional model name used to filter feature scores.
    top_n
        Optional positive number of highest-scoring features to display.
    show_pvalues
        If True, annotate bars with ``CorrectedPValue`` when available,
        otherwise ``PValue``.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    coco_pipe.viz.interactive.decoding.plot_feature_scores : Interactive Plotly version.
    plot_feature_importance : Ranked feature importances.
    plot_feature_stability : Feature-selection stability across folds.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM"] * 4,
    ...         "FeatureName": ["f1", "f2", "f3", "f4"],
    ...         "Score": [12.3, 8.1, 5.4, 2.0],
    ...     }
    ... )
    >>> fig, ax = viz.plot_feature_scores(df)
    """
    with coco_theme():
        frame = coerce_decoding_frame(result, accessor="get_feature_scores")
        frame = select_rows(frame, model=model)
        require_columns(frame, ["FeatureName", "Score"], context="get_feature_scores")
        require_non_empty(frame, "feature scores")
        scores = (
            frame.groupby("FeatureName")["Score"].mean().sort_values(ascending=False)
        )
        if top_n is not None:
            if top_n <= 0:
                raise ValueError("top_n must be positive when provided.")
            scores = scores.head(top_n)
        plot_scores = scores.sort_values()
        fig, ax = plot_bar(
            plot_scores,
            cmap=_palette_for(False),
            orientation="horizontal",
            ax=ax,
            figsize=figsize or (8, max(4, len(scores) * 0.28)),
            xlabel="Score",
            title="Feature Scores",
        )
        p_col = "CorrectedPValue" if "CorrectedPValue" in frame else "PValue"
        if show_pvalues and p_col in frame:
            pvals = (
                frame.groupby("FeatureName")[p_col].mean().reindex(plot_scores.index)
            )
            for patch, pval in zip(ax.patches, pvals):
                if pd.notna(pval):
                    ax.text(
                        patch.get_width(),
                        patch.get_y() + patch.get_height() / 2,
                        f" p={pval:.2g}",
                        va="center",
                    )
        return fig, ax


def plot_decoding_topomap(
    sensor_df: pd.DataFrame,
    value: str,
    info=None,
    coords=None,
    center: float | None = None,
    minimum_half_range: float = 0.02,
    times: list[float] | None = None,
    mask: np.ndarray | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a topomap from sensor-level decoding values.

    Parameters
    ----------
    sensor_df
        DataFrame containing ``FeatureName`` and the selected value column.
        ``FeatureName`` entries must match the provided sensor layout.
    value
        Numeric column to aggregate and plot.
    info
        Optional MNE ``Info`` object used to resolve sensor positions.
    coords
        Optional coordinate table used to resolve sensor positions when
        ``info`` is not provided.
    center
        Optional numeric center for diverging color limits. When provided,
        ``vmin`` and ``vmax`` are computed symmetrically around this value.
    minimum_half_range
        Minimum distance from ``center`` to either color limit.
    times
        Optional list of time values used to filter rows when ``Time`` exists.
    mask
        Optional boolean mask applied after time filtering.
    title
        Optional axes title. Defaults to the plotted value name.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    plot_sensor_feature_heatmap : Feature-family importance summarized by sensor.
    plot_sensor_feature_profile : Feature importances for one sensor.
    plot_feature_sensor_profile : Sensor topomap for one feature family.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> names = [f"EEG{i:03d}" for i in range(5)]
    >>> df = pd.DataFrame({"FeatureName": names,
    ...                           "Importance": np.linspace(0.1, 0.9, 5)})
    >>> coords = pd.DataFrame(
    ...     {
    ...         "ch_name": names,
    ...         "x": np.cos(np.linspace(0, 2 * np.pi, 5, endpoint=False)),
    ...         "y": np.sin(np.linspace(0, 2 * np.pi, 5, endpoint=False)),
    ...     }
    ... )
    >>> fig, ax = viz.plot_decoding_topomap(df, "Importance", coords=coords)
    """
    with coco_theme():
        require_columns(
            sensor_df, ["FeatureName", value], context="plot_decoding_topomap"
        )
        if info is None and coords is None:
            raise ValueError("plot_decoding_topomap requires either info or coords.")
        data = sensor_df.copy()
        if times is not None and "Time" in data:
            data = data[data["Time"].isin(times)]
            require_non_empty(data, "decoding topomap after time filtering")
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if len(mask_arr) != len(data):
                raise ValueError(
                    "mask length must match the number of topomap rows after filtering."
                )
            data = data.loc[mask_arr]
            require_non_empty(data, "decoding topomap after mask filtering")
        require_non_empty(data, "decoding topomap")
        values = data.groupby("FeatureName")[value].mean()
        vmin = vmax = None
        if center is not None:
            from coco_pipe.viz.base import centered_color_limits

            vmin, vmax = centered_color_limits(
                pd.to_numeric(values, errors="coerce"),
                center=float(center),
                minimum_half_range=minimum_half_range,
            )
        return plot_topomap(
            values,
            coords=coords,
            info=info,
            vmin=vmin,
            vmax=vmax,
            cmap=DIVERGING if center is not None else SEQUENTIAL,
            symmetric=center is not None,
            title=title or value,
            cbar_label=value,
            ax=ax,
            figsize=figsize or (5, 5),
        )


def plot_sensor_feature_heatmap(
    result: Any,
    feature_metadata: pd.DataFrame,
    feature_group: str | None = None,
    sensor_order: list[str] | None = None,
    agg: str = "mean",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot feature-family importance summarized by sensor.

    Parameters
    ----------
    result
        Experiment result with ``get_feature_importances()`` or a compatible
        feature-importance DataFrame.
    feature_metadata
        Explicit metadata with ``FeatureName``, ``Sensor``, and
        ``FeatureFamily`` columns.
    feature_group
        Optional feature family value used to filter rows. The parameter name
        is retained for compatibility; it matches ``FeatureFamily`` values.
    sensor_order
        Optional ordered list of sensors to display as columns.
    agg
        Aggregation function passed to ``pandas.pivot_table``.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    plot_decoding_topomap : Sensor topomap for a single score column.
    plot_sensor_feature_profile : Feature importances for one sensor.
    plot_feature_sensor_profile : Sensor topomap for one feature family.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> feat_meta = pd.DataFrame(
    ...     {
    ...         "FeatureName": ["f1", "f2", "f3", "f4"],
    ...         "Sensor": ["S1", "S1", "S2", "S2"],
    ...         "FeatureFamily": ["band1", "band2", "band1", "band2"],
    ...     }
    ... )
    >>> importance = pd.DataFrame(
    ...     {"FeatureName": ["f1", "f2", "f3", "f4"], "Mean": [0.5, 0.3, 0.4, 0.2]}
    ... )
    >>> fig, ax = viz.plot_sensor_feature_heatmap(importance, feat_meta)
    """
    with coco_theme():
        data = _importance_with_metadata(
            result,
            feature_metadata,
            metadata_columns=["FeatureName", "Sensor", "FeatureFamily"],
        )
        if feature_group is not None:
            data = data[data["FeatureFamily"] == feature_group]
            require_non_empty(
                data, "sensor feature heatmap after feature_group filtering"
            )
        matrix = data.pivot_table(
            index="FeatureFamily",
            columns="Sensor",
            values="_ImportanceValue",
            aggfunc=agg,
        )
        if sensor_order is not None:
            matrix = matrix.reindex(columns=sensor_order)
            if matrix.dropna(axis=1, how="all").empty:
                raise ValueError(
                    "No sensor feature heatmap columns remain after "
                    "sensor_order filtering."
                )
        require_non_empty(matrix.dropna(how="all"), "sensor feature heatmap")
        fig, ax = plot_heatmap(
            matrix,
            cmap=SEQUENTIAL,
            aspect="auto",
            colorbar_label="Importance",
            xlabel="Sensor",
            ylabel="FeatureFamily",
            title="Sensor Feature Heatmap",
            ax=ax,
            figsize=figsize
            or (max(6, matrix.shape[1] * 0.35), max(4, matrix.shape[0] * 0.35)),
        )
        return fig, ax


def plot_sensor_feature_profile(
    result: Any,
    feature_metadata: pd.DataFrame,
    sensor: str,
    model: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot feature importances for one sensor.

    Parameters
    ----------
    result
        Experiment result with ``get_feature_importances()`` or a compatible
        feature-importance DataFrame.
    feature_metadata
        Explicit metadata with ``FeatureName`` and ``Sensor`` columns.
    sensor
        Sensor name used to select features.
    model
        Optional model name used to filter importances.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    plot_sensor_feature_heatmap : Feature-family importance summarized by sensor.
    plot_feature_sensor_profile : Sensor topomap for one feature family.
    plot_feature_importance : Ranked feature importances.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> feat_meta = pd.DataFrame(
    ...     {"FeatureName": ["f1", "f2", "f3"], "Sensor": ["S1", "S1", "S1"]}
    ... )
    >>> importance = pd.DataFrame({"FeatureName": ["f1",
    ...                           "f2",
    ...                           "f3"],
    ...                           "Mean": [0.5, 0.3, 0.1]})
    >>> fig, ax = viz.plot_sensor_feature_profile(importance, feat_meta, sensor="S1")
    """
    with coco_theme():
        data = _importance_with_metadata(
            result,
            feature_metadata,
            model=model,
            metadata_columns=["FeatureName", "Sensor"],
        )
        data = data[data["Sensor"] == sensor]
        require_non_empty(data, "sensor feature profile")
        series = data.groupby("FeatureName")["_ImportanceValue"].mean().sort_values()
        fig, ax = plot_bar(
            series,
            cmap=_palette_for(False),
            orientation="horizontal",
            ax=ax,
            figsize=figsize or (8, max(4, len(series) * 0.28)),
            xlabel="Importance",
            title=f"{sensor} Feature Profile",
        )
        return fig, ax


def plot_feature_sensor_profile(
    result: Any,
    feature_metadata: pd.DataFrame,
    feature_family: str,
    info=None,
    coords=None,
    model: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a sensor topomap for one feature family.

    Parameters
    ----------
    result
        Experiment result with ``get_feature_importances()`` or a compatible
        feature-importance DataFrame.
    feature_metadata
        Explicit metadata with ``FeatureName``, ``Sensor``, and
        ``FeatureFamily`` columns.
    feature_family
        Feature-family value to aggregate over sensors.
    info
        Optional MNE ``Info`` object used to resolve sensor positions.
    coords
        Optional coordinate table used to resolve sensor positions when
        ``info`` is not provided.
    model
        Optional model name used to filter importances.
    title
        Optional topomap title. Defaults to ``"<feature_family> Sensor Profile"``.
    ax
        Existing Matplotlib axes to draw into.
    figsize
        Figure size used when creating a new axes.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created or reused figure and axes.

    See Also
    --------
    plot_sensor_feature_heatmap : Feature-family importance summarized by sensor.
    plot_sensor_feature_profile : Feature importances for one sensor.
    plot_decoding_topomap : Sensor topomap for a single score column.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from coco_pipe.viz import decoding as viz
    >>> feat_meta = pd.DataFrame(
    ...     {
    ...         "FeatureName": ["f1", "f2"],
    ...         "Sensor": ["S1", "S2"],
    ...         "FeatureFamily": ["band1", "band1"],
    ...     }
    ... )
    >>> importance = pd.DataFrame({"FeatureName": ["f1", "f2"], "Mean": [0.5, 0.3]})
    >>> coords = pd.DataFrame({"ch_name": ["S1",
    ...                           "S2"],
    ...                           "x": [0.1, -0.1],
    ...                           "y": [0.1, -0.1]})
    >>> fig, ax = viz.plot_feature_sensor_profile(importance, feat_meta,
    ...                           "band1", coords=coords)
    """
    with coco_theme():
        if info is None and coords is None:
            raise ValueError(
                "plot_feature_sensor_profile requires either info or coords."
            )
        data = _importance_with_metadata(
            result,
            feature_metadata,
            model=model,
            metadata_columns=["FeatureName", "Sensor", "FeatureFamily"],
        )
        data = data[data["FeatureFamily"] == feature_family]
        require_non_empty(data, "feature sensor profile after feature_family filtering")
        sensor_values = (
            data.groupby("Sensor")["_ImportanceValue"].mean().rename_axis("FeatureName")
        )
        return plot_topomap(
            sensor_values,
            coords=coords,
            info=info,
            cmap=SEQUENTIAL,
            symmetric=False,
            title=title or f"{feature_family} Sensor Profile",
            cbar_label="Importance",
            ax=ax,
            figsize=figsize or (5, 5),
        )


def plot_head_to_head(
    frame: pd.DataFrame,
    *,
    label: str,
    value: str,
    error: str | None = None,
    reference: float | None = None,
    title: str = "Head-to-Head Comparison",
    ylabel: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot labelled estimates for cross-result head-to-head comparisons."""
    require_columns(frame, [label, value], context="plot_head_to_head")
    columns = [label, value] + ([error] if error else [])
    data = frame[columns].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna(subset=[label, value])
    require_non_empty(data, "head-to-head comparison")
    values = pd.Series(data[value].to_numpy(), index=data[label].astype(str))
    errors = (
        pd.Series(
            pd.to_numeric(data[error], errors="coerce").to_numpy(),
            index=values.index,
        )
        if error is not None and error in data
        else None
    )
    fig, ax = plot_bar(
        values,
        errors=errors,
        sort=False,
        title=title,
        ylabel=ylabel or value,
        ax=ax,
        figsize=figsize,
    )
    if reference is not None:
        ax.axhline(reference, color="0.4", linestyle="--", linewidth=1)
    return fig, ax


def plot_paired_delta(
    frame: pd.DataFrame,
    *,
    label: str,
    delta: str,
    lower: str | None = None,
    upper: str | None = None,
    title: str = "Paired Difference",
    xlabel: str = "Comparison",
    ylabel: str = "Delta",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot paired deltas with optional confidence intervals."""
    require_columns(frame, [label, delta], context="plot_paired_delta")
    data = frame.copy()
    data[delta] = pd.to_numeric(data[delta], errors="coerce")
    data = data.dropna(subset=[label, delta])
    require_non_empty(data, "paired delta comparison")
    yerr = None
    if lower is not None and upper is not None:
        require_columns(data, [lower, upper], context="plot_paired_delta")
        low = pd.to_numeric(data[lower], errors="coerce").to_numpy()
        high = pd.to_numeric(data[upper], errors="coerce").to_numpy()
        estimate = data[delta].to_numpy(dtype=float)
        yerr = np.vstack([estimate - low, high - estimate])
    return plot_error_points(
        np.arange(len(data)),
        data[delta].to_numpy(dtype=float),
        yerr=yerr,
        labels=data[label].astype(str).tolist(),
        reference_y=0.0,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        figsize=figsize,
    )
