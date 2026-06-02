"""Interactive Plotly visualization helpers for decoding result tables."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .._utils import (
    prepare_confusion_matrix,
    prepare_curve_group_data,
    prepare_decoding_curve_frame,
    prepare_decoding_score_data,
    prepare_feature_score_series,
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
)
from ..theme import _COLORBLIND_COLORS, SEQUENTIAL
from ._utils import _apply_layout

__all__ = [
    "plot_calibration_curve",
    "plot_confusion_matrix",
    "plot_decoding_scores",
    "plot_feature_scores",
    "plot_feature_stability",
    "plot_fit_diagnostics",
    "plot_fold_score_dispersion",
    "plot_group_summary",
    "plot_model_comparison",
    "plot_null_interval_summary",
    "plot_pr_curve",
    "plot_probability_diagnostics",
    "plot_regression_diagnostics",
    "plot_roc_curve",
    "plot_search_results",
    "plot_subject_diagnostics",
    "plot_temporal_generalization_matrix",
    "plot_temporal_score_curve",
    "plot_temporal_statistical_assessment",
    "plot_training_history",
]


def plot_confusion_matrix(
    result_or_matrix: Any,
    model: Optional[str] = None,
    fold: Optional[int] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot an aggregated confusion matrix interactively.

    Parameters
    ----------
    result_or_matrix
        Experiment result with ``get_confusion_matrices()`` or a tidy DataFrame
        containing ``TrueLabel``, ``PredictedLabel``, and ``Value``.
    model
        Optional model name used to filter rows before aggregation.
    fold
        Optional fold index used to filter rows before aggregation.
    title
        Optional figure title. Defaults to ``"Confusion Matrix"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive confusion matrix heatmap.

    See Also
    --------
    coco_pipe.viz.decoding.plot_confusion_matrix : Static Matplotlib version.
    plot_probability_diagnostics :
        Probability quality diagnostics for classifier output.
    plot_calibration_curve : Calibration reliability curve for probability estimates.
    plot_roc_curve : ROC curve complementing confusion matrix analysis.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "TrueLabel": ["A", "B", "A"],
    ...         "PredictedLabel": ["A", "B", "B"],
    ...         "Value": [5, 3, 2],
    ...     }
    ... )
    >>> fig = viz.plot_confusion_matrix(df)
    """
    matrix = prepare_confusion_matrix(result_or_matrix, model=model, fold=fold)
    z = matrix.values
    text = [[str(int(v)) for v in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=matrix.columns.astype(str).tolist(),
            y=matrix.index.astype(str).tolist(),
            colorscale=SEQUENTIAL,
            colorbar=dict(title="Count"),
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=12),
        )
    )
    _apply_layout(
        fig,
        title=title or "Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert a hex color string to an rgba() CSS string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _add_curve_band(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    fillcolor: str = "rgba(68,68,68,0.12)",
) -> None:
    """Add a shaded std-deviation band around a mean curve."""
    upper = y + yerr
    lower = y - yerr
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=fillcolor,
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )


def plot_roc_curve(
    result_or_curve: Any,
    model: Optional[str] = None,
    fold: Optional[int] = None,
    title: Optional[str] = None,
    mean_only: bool = False,
) -> go.Figure:
    """
    Plot receiver-operating-characteristic curves interactively.

    Parameters
    ----------
    result_or_curve
        Experiment result with ``get_roc_curve()`` or a DataFrame containing
        ``Model``, ``FPR``, and ``TPR``.
    model
        Optional model name to display.
    fold
        Optional fold index to display.
    title
        Optional figure title. Defaults to ``"ROC Curve"``.
    mean_only
        If True, interpolate fold curves onto a common grid and draw the mean
        curve with a standard-deviation band.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive ROC curve figure.

    See Also
    --------
    coco_pipe.viz.decoding.plot_roc_curve : Static Matplotlib version.
    plot_pr_curve : Precision-recall curve for imbalanced-class problems.
    plot_calibration_curve : Calibration reliability curve.
    plot_confusion_matrix : Confusion matrix at a fixed operating point.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame({"Model": ["SVM",
    ...                           "SVM"],
    ...                           "FPR": [0.0, 1.0],
    ...                           "TPR": [0.0, 1.0]})
    >>> fig = viz.plot_roc_curve(df)
    """
    frame = prepare_decoding_curve_frame(
        result_or_curve,
        accessor="get_roc_curve",
        required_columns=["Model", "FPR", "TPR"],
        context="ROC curve",
        model=model,
        fold=fold,
    )
    fig = go.Figure()
    for curve in prepare_curve_group_data(frame, "FPR", "TPR", mean_only):
        fig.add_trace(
            go.Scatter(
                x=curve["x"],
                y=curve["y"],
                name=curve["label"],
                mode="lines",
                line=dict(width=2 if curve["kind"] == "mean" else 1),
                opacity=1.0 if curve["kind"] == "mean" else 0.55,
            )
        )
        if curve["kind"] == "mean" and curve["yerr"] is not None:
            _add_curve_band(fig, curve["x"], curve["y"], curve["yerr"])
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        )
    )
    _apply_layout(
        fig,
        title=title or "ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
    )
    return fig


def plot_pr_curve(
    result_or_curve: Any,
    model: Optional[str] = None,
    fold: Optional[int] = None,
    title: Optional[str] = None,
    mean_only: bool = False,
) -> go.Figure:
    """
    Plot precision-recall curves interactively.

    Parameters
    ----------
    result_or_curve
        Experiment result with ``get_pr_curve()`` or a DataFrame containing
        ``Model``, ``Recall``, and ``Precision``.
    model
        Optional model name to display.
    fold
        Optional fold index to display.
    title
        Optional figure title. Defaults to ``"Precision-Recall Curve"``.
    mean_only
        If True, draw the mean curve with a standard-deviation band.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive precision-recall curve figure.

    See Also
    --------
    coco_pipe.viz.decoding.plot_pr_curve : Static Matplotlib version.
    plot_roc_curve : ROC curve for alternative threshold analysis.
    plot_calibration_curve : Calibration reliability curve.
    plot_fold_score_dispersion : Fold-level score variability.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {"Model": ["SVM", "SVM"], "Recall": [0.0, 1.0], "Precision": [1.0, 0.5]}
    ... )
    >>> fig = viz.plot_pr_curve(df)
    """
    frame = prepare_decoding_curve_frame(
        result_or_curve,
        accessor="get_pr_curve",
        required_columns=["Model", "Recall", "Precision"],
        context="precision-recall curve",
        model=model,
        fold=fold,
    )
    fig = go.Figure()
    for curve in prepare_curve_group_data(frame, "Recall", "Precision", mean_only):
        fig.add_trace(
            go.Scatter(
                x=curve["x"],
                y=curve["y"],
                name=curve["label"],
                mode="lines",
                line=dict(width=2 if curve["kind"] == "mean" else 1),
                opacity=1.0 if curve["kind"] == "mean" else 0.55,
            )
        )
        if curve["kind"] == "mean" and curve["yerr"] is not None:
            _add_curve_band(fig, curve["x"], curve["y"], curve["yerr"])
    _apply_layout(
        fig,
        title=title or "Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=450,
    )
    return fig


def plot_calibration_curve(
    result_or_curve: Any,
    model: Optional[str] = None,
    fold: Optional[int] = None,
    title: Optional[str] = None,
    mean_only: bool = False,
) -> go.Figure:
    """
    Plot calibration reliability curves interactively.

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
        Optional figure title. Defaults to ``"Calibration Curve"``.
    mean_only
        If True, draw the mean curve with a standard-deviation band.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive calibration curve figure.

    See Also
    --------
    coco_pipe.viz.decoding.plot_calibration_curve : Static Matplotlib version.
    plot_roc_curve : ROC curve complementing calibration analysis.
    plot_pr_curve : Precision-recall curve for classifier evaluation.
    plot_probability_diagnostics : Scalar probability-quality metrics.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["LR", "LR"],
    ...         "MeanPredictedProbability": [0.2, 0.8],
    ...         "FractionPositive": [0.15, 0.85],
    ...     }
    ... )
    >>> fig = viz.plot_calibration_curve(df)
    """
    frame = prepare_decoding_curve_frame(
        result_or_curve,
        accessor="get_calibration_curve",
        required_columns=["Model", "MeanPredictedProbability", "FractionPositive"],
        context="calibration curve",
        model=model,
        fold=fold,
    )
    fig = go.Figure()
    for curve in prepare_curve_group_data(
        frame, "MeanPredictedProbability", "FractionPositive", mean_only
    ):
        fig.add_trace(
            go.Scatter(
                x=curve["x"],
                y=curve["y"],
                name=curve["label"],
                mode="lines",
                line=dict(width=2 if curve["kind"] == "mean" else 1),
                opacity=1.0 if curve["kind"] == "mean" else 0.55,
            )
        )
        if curve["kind"] == "mean" and curve["yerr"] is not None:
            _add_curve_band(fig, curve["x"], curve["y"], curve["yerr"])
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        )
    )
    _apply_layout(
        fig,
        title=title or "Calibration Curve",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction Positive",
        height=450,
    )
    return fig


def plot_fold_score_dispersion(
    result_or_scores: Any,
    metric: Optional[str] = None,
    model: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot fold-level scalar score distributions by model and metric interactively.

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
        Optional figure title. Defaults to ``"Fold Score Dispersion"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive fold score box plot.

    See Also
    --------
    coco_pipe.viz.decoding.plot_fold_score_dispersion : Static Matplotlib version.
    plot_decoding_scores : Aggregate score bar chart with error bars.
    plot_model_comparison : Pairwise score-difference comparison.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM"] * 4,
    ...         "Fold": [0, 1, 2, 3],
    ...         "Metric": ["accuracy"] * 4,
    ...         "Value": [0.8, 0.85, 0.82, 0.78],
    ...     }
    ... )
    >>> fig = viz.plot_fold_score_dispersion(df)
    """
    data = prepare_fold_score_data(result_or_scores, model=model, metric=metric)
    fig = go.Figure()
    for (model_name, metric_name), group in data.groupby(["Model", "Metric"]):
        label = f"{model_name} / {metric_name}"
        fig.add_trace(
            go.Box(
                y=group["Value"].astype(float).tolist(),
                name=label,
                boxpoints="all",
                jitter=0.3,
                pointpos=0,
            )
        )
    _apply_layout(
        fig,
        title=title or "Fold Score Dispersion",
        yaxis_title="Value",
        height=450,
    )
    return fig


def plot_temporal_score_curve(
    result_or_scores: Any,
    metric: Optional[str] = None,
    model: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[dict] = None,
    smooth_window: Optional[int] = None,
) -> go.Figure:
    """
    Plot mean temporal decoding score curves interactively.

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
        Optional figure title. Defaults to ``"Temporal Decoding Value"``.
    colors
        Optional dict mapping model names to CSS color strings.
    smooth_window
        Optional integer window size for smoothing the curves using a centered
        moving average.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive temporal score curve figure.

    See Also
    --------
    coco_pipe.viz.decoding.plot_temporal_score_curve : Static Matplotlib version.
    plot_temporal_generalization_matrix : Train-time by test-time generalization matrix.
    plot_temporal_statistical_assessment :
        Statistical significance overlay on temporal curves.
    plot_null_interval_summary : Null permutation interval for scalar score comparison.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM"] * 3,
    ...         "Metric": ["accuracy"] * 3,
    ...         "Time": [0.1, 0.2, 0.3],
    ...         "Mean": [0.6, 0.75, 0.7],
    ...     }
    ... )
    >>> fig = viz.plot_temporal_score_curve(df)
    """
    curve_data = prepare_temporal_score_curve_frame(
        result_or_scores,
        model=model,
        metric=metric,
    )
    fig = go.Figure()
    for (model_name, metric_name), group in curve_data.groupby(["Model", "Metric"]):
        numeric = pd.to_numeric(group["Time"], errors="coerce")
        x_vals = numeric.to_numpy() if numeric.notna().all() else np.arange(len(group))

        y_vals_s = group["Mean"].astype(float)
        has_std = "Std" in group.columns
        std_vals_s = group["Std"].fillna(0).astype(float) if has_std else None

        if smooth_window is not None and smooth_window > 1:
            y_vals_s = y_vals_s.rolling(
                smooth_window, center=True, min_periods=1
            ).mean()
            if std_vals_s is not None:
                std_vals_s = std_vals_s.rolling(
                    smooth_window, center=True, min_periods=1
                ).mean()

        y_vals = y_vals_s.to_numpy()
        std_vals = std_vals_s.to_numpy() if std_vals_s is not None else None

        line_color = (colors or {}).get(model_name)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=f"{model_name} / {metric_name}",
                line=dict(width=2, color=line_color),
            )
        )
        if std_vals is not None:
            fillcolor = (
                _hex_to_rgba(line_color, 0.15) if line_color else "rgba(68,68,68,0.12)"
            )
            _add_curve_band(fig, x_vals, y_vals, std_vals, fillcolor=fillcolor)
    _apply_layout(
        fig,
        title=title or "Temporal Decoding Value",
        xaxis_title="Time",
        yaxis_title="Value",
        height=450,
    )
    return fig


def plot_temporal_generalization_matrix(
    result_or_scores: Any,
    metric: Optional[str] = None,
    model: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot a train-time by test-time temporal generalization matrix interactively.

    Parameters
    ----------
    result_or_scores
        Experiment result with ``get_temporal_score_summary()`` or a DataFrame
        containing ``Model``, ``Metric``, ``TrainTime``, ``TestTime``, and ``Mean``.
    metric
        Optional metric name.
    model
        Optional model name.
    title
        Optional figure title. Defaults to ``"<model> / <metric>"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive generalization matrix heatmap.

    See Also
    --------
    coco_pipe.viz.decoding.plot_temporal_generalization_matrix :
        Static Matplotlib version.
    plot_temporal_score_curve : Mean temporal score curve across time.
    plot_temporal_statistical_assessment : Statistical assessment of temporal decoding.
    plot_null_interval_summary : Null permutation interval summary.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM"] * 4,
    ...         "Metric": ["accuracy"] * 4,
    ...         "TrainTime": [0.1, 0.1, 0.2, 0.2],
    ...         "TestTime": [0.1, 0.2, 0.1, 0.2],
    ...         "Mean": [0.8, 0.6, 0.65, 0.82],
    ...     }
    ... )
    >>> fig = viz.plot_temporal_generalization_matrix(df)
    """
    matrix, first = prepare_temporal_generalization_matrix(
        result_or_scores,
        model=model,
        metric=metric,
    )
    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=matrix.columns.astype(str).tolist(),
            y=matrix.index.astype(str).tolist(),
            colorscale=SEQUENTIAL,
            colorbar=dict(title="Value"),
        )
    )
    _apply_layout(
        fig,
        title=title or f"{first['Model']} / {first['Metric']}",
        xaxis_title="Test Time",
        yaxis_title="Train Time",
        height=500,
    )
    return fig


def plot_temporal_statistical_assessment(
    result_or_assessment: Any,
    metric: Optional[str] = None,
    model: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot temporal statistical assessment results interactively.

    Parameters
    ----------
    result_or_assessment
        Experiment result with ``get_statistical_assessment()`` or a DataFrame
        containing ``Model``, ``Metric``, ``Observed``, and ``Time``.
    metric
        Optional metric name.
    model
        Optional model name.
    title
        Optional figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive temporal statistical assessment figure.

    See Also
    --------
    coco_pipe.viz.decoding.plot_temporal_statistical_assessment :
        Static Matplotlib version.
    plot_temporal_score_curve : Mean temporal decoding score curve.
    plot_null_interval_summary : Scalar null interval summary across models.
    plot_temporal_generalization_matrix : Full train-by-test generalization matrix.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM"] * 3,
    ...         "Metric": ["accuracy"] * 3,
    ...         "Observed": [0.6, 0.75, 0.7],
    ...         "Time": [0.1, 0.2, 0.3],
    ...     }
    ... )
    >>> fig = viz.plot_temporal_statistical_assessment(df)
    """
    frame = prepare_temporal_statistical_frame(
        result_or_assessment,
        model=model,
        metric=metric,
    )
    first = frame.iloc[0]
    numeric = pd.to_numeric(frame["Time"], errors="coerce")
    x_vals = numeric.to_numpy() if numeric.notna().all() else np.arange(len(frame))
    observed = frame["Observed"].astype(float).to_numpy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=observed,
            mode="lines+markers",
            name="Observed",
            line=dict(width=2),
        )
    )
    if {"NullLower", "NullUpper"}.issubset(frame.columns):
        lower = pd.to_numeric(frame["NullLower"], errors="coerce")
        upper = pd.to_numeric(frame["NullUpper"], errors="coerce")
        if lower.notna().any() and upper.notna().any():
            l_vals = lower.ffill().to_numpy()
            u_vals = upper.ffill().to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_vals, x_vals[::-1]]),
                    y=np.concatenate([u_vals, l_vals[::-1]]),
                    fill="toself",
                    fillcolor="rgba(128,128,128,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Permutation null band",
                )
            )
    if "Significant" in frame.columns:
        sig = frame["Significant"].fillna(False).to_numpy(dtype=bool)
        if sig.any():
            fig.add_trace(
                go.Scatter(
                    x=x_vals[sig],
                    y=observed[sig],
                    mode="markers",
                    marker=dict(symbol="square", color="black", size=10),
                    name="Significant",
                )
            )
    _apply_layout(
        fig,
        title=title or f"{first['Model']} / {first['Metric']} statistical assessment",
        xaxis_title="Time",
        yaxis_title="Value",
        height=450,
    )
    return fig


def plot_null_interval_summary(
    result_or_assessment: Any,
    metric: Optional[str] = None,
    model: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot observed scalar scores against null interval summaries interactively.

    Parameters
    ----------
    result_or_assessment
        Experiment result with ``get_statistical_assessment()`` or a DataFrame
        containing ``Model``, ``Metric``, and ``Observed``.
    metric
        Optional metric name used to filter assessment rows.
    model
        Optional model name used to filter assessment rows.
    title
        Optional figure title. Defaults to ``"Null Interval Summary"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive null interval summary figure.

    See Also
    --------
    coco_pipe.viz.decoding.plot_null_interval_summary : Static Matplotlib version.
    plot_temporal_statistical_assessment :
        Temporal null band overlaid on the score curve.
    plot_temporal_score_curve : Mean temporal decoding score curve.
    plot_decoding_scores : Aggregate score bar chart without null bands.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM", "LDA"],
    ...         "Metric": ["accuracy", "accuracy"],
    ...         "Observed": [0.82, 0.74],
    ...     }
    ... )
    >>> fig = viz.plot_null_interval_summary(df)
    """
    frame = prepare_null_interval_frame(
        result_or_assessment,
        model=model,
        metric=metric,
    )
    labels = [f"{row.Model} / {row.Metric}" for row in frame.itertuples()]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=frame["Observed"].astype(float).tolist(),
            mode="markers",
            marker=dict(size=9),
            name="Observed",
        )
    )
    if {"NullLower", "NullUpper"}.issubset(frame.columns):
        lower = pd.to_numeric(frame["NullLower"], errors="coerce")
        upper = pd.to_numeric(frame["NullUpper"], errors="coerce")
        center = (lower + upper) / 2
        error_plus = (upper - center).to_numpy()
        error_minus = (center - lower).to_numpy()
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=center.tolist(),
                mode="markers",
                error_y=dict(
                    type="data",
                    array=error_plus.tolist(),
                    arrayminus=error_minus.tolist(),
                ),
                name="Null band",
            )
        )
    if "NullMedian" in frame.columns and frame["NullMedian"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=frame["NullMedian"].astype(float).tolist(),
                mode="markers",
                marker=dict(symbol="x", size=9),
                name="Null median",
            )
        )
    _apply_layout(
        fig,
        title=title or "Null Interval Summary",
        yaxis_title="Value",
        height=450,
    )
    return fig


def plot_training_history(
    result_or_artifacts: Any,
    model: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot neural training-history artifacts interactively.

    Parameters
    ----------
    result_or_artifacts
        Experiment result with ``get_model_artifacts()`` or an artifact
        DataFrame containing ``Model``, ``Key``, ``ArtifactType``, and
        ``Value``.
    model
        Optional model name used to filter artifacts.
    title
        Optional figure title. Defaults to ``"Training History"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive training history line chart.

    See Also
    --------
    coco_pipe.viz.decoding.plot_training_history : Static Matplotlib version.
    plot_fit_diagnostics : Fit-time diagnostics including total training time.
    plot_decoding_scores : Final evaluation scores after training.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["CNN"],
    ...         "Key": ["training"],
    ...         "ArtifactType": ["history"],
    ...         "Value": [[{"epoch": 0, "loss": 1.0}, {"epoch": 1, "loss": 0.5}]],
    ...     }
    ... )
    >>> fig = viz.plot_training_history(df)
    """
    rows = prepare_training_history_artifacts(result_or_artifacts, model=model)
    fig = go.Figure()
    plotted = False
    for row in rows.itertuples():
        history = row.Value if row.Value is not None else []
        frame = pd.DataFrame(history)
        if frame.empty:
            continue
        if "epoch" not in frame.columns:
            frame = frame.reset_index(names="epoch")
        for column in [col for col in frame.columns if col != "epoch"]:
            fig.add_trace(
                go.Scatter(
                    x=frame["epoch"],
                    y=frame[column],
                    mode="lines+markers",
                    name=f"{row.Model} {column}",
                )
            )
            plotted = True
    if not plotted:
        raise ValueError("No plottable training history artifacts available.")
    _apply_layout(
        fig,
        title=title or "Training History",
        xaxis_title="Epoch",
        yaxis_title="Value",
        height=420,
    )
    return fig


def plot_decoding_scores(
    result: Any,
    metric: Optional[str] = None,
    model: Optional[str] = None,
    aggregate: Literal["mean", "median"] = "mean",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot aggregate scalar decoding scores by model and metric interactively.

    Parameters
    ----------
    result
        Experiment result with ``get_detailed_scores()`` or a detailed score
        DataFrame containing scalar ``Value`` rows.
    metric
        Optional metric name used to filter scores.
    model
        Optional model name used to filter scores.
    aggregate
        Summary statistic: ``"mean"`` or ``"median"``.
    title
        Optional figure title. Defaults to ``"Decoding Scores"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive decoding score bar chart with error bars.

    See Also
    --------
    coco_pipe.viz.decoding.plot_decoding_scores : Static Matplotlib version.
    plot_fold_score_dispersion : Per-fold score distribution box plot.
    plot_model_comparison : Pairwise model score-difference figure.
    plot_null_interval_summary : Observed scores against null permutation intervals.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM"] * 3,
    ...         "Fold": [0, 1, 2],
    ...         "Metric": ["accuracy"] * 3,
    ...         "Value": [0.8, 0.85, 0.82],
    ...     }
    ... )
    >>> fig = viz.plot_decoding_scores(df)
    """
    data = prepare_decoding_score_data(result, model=model, metric=metric)
    grouped = data.groupby(["Model", "Metric"], dropna=False)["Value"]
    summary = grouped.agg(["mean", "median", "std", "count"]).reset_index()
    summary["center"] = summary[aggregate]
    summary["sem"] = summary["std"].fillna(0) / np.sqrt(summary["count"].clip(lower=1))
    labels = summary.apply(lambda row: f"{row['Model']} / {row['Metric']}", axis=1)
    fig = go.Figure(
        go.Bar(
            x=labels.tolist(),
            y=summary["center"].astype(float).tolist(),
            error_y=dict(
                type="data",
                array=summary["sem"].astype(float).tolist(),
            ),
            name="Score",
        )
    )
    _apply_layout(
        fig,
        title=title or "Decoding Scores",
        yaxis_title="Value",
        height=420,
    )
    return fig


def plot_model_comparison(
    result: Any,
    metric: str = "accuracy",
    reference: Optional[str] = None,
    paired: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot model-comparison score differences interactively.

    Parameters
    ----------
    result
        Experiment result or a comparison DataFrame with a ``Difference``
        column. Optional ``ModelA``, ``ModelB``, ``CILower``, ``CIUpper``
        columns are also used when present.
    metric
        Metric used when computing comparisons.
    reference
        Optional reference model for paired comparisons.
    paired
        If True and ``reference`` is provided, use ``compare_models_paired``.
    title
        Optional figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive model comparison scatter figure.

    See Also
    --------
    coco_pipe.viz.decoding.plot_model_comparison : Static Matplotlib version.
    plot_decoding_scores : Aggregate score bar chart per model.
    plot_fold_score_dispersion : Per-fold score variability.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame({"Difference": [0.05, -0.02, 0.08]})
    >>> fig = viz.plot_model_comparison(df)
    """
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
    error_x = None
    if {"CILower", "CIUpper"}.issubset(comp.columns):
        lower = comp["CILower"].astype(float).to_numpy()[order]
        upper = comp["CIUpper"].astype(float).to_numpy()[order]
        error_x = dict(
            type="data",
            array=(upper - diffs).tolist(),
            arrayminus=(diffs - lower).tolist(),
        )
    fig = go.Figure(
        go.Scatter(
            x=diffs.tolist(),
            y=labels,
            mode="markers",
            error_x=error_x,
            marker=dict(color="black", size=9),
            name="Difference",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    _apply_layout(
        fig,
        title=title or f"Model Comparison ({metric})",
        xaxis_title=f"{metric} difference",
        height=max(300, len(labels) * 35 + 100),
    )
    return fig


def plot_fit_diagnostics(
    result: Any,
    by: Literal["Model", "Fold"] = "Model",
    show_warnings: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot fit-time diagnostics by model or fold interactively.

    Parameters
    ----------
    result
        Experiment result with ``get_fit_diagnostics()`` or a diagnostics
        DataFrame containing ``TotalTime`` plus the selected ``by`` column.
    by
        Column used for grouping, usually ``"Model"`` or ``"Fold"``.
    show_warnings
        If True, annotate the figure with the warning count when present.
    title
        Optional figure title. Defaults to ``"Fit Diagnostics"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive fit diagnostics bar chart.

    See Also
    --------
    coco_pipe.viz.decoding.plot_fit_diagnostics : Static Matplotlib version.
    plot_training_history : Epoch-level training metrics for neural models.
    plot_search_results : Hyperparameter-search results sorted by rank.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame({"Model": ["SVM",
    ...                           "LDA"],
    ...                           "Fold": [0, 0],
    ...                           "TotalTime": [1.2, 0.4]})
    >>> fig = viz.plot_fit_diagnostics(df)
    """
    frame, data = prepare_fit_diagnostics_frame(result, by=by)
    summary = data.groupby(by)["TotalTime"].mean().sort_values()
    fig = go.Figure(
        go.Bar(
            x=summary.index.astype(str).tolist(),
            y=summary.values.tolist(),
            name="Mean Total Time",
        )
    )
    _apply_layout(
        fig,
        title=title or "Fit Diagnostics",
        xaxis_title=by,
        yaxis_title="Seconds",
        height=420,
    )
    if show_warnings and "WarningMessage" in frame.columns:
        n_warn = int(frame["WarningMessage"].notna().sum())
        if n_warn:
            fig.add_annotation(
                text=f"Warnings: {n_warn}",
                xref="paper",
                yref="paper",
                x=0.99,
                y=0.97,
                showarrow=False,
                xanchor="right",
            )
    return fig


def plot_probability_diagnostics(
    result: Any,
    model: Optional[str] = None,
    metric: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot probability-quality diagnostics interactively.

    Parameters
    ----------
    result
        Experiment result with ``get_probability_diagnostics()`` or a
        DataFrame containing ``Model``, ``Metric``, and ``Value``.
    model
        Optional model name used to filter diagnostics.
    metric
        Optional diagnostic metric used to filter rows.
    title
        Optional figure title. Defaults to ``"Probability Diagnostics"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive probability diagnostics bar chart.

    See Also
    --------
    coco_pipe.viz.decoding.plot_probability_diagnostics : Static Matplotlib version.
    plot_confusion_matrix : Confusion matrix for a complementary classifier view.
    plot_calibration_curve : Reliability curve for probability estimates.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["LR", "LR"],
    ...         "Metric": ["brier_score", "log_loss"],
    ...         "Value": [0.12, 0.35],
    ...     }
    ... )
    >>> fig = viz.plot_probability_diagnostics(df)
    """
    summary = prepare_probability_diagnostics_summary(
        result,
        model=model,
        metric=metric,
    )
    labels = summary.apply(lambda row: f"{row['Model']} / {row['Metric']}", axis=1)
    fig = go.Figure(
        go.Bar(
            x=labels.tolist(),
            y=summary["Value"].astype(float).tolist(),
            name="Mean diagnostic value",
        )
    )
    _apply_layout(
        fig,
        title=title or "Probability Diagnostics",
        yaxis_title="Mean diagnostic value",
        height=420,
    )
    return fig


def plot_subject_diagnostics(
    result: Any,
    unit: str = "Subject",
    metric: str = "accuracy",
    model: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot per-unit prediction accuracy diagnostics interactively.

    Parameters
    ----------
    result
        Experiment result with ``get_predictions()`` or a prediction DataFrame
        containing ``Model``, ``y_true``, ``y_pred``, and the selected ``unit``
        column.
    unit
        Metadata column used as the unit of aggregation.
    metric
        Metric to compute. Currently only ``"accuracy"`` is supported.
    model
        Optional model name used to filter predictions.
    title
        Optional figure title. Defaults to ``"<unit> Diagnostics"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive subject diagnostics horizontal bar chart.

    See Also
    --------
    coco_pipe.viz.decoding.plot_subject_diagnostics : Static Matplotlib version.
    plot_group_summary : Group-level accuracy summary with fold-level box plots.
    plot_regression_diagnostics : Observed vs predicted scatter for regression models.
    plot_fit_diagnostics : Per-model or per-fold timing diagnostics.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM"] * 4,
    ...         "Subject": ["S1", "S1", "S2", "S2"],
    ...         "y_true": ["A", "B", "A", "B"],
    ...         "y_pred": ["A", "B", "A", "A"],
    ...     }
    ... )
    >>> fig = viz.plot_subject_diagnostics(df)
    """
    if metric != "accuracy":
        raise ValueError(
            "plot_subject_diagnostics currently supports metric='accuracy'."
        )
    scores = prepare_prediction_accuracy_scores(
        result,
        [unit],
        model=model,
        context="subject diagnostics",
    )
    scores = scores.sort_values("Value")
    fig = go.Figure(
        go.Bar(
            x=scores["Value"].tolist(),
            y=scores[unit].astype(str).tolist(),
            orientation="h",
            name=metric,
        )
    )
    _apply_layout(
        fig,
        title=title or f"{unit} Diagnostics",
        xaxis_title=metric,
        height=max(300, scores[unit].nunique() * 22 + 100),
    )
    return fig


def plot_group_summary(
    result: Any,
    group: str = "Group",
    metric: str = "accuracy",
    model: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot group-level prediction accuracy summaries interactively.

    Parameters
    ----------
    result
        Experiment result with ``get_predictions()`` or a prediction DataFrame
        containing ``Model``, ``Fold``, ``y_true``, ``y_pred``, and the
        selected grouping column.
    group
        Prediction metadata column used to define groups.
    metric
        Metric to compute. Currently only ``"accuracy"`` is supported.
    model
        Optional model name used to filter predictions.
    title
        Optional figure title. Defaults to ``"Group Summary"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive group summary box plot.

    See Also
    --------
    coco_pipe.viz.decoding.plot_group_summary : Static Matplotlib version.
    plot_subject_diagnostics : Per-subject accuracy horizontal bar chart.
    plot_regression_diagnostics : Observed vs predicted for regression outputs.
    plot_fold_score_dispersion : Fold-level score variability per model.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {
    ...         "Model": ["SVM"] * 4,
    ...         "Group": ["G1", "G1", "G2", "G2"],
    ...         "Fold": [0, 1, 0, 1],
    ...         "y_true": ["A", "B", "A", "B"],
    ...         "y_pred": ["A", "A", "A", "B"],
    ...     }
    ... )
    >>> fig = viz.plot_group_summary(df)
    """
    if metric != "accuracy":
        raise ValueError("plot_group_summary currently supports metric='accuracy'.")
    scores = prepare_prediction_accuracy_scores(
        result,
        [group, "Fold"],
        model=model,
        context="group summary",
    )
    fig = go.Figure()
    for keys, grp in scores.groupby(["Model", group], dropna=False):
        label = f"{keys[0]} / {keys[1]}"
        fig.add_trace(
            go.Box(
                y=grp["Value"].tolist(),
                name=label,
                boxpoints="all",
                jitter=0.3,
                pointpos=0,
            )
        )
    _apply_layout(fig, title=title or "Group Summary", yaxis_title=metric, height=450)
    return fig


def plot_regression_diagnostics(
    result: Any,
    model: Optional[str] = None,
    fold: Optional[int] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot regression prediction diagnostics interactively.

    Parameters
    ----------
    result
        Experiment result with ``get_predictions()`` or a prediction DataFrame
        containing numeric ``y_true`` and ``y_pred`` columns.
    model
        Optional model name used to filter predictions.
    fold
        Optional fold index used to filter predictions.
    title
        Optional figure title. Defaults to ``"Regression Diagnostics"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive observed vs predicted scatter figure.

    See Also
    --------
    coco_pipe.viz.decoding.plot_regression_diagnostics : Static Matplotlib version.
    plot_subject_diagnostics : Per-subject accuracy for classification models.
    plot_group_summary : Group-level accuracy with fold-level variability.
    plot_fit_diagnostics : Fit-time and warning diagnostics.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame({"y_true": [1.0, 2.0, 3.0, 4.0],
    ...                           "y_pred": [1.1, 1.9, 3.2, 3.8]})
    >>> fig = viz.plot_regression_diagnostics(df)
    """
    y_true, y_pred = prepare_regression_prediction_data(
        result,
        model=model,
        fold=fold,
    )
    lims = [
        float(min(y_true.min(), y_pred.min())),
        float(max(y_true.max(), y_pred.max())),
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true.tolist(),
            y=y_pred.tolist(),
            mode="markers",
            opacity=0.7,
            marker=dict(size=5),
            name="Predictions",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lims,
            y=lims,
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Ideal",
        )
    )
    _apply_layout(
        fig,
        title=title or "Regression Diagnostics",
        xaxis_title="Observed",
        yaxis_title="Predicted",
        height=450,
    )
    return fig


def plot_search_results(
    result: Any,
    model: Optional[str] = None,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot compact hyperparameter-search results interactively.

    Parameters
    ----------
    result
        Experiment result with ``get_search_results()`` or a search-results
        DataFrame containing ``Model``, ``Rank``, and ``MeanTestScore``.
    model
        Optional model name used to filter search rows.
    top_n
        Optional positive number of top-ranked candidates to keep per model.
    title
        Optional figure title. Defaults to ``"Search Results"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive search results bar chart.

    See Also
    --------
    coco_pipe.viz.decoding.plot_search_results : Static Matplotlib version.
    plot_fit_diagnostics : Fit-time diagnostics complementing search analysis.
    plot_decoding_scores : Final evaluation scores after hyperparameter selection.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {"Model": ["SVM"] * 3,
    ...                           "Rank": [1, 2, 3],
    ...                           "MeanTestScore": [0.88, 0.85, 0.82]}
    ... )
    >>> fig = viz.plot_search_results(df)
    """
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
        return " / ".join(parts)

    labels = frame.apply(_search_label, axis=1).tolist()
    values = frame["MeanTestScore"].astype(float).tolist()
    fig = go.Figure(go.Bar(x=labels, y=values, name="Mean Test Score"))
    _apply_layout(
        fig,
        title=title or "Search Results",
        yaxis_title="Mean Test Score",
        height=420,
    )
    return fig


def plot_feature_stability(
    result: Any,
    model: Optional[str] = None,
    top_n: Optional[int] = 25,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot feature-selection stability interactively.

    Parameters
    ----------
    result
        Experiment result with ``get_feature_stability()`` or a DataFrame
        containing ``FeatureName`` and ``SelectionFrequency``.
    model
        Optional model name used to filter feature rows.
    top_n
        Optional number of most stable features to display.
    title
        Optional figure title. Defaults to ``"Feature Stability"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive feature stability horizontal bar chart.

    See Also
    --------
    coco_pipe.viz.decoding.plot_feature_stability : Static Matplotlib version.
    plot_feature_scores : Univariate feature-selector scores.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame(
    ...     {"FeatureName": ["F1", "F2", "F3"], "SelectionFrequency": [0.9, 0.6, 0.3]}
    ... )
    >>> fig = viz.plot_feature_stability(df)
    """
    series = prepare_feature_stability_series(result, model=model, top_n=top_n)
    fig = go.Figure(
        go.Bar(
            x=series.values[::-1].tolist(),
            y=series.index.astype(str).tolist()[::-1],
            orientation="h",
            name="Selection Frequency",
            marker_color=_COLORBLIND_COLORS[0],
        )
    )
    _apply_layout(
        fig,
        title=title or "Feature Stability",
        xaxis_title="Selection Frequency",
        height=max(400, len(series) * 22 + 100),
    )
    return fig


def plot_feature_scores(
    result: Any,
    model: Optional[str] = None,
    top_n: Optional[int] = 25,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot univariate feature-selector scores interactively.

    Parameters
    ----------
    result
        Experiment result with ``get_feature_scores()`` or a feature-score
        DataFrame containing ``FeatureName`` and ``Score``.
    model
        Optional model name used to filter feature scores.
    top_n
        Optional positive number of highest-scoring features to display.
    title
        Optional figure title. Defaults to ``"Feature Scores"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive feature scores horizontal bar chart.

    See Also
    --------
    coco_pipe.viz.decoding.plot_feature_scores : Static Matplotlib version.
    plot_feature_stability : Feature-selection stability across folds.

    Examples
    --------
    >>> import pandas as pd
    >>> from coco_pipe.viz.interactive import decoding as viz
    >>> df = pd.DataFrame({"FeatureName": ["F1",
    ...                           "F2",
    ...                           "F3"],
    ...                           "Score": [0.72, 0.55, 0.31]})
    >>> fig = viz.plot_feature_scores(df)
    """
    scores = prepare_feature_score_series(result, model=model, top_n=top_n)
    fig = go.Figure(
        go.Bar(
            x=scores.values[::-1].tolist(),
            y=scores.index.astype(str).tolist()[::-1],
            orientation="h",
            name="Score",
            marker_color=_COLORBLIND_COLORS[1],
        )
    )
    _apply_layout(
        fig,
        title=title or "Feature Scores",
        xaxis_title="Score",
        height=max(400, len(scores) * 22 + 100),
    )
    return fig
