#!/usr/bin/env python3
"""Curated plotting helpers for coco_pipe."""

from .decoding import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_fold_score_dispersion,
    plot_pr_curve,
    plot_roc_curve,
    plot_statistical_null_distribution,
    plot_temporal_generalization_matrix,
    plot_temporal_score_curve,
    plot_temporal_statistical_assessment,
    plot_training_history,
)
from .dim_reduction import (
    plot_eigenvalues,
    plot_embedding,
    plot_feature_correlation_heatmap,
    plot_feature_importance,
    plot_interpretation,
    plot_local_metrics,
    plot_loss_history,
    plot_metrics,
    plot_shepard_diagram,
    plot_streamlines,
    plot_trajectory,
    plot_trajectory_metric_series,
)
from .plotly_utils import plot_channel_traces_interactive
from .plots import plot_bar, plot_scatter2d, plot_topomap

__all__ = [
    "plot_topomap",
    "plot_bar",
    "plot_scatter2d",
    "plot_embedding",
    "plot_metrics",
    "plot_loss_history",
    "plot_eigenvalues",
    "plot_shepard_diagram",
    "plot_streamlines",
    "plot_feature_importance",
    "plot_feature_correlation_heatmap",
    "plot_interpretation",
    "plot_trajectory",
    "plot_trajectory_metric_series",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_calibration_curve",
    "plot_fold_score_dispersion",
    "plot_statistical_null_distribution",
    "plot_training_history",
    "plot_temporal_score_curve",
    "plot_temporal_generalization_matrix",
    "plot_temporal_statistical_assessment",
    "plot_local_metrics",
    "plot_channel_traces_interactive",
]
