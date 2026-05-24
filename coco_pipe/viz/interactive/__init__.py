"""Interactive (Plotly) visualization helpers for coco_pipe.

Organized into sub-modules:

- :mod:`coco_pipe.viz.interactive.base` — general-purpose plots (bar,
  distribution groups, heatmap) mirroring :mod:`coco_pipe.viz.base`.
- :mod:`coco_pipe.viz.interactive.dim_reduction` — embedding, metrics, trajectory,
  feature-importance, and co-ranking plots.
- :mod:`coco_pipe.viz.interactive.decoding` — confusion matrix, ROC/PR/calibration
  curves, temporal analysis, and feature diagnostics plots.
"""

from . import base, decoding, dim_reduction
from .base import (
    plot_bar,
    plot_distribution_groups,
    plot_heatmap,
)
from .decoding import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_decoding_scores,
    plot_feature_scores,
    plot_feature_stability,
    plot_fit_diagnostics,
    plot_fold_score_dispersion,
    plot_group_summary,
    plot_model_comparison,
    plot_null_interval_summary,
    plot_pr_curve,
    plot_probability_diagnostics,
    plot_regression_diagnostics,
    plot_roc_curve,
    plot_search_results,
    plot_subject_diagnostics,
    plot_temporal_generalization_matrix,
    plot_temporal_score_curve,
    plot_temporal_statistical_assessment,
    plot_training_history,
)
from .dim_reduction import (
    plot_channel_traces,
    plot_component_loadings,
    plot_coranking_matrix,
    plot_eigenvalues,
    plot_embedding,
    plot_feature_correlation_heatmap,
    plot_feature_importance,
    plot_loss_history,
    plot_metrics,
    plot_phase_portrait,
    plot_radar_comparison,
    plot_raw_preview,
    plot_shepard_diagram,
    plot_streamlines,
    plot_trajectory,
    plot_trajectory_metric_series,
    plot_trajectory_separation,
)

__all__ = [
    "base",
    "decoding",
    "dim_reduction",
    # base (general)
    "plot_bar",
    "plot_distribution_groups",
    "plot_heatmap",
    # dim_reduction
    "plot_channel_traces",
    "plot_coranking_matrix",
    "plot_component_loadings",
    "plot_eigenvalues",
    "plot_embedding",
    "plot_feature_correlation_heatmap",
    "plot_feature_importance",
    "plot_loss_history",
    "plot_metrics",
    "plot_phase_portrait",
    "plot_radar_comparison",
    "plot_raw_preview",
    "plot_shepard_diagram",
    "plot_streamlines",
    "plot_trajectory",
    "plot_trajectory_metric_series",
    "plot_trajectory_separation",
    # decoding
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
