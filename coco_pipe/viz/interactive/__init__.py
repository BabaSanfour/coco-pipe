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
    plot_grouped_bar,
    plot_heatmap,
    plot_ranked_bar,
    plot_scatter,
    plot_timecourses,
)
from .decoding import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_decoding_scores,
    plot_feature_importance,
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
    plot_component_loadings,
    plot_coranking_matrix,
    plot_embedding,
    plot_feature_correlation_heatmap,
    plot_loss_history,
    plot_metrics,
    plot_phase_portrait,
    plot_radar_comparison,
    plot_raw_preview,
    plot_scree,
    plot_shepard_diagram,
    plot_streamlines,
    plot_trajectory,
    plot_trajectory_metric_series,
    plot_trajectory_separation,
)
from .dim_reduction import (
    plot_feature_importance as plot_reduction_feature_importance,
)

__all__ = [
    "base",
    "decoding",
    "dim_reduction",
    # base (general)
    "plot_bar",
    # decoding
    "plot_calibration_curve",
    "plot_component_loadings",
    "plot_confusion_matrix",
    # dim_reduction
    "plot_coranking_matrix",
    "plot_decoding_scores",
    "plot_distribution_groups",
    "plot_embedding",
    "plot_feature_correlation_heatmap",
    "plot_feature_importance",
    "plot_feature_scores",
    "plot_feature_stability",
    "plot_fit_diagnostics",
    "plot_fold_score_dispersion",
    "plot_group_summary",
    "plot_grouped_bar",
    "plot_heatmap",
    "plot_loss_history",
    "plot_metrics",
    "plot_model_comparison",
    "plot_null_interval_summary",
    "plot_phase_portrait",
    "plot_pr_curve",
    "plot_probability_diagnostics",
    "plot_radar_comparison",
    "plot_ranked_bar",
    "plot_raw_preview",
    "plot_reduction_feature_importance",
    "plot_regression_diagnostics",
    "plot_roc_curve",
    "plot_scatter",
    "plot_scree",
    "plot_search_results",
    "plot_shepard_diagram",
    "plot_streamlines",
    "plot_subject_diagnostics",
    "plot_temporal_generalization_matrix",
    "plot_temporal_score_curve",
    "plot_temporal_statistical_assessment",
    "plot_timecourses",
    "plot_training_history",
    "plot_trajectory",
    "plot_trajectory_metric_series",
    "plot_trajectory_separation",
]
