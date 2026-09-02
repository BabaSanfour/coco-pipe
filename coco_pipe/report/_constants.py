"""Report-layer constants shared across coco-pipe reporting modules."""

from __future__ import annotations

from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
"""Filesystem path to the ``coco_pipe.report`` package."""

TEMPLATE_DIR = MODULE_DIR / "templates"
"""Filesystem path to bundled report Jinja templates."""

VENDORED_URLS: dict[str, str] = {
    "plotly": "https://cdn.plot.ly/plotly-2.27.0.min.js",
    "tailwind": "https://cdn.tailwindcss.com",
    "pako": "https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js",
}
"""Default CDN URLs for report assets that can be vendored inline."""

INLINE_SENTINEL = "inline"
"""Sentinel value requesting inline vendored report assets."""

ASSET_USER_AGENT = "coco-pipe/asset-vendor (+https://github.com/BabaSanfour/coco-pipe)"
"""User-Agent sent when downloading vendored report assets."""

DEFAULT_DECODING_RESULT_COLUMN_LABELS = {
    "scope": "Scope",
    "condition": "Condition",
    "target": "Target",
    "analysis_mode": "Analysis Mode",
    "unit_name": "Analysis Unit",
    "family": "Family",
    "subfamily": "Subfamily",
    "model": "Model",
    "model_key": "Model Key",
    "train_mode": "Training Mode",
    "selection_mode": "Feature Selection",
    "status": "Status",
    "reason": "Reason",
    "primary_metric_name": "Primary Metric",
    "primary_metric": "Primary Score",
    "n_samples": "N Observations",
    "n_groups": "N Subjects",
    "accuracy_mean": "Accuracy",
    "accuracy_std": "Accuracy SD",
    "balanced_accuracy_mean": "Balanced Accuracy",
    "balanced_accuracy_std": "Balanced Accuracy SD",
    "f1_mean": "F1",
    "f1_std": "F1 SD",
    "precision_mean": "Precision",
    "precision_std": "Precision SD",
    "recall_mean": "Recall",
    "recall_std": "Recall SD",
    "roc_auc_mean": "ROC AUC",
    "roc_auc_std": "ROC AUC SD",
    "p_value": "P Value",
    "p_value_fdr": "FDR P Value",
    "significant_fdr": "FDR Significant",
    "cv_signature": "CV Design",
    "cohort_signature": "Cohort Design",
}
"""Default sweep-result column labels for decoding report tables."""

DEFAULT_DECODING_RESULT_COLUMN_ORDER = (
    "target",
    "unit_name",
    "family",
    "subfamily",
    "model",
    "selection_mode",
    "status",
    "reason",
    "primary_metric_name",
    "primary_metric",
    "n_samples",
    "n_groups",
    "accuracy_mean",
    "accuracy_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "f1_mean",
    "f1_std",
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
    "roc_auc_mean",
    "roc_auc_std",
    "p_value",
    "p_value_fdr",
    "significant_fdr",
)
"""Default score-led decoding display order.

Deliberately excludes scope/condition/analysis-mode/model-key/train-mode and
CV/cohort-signature columns so ``display_frame`` default tables lead with scores
and hide audit noise. Those columns still relabel via
:data:`DEFAULT_DECODING_RESULT_COLUMN_LABELS` when a caller passes them in an
explicit ``order``."""

PRIMARY_TIE_BREAKERS = (("p_value_fdr", True), ("p_value", True))
"""Default tie-breakers for primary-metric decoding leaderboards."""

CV_SIGNATURE_COLUMNS = (
    "cohort_signature",
    "cv_signature",
    "cv_strategy",
    "effective_n_splits",
    "cv_random_state",
)
"""Scientific-design fields that must match before two rows are paired."""

SUBJECT_ALIGNMENT_METRIC_LABELS = {
    "total_sample_variance": "Total sample variance",
    "variance_participation_ratio": "Variance participation ratio (effective rank)",
    "variance_participation_ratio_fraction": "Participation-ratio rank fraction",
    "subject_probe_excess_above_chance": "Subject probe excess above chance",
    "between_subject_excess_over_null": "Subject variance excess over null",
    "marginal_label_excess_over_null": "Label variance excess over null",
    "permutation_p_between_subject_eta2": "Subject variance permutation p",
    "permutation_p_marginal_label_eta2": "Label variance permutation p",
    "partial_omega2_subject_within_label": "Subject-within-label partial omega²",
    "partial_omega2_subject": "Unique subject partial omega²",
    "partial_omega2_label": "Unique label partial omega²",
}
"""Display labels for quantitative subject-alignment diagnostic metrics."""

SUBJECT_ALIGNMENT_IDENTITY_METRICS = frozenset(
    {
        "subject_probe_excess_above_chance",
        "between_subject_excess_over_null",
    }
)
"""Metrics measuring residual subject-identifiable structure."""

SUBJECT_ALIGNMENT_LABEL_METRICS = frozenset({"marginal_label_excess_over_null"})
"""Metrics measuring retained label-relevant structure."""

SUBJECT_ALIGNMENT_AUDIT_METRICS = (
    SUBJECT_ALIGNMENT_IDENTITY_METRICS
    | SUBJECT_ALIGNMENT_LABEL_METRICS
    | frozenset(
        {
            "total_sample_variance",
            "variance_participation_ratio",
            "variance_participation_ratio_fraction",
            "permutation_p_between_subject_eta2",
            "permutation_p_marginal_label_eta2",
            "partial_omega2_subject_within_label",
            "partial_omega2_subject",
            "partial_omega2_label",
        }
    )
)
"""Metrics included in the quantitative alignment audit table."""

SUBJECT_ALIGNMENT_PAIRING_COLUMNS = (
    "cohort_name",
    "population",
    "selection_fingerprint",
    "diagnostic_pair",
    "scope",
    "eval_name",
    "target_col",
    "design",
)
"""Scientific-design fields that must match before computing raw deltas."""

SUBJECT_ALIGNMENT_REQUIRED_COLUMNS = frozenset(
    {
        "transform",
        "cohort_name",
        "population",
        "selection_fingerprint",
        "scope",
        "eval_name",
        "target_col",
        "metric",
        "value",
    }
)
"""Required schema for population-specific subject-alignment report input."""

SUBJECT_ALIGNMENT_ASSESSMENT_COLUMNS = (
    "population",
    "scope",
    "eval_name",
    "target_col",
    "diagnostic_pair",
    "design",
)
"""Context fields identifying a displayed alignment assessment."""

MAX_INLINE_DIAGNOSTIC_RESULTS = 5
"""Maximum top-ranked artifacts to load inline for diagnostic tabs."""

DEFAULT_MODE_RESULT_CONTEXT_COLUMNS = (
    "scope",
    "target",
    "analysis_mode",
    "unit_key",
    "unit_name",
    "family",
    "subfamily",
    "selection_mode",
    "transform",
    "reduction_mode",
)
"""Default context columns used to collect decoding sweep artifacts."""

DEFAULT_BEST_RESULT_GROUP_COLUMNS = (
    "scope",
    "target",
    "analysis_mode",
    "selection_mode",
    "transform",
    "reduction_mode",
)
"""Default grouping for selecting top decoding-sweep diagnostic artifacts."""

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
"""Named presets for per-result decoding reports."""

DEFAULT_SECTIONS = DECODING_PRESETS["default"]
"""Default per-result decoding report section sequence."""

SECTION_ALIASES = {
    "cv_summary": "cv",
    "confusion_probability": "probability",
}
"""Backward-compatible decoding report section aliases."""

INTERACTIVE_AWARE_SECTIONS = {
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
"""Decoding report sections that accept the ``interactive`` option."""

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
"""Default dimensionality-reduction report section sequence."""

VALID_REDUCTION_SECTIONS = set(DEFAULT_REDUCTION_SECTIONS)
"""Valid dimensionality-reduction report section keys."""

FAMILY_DIAGNOSTIC_COLUMNS: dict[str, tuple[str, str, list[str]]] = {
    "band": (
        "Band Power Sanity",
        "\U0001f39a\ufe0f",
        [
            "band_abs_negative_rate",
            "band_rel_out_of_range_rate",
            "band_corr_rel_out_of_range_rate",
            "band_ratio_nan_rate",
        ],
    ),
    "param": (
        "FOOOF Fit Quality",
        "\U0001f4d0",
        [
            "param_r_squared_median",
            "param_r_squared_p05",
            "param_fit_error_median",
            "param_fit_error_p95",
            "param_peak_count_missing_rate",
            "param_alpha_peak_freq_missing_rate",
        ],
    ),
    "complexity": (
        "Complexity Measure Sanity",
        "\U0001f300",
        [
            "complexity_measure_missingness_max",
            "complexity_measure_missingness_median",
            "complexity_nonfinite_rate",
        ],
    ),
}
"""Family-specific descriptor-QC diagnostic section definitions."""

__all__ = [
    "ASSET_USER_AGENT",
    "CV_SIGNATURE_COLUMNS",
    "DECODING_PRESETS",
    "DEFAULT_BEST_RESULT_GROUP_COLUMNS",
    "DEFAULT_DECODING_RESULT_COLUMN_LABELS",
    "DEFAULT_DECODING_RESULT_COLUMN_ORDER",
    "DEFAULT_MODE_RESULT_CONTEXT_COLUMNS",
    "DEFAULT_REDUCTION_SECTIONS",
    "DEFAULT_SECTIONS",
    "FAMILY_DIAGNOSTIC_COLUMNS",
    "INLINE_SENTINEL",
    "INTERACTIVE_AWARE_SECTIONS",
    "MAX_INLINE_DIAGNOSTIC_RESULTS",
    "MODULE_DIR",
    "PRIMARY_TIE_BREAKERS",
    "SECTION_ALIASES",
    "SUBJECT_ALIGNMENT_ASSESSMENT_COLUMNS",
    "SUBJECT_ALIGNMENT_AUDIT_METRICS",
    "SUBJECT_ALIGNMENT_IDENTITY_METRICS",
    "SUBJECT_ALIGNMENT_LABEL_METRICS",
    "SUBJECT_ALIGNMENT_METRIC_LABELS",
    "SUBJECT_ALIGNMENT_PAIRING_COLUMNS",
    "SUBJECT_ALIGNMENT_REQUIRED_COLUMNS",
    "TEMPLATE_DIR",
    "VALID_REDUCTION_SECTIONS",
    "VENDORED_URLS",
]
