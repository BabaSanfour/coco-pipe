"""
coco_pipe.report
================

Reporting module for generating single-file HTML quality control reports.
"""

from importlib import import_module

# Eagerly load decoding/dim_reduction so they attach their `add_*` methods to
# Report. Lazy loading is fine for everything else, but the bound methods are
# part of Report's public surface and must exist before any user code calls
# `report.add_decoding_overview(...)`.
from . import decoding, dim_reduction  # noqa: F401

_LAZY_EXPORTS = {
    "Report": ("coco_pipe.report.core", "Report"),
    "Section": ("coco_pipe.report.core", "Section"),
    "PlotlyElement": ("coco_pipe.report.core", "PlotlyElement"),
    "TableElement": ("coco_pipe.report.core", "TableElement"),
    "InteractiveTableElement": ("coco_pipe.report.elements", "InteractiveTableElement"),
    "ImageElement": ("coco_pipe.report.core", "ImageElement"),
    "AccordionElement": ("coco_pipe.report.elements", "AccordionElement"),
    "CalloutElement": ("coco_pipe.report.elements", "CalloutElement"),
    "ColumnsElement": ("coco_pipe.report.elements", "ColumnsElement"),
    "ContainerElement": ("coco_pipe.report.elements", "ContainerElement"),
    "StatCardElement": ("coco_pipe.report.elements", "StatCardElement"),
    "TabsElement": ("coco_pipe.report.elements", "TabsElement"),
    "from_container": ("coco_pipe.report.api", "from_container"),
    "from_bids": ("coco_pipe.report.api", "from_bids"),
    "from_tabular": ("coco_pipe.report.api", "from_tabular"),
    "from_embeddings": ("coco_pipe.report.api", "from_embeddings"),
    "from_reductions": ("coco_pipe.report.api", "from_reductions"),
    "from_experiment_result": ("coco_pipe.report.api", "from_experiment_result"),
    "from_experiment_results": ("coco_pipe.report.api", "from_experiment_results"),
    "from_decoding_sweep": ("coco_pipe.report.api", "from_decoding_sweep"),
    "from_foundation_sweep": ("coco_pipe.report.api", "from_foundation_sweep"),
    "from_head_to_head": ("coco_pipe.report.api", "from_head_to_head"),
    "merge_reports": ("coco_pipe.report.api", "merge_reports"),
    "make_decoding_report": ("coco_pipe.report.decoding_sweep", "make_decoding_report"),
    "make_decoding_result_report": (
        "coco_pipe.report.decoding",
        "make_decoding_result_report",
    ),
    "render_unit_reports": ("coco_pipe.report.decoding", "render_unit_reports"),
    "ResultCollection": (
        "coco_pipe.report.decoding_sweep",
        "ResultCollection",
    ),
    "collect_results": (
        "coco_pipe.report.decoding_sweep",
        "collect_results",
    ),
    "build_comparison_section": (
        "coco_pipe.report.decoding_sweep",
        "build_comparison_section",
    ),
    "build_result_tabs": (
        "coco_pipe.report.decoding_sweep",
        "build_result_tabs",
    ),
    "make_experiment_results_report": (
        "coco_pipe.report.decoding_sweep",
        "make_experiment_results_report",
    ),
    "add_scientific_overview": (
        "coco_pipe.report.decoding_sweep",
        "add_scientific_overview",
    ),
    "best_result_tabs": (
        "coco_pipe.report.decoding_sweep",
        "best_result_tabs",
    ),
    "collect_mode_results": (
        "coco_pipe.report.decoding_sweep",
        "collect_mode_results",
    ),
    "descriptor_section": (
        "coco_pipe.report.decoding_sweep",
        "descriptor_section",
    ),
    "failures_section": (
        "coco_pipe.report.decoding_sweep",
        "failures_section",
    ),
    "feature_selection_section": (
        "coco_pipe.report.decoding_sweep",
        "feature_selection_section",
    ),
    "flat_section": (
        "coco_pipe.report.decoding_sweep",
        "flat_section",
    ),
    "grouped_section": (
        "coco_pipe.report.decoding_sweep",
        "grouped_section",
    ),
    "hp_tuning_section": (
        "coco_pipe.report.decoding_sweep",
        "hp_tuning_section",
    ),
    "leaderboard_section": (
        "coco_pipe.report.decoding_sweep",
        "leaderboard_section",
    ),
    "paired_delta_vs_baseline": (
        "coco_pipe.report.decoding_sweep",
        "paired_delta_vs_baseline",
    ),
    "sensor_section": (
        "coco_pipe.report.decoding_sweep",
        "sensor_section",
    ),
    "summary_collection": (
        "coco_pipe.report.decoding_sweep",
        "summary_collection",
    ),
    "prepare_sweep_frame": (
        "coco_pipe.report.decoding_sweep",
        "prepare_sweep_frame",
    ),
    "make_decoding_sweep_report": (
        "coco_pipe.report.decoding_sweep",
        "make_decoding_sweep_report",
    ),
    "build_capability_matrix_section": (
        "coco_pipe.report.decoding_sweep",
        "build_capability_matrix_section",
    ),
    "build_subject_alignment_diagnostics_section": (
        "coco_pipe.report.variance",
        "build_subject_alignment_diagnostics_section",
    ),
    "build_alignment_coverage_section": (
        "coco_pipe.report.variance",
        "build_alignment_coverage_section",
    ),
    "build_alignment_tradeoff_section": (
        "coco_pipe.report.variance",
        "build_alignment_tradeoff_section",
    ),
    "select_subject_alignment_diagnostics": (
        "coco_pipe.report.variance",
        "select_subject_alignment_diagnostics",
    ),
    "validate_subject_alignment_diagnostics": (
        "coco_pipe.report.variance",
        "validate_subject_alignment_diagnostics",
    ),
    "build_foundation_comparison_sections": (
        "coco_pipe.report.decoding_sweep",
        "build_foundation_comparison_sections",
    ),
    "build_classical_taxonomy_sections": (
        "coco_pipe.report.decoding_sweep",
        "build_classical_taxonomy_sections",
    ),
    "build_classical_mode_elements": (
        "coco_pipe.report.decoding_sweep",
        "build_classical_mode_elements",
    ),
    "CLASSICAL_MODE_TITLES": (
        "coco_pipe.report.decoding_sweep",
        "CLASSICAL_MODE_TITLES",
    ),
    "DEFAULT_CLASSICAL_SECTION_BUILDERS": (
        "coco_pipe.report.decoding_sweep",
        "DEFAULT_CLASSICAL_SECTION_BUILDERS",
    ),
    "enrich_head_to_head_frame": (
        "coco_pipe.report.decoding_sweep",
        "enrich_head_to_head_frame",
    ),
    "normalize_head_to_head_frame": (
        "coco_pipe.report.decoding_sweep",
        "normalize_head_to_head_frame",
    ),
    "collect_comparison_runs": (
        "coco_pipe.report.decoding_sweep",
        "collect_comparison_runs",
    ),
    "HEAD_TO_HEAD_DISPLAY_COLUMNS": (
        "coco_pipe.report.decoding_sweep",
        "HEAD_TO_HEAD_DISPLAY_COLUMNS",
    ),
    "make_head_to_head_report": (
        "coco_pipe.report.decoding_sweep",
        "make_head_to_head_report",
    ),
    "make_foundation_embedding_report": (
        "coco_pipe.report.foundation",
        "make_foundation_embedding_report",
    ),
    "make_foundation_decoding_report": (
        "coco_pipe.report.foundation",
        "make_foundation_decoding_report",
    ),
    "generate_descriptor_subject_report": (
        "coco_pipe.report.descriptor_qc",
        "generate_descriptor_subject_report",
    ),
    "generate_descriptor_dataset_report": (
        "coco_pipe.report.descriptor_qc",
        "generate_descriptor_dataset_report",
    ),
    "make_reduction_report": (
        "coco_pipe.report.dim_reduction",
        "make_reduction_report",
    ),
    "reduction_embedding_element": (
        "coco_pipe.report.dim_reduction",
        "reduction_embedding_element",
    ),
    "reduction_loadings_element": (
        "coco_pipe.report.dim_reduction",
        "reduction_loadings_element",
    ),
    "merge_fit_eval": ("coco_pipe.report.dim_reduction_sweep", "merge_fit_eval"),
    "rank_reduction_runs": (
        "coco_pipe.report.dim_reduction_sweep",
        "rank_reduction_runs",
    ),
    "build_reduction_rollup_report": (
        "coco_pipe.report.dim_reduction_sweep",
        "build_reduction_rollup_report",
    ),
    "add_reduction_best_run_cards": (
        "coco_pipe.report.dim_reduction_sweep",
        "add_reduction_best_run_cards",
    ),
    "build_reduction_eval_results_section": (
        "coco_pipe.report.dim_reduction_sweep",
        "build_reduction_eval_results_section",
    ),
    "build_reduction_condition_ranking_section": (
        "coco_pipe.report.dim_reduction_sweep",
        "build_reduction_condition_ranking_section",
    ),
    "DEFAULT_REDUCTION_TIE_BREAKERS": (
        "coco_pipe.report.dim_reduction_sweep",
        "DEFAULT_REDUCTION_TIE_BREAKERS",
    ),
    "DimReductionReportContext": (
        "coco_pipe.report.dim_reduction_sweep",
        "DimReductionReportContext",
    ),
    "build_dataset_report": (
        "coco_pipe.report.dim_reduction_sweep",
        "build_dataset_report",
    ),
    "build_meta_dict": ("coco_pipe.report.dim_reduction_sweep", "build_meta_dict"),
    "build_best_fit_plots": (
        "coco_pipe.report.dim_reduction_sweep",
        "build_best_fit_plots",
    ),
    "build_flat_condition_section": (
        "coco_pipe.report.dim_reduction_sweep",
        "build_flat_condition_section",
    ),
    "build_nonflat_condition_section": (
        "coco_pipe.report.dim_reduction_sweep",
        "build_nonflat_condition_section",
    ),
    "build_pooled_section": (
        "coco_pipe.report.dim_reduction_sweep",
        "build_pooled_section",
    ),
    "build_unit_summary": (
        "coco_pipe.report.dim_reduction_sweep",
        "build_unit_summary",
    ),
    "build_data_availability_summary": (
        "coco_pipe.report.dim_reduction_sweep",
        "build_data_availability_summary",
    ),
    "build_failure_sections": (
        "coco_pipe.report.dim_reduction_sweep",
        "build_failure_sections",
    ),
    "split_by_status": ("coco_pipe.report.tables", "split_by_status"),
    "sort_by_metrics": ("coco_pipe.report.tables", "sort_by_metrics"),
    "best_rows": ("coco_pipe.report.tables", "best_rows"),
    "display_frame": ("coco_pipe.report.tables", "display_frame"),
    "relabel_columns": ("coco_pipe.report.tables", "relabel_columns"),
    "selector_columns": ("coco_pipe.report.tables", "selector_columns"),
    "primary_metric_column": ("coco_pipe.report.tables", "primary_metric_column"),
    "ensure_primary_metric": ("coco_pipe.report.tables", "ensure_primary_metric"),
    "make_cv_signature": ("coco_pipe.report.tables", "make_cv_signature"),
    "signature_compatibility": ("coco_pipe.report.tables", "signature_compatibility"),
    "ASSET_USER_AGENT": ("coco_pipe.report._constants", "ASSET_USER_AGENT"),
    "CV_SIGNATURE_COLUMNS": ("coco_pipe.report._constants", "CV_SIGNATURE_COLUMNS"),
    "DECODING_PRESETS": ("coco_pipe.report._constants", "DECODING_PRESETS"),
    "DEFAULT_BEST_RESULT_GROUP_COLUMNS": (
        "coco_pipe.report._constants",
        "DEFAULT_BEST_RESULT_GROUP_COLUMNS",
    ),
    "DEFAULT_DECODING_RESULT_COLUMN_LABELS": (
        "coco_pipe.report._constants",
        "DEFAULT_DECODING_RESULT_COLUMN_LABELS",
    ),
    "DEFAULT_DECODING_RESULT_COLUMN_ORDER": (
        "coco_pipe.report._constants",
        "DEFAULT_DECODING_RESULT_COLUMN_ORDER",
    ),
    "DEFAULT_MODE_RESULT_CONTEXT_COLUMNS": (
        "coco_pipe.report._constants",
        "DEFAULT_MODE_RESULT_CONTEXT_COLUMNS",
    ),
    "DEFAULT_REDUCTION_SECTIONS": (
        "coco_pipe.report._constants",
        "DEFAULT_REDUCTION_SECTIONS",
    ),
    "DEFAULT_SECTIONS": ("coco_pipe.report._constants", "DEFAULT_SECTIONS"),
    "FAMILY_DIAGNOSTIC_COLUMNS": (
        "coco_pipe.report._constants",
        "FAMILY_DIAGNOSTIC_COLUMNS",
    ),
    "INTERACTIVE_AWARE_SECTIONS": (
        "coco_pipe.report._constants",
        "INTERACTIVE_AWARE_SECTIONS",
    ),
    "INLINE_SENTINEL": ("coco_pipe.report._constants", "INLINE_SENTINEL"),
    "MAX_INLINE_DIAGNOSTIC_RESULTS": (
        "coco_pipe.report._constants",
        "MAX_INLINE_DIAGNOSTIC_RESULTS",
    ),
    "MODULE_DIR": ("coco_pipe.report._constants", "MODULE_DIR"),
    "PRIMARY_TIE_BREAKERS": ("coco_pipe.report._constants", "PRIMARY_TIE_BREAKERS"),
    "SECTION_ALIASES": ("coco_pipe.report._constants", "SECTION_ALIASES"),
    "SUBJECT_ALIGNMENT_REQUIRED_COLUMNS": (
        "coco_pipe.report._constants",
        "SUBJECT_ALIGNMENT_REQUIRED_COLUMNS",
    ),
    "TEMPLATE_DIR": ("coco_pipe.report._constants", "TEMPLATE_DIR"),
    "VALID_REDUCTION_SECTIONS": (
        "coco_pipe.report._constants",
        "VALID_REDUCTION_SECTIONS",
    ),
    "VENDORED_URLS": ("coco_pipe.report._constants", "VENDORED_URLS"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_LAZY_EXPORTS)
