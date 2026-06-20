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
from . import decoding as _decoding  # noqa: F401
from . import dim_reduction as _dim_reduction  # noqa: F401

_LAZY_EXPORTS = {
    "Report": ("coco_pipe.report.core", "Report"),
    "Section": ("coco_pipe.report.core", "Section"),
    "PlotlyElement": ("coco_pipe.report.core", "PlotlyElement"),
    "TableElement": ("coco_pipe.report.core", "TableElement"),
    "InteractiveTableElement": ("coco_pipe.report.elements", "InteractiveTableElement"),
    "ImageElement": ("coco_pipe.report.core", "ImageElement"),
    "from_container": ("coco_pipe.report.api", "from_container"),
    "from_bids": ("coco_pipe.report.api", "from_bids"),
    "from_tabular": ("coco_pipe.report.api", "from_tabular"),
    "from_embeddings": ("coco_pipe.report.api", "from_embeddings"),
    "from_reductions": ("coco_pipe.report.api", "from_reductions"),
    "from_experiment_result": ("coco_pipe.report.api", "from_experiment_result"),
    "from_experiment_results": ("coco_pipe.report.api", "from_experiment_results"),
    "merge_reports": ("coco_pipe.report.api", "merge_reports"),
    "make_decoding_report": ("coco_pipe.report.decoding", "make_decoding_report"),
    "ResultCollection": (
        "coco_pipe.report.decoding_comparison",
        "ResultCollection",
    ),
    "collect_results": (
        "coco_pipe.report.decoding_comparison",
        "collect_results",
    ),
    "build_comparison_section": (
        "coco_pipe.report.decoding_comparison",
        "build_comparison_section",
    ),
    "build_result_tabs": (
        "coco_pipe.report.decoding_comparison",
        "build_result_tabs",
    ),
    "make_experiment_results_report": (
        "coco_pipe.report.decoding_comparison",
        "make_experiment_results_report",
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
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_LAZY_EXPORTS)
