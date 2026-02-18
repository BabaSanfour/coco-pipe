"""
coco_pipe.report
================

Reporting module for generating single-file HTML quality control reports.
"""


def __getattr__(name):
    if name == "Report":
        from .core import Report

        return Report
    if name in [
        "from_container",
        "from_bids",
        "from_tabular",
        "from_embeddings",
        "from_reductions",
    ]:
        from .api import (
            from_bids,  # noqa: F401
            from_container,  # noqa: F401
            from_embeddings,  # noqa: F401
            from_reductions,  # noqa: F401
            from_tabular,  # noqa: F401
        )

        return locals()[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "Report",
    "from_container",
    "from_bids",
    "from_tabular",
    "from_embeddings",
    "from_reductions",
]
