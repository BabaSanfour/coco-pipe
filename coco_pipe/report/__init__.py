"""
coco_pipe.report
================

Reporting module for generating single-file HTML quality control reports.
"""

from .api import (
    from_bids,
    from_container,
    from_embeddings,
    from_reductions,
    from_tabular,
)
from .core import Report

__all__ = [
    "Report",
    "from_container",
    "from_bids",
    "from_tabular",
    "from_embeddings",
    "from_reductions",
]
