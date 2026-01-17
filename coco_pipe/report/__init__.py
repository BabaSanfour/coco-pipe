"""
coco_pipe.report
================

Reporting module for generating single-file HTML quality control reports.
"""

from .core import Report, Section
from .api import from_container, from_bids, from_tabular, from_embeddings, from_reductions
from .engine import render_template
