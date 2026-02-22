#!/usr/bin/env python3
"""
coco_pipe.viz
-------------
Lightweight, backend-agnostic plotting helpers for common analyses.

Public API:
- plot_topomap: draw a 2D sensor topography given sensor coordinates and values.
- plot_bar: draw a sorted bar plot (vertical or horizontal) with optional error bars.
"""

from .dim_reduction import (  # noqa: F401
    plot_embedding,
    plot_shepard_diagram,
    plot_streamlines,
)
from .plotly_utils import (
    plot_channel_traces_interactive,  # noqa: F401
)
from .plots import plot_bar, plot_scatter2d, plot_topomap  # noqa: F401

__all__ = [
    "plot_topomap",
    "plot_bar",
    "plot_scatter2d",
    "plot_embedding",
    "plot_shepard_diagram",
    "plot_streamlines",
    "plot_channel_traces_interactive",
]
