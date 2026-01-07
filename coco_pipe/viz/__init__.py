#!/usr/bin/env python3
"""
coco_pipe.viz
-------------
Lightweight, backend-agnostic plotting helpers for common analyses.

Public API:
- plot_topomap: draw a 2D sensor topography given sensor coordinates and values.
- plot_bar: draw a sorted bar plot (vertical or horizontal) with optional error bars.
"""
from .plots import plot_topomap, plot_bar, plot_scatter2d
from .dim_reduction import plot_embedding, plot_shepard_diagram, plot_streamlines

__all__ = [
    "plot_topomap",
    "plot_bar",
    "plot_scatter2d",
    "plot_embedding",
    "plot_shepard_diagram",
    "plot_streamlines"
]
