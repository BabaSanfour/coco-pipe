"""
Foundation model pipelines for CoCo Pipe.

This module hosts pipelines built on top of the CBRAMod foundation model.
"""

from .pipeline import CBRAModRegressionPipeline, FoundationRegressor

__all__ = [
    "CBRAModRegressionPipeline",
    "FoundationRegressor",
]
