"""
Package initializer for the coco_pipe package.
"""

from .ml import (
    MLPipeline
)

from .dim_reduction import (
    DimReduction,
    METHODS,
    METHODS_DICT,
    BaseReducer,
    PCAReducer,
    IsomapReducer,
    LLEReducer,
    MDSReducer,
    SpectralEmbeddingReducer,
    TSNEReducer,
    UMAPReducer,
    PacmapReducer,
    TrimapReducer,
    PHATEReducer,
    DMDReducer,
    TRCAReducer,
    IVISReducer,
    TopologicalAEReducer,
    trustworthiness,
    continuity,
    lcmc,
    shepard_diagram_data
)

__all__ = [
    "DimReduction",
    "METHODS",
    "METHODS_DICT",
    "BaseReducer",
    "PCAReducer",
    "IsomapReducer",
    "LLEReducer",
    "MDSReducer",
    "SpectralEmbeddingReducer",
    "TSNEReducer",
    "UMAPReducer",
    "PacmapReducer",
    "TrimapReducer",
    "PHATEReducer",
    "DMDReducer",
    "TRCAReducer",
    "IVISReducer",
    "TopologicalAEReducer",
    "trustworthiness",
    "continuity",
    "lcmc",
    "shepard_diagram_data",
]
