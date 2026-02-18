"""
Package initializer for the coco_pipe package.
"""

from .dim_reduction import (
    METHODS,
    BaseReducer,
    DimReduction,
    IncrementalPCAReducer,
    IsomapReducer,
    LLEReducer,
    MDSReducer,
    PCAReducer,
    SpectralEmbeddingReducer,
    TSNEReducer,
    continuity,
    lcmc,
    shepard_diagram_data,
    trustworthiness,
)

# Core exports
__all__ = [
    "DimReduction",
    "METHODS",
    "BaseReducer",
    "PCAReducer",
    "IncrementalPCAReducer",
    "IsomapReducer",
    "LLEReducer",
    "MDSReducer",
    "SpectralEmbeddingReducer",
    "TSNEReducer",
    "trustworthiness",
    "continuity",
    "lcmc",
    "shepard_diagram_data",
    # Optional (Lazy)
    "UMAPReducer",
    "PacmapReducer",
    "TrimapReducer",
    "PHATEReducer",
    "DMDReducer",
    "TRCAReducer",
    "IVISReducer",
    "TopologicalAEReducer",
    "DaskPCAReducer",
    "DaskTruncatedSVDReducer",
    "ParametricUMAPReducer",
]


def __getattr__(name):
    # Lazily fetch optional members from dim_reduction
    if name in __all__:
        import importlib

        return getattr(
            importlib.import_module(".dim_reduction", package=__name__), name
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
