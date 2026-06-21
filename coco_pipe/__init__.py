"""
Package initializer for the coco_pipe package.
"""

from .descriptors import (
    DescriptorConfig,
    DescriptorPipeline,
)
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
    interpret_features,
    lcmc,
    shepard_diagram_data,
    trustworthiness,
)
from .utils import get_environment_info, get_git_revision_hash, get_package_version

# Core exports
__all__ = [
    "METHODS",
    "BaseReducer",
    "DMDReducer",
    "DaskPCAReducer",
    "DaskTruncatedSVDReducer",
    "DescriptorConfig",
    "DescriptorPipeline",
    "DimReduction",
    "IVISReducer",
    "IncrementalPCAReducer",
    "IsomapReducer",
    "LLEReducer",
    "MDSReducer",
    "PCAReducer",
    "PHATEReducer",
    "PacmapReducer",
    "ParametricUMAPReducer",
    "SpectralEmbeddingReducer",
    "TRCAReducer",
    "TSNEReducer",
    "TopologicalAEReducer",
    "TrimapReducer",
    # Optional (Lazy)
    "UMAPReducer",
    "continuity",
    "get_environment_info",
    "get_git_revision_hash",
    "get_package_version",
    "interpret_features",
    "lcmc",
    "shepard_diagram_data",
    "trustworthiness",
]

_LAZY_DIM_REDUCTION_EXPORTS = {
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
}


def __getattr__(name):
    # Lazily fetch optional members from dim_reduction
    if name in _LAZY_DIM_REDUCTION_EXPORTS:
        import importlib

        return getattr(
            importlib.import_module(".dim_reduction", package=__name__), name
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
