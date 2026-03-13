from .analysis import interpret_features
from .config import METHODS
from .core import DimReduction
from .evaluation.geometry import (
    trajectory_curvature,
    trajectory_separation,
    trajectory_speed,
)
from .evaluation.metrics import continuity, lcmc, shepard_diagram_data, trustworthiness
from .reducers import (
    BaseReducer,
    IncrementalPCAReducer,
    IsomapReducer,
    LLEReducer,
    MDSReducer,
    PCAReducer,
    SpectralEmbeddingReducer,
    TSNEReducer,
)

# Core exports
__all__ = [
    "DimReduction",
    "METHODS",
    "interpret_features",
    "trustworthiness",
    "continuity",
    "lcmc",
    "shepard_diagram_data",
    "trajectory_speed",
    "trajectory_curvature",
    "trajectory_separation",
    "BaseReducer",
    "PCAReducer",
    "IncrementalPCAReducer",
    "IsomapReducer",
    "LLEReducer",
    "MDSReducer",
    "SpectralEmbeddingReducer",
    "TSNEReducer",
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

_LAZY_REDUCER_EXPORTS = {
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
    # Lazily import optional reducers from .reducers package
    if name in _LAZY_REDUCER_EXPORTS:
        import importlib

        return getattr(importlib.import_module(".reducers", package=__name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
