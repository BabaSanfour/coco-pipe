from .base import BaseReducer
from .linear import (
    IncrementalPCAReducer,
    PCAReducer,
)
from .manifold import IsomapReducer, LLEReducer, MDSReducer, SpectralEmbeddingReducer
from .neighbor import TSNEReducer

# Define Core exports
__all__ = [
    "BaseReducer",
    "PCAReducer",
    "IncrementalPCAReducer",
    "IsomapReducer",
    "LLEReducer",
    "MDSReducer",
    "SpectralEmbeddingReducer",
    "TSNEReducer",
    # Optional Reducers (Lazy)
    "DaskPCAReducer",
    "DaskTruncatedSVDReducer",
    "UMAPReducer",
    "ParametricUMAPReducer",
    "PacmapReducer",
    "TrimapReducer",
    "PHATEReducer",
    "DMDReducer",
    "TRCAReducer",
    "IVISReducer",
    "TopologicalAEReducer",
]

# Map optional class names to their module paths
_OPTIONAL_REDUCERS = {
    "DaskPCAReducer": ".linear",
    "DaskTruncatedSVDReducer": ".linear",
    "UMAPReducer": ".neighbor",
    "ParametricUMAPReducer": ".neighbor",
    "PacmapReducer": ".neighbor",
    "TrimapReducer": ".neighbor",
    "PHATEReducer": ".neighbor",
    "DMDReducer": ".spatiotemporal",
    "TRCAReducer": ".spatiotemporal",
    "IVISReducer": ".neural",
    "TopologicalAEReducer": ".topology",
}

def __getattr__(name):
    if name in _OPTIONAL_REDUCERS:
        import importlib
        module_path = _OPTIONAL_REDUCERS[name]
        module = importlib.import_module(module_path, package=__package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
