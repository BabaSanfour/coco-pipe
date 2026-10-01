from .base import BaseReducer
from .linear import IncrementalPCAReducer, PCAReducer
from .manifold import IsomapReducer, LLEReducer, MDSReducer, SpectralEmbeddingReducer
from .neighbor import TSNEReducer

# Map optional class names to their module paths. These reducers pull in heavy
# or optional third-party dependencies, so they are resolved lazily through the
# module-level ``__getattr__`` below (PEP 562) rather than imported eagerly.
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

# Eagerly imported reducers plus the lazily resolved optional ones. The optional
# names are spread in from ``_OPTIONAL_REDUCERS`` so they remain part of the
# public surface (and ``import *``) without being statically undefined.
__all__ = [
    "BaseReducer",
    "IncrementalPCAReducer",
    "IsomapReducer",
    "LLEReducer",
    "MDSReducer",
    "PCAReducer",
    "SpectralEmbeddingReducer",
    "TSNEReducer",
    *sorted(_OPTIONAL_REDUCERS),
]


def __getattr__(name):
    if name in _OPTIONAL_REDUCERS:
        import importlib

        module_path = _OPTIONAL_REDUCERS[name]
        module = importlib.import_module(module_path, package=__package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
