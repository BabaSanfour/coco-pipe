"""
Package initializer for the coco_pipe package.
"""

from typing import TYPE_CHECKING

from .descriptors import (
    DescriptorConfig,
    DescriptorPipeline,
)

from .utils import get_environment_info, get_git_revision_hash, get_package_version

if TYPE_CHECKING:
    from .dim_reduction import (
        DaskPCAReducer as DaskPCAReducer,
    )
    from .dim_reduction import (
        DaskTruncatedSVDReducer as DaskTruncatedSVDReducer,
    )
    from .dim_reduction import (
        DMDReducer as DMDReducer,
    )
    from .dim_reduction import (
        IVISReducer as IVISReducer,
    )
    from .dim_reduction import (
        PacmapReducer as PacmapReducer,
    )
    from .dim_reduction import (
        ParametricUMAPReducer as ParametricUMAPReducer,
    )
    from .dim_reduction import (
        PHATEReducer as PHATEReducer,
    )
    from .dim_reduction import (
        TopologicalAEReducer as TopologicalAEReducer,
    )
    from .dim_reduction import (
        TRCAReducer as TRCAReducer,
    )
    from .dim_reduction import (
        TrimapReducer as TrimapReducer,
    )
    from .dim_reduction import (
        UMAPReducer as UMAPReducer,
    )

# Core exports (non-dim_reduction names only; dim_reduction names are added
# below via __all__.extend(_LAZY_DIM_REDUCTION_EXPORTS) to avoid duplicates)
__all__ = [
    "DescriptorConfig",
    "DescriptorPipeline",
    "get_environment_info",
    "get_git_revision_hash",
    "get_package_version",
]

_LAZY_DIM_REDUCTION_EXPORTS = {
    # sklearn-based reducers — always available once scikit-learn is installed
    "METHODS",
    "BaseReducer",
    "DimReduction",
    "IncrementalPCAReducer",
    "IsomapReducer",
    "LLEReducer",
    "MDSReducer",
    "PCAReducer",
    "SpectralEmbeddingReducer",
    "TSNEReducer",
    # evaluation helpers
    "continuity",
    "interpret_features",
    "lcmc",
    "shepard_diagram_data",
    "trustworthiness",
    # optional reducers that require extra dependencies
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

__all__.extend(_LAZY_DIM_REDUCTION_EXPORTS)


def __getattr__(name):
    # Lazily fetch optional members from dim_reduction
    if name in _LAZY_DIM_REDUCTION_EXPORTS:
        import importlib

        return getattr(
            importlib.import_module(".dim_reduction", package=__name__), name
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
