"""
Package initializer for the coco_pipe package.
"""

from typing import TYPE_CHECKING

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

# Core exports
__all__ = [
    "METHODS",
    "BaseReducer",
    "DescriptorConfig",
    "DescriptorPipeline",
    "DimReduction",
    "IncrementalPCAReducer",
    "IsomapReducer",
    "LLEReducer",
    "MDSReducer",
    "PCAReducer",
    "SpectralEmbeddingReducer",
    "TSNEReducer",
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

__all__.extend(_LAZY_DIM_REDUCTION_EXPORTS)


def __getattr__(name):
    # Lazily fetch optional members from dim_reduction
    if name in _LAZY_DIM_REDUCTION_EXPORTS:
        import importlib

        return getattr(
            importlib.import_module(".dim_reduction", package=__name__), name
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
