from .config import (
    BaseDatasetConfig,
    BIDSConfig,
    DatasetConfig,
    EmbeddingConfig,
    TabularConfig,
)
from .load import load_data
from .structures import DataContainer
from .transform import SklearnWrapper, SpatialWhitener

__all__ = [
    "DataContainer",
    "SklearnWrapper",
    "SpatialWhitener",
    "load_data",
    "BIDSDataset",
    "TabularDataset",
    "EmbeddingDataset",
    "BaseDatasetConfig",
    "BIDSConfig",
    "TabularConfig",
    "EmbeddingConfig",
    "DatasetConfig",
]


def __getattr__(name):
    if name in {"BIDSDataset", "EmbeddingDataset", "TabularDataset"}:
        from .dataset import (  # noqa: F401
            BIDSDataset,
            EmbeddingDataset,
            TabularDataset,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
