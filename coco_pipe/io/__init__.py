from .config import (
    BaseDatasetConfig,
    BIDSConfig,
    DatasetConfig,
    EmbeddingConfig,
    TabularConfig,
)
from .descriptors import load_descriptor_table, parse_descriptor_feature_column
from .load import load_data
from .structures import DataContainer
from .transform import SklearnWrapper, SpatialWhitener
from .utils import normalize_subject_value, read_table

__all__ = [
    "DataContainer",
    "SklearnWrapper",
    "SpatialWhitener",
    "load_data",
    "load_descriptor_table",
    "parse_descriptor_feature_column",
    "normalize_subject_value",
    "read_table",
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
