from typing import TYPE_CHECKING

from ..utils import NoDataError
from ._constants import (
    AGGREGATION_LEVELS,
    ANALYSIS_MODES,
    DESCRIPTOR_ONLY_ANALYSIS_MODES,
    EMBEDDING_CONFIG_KEYS,
    EMBEDDING_VIEWS,
)
from ._serialization import (
    load_object,
    read_json,
    read_table,
    save_npz,
    save_object,
    write_json,
)
from .config import (
    BIDSConfig,
    DatasetConfig,
    EmbeddingConfig,
    TabularConfig,
)
from .load import load_data
from .provenance import fingerprint_container
from .quality import (
    CheckResult,
    EpochDropRecord,
    QCResult,
    SubjectDropRecord,
    compute_row_outlier_scores,
    compute_subject_outlier_burden,
    drop_epoch_outliers,
    drop_subject_outliers,
    group_labels,
    make_qc_flag,
    resolve_qc_status,
    run_qc,
)
from .structures import DataContainer, aggregate_recordings
from .transform import SklearnWrapper
from .units import iter_analysis_units, split_unit_sensor
from .utils import (
    compute_constant_feature_summary,
    compute_feature_missingness,
    normalize_subject_value,
)

if TYPE_CHECKING:
    from .dataset import BIDSDataset as BIDSDataset
    from .dataset import TabularDataset as TabularDataset
    from .embeddings import consolidate_embeddings as consolidate_embeddings
    from .embeddings import embedding_label as embedding_label
    from .embeddings import embedding_path as embedding_path
    from .embeddings import embeddings_up_to_date as embeddings_up_to_date
    from .embeddings import find_embeddings as find_embeddings
    from .embeddings import load_embeddings as load_embeddings
    from .embeddings import write_embedding_dataset as write_embedding_dataset
    from .embeddings import write_embedding_manifest as write_embedding_manifest
    from .embeddings import write_recording_embeddings as write_recording_embeddings
    from .embeddings import (
        write_transformed_embeddings as write_transformed_embeddings,
    )

__all__ = [
    "AGGREGATION_LEVELS",
    "ANALYSIS_MODES",
    "DESCRIPTOR_ONLY_ANALYSIS_MODES",
    "EMBEDDING_CONFIG_KEYS",
    "EMBEDDING_VIEWS",
    "BIDSConfig",
    "CheckResult",
    "DataContainer",
    "DatasetConfig",
    "EmbeddingConfig",
    "EpochDropRecord",
    "NoDataError",
    "QCResult",
    "SklearnWrapper",
    "SubjectDropRecord",
    "TabularConfig",
    "aggregate_recordings",
    "compute_constant_feature_summary",
    "compute_feature_missingness",
    "compute_row_outlier_scores",
    "compute_subject_outlier_burden",
    "drop_epoch_outliers",
    "drop_subject_outliers",
    "fingerprint_container",
    "group_labels",
    "iter_analysis_units",
    "load_data",
    "load_object",
    "make_qc_flag",
    "normalize_subject_value",
    "read_json",
    "read_table",
    "resolve_qc_status",
    "run_qc",
    "save_npz",
    "save_object",
    "split_unit_sensor",
    "write_json",
]

_LAZY_DATASET_EXPORTS = ["BIDSDataset", "TabularDataset"]
# They import mne-bids and h5netcdf, which `import coco_pipe.io` does not need.
_LAZY_EMBEDDING_EXPORTS = [
    "consolidate_embeddings",
    "embedding_label",
    "embedding_path",
    "embeddings_up_to_date",
    "find_embeddings",
    "load_embeddings",
    "write_embedding_dataset",
    "write_embedding_manifest",
    "write_recording_embeddings",
    "write_transformed_embeddings",
]
__all__.extend(_LAZY_DATASET_EXPORTS + _LAZY_EMBEDDING_EXPORTS)


def __getattr__(name):
    if name in _LAZY_DATASET_EXPORTS:
        from . import dataset

        return getattr(dataset, name)
    if name in _LAZY_EMBEDDING_EXPORTS:
        from . import embeddings

        return getattr(embeddings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
