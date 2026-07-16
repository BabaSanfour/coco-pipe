from typing import TYPE_CHECKING

from ._constants import (
    AGGREGATION_LEVELS,
    ANALYSIS_MODES,
    DESCRIPTOR_ONLY_ANALYSIS_MODES,
)
from ._serialization import (
    default_id_extractor,
    load_object,
    read_json,
    read_table,
    save_npz,
    save_object,
    smart_reader,
    write_json,
)
from .config import (
    BaseDatasetConfig,
    BIDSConfig,
    DatasetConfig,
    EmbeddingConfig,
    TabularConfig,
)
from .embeddings import (
    combined_embedding_table_path,
    discover_embedding_derivatives,
    embedding_observation_id,
    embedding_sidecar_path,
    load_combined_embedding_table,
    load_embedding_derivatives,
    save_embedding_derivative,
    validate_embedding_derivative,
    write_embedding_dataset_description,
    write_embedding_manifest,
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
from .structures import DataContainer
from .transform import SklearnWrapper, SpatialWhitener
from .units import iter_analysis_units, split_unit_sensor
from .utils import (
    compute_constant_feature_summary,
    compute_feature_missingness,
    normalize_subject_value,
    row_quality_score,
)

if TYPE_CHECKING:
    from .dataset import BIDSDataset as BIDSDataset
    from .dataset import EmbeddingDataset as EmbeddingDataset
    from .dataset import TabularDataset as TabularDataset

__all__ = [
    "AGGREGATION_LEVELS",
    "ANALYSIS_MODES",
    "DESCRIPTOR_ONLY_ANALYSIS_MODES",
    "BIDSConfig",
    "BaseDatasetConfig",
    "CheckResult",
    "DataContainer",
    "DatasetConfig",
    "EmbeddingConfig",
    "EpochDropRecord",
    "QCResult",
    "SklearnWrapper",
    "SpatialWhitener",
    "SubjectDropRecord",
    "TabularConfig",
    "combined_embedding_table_path",
    "compute_constant_feature_summary",
    "compute_feature_missingness",
    "compute_row_outlier_scores",
    "compute_subject_outlier_burden",
    "default_id_extractor",
    "discover_embedding_derivatives",
    "drop_epoch_outliers",
    "drop_subject_outliers",
    "embedding_observation_id",
    "embedding_sidecar_path",
    "fingerprint_container",
    "group_labels",
    "iter_analysis_units",
    "load_combined_embedding_table",
    "load_data",
    "load_embedding_derivatives",
    "load_object",
    "make_qc_flag",
    "normalize_subject_value",
    "read_json",
    "read_table",
    "resolve_qc_status",
    "row_quality_score",
    "run_qc",
    "save_embedding_derivative",
    "save_npz",
    "save_object",
    "smart_reader",
    "split_unit_sensor",
    "validate_embedding_derivative",
    "write_embedding_dataset_description",
    "write_embedding_manifest",
    "write_json",
]

_LAZY_DATASET_EXPORTS = ["BIDSDataset", "EmbeddingDataset", "TabularDataset"]
__all__.extend(_LAZY_DATASET_EXPORTS)


def __getattr__(name):
    if name in _LAZY_DATASET_EXPORTS:
        # Delegate to load's lazy resolver so dataset-class resolution (and
        # test monkeypatching) lives in a single place.
        from .load import _resolve_dataset_class

        return _resolve_dataset_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
