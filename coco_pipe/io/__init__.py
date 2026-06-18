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
from .descriptors import (
    check_feature_column_consistency,
    load_descriptor_table,
    parse_descriptor_feature_column,
    save_descriptor_table,
)
from .embeddings import (
    discover_embedding_derivatives,
    embedding_sidecar_path,
    load_embedding_derivatives,
    save_embedding_derivative,
    validate_embedding_derivative,
    write_embedding_dataset_description,
    write_embedding_manifest,
)
from .load import load_data
from .quality import (
    CheckResult,
    EpochDropRecord,
    QCResult,
    SubjectDropRecord,
    compute_constant_feature_summary,
    compute_feature_missingness,
    compute_row_outlier_scores,
    compute_subject_outlier_burden,
    drop_epoch_outliers,
    drop_subject_outliers,
    make_qc_flag,
    resolve_qc_status,
    row_quality_score,
    run_qc,
)
from .structures import DataContainer
from .transform import SklearnWrapper, SpatialWhitener
from .units import iter_analysis_units
from .utils import normalize_subject_value

__all__ = [
    "DataContainer",
    "SklearnWrapper",
    "SpatialWhitener",
    "load_data",
    "load_descriptor_table",
    "parse_descriptor_feature_column",
    "save_descriptor_table",
    "check_feature_column_consistency",
    "discover_embedding_derivatives",
    "embedding_sidecar_path",
    "load_embedding_derivatives",
    "save_embedding_derivative",
    "validate_embedding_derivative",
    "write_embedding_dataset_description",
    "write_embedding_manifest",
    "CheckResult",
    "EpochDropRecord",
    "QCResult",
    "SubjectDropRecord",
    "compute_constant_feature_summary",
    "compute_feature_missingness",
    "compute_row_outlier_scores",
    "compute_subject_outlier_burden",
    "drop_epoch_outliers",
    "drop_subject_outliers",
    "make_qc_flag",
    "resolve_qc_status",
    "row_quality_score",
    "run_qc",
    "normalize_subject_value",
    "read_table",
    "smart_reader",
    "default_id_extractor",
    "read_json",
    "write_json",
    "save_object",
    "load_object",
    "save_npz",
    "iter_analysis_units",
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
        # Delegate to load's lazy resolver so dataset-class resolution (and
        # test monkeypatching) lives in a single place.
        from .load import _resolve_dataset_class

        return _resolve_dataset_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
