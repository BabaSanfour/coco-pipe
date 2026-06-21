from ._constants import (
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
    discover_embedding_derivatives,
    embedding_sidecar_path,
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
    "ANALYSIS_MODES",
    "DESCRIPTOR_ONLY_ANALYSIS_MODES",
    "BIDSConfig",
    "BIDSDataset",
    "BaseDatasetConfig",
    "CheckResult",
    "DataContainer",
    "DatasetConfig",
    "EmbeddingConfig",
    "EmbeddingDataset",
    "EpochDropRecord",
    "QCResult",
    "SklearnWrapper",
    "SpatialWhitener",
    "SubjectDropRecord",
    "TabularConfig",
    "TabularDataset",
    "compute_constant_feature_summary",
    "compute_feature_missingness",
    "compute_row_outlier_scores",
    "compute_subject_outlier_burden",
    "default_id_extractor",
    "discover_embedding_derivatives",
    "drop_epoch_outliers",
    "drop_subject_outliers",
    "embedding_sidecar_path",
    "fingerprint_container",
    "iter_analysis_units",
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
    "validate_embedding_derivative",
    "write_embedding_dataset_description",
    "write_embedding_manifest",
    "write_json",
]


def __getattr__(name):
    if name in {"BIDSDataset", "EmbeddingDataset", "TabularDataset"}:
        # Delegate to load's lazy resolver so dataset-class resolution (and
        # test monkeypatching) lives in a single place.
        from .load import _resolve_dataset_class

        return _resolve_dataset_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
