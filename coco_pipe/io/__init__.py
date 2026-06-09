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
from .utils import normalize_subject_value, read_table

__all__ = [
    "DataContainer",
    "SklearnWrapper",
    "SpatialWhitener",
    "load_data",
    "load_descriptor_table",
    "parse_descriptor_feature_column",
    "save_descriptor_table",
    "check_feature_column_consistency",
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
        from .dataset import (  # noqa: F401
            BIDSDataset,
            EmbeddingDataset,
            TabularDataset,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
