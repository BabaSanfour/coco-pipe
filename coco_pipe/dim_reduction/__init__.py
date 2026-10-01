from typing import TYPE_CHECKING

from .analysis import interpret_features
from .artifacts import (
    EVAL_METRIC_COLUMNS,
    EVAL_RUN_KEY_FIELDS,
    FIT_METRIC_COLUMNS,
    FIT_RUN_KEY_FIELDS,
    SEPARATION_METRIC_KEY,
    SEPARATION_RF_METRIC_KEY,
    build_availability_record,
    build_record,
    load_fit_artifact,
    load_fit_runs,
    save_eval_artifact,
    save_fit_artifact,
    update_runs,
    write_run_status,
)
from .config import DEFAULT_EVAL_GROUP_COL, METHODS, parse_eval_specs
from .core import DimReduction
from .evaluation.geometry import (
    moving_average,
    trajectory_acceleration,
    trajectory_auc_speed,
    trajectory_cohesion,
    trajectory_curvature,
    trajectory_dispersion,
    trajectory_displacement,
    trajectory_distance_from_center,
    trajectory_intra_spread,
    trajectory_jerk,
    trajectory_path_length,
    trajectory_separation,
    trajectory_speed,
    trajectory_tortuosity,
    trajectory_turning_angle,
)
from .evaluation.metrics import (
    compute_coranking_matrix,
    compute_mrre,
    continuity,
    lcmc,
    shepard_diagram_data,
    trustworthiness,
)
from .evaluation.stats import grouped_condition_stats, paired_condition_stats
from .evaluation.velocity import compute_velocity_fields
from .pipeline import (
    POOLED_CONDITION,
    build_auto_pooled_eval_spec,
    build_eval_request,
    build_fit_request,
    occurrence_aligned_positions,
    prepare_eval_inputs,
    run_eval,
    run_fit,
    run_fit_group,
    supports_nested_components,
    valid_component_sweep,
    valid_n_components_for_container,
)
from .preprocessing import apply_pca_score_baseline, flip_pc_scores_for_consistency
from .reducers import (
    BaseReducer,
    IncrementalPCAReducer,
    IsomapReducer,
    LLEReducer,
    MDSReducer,
    PCAReducer,
    SpectralEmbeddingReducer,
    TSNEReducer,
)

if TYPE_CHECKING:
    from .reducers import (
        DaskPCAReducer as DaskPCAReducer,
    )
    from .reducers import (
        DaskTruncatedSVDReducer as DaskTruncatedSVDReducer,
    )
    from .reducers import (
        DMDReducer as DMDReducer,
    )
    from .reducers import (
        IVISReducer as IVISReducer,
    )
    from .reducers import (
        PacmapReducer as PacmapReducer,
    )
    from .reducers import (
        ParametricUMAPReducer as ParametricUMAPReducer,
    )
    from .reducers import (
        PHATEReducer as PHATEReducer,
    )
    from .reducers import (
        TopologicalAEReducer as TopologicalAEReducer,
    )
    from .reducers import (
        TRCAReducer as TRCAReducer,
    )
    from .reducers import (
        TrimapReducer as TrimapReducer,
    )
    from .reducers import (
        UMAPReducer as UMAPReducer,
    )

# Core exports
__all__ = [
    "DEFAULT_EVAL_GROUP_COL",
    "EVAL_METRIC_COLUMNS",
    "EVAL_RUN_KEY_FIELDS",
    "FIT_METRIC_COLUMNS",
    "FIT_RUN_KEY_FIELDS",
    "METHODS",
    # Pipeline
    "POOLED_CONDITION",
    # Artifact constants
    "SEPARATION_METRIC_KEY",
    "SEPARATION_RF_METRIC_KEY",
    "BaseReducer",
    "DimReduction",
    "IncrementalPCAReducer",
    "IsomapReducer",
    "LLEReducer",
    "MDSReducer",
    "PCAReducer",
    "SpectralEmbeddingReducer",
    "TSNEReducer",
    "apply_pca_score_baseline",
    "build_auto_pooled_eval_spec",
    "build_availability_record",
    "build_eval_request",
    "build_fit_request",
    "build_record",
    "compute_coranking_matrix",
    "compute_mrre",
    "compute_velocity_fields",
    "continuity",
    "flip_pc_scores_for_consistency",
    "grouped_condition_stats",
    "interpret_features",
    "lcmc",
    "load_fit_artifact",
    "load_fit_runs",
    "moving_average",
    "occurrence_aligned_positions",
    "paired_condition_stats",
    "parse_eval_specs",
    "prepare_eval_inputs",
    "run_eval",
    "run_fit",
    "run_fit_group",
    "save_eval_artifact",
    # Artifact persistence
    "save_fit_artifact",
    "shepard_diagram_data",
    "supports_nested_components",
    "trajectory_acceleration",
    "trajectory_auc_speed",
    "trajectory_cohesion",
    "trajectory_curvature",
    "trajectory_dispersion",
    "trajectory_displacement",
    "trajectory_distance_from_center",
    "trajectory_intra_spread",
    "trajectory_jerk",
    "trajectory_path_length",
    "trajectory_separation",
    "trajectory_speed",
    "trajectory_tortuosity",
    "trajectory_turning_angle",
    "trustworthiness",
    "update_runs",
    "valid_component_sweep",
    "valid_n_components_for_container",
    "write_run_status",
]

_LAZY_REDUCER_EXPORTS = {
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

__all__.extend(_LAZY_REDUCER_EXPORTS)


def __getattr__(name):
    # Lazily import optional reducers from .reducers package
    if name in _LAZY_REDUCER_EXPORTS:
        import importlib

        return getattr(importlib.import_module(".reducers", package=__name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
