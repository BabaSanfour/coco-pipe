"""
Dimensionality Reduction Configuration
======================================

Strict configuration models and reducer registry for the dim-reduction module.

This module defines:

- canonical reducer names and lazy registry lookup
- strict pydantic configs for each supported reducer
- evaluation configuration with early validation for metric and ranking choices

The config layer follows the same explicit design as the rest of the module:
exact method names, no aliasing, no compatibility wrappers, and no permissive
extra fields.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

import importlib
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "METHODS",
    "get_reducer_class",
    "BaseReducerConfig",
    "StochasticReducerConfig",
    "PCAConfig",
    "IncrementalPCAConfig",
    "DaskPCAConfig",
    "DaskTruncatedSVDConfig",
    "UMAPConfig",
    "TSNEConfig",
    "PacmapConfig",
    "TrimapConfig",
    "PHATEConfig",
    "IsomapConfig",
    "LLEConfig",
    "MDSConfig",
    "SpectralEmbeddingConfig",
    "DMDConfig",
    "TRCAConfig",
    "TopologicalAEConfig",
    "IVISConfig",
    "ParametricUMAPConfig",
    "EvaluationConfig",
]

# --- Registry & Lazy Loading ---

_METHOD_REGISTRY = {
    # Linear
    "PCA": ("coco_pipe.dim_reduction.reducers.linear", "PCAReducer"),
    "IncrementalPCA": (
        "coco_pipe.dim_reduction.reducers.linear",
        "IncrementalPCAReducer",
    ),
    "DaskPCA": ("coco_pipe.dim_reduction.reducers.linear", "DaskPCAReducer"),
    "DaskTruncatedSVD": (
        "coco_pipe.dim_reduction.reducers.linear",
        "DaskTruncatedSVDReducer",
    ),
    # Manifold
    "Isomap": ("coco_pipe.dim_reduction.reducers.manifold", "IsomapReducer"),
    "LLE": ("coco_pipe.dim_reduction.reducers.manifold", "LLEReducer"),
    "MDS": ("coco_pipe.dim_reduction.reducers.manifold", "MDSReducer"),
    "SpectralEmbedding": (
        "coco_pipe.dim_reduction.reducers.manifold",
        "SpectralEmbeddingReducer",
    ),
    # Neighbor (t-SNE is core sklearn)
    "TSNE": ("coco_pipe.dim_reduction.reducers.neighbor", "TSNEReducer"),
    "UMAP": ("coco_pipe.dim_reduction.reducers.neighbor", "UMAPReducer"),
    "ParametricUMAP": (
        "coco_pipe.dim_reduction.reducers.neighbor",
        "ParametricUMAPReducer",
    ),
    "Pacmap": ("coco_pipe.dim_reduction.reducers.neighbor", "PacmapReducer"),
    "Trimap": ("coco_pipe.dim_reduction.reducers.neighbor", "TrimapReducer"),
    "PHATE": ("coco_pipe.dim_reduction.reducers.neighbor", "PHATEReducer"),
    # Spatiotemporal
    "DMD": ("coco_pipe.dim_reduction.reducers.spatiotemporal", "DMDReducer"),
    "TRCA": ("coco_pipe.dim_reduction.reducers.spatiotemporal", "TRCAReducer"),
    # Neural / Topology
    "IVIS": ("coco_pipe.dim_reduction.reducers.neural", "IVISReducer"),
    "TopologicalAE": (
        "coco_pipe.dim_reduction.reducers.topology",
        "TopologicalAEReducer",
    ),
}

_OPTIONAL_METHODS = frozenset(
    {
        "DaskPCA",
        "DaskTruncatedSVD",
        "UMAP",
        "ParametricUMAP",
        "Pacmap",
        "Trimap",
        "PHATE",
        "DMD",
        "TRCA",
        "IVIS",
        "TopologicalAE",
    }
)

METHODS = tuple(_METHOD_REGISTRY)

_VALID_EVALUATION_METRICS = frozenset(
    {
        "trustworthiness",
        "continuity",
        "lcmc",
        "mrre_intrusion",
        "mrre_extrusion",
        "mrre_total",
        "shepard_correlation",
        "trajectory_speed",
        "trajectory_acceleration",
        "trajectory_curvature",
        "trajectory_turning_angle",
        "trajectory_dispersion",
        "trajectory_path_length",
        "trajectory_displacement",
        "trajectory_tortuosity",
        "trajectory_separation",
    }
)

_VALID_RANKING_METRICS = frozenset(
    {
        "trustworthiness",
        "continuity",
        "lcmc",
        "shepard_correlation",
        "mrre_intrusion",
        "mrre_extrusion",
        "mrre_total",
    }
)

_VALID_SEPARATION_METHODS = frozenset(
    {
        "centroid",
        "within_between_ratio",
        "mahalanobis",
        "distributional",
        "margin",
    }
)


def get_reducer_class(method: str):
    """
    Return the reducer class registered for one canonical method name.

    Parameters
    ----------
    method : str
        Canonical public name of the reduction method.

    Returns
    -------
    class
        The reducer class (subclass of BaseReducer).

    Raises
    ------
    ValueError
        If ``method`` is not one of the canonical names in ``METHODS``.
    ImportError
        If the reducer backend cannot be imported.

    Notes
    -----
    Registry lookup is exact and case-sensitive. The dim-reduction module does
    not support aliasing or case normalization.

    See Also
    --------
    METHODS
        Canonical public method names accepted by the registry.
    BaseReducerConfig
        Base type for typed reducer configuration objects.

    Examples
    --------
    >>> cls = get_reducer_class("PCA")
    >>> cls.__name__
    'PCAReducer'
    """

    if method not in _METHOD_REGISTRY:
        valid = ", ".join(METHODS)
        raise ValueError(f"Unknown method '{method}'. Valid options are: {valid}")
    mod_path, cls_name = _METHOD_REGISTRY[method]

    try:
        module = importlib.import_module(mod_path)
        return getattr(module, cls_name)
    except ImportError as e:
        if method in _OPTIONAL_METHODS:
            raise ImportError(
                f"Could not import reducer '{method}'. "
                f"Ensure required dependencies are installed. Error: {e}"
            )
        raise e


class _StrictConfigModel(BaseModel):
    """Shared strict pydantic behavior for dim-reduction configs."""

    model_config = ConfigDict(extra="forbid")


# --- Base Config ---
class BaseReducerConfig(_StrictConfigModel):
    """
    Base configuration shared by all reducer configs.

    Notes
    -----
    All reducer configs are strict. Unknown fields are rejected at parse time.
    Subclasses must expose a canonical ``method`` literal and may override
    ``to_reducer_kwargs()`` when the reducer constructor needs renamed fields.

    See Also
    --------
    get_reducer_class
        Registry lookup for canonical method names.
    EvaluationConfig
        Post-hoc scoring and ranking configuration.
    """

    n_components: int = Field(2, gt=0, description="Target dimensionality")

    def to_reducer_kwargs(self) -> dict[str, Any]:
        """Return reducer keyword arguments for this config."""
        return self.model_dump(exclude={"method", "n_components"})


class StochasticReducerConfig(_StrictConfigModel):
    """Mixin for reducers that expose a random seed."""

    random_state: Optional[int] = Field(42, description="Seed for reproducibility")


# --- Specific Reducer Configs ---
class PCAConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for PCA."""

    method: Literal["PCA"] = "PCA"
    whiten: bool = Field(
        False,
        description="Whiten projected components to unit variance.",
    )
    svd_solver: str = Field("auto", description="Solver for SVD.")


class IncrementalPCAConfig(BaseReducerConfig):
    """Configuration for Incremental PCA."""

    method: Literal["IncrementalPCA"] = "IncrementalPCA"
    batch_size: Optional[int] = Field(None, description="Batch size.")
    whiten: bool = Field(False, description="Whiten.")


class DaskPCAConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for Dask PCA."""

    method: Literal["DaskPCA"] = "DaskPCA"
    svd_solver: str = Field(
        "auto", description="SVD solver: auto, full, tsqr, or randomized."
    )


class DaskTruncatedSVDConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for Dask TruncatedSVD."""

    method: Literal["DaskTruncatedSVD"] = "DaskTruncatedSVD"
    algorithm: str = Field("tsqr", description="SVD Algorithm.")


class UMAPConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for UMAP."""

    method: Literal["UMAP"] = "UMAP"
    n_neighbors: int = Field(15, ge=2, description="Size of local neighborhood.")
    min_dist: float = Field(
        0.1, ge=0.0, description="Minimum distance between points in low-dim space."
    )
    metric: str = Field("euclidean", description="Metric for distance computation.")
    n_epochs: Optional[int] = Field(None, description="Number of training epochs.")
    spread: float = Field(1.0, description="Effective scale of embedded points.")
    set_op_mix_ratio: float = Field(
        1.0, description="Interpolate between intersection and union (1.0 is union)."
    )


class TSNEConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for TSNE."""

    method: Literal["TSNE"] = "TSNE"
    perplexity: float = Field(
        30.0, ge=5.0, description="Perplexity related to number of nearest neighbors."
    )
    early_exaggeration: float = Field(
        12.0,
        description="Controls how tight natural clusters are in the embedding space.",
    )
    learning_rate: float | str = Field(
        "auto", description="Learning rate for t-SNE optimization."
    )
    max_iter: int = Field(1000, ge=250, description="Maximum number of iterations.")
    init: str = Field("pca", description="Initialization of embedding: random or pca.")


class PacmapConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for Pacmap."""

    method: Literal["Pacmap"] = "Pacmap"
    n_neighbors: int = Field(10, description="Number of neighbors.")
    MN_ratio: float = Field(0.5, description="Ratio of mid-near pairs.")
    FP_ratio: float = Field(2.0, description="Ratio of far pairs.")
    nn_backend: str = Field(
        "faiss",
        description=(
            "Nearest-neighbor backend. Recent PaCMAP versions support "
            "faiss, annoy, and voyager."
        ),
    )
    init: str = Field("pca", description="Initialization method.")


class TrimapConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for Trimap."""

    method: Literal["Trimap"] = "Trimap"
    n_inliers: int = Field(10, description="Number of inlier triplets.")
    n_outliers: int = Field(5, description="Number of outlier triplets.")
    n_random: int = Field(5, description="Number of random triplets.")


class PHATEConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for PHATE."""

    method: Literal["PHATE"] = "PHATE"
    knn: int = Field(5, description="Number of nearest neighbors.")
    decay: int = Field(40, description="Decay rate for kernel.")
    t: int | str = Field("auto", description="Diffusion time.")


class IsomapConfig(BaseReducerConfig):
    """Configuration for Isomap."""

    method: Literal["Isomap"] = "Isomap"
    n_neighbors: int = Field(5, description="Number of neighbors.")
    metric: str = Field("minkowski", description="Metric.")
    p: int = Field(2, description="Power for Minkowski.")


class LLEConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for LLE."""

    method: Literal["LLE"] = "LLE"
    n_neighbors: int = Field(5, description="Number of neighbors.")
    lle_method: str = Field(
        "standard",
        description=(
            "LLE method (standard, hessian, modified, ltsa). Named "
            "'lle_method' because 'method' is reserved for reducer selection."
        ),
    )

    def to_reducer_kwargs(self) -> dict[str, Any]:
        """Return reducer keyword arguments with sklearn-compatible names."""
        reducer_kwargs = super().to_reducer_kwargs()
        reducer_kwargs["method"] = reducer_kwargs.pop("lle_method")
        return reducer_kwargs


class MDSConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for MDS."""

    method: Literal["MDS"] = "MDS"
    metric: bool = Field(
        True, description="Use metric MDS when True, non-metric MDS when False."
    )
    n_init: int = Field(4, description="Number of initializations.")
    max_iter: int = Field(300, description="Max iterations.")
    dissimilarity: str = Field("euclidean", description="Dissimilarity measure.")


class SpectralEmbeddingConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for Spectral Embedding."""

    method: Literal["SpectralEmbedding"] = "SpectralEmbedding"
    affinity: str = Field(
        "nearest_neighbors", description="Affinity (nearest_neighbors, rbf, etc)."
    )
    gamma: Optional[float] = Field(None, description="Kernel coefficient for rbf.")


class DMDConfig(BaseReducerConfig):
    """Configuration for DMD."""

    method: Literal["DMD"] = "DMD"
    force_transpose: bool = Field(
        False,
        description=(
            "Transpose input from (n_snapshots, n_features) to "
            "(n_features, n_snapshots)."
        ),
    )
    tlsq_rank: int = Field(0, description="Rank for Total Least Squares processing.")
    exact: bool = Field(
        False, description="Compute exact DMD (True) or projected DMD (False)."
    )
    opt: bool = Field(False, description="Amplitudes optimization.")


class TRCAConfig(BaseReducerConfig):
    """Configuration for TRCA."""

    method: Literal["TRCA"] = "TRCA"
    sfreq: float = Field(250.0, description="Sampling frequency in Hertz.")
    filterbank: Optional[list] = Field(
        None,
        description=(
            "Optional filterbank definition as [(passband), (stopband)] groups."
        ),
    )


class TopologicalAEConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for Topological Autoencoder."""

    method: Literal["TopologicalAE"] = "TopologicalAE"
    hidden_dims: list[int] = Field(
        default_factory=lambda: [128, 64],
        description="Encoder hidden layer dimensions.",
    )
    lam: float = Field(0.0, description="Regularization strength for topological loss.")
    lr: float = Field(1e-3, description="Learning rate.")
    batch_size: int = Field(64, description="Batch size.")
    epochs: int = Field(50, description="Number of training epochs.")
    device: str = Field("auto", description="Device to use (cpu, cuda, mps, auto).")
    verbose: int = Field(0, description="Skorch verbosity level.")


class IVISConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for IVIS."""

    method: Literal["IVIS"] = "IVIS"
    k: int = Field(15, description="Number of neighbors.")
    model: str = Field("szubert", description="Network architecture.")
    n_epochs_without_progress: int = Field(15, description="Early stopping patience.")
    supervise_metric: str = Field(
        "softmax_cross_entropy", description="Metric for supervised training."
    )


class ParametricUMAPConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for Parametric UMAP."""

    method: Literal["ParametricUMAP"] = "ParametricUMAP"
    n_neighbors: int = Field(15, description="Number of neighbors.")
    min_dist: float = Field(0.1, description="Minimum distance.")
    metric: str = Field("euclidean", description="Metric.")
    n_epochs: Optional[int] = Field(None, description="Number of epochs.")
    batch_size: int = Field(1000, description="Batch size.")
    verbose: bool = Field(False, description="Verbose.")


# --- Evaluation Config ---
class EvaluationConfig(_StrictConfigModel):
    """
    Configuration for post-hoc evaluation and method comparison.

    Parameters
    ----------
    metrics : list of str, optional
        Metric families to compute. Must use canonical evaluator metric names.
    k_range : list of int, optional
        Neighborhood sizes used for standard structure-preservation metrics.
    selection_metric : str, optional
        Primary ranking metric. Must be one of the supported ranking metrics
        and also appear in ``metrics``.
    selection_k : int, optional
        Neighborhood size used when ranking a k-scoped metric.
    tie_breakers : list of str, optional
        Additional ranking metrics applied in order. Each value must also be
        present in ``metrics``.
    separation_method : str, default="centroid"
        Separation definition used for trajectory separation scoring.

    Notes
    -----
    ``EvaluationConfig`` validates semantic consistency at parse time. Invalid
    metric names, duplicate entries, invalid separation methods, and ranking
    metrics that are not part of ``metrics`` all fail early.

    See Also
    --------
    coco_pipe.dim_reduction.evaluation.core.evaluate_embedding
        Pure evaluator that consumes these settings.
    coco_pipe.dim_reduction.evaluation.core.MethodSelector
        Post-hoc collector and ranker for scored reducers.

    Examples
    --------
    >>> config = EvaluationConfig(
    ...     metrics=["trustworthiness", "continuity"],
    ...     k_range=[5, 10],
    ...     selection_metric="trustworthiness",
    ...     selection_k=10,
    ...     tie_breakers=["continuity"],
    ... )
    >>> config.selection_metric
    'trustworthiness'
    """

    metrics: list[str] = Field(
        default_factory=lambda: [
            "trustworthiness",
            "continuity",
            "lcmc",
            "mrre_total",
        ],
        description=(
            "Quality metrics to compute. Trajectory selectors such as "
            "'trajectory_speed', 'trajectory_curvature', 'trajectory_dispersion', "
            "and 'trajectory_separation' apply only to native 3D trajectory "
            "embeddings. Evaluation-level 'trajectory_dispersion' uses the "
            "global unlabeled dispersion definition."
        ),
    )
    k_range: list[int] = Field(
        default_factory=lambda: [5, 10, 20, 50, 100],
        description="Neighborhood sizes (k) for multi-scale evaluation.",
    )
    selection_metric: Optional[str] = Field(
        default=None,
        description="Primary metric used for automatic method ranking.",
    )
    selection_k: Optional[int] = Field(
        default=None,
        description="Neighborhood size to compare for k-scoped ranking metrics.",
    )
    tie_breakers: list[str] = Field(
        default_factory=list,
        description="Additional metrics used in order to break ranking ties.",
    )
    separation_method: str = Field(
        default="centroid",
        description=(
            "Separation definition passed to trajectory_separation when "
            "trajectory labels are available. Supported values mirror "
            "geometry.trajectory_separation(..., method=...)."
        ),
    )

    @field_validator("metrics")
    @classmethod
    def _validate_metrics(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("`metrics` must contain at least one metric name.")
        if len(set(value)) != len(value):
            raise ValueError("`metrics` must not contain duplicate entries.")
        invalid = sorted(set(value) - _VALID_EVALUATION_METRICS)
        if invalid:
            valid = ", ".join(sorted(_VALID_EVALUATION_METRICS))
            raise ValueError(
                f"Unknown evaluation metric(s): {invalid}. Valid options are: {valid}"
            )
        return value

    @field_validator("k_range")
    @classmethod
    def _validate_k_range(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("`k_range` must not contain duplicate entries.")
        if any(k <= 0 for k in value):
            raise ValueError("`k_range` values must be positive integers.")
        return value

    @field_validator("selection_metric")
    @classmethod
    def _validate_selection_metric(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value not in _VALID_RANKING_METRICS:
            valid = ", ".join(sorted(_VALID_RANKING_METRICS))
            raise ValueError(
                f"`selection_metric` must be one of: {valid}. Got {value!r}."
            )
        return value

    @field_validator("selection_k")
    @classmethod
    def _validate_selection_k(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("`selection_k` must be a positive integer.")
        return value

    @field_validator("tie_breakers")
    @classmethod
    def _validate_tie_breakers(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("`tie_breakers` must not contain duplicate entries.")
        invalid = sorted(set(value) - _VALID_RANKING_METRICS)
        if invalid:
            valid = ", ".join(sorted(_VALID_RANKING_METRICS))
            raise ValueError(
                f"`tie_breakers` must use ranking metrics only. "
                f"Invalid values: {invalid}. Valid options are: {valid}"
            )
        return value

    @field_validator("separation_method")
    @classmethod
    def _validate_separation_method(cls, value: str) -> str:
        if value not in _VALID_SEPARATION_METHODS:
            valid = ", ".join(sorted(_VALID_SEPARATION_METHODS))
            raise ValueError(
                f"`separation_method` must be one of: {valid}. Got {value!r}."
            )
        return value

    @model_validator(mode="after")
    def _validate_metric_dependencies(self) -> "EvaluationConfig":
        if self.selection_metric and self.selection_metric not in self.metrics:
            raise ValueError("`selection_metric` must also be present in `metrics`.")
        missing_tie_breakers = [
            metric for metric in self.tie_breakers if metric not in self.metrics
        ]
        if missing_tie_breakers:
            raise ValueError(
                "`tie_breakers` must also be present in `metrics`. "
                f"Missing: {missing_tie_breakers}"
            )
        return self
