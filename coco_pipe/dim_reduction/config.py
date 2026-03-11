"""
Configuration Schemas
=====================

Pydantic models for strictly validating configuration in the dimensionality
reduction module.

This module defines specific configuration classes for each supported reducer
(e.g., UMAPConfig, PCAConfig) and a generic container (DimReductionConfig).

Classes
-------
DimReductionConfig
    Union container for any reducer configuration.
EvaluationConfig
    Configuration for the evaluation/benchmarking process.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
Date: 2026-01-08
"""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

# --- Registry & Lazy Loading ---

# 1. Core Reducers (Sklearn/Scipy based - always available)
CORE_METHODS = {
    # Linear
    "PCA": ("coco_pipe.dim_reduction.reducers.linear", "PCAReducer"),
    "INCREMENTALPCA": (
        "coco_pipe.dim_reduction.reducers.linear",
        "IncrementalPCAReducer",
    ),
    # Manifold
    "ISOMAP": ("coco_pipe.dim_reduction.reducers.manifold", "IsomapReducer"),
    "LLE": ("coco_pipe.dim_reduction.reducers.manifold", "LLEReducer"),
    "MDS": ("coco_pipe.dim_reduction.reducers.manifold", "MDSReducer"),
    "SPECTRALEMBEDDING": (
        "coco_pipe.dim_reduction.reducers.manifold",
        "SpectralEmbeddingReducer",
    ),
    # Neighbor (t-SNE is core sklearn)
    "TSNE": ("coco_pipe.dim_reduction.reducers.neighbor", "TSNEReducer"),
}

# 2. Optional Reducers (Requires extra deps or heavy imports)
OPTIONAL_METHODS = {
    # Linear
    "DASKPCA": ("coco_pipe.dim_reduction.reducers.linear", "DaskPCAReducer"),
    "DASKTRUNCATEDSVD": (
        "coco_pipe.dim_reduction.reducers.linear",
        "DaskTruncatedSVDReducer",
    ),
    # Neighbor
    "UMAP": ("coco_pipe.dim_reduction.reducers.neighbor", "UMAPReducer"),
    "PARAMETRICUMAP": (
        "coco_pipe.dim_reduction.reducers.neighbor",
        "ParametricUMAPReducer",
    ),
    "PACMAP": ("coco_pipe.dim_reduction.reducers.neighbor", "PacmapReducer"),
    "TRIMAP": ("coco_pipe.dim_reduction.reducers.neighbor", "TrimapReducer"),
    "PHATE": ("coco_pipe.dim_reduction.reducers.neighbor", "PHATEReducer"),
    # Spatiotemporal
    "DMD": ("coco_pipe.dim_reduction.reducers.spatiotemporal", "DMDReducer"),
    "TRCA": ("coco_pipe.dim_reduction.reducers.spatiotemporal", "TRCAReducer"),
    # Neural / Topology
    "IVIS": ("coco_pipe.dim_reduction.reducers.neural", "IVISReducer"),
    "TOPOLOGICALAE": (
        "coco_pipe.dim_reduction.reducers.topology",
        "TopologicalAEReducer",
    ),
}

# User-friendly aliases
METHOD_ALIASES = {
    # Variations
    "t-SNE": "TSNE",
    "tsne": "TSNE",
    "PaCMAP": "PACMAP",
    "pacmap": "PACMAP",
    "TopoAE": "TOPOLOGICALAE",
    "topoae": "TOPOLOGICALAE",
    "IncrPCA": "INCREMENTALPCA",
    "incrementalpca": "INCREMENTALPCA",
    "Spectral": "SPECTRALEMBEDDING",
    "spectral": "SPECTRALEMBEDDING",
    "Isomap": "ISOMAP",
    "isomap": "ISOMAP",
    "LLE": "LLE",
    "lle": "LLE",
    "MDS": "MDS",
    "mds": "MDS",
    "PCA": "PCA",
    "pca": "PCA",
    "UMAP": "UMAP",
    "umap": "UMAP",
    "DMD": "DMD",
    "dmd": "DMD",
    "TRCA": "TRCA",
    "trca": "TRCA",
    "PHATE": "PHATE",
    "phate": "PHATE",
    "IVIS": "IVIS",
    "ivis": "IVIS",
    "TriMap": "TRIMAP",
    "trimap": "TRIMAP",
    "ParametricUMAP": "PARAMETRICUMAP",
    # Hyphenated
    "t-sne": "TSNE",
    "Dask-PCA": "DASKPCA",
    "DaskPCA": "DASKPCA",
}

# For validation
METHODS = list(CORE_METHODS.keys()) + list(OPTIONAL_METHODS.keys())


def normalize_method_name(method: str) -> str:
    """
    Normalize method name to canonical ID using aliases.

    Parameters
    ----------
    method : str
        Input method name (e.g. 't-SNE', 'PaCMAP').

    Returns
    -------
    str
        Canonical method ID (e.g. 'TSNE', 'PACMAP').
    """
    # 1. Exact Alias Match
    if method in METHOD_ALIASES:
        return METHOD_ALIASES[method]

    # 2. Upper Case Fallback
    upper_method = method.upper()
    if upper_method in METHOD_ALIASES:
        return METHOD_ALIASES[upper_method]

    # 3. Default to Upper (Core/Optional keys are upper)
    return upper_method


def get_reducer_class(method: str):
    """
    Factory to retrieve the reducer class for a given method name.
    Lazily imports the module to avoid overhead for unused methods.

    Parameters
    ----------
    method : str
        Name of the reduction method (case-insensitive).

    Returns
    -------
    class
        The reducer class (subclass of BaseReducer).

    Raises
    ------
    ValueError
        If method is unknown.
    ImportError
        If the module cannot be imported.
    """

    method = normalize_method_name(method)

    # Check Core
    if method in CORE_METHODS:
        mod_path, cls_name = CORE_METHODS[method]
    elif method in OPTIONAL_METHODS:
        mod_path, cls_name = OPTIONAL_METHODS[method]
    else:
        valid = ", ".join(METHODS)
        raise ValueError(f"Unknown method '{method}'. Valid options are: {valid}")

    try:
        import importlib

        module = importlib.import_module(mod_path)
        return getattr(module, cls_name)
    except ImportError as e:
        # Provide helpful error for optional dependencies
        if method in OPTIONAL_METHODS:
            raise ImportError(
                f"Could not import reducer '{method}'. "
                f"Ensure required dependencies are installed. Error: {e}"
            )
        raise e


# --- Base Config ---
class BaseReducerConfig(BaseModel):
    n_components: int = Field(2, gt=0, description="Target dimensionality")


class StochasticReducerConfig(BaseModel):
    random_state: Optional[int] = Field(42, description="Seed for reproducibility")


# --- Specific Reducer Configs ---


class PCAConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for PCA."""

    method: Literal["PCA"] = "PCA"
    whiten: bool = Field(
        False,
        description="When True, false, vectors are multiplied by specified number.",
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
    svd_solver: str = Field("auto", description="Solver: auto, full, tsqr, randomized.")


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
    """Configuration for t-SNE."""

    method: Literal["TSNE"] = "TSNE"
    perplexity: float = Field(
        30.0, ge=5.0, description="Perplexity related to number of nearest neighbors."
    )
    early_exaggeration: float = Field(
        12.0,
        description="Controls how tight natural clusters are in the embedding space.",
    )
    learning_rate: Union[float, str] = Field(
        "auto", description="Learning rate for t-SNE optimization."
    )
    max_iter: int = Field(1000, ge=250, description="Maximum number of iterations.")
    init: str = Field("pca", description="Initialization of embedding: random or pca.")


class PacmapConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for PaCMAP."""

    method: Literal["PaCMAP"] = "PaCMAP"
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
    """Configuration for TriMap."""

    method: Literal["TriMap"] = "TriMap"
    n_inliers: int = Field(10, description="Number of inlier triplets.")
    n_outliers: int = Field(5, description="Number of outlier triplets.")
    n_random: int = Field(5, description="Number of random triplets.")


class PHATEConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for PHATE."""

    method: Literal["PHATE"] = "PHATE"
    knn: int = Field(5, description="Number of nearest neighbors.")
    decay: int = Field(40, description="Decay rate for kernel.")
    t: Union[int, str] = Field("auto", description="Diffusion time.")


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


class MDSConfig(BaseReducerConfig, StochasticReducerConfig):
    """Configuration for MDS."""

    method: Literal["MDS"] = "MDS"
    metric: bool = Field(True, description="True for metric MDS, False for non-metric.")
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
    hidden_dims: List[int] = Field(
        [128, 64], description="Encoder hidden layer dimensions."
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


# --- Union Container ---


class DimReductionConfig(BaseModel):
    """
    Master configuration container.
    """

    config: Union[
        PCAConfig,
        IncrementalPCAConfig,
        DaskPCAConfig,
        DaskTruncatedSVDConfig,
        UMAPConfig,
        TSNEConfig,
        PacmapConfig,
        TrimapConfig,
        PHATEConfig,
        IsomapConfig,
        LLEConfig,
        MDSConfig,
        SpectralEmbeddingConfig,
        DMDConfig,
        TRCAConfig,
        TopologicalAEConfig,
        IVISConfig,
        ParametricUMAPConfig,
    ] = Field(..., discriminator="method")


# --- Evaluation Config ---


class EvaluationConfig(BaseModel):
    """
    Configuration for the Method Evaluation process.
    """

    metrics: List[str] = Field(
        default=["trustworthiness", "continuity", "lcmc", "mrre_total"],
        description="Quality metrics to compute.",
    )
    k_range: List[int] = Field(
        default=[5, 10, 20, 50, 100],
        description="Neighborhood sizes (k) for multi-scale evaluation.",
    )
    viz_metric: str = Field(
        "trustworthiness",
        description="Primary metric for plotting comparison curves.",
    )
    selection_metric: Optional[str] = Field(
        default=None,
        description="Primary metric used for automatic method ranking.",
    )
    selection_k: Optional[int] = Field(
        default=None,
        description="Neighborhood size to compare for k-scoped ranking metrics.",
    )
    tie_breakers: List[str] = Field(
        default_factory=list,
        description="Additional metrics used in order to break ranking ties.",
    )
