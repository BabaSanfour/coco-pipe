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

from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator

# Import Reducers for Registry
from .reducers.linear import PCAReducer, IncrementalPCAReducer, DaskPCAReducer, DaskTruncatedSVDReducer
from .reducers.manifold import IsomapReducer, LLEReducer, MDSReducer, SpectralEmbeddingReducer
from .reducers.neighbor import TSNEReducer, UMAPReducer, PacmapReducer, TrimapReducer, PHATEReducer, ParametricUMAPReducer
from .reducers.spatiotemporal import DMDReducer, TRCAReducer
from .reducers.neural import IVISReducer
from .reducers.topology import TopologicalAEReducer

# --- Base Config ---
class BaseReducerConfig(BaseModel):
    n_components: int = Field(2, gt=0, description="Target dimensionality")
    random_state: Optional[int] = Field(42, description="Seed for reproducibility")

# --- Specific Reducer Configs ---

class PCAConfig(BaseReducerConfig):
    """Configuration for PCA."""
    method: Literal["PCA"] = "PCA"
    whiten: bool = Field(False, description="When True, false, vectors are multiplied by specified number.")
    svd_solver: str = Field("auto", description="Solver for SVD.")

class IncrementalPCAConfig(BaseReducerConfig):
    """Configuration for Incremental PCA."""
    method: Literal["IncrementalPCA"] = "IncrementalPCA"
    batch_size: Optional[int] = Field(None, description="Batch size.")
    whiten: bool = Field(False, description="Whiten.")

class DaskPCAConfig(BaseReducerConfig):
    """Configuration for Dask PCA."""
    method: Literal["DaskPCA"] = "DaskPCA"
    svd_solver: str = Field("auto", description="Solver: auto, full, tsqr, randomized.")

class DaskTruncatedSVDConfig(BaseReducerConfig):
    """Configuration for Dask TruncatedSVD."""
    method: Literal["DaskTruncatedSVD"] = "DaskTruncatedSVD"
    algorithm: str = Field("tsqr", description="SVD Algorithm.")

class UMAPConfig(BaseReducerConfig):
    """Configuration for UMAP."""
    method: Literal["UMAP"] = "UMAP"
    n_neighbors: int = Field(15, ge=2, description="Size of local neighborhood.")
    min_dist: float = Field(0.1, ge=0.0, description="Minimum distance between points in low-dim space.")
    metric: str = Field("euclidean", description="Metric for distance computation.")
    n_epochs: Optional[int] = Field(None, description="Number of training epochs.")
    spread: float = Field(1.0, description="Effective scale of embedded points.")
    set_op_mix_ratio: float = Field(1.0, description="Interpolate between intersection and union (1.0 is union).")

class TSNEConfig(BaseReducerConfig):
    """Configuration for t-SNE."""
    method: Literal["TSNE"] = "TSNE"
    perplexity: float = Field(30.0, ge=5.0, description="Perplexity related to number of nearest neighbors.")
    early_exaggeration: float = Field(12.0, description="Controls how tight natural clusters are in the embedding space.")
    learning_rate: Union[float, str] = Field("auto", description="Learning rate for t-SNE optimization.")
    n_iter: int = Field(1000, ge=250, description="Maximum number of iterations.")
    init: str = Field("pca", description="Initialization of embedding: random or pca.")

class PacmapConfig(BaseReducerConfig):
    """Configuration for PaCMAP."""
    method: Literal["PaCMAP"] = "PaCMAP"
    n_neighbors: int = Field(10, description="Number of neighbors.")
    MN_ratio: float = Field(0.5, description="Ratio of mid-near pairs.")
    FP_ratio: float = Field(2.0, description="Ratio of far pairs.")
    init: str = Field("pca", description="Initialization method.")

class TrimapConfig(BaseReducerConfig):
    """Configuration for TriMap."""
    method: Literal["TriMap"] = "TriMap"
    n_inliers: int = Field(10, description="Number of inlier triplets.")
    n_outliers: int = Field(5, description="Number of outlier triplets.")
    n_random: int = Field(5, description="Number of random triplets.")

class PHATEConfig(BaseReducerConfig):
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

class LLEConfig(BaseReducerConfig):
    """Configuration for LLE."""
    method: Literal["LLE"] = "LLE"
    n_neighbors: int = Field(5, description="Number of neighbors.")
    method_variant: str = Field("standard", alias="variant", description="LLE Variant (standard, hessian, modified, ltsa).")

class MDSConfig(BaseReducerConfig):
    """Configuration for MDS."""
    method: Literal["MDS"] = "MDS"
    metric: bool = Field(True, description="True for metric MDS, False for non-metric.")
    n_init: int = Field(4, description="Number of initializations.")
    max_iter: int = Field(300, description="Max iterations.")
    dissimilarity: str = Field("euclidean", description="Dissimilarity measure.")

class SpectralEmbeddingConfig(BaseReducerConfig):
    """Configuration for Spectral Embedding."""
    method: Literal["SpectralEmbedding"] = "SpectralEmbedding"
    affinity: str = Field("nearest_neighbors", description="Affinity (nearest_neighbors, rbf, etc).")
    gamma: Optional[float] = Field(None, description="Kernel coefficient for rbf.")

class DMDConfig(BaseReducerConfig):
    """Configuration for DMD."""
    method: Literal["DMD"] = "DMD"
    tlsq_rank: int = Field(0, description="Rank for Total Least Squares processing.")
    exact: bool = Field(False, description="Compute exact DMD (True) or projected DMD (False).")
    opt: bool = Field(False, description="Amplitudes optimization.")

class TRCAConfig(BaseReducerConfig):
    """Configuration for TRCA."""
    method: Literal["TRCA"] = "TRCA"
    # TRCA generally takes simple args found in Base (n_components)
    # Adding any future specific args here if needed.

class TopologicalAEConfig(BaseReducerConfig):
    """Configuration for Topological Autoencoder."""
    method: Literal["TopologicalAE"] = "TopologicalAE"
    hidden_dims: List[int] = Field([128, 64], description="Encoder hidden layer dimensions.")
    lam: float = Field(0.0, description="Regularization strength for topological loss.")
    lr: float = Field(1e-3, description="Learning rate.")
    batch_size: int = Field(64, description="Batch size.")
    epochs: int = Field(50, description="Number of training epochs.")
    device: str = Field("cpu", description="Device to use (cpu, cuda, mps, auto).")

class IVISConfig(BaseReducerConfig):
    """Configuration for IVIS."""
    method: Literal["IVIS"] = "IVIS"
    k: int = Field(15, description="Number of neighbors.")
    model: str = Field("szubert", description="Network architecture.")
    n_epochs_without_progress: int = Field(15, description="Early stopping patience.")
    supervise_metric: str = Field("softmax_cross_entropy", description="Metric for supervised training.")

class ParametricUMAPConfig(BaseReducerConfig):
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
        ParametricUMAPConfig
    ] = Field(..., discriminator='method')


# --- Evaluation Config ---

class EvaluationConfig(BaseModel):
    """
    Configuration for the Method Evaluation process.
    """
    metrics: List[str] = Field(
        default=["trustworthiness", "continuity", "lcmc", "mrre_total"],
        description="Quality metrics to compute."
    )
    k_range: List[int] = Field(
        default=[5, 10, 20, 50, 100],
        description="Neighborhood sizes (k) for multi-scale evaluation."
    )
    viz_metric: str = Field(
        "trustworthiness",
        description="Primary metric for plotting comparison curves."
    )


# --- Registry ---
# Maps string names to their implementation classes
# Used by DimReduction core to instantiate methods.

METHODS_DICT = {
    # Linear
    "PCA": PCAReducer,
    "INCREMENTALPCA": IncrementalPCAReducer,
    "DASKPCA": DaskPCAReducer,
    "DASKTRUNCATEDSVD": DaskTruncatedSVDReducer,
    # Manifold
    "ISOMAP": IsomapReducer,
    "LLE": LLEReducer,
    "MDS": MDSReducer,
    "SPECTRALEMBEDDING": SpectralEmbeddingReducer,
    # Neighbor
    "TSNE": TSNEReducer,
    "UMAP": UMAPReducer,
    "PACMAP": PacmapReducer,
    "TRIMAP": TrimapReducer,
    "PHATE": PHATEReducer,
    # Spatiotemporal
    "DMD": DMDReducer,
    "TRCA": TRCAReducer,
    # Neural / Topological
    "IVIS": IVISReducer,
    "TOPOLOGICALAE": TopologicalAEReducer,
    "PARAMETRICUMAP": ParametricUMAPReducer
}

METHODS = list(METHODS_DICT.keys())
