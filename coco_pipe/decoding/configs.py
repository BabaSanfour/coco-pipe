"""
Decoding Configurations
=======================

Comprehensive Pydantic models for strict validation of Decoding/ML experiments.
These models ensure that all parameters are scientifically sound before
any computation begins.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._constants import MetricTask

# --- Base Schemas ---


class BaseEstimatorConfig(BaseModel):
    """Base configuration for any estimator."""

    model_config = ConfigDict(extra="forbid")


class ClassicalEstimatorConfig(BaseEstimatorConfig):
    """Base for scikit-learn compatible classical estimators."""

    kind: Literal["classical"] = "classical"


# --- Mixins ---


class LinearMixin(BaseModel):
    """Common parameters for linear models."""

    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: int | None = None


class RegularizedLinearMixin(BaseModel):
    """Parameters for regularized linear models."""

    fit_intercept: bool = True
    copy_X: bool = True
    tol: float = 1e-3
    max_iter: int | None = None
    positive: bool = False
    random_state: int | None = Field(42, description="Random seed for reproducibility.")


class TreeMixin(BaseModel):
    """Common parameters for Tree-based models."""

    n_estimators: int = Field(100, ge=1)
    max_depth: int | None = None
    min_samples_split: int | float = 2
    min_samples_leaf: int | float = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: str | int | float | None = "sqrt"
    max_leaf_nodes: int | None = None
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0
    n_jobs: int | None = None
    random_state: int | None = Field(42, description="Random seed for reproducibility.")
    verbose: int = 0
    warm_start: bool = False


class SupportVectorMixin(BaseModel):
    """Common parameters for Support Vector Machines."""

    C: float = Field(1.0, gt=0.0)
    kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf"
    degree: int = 3
    gamma: str | float = "scale"
    coef0: float = 0.0
    tol: float = 1e-3
    verbose: bool = False
    max_iter: int = -1
    shrinking: bool = True
    cache_size: float = 200


class SGDMixin(BaseModel):
    """Stochastic Gradient Descent parameters."""

    loss: str = "hinge"
    penalty: Literal["l2", "l1", "elasticnet"] | None = "l2"
    alpha: float = 0.0001
    l1_ratio: float = 0.15
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float = 1e-3
    shuffle: bool = True
    verbose: int = 0
    epsilon: float = 0.1
    learning_rate: str = "optimal"
    eta0: float = 0.01
    power_t: float = 0.5
    early_stopping: bool = False


class MLPMixin(BaseModel):
    """Common parameters for Multi-layer Perceptron models."""

    hidden_layer_sizes: tuple = (100,)
    activation: Literal["identity", "logistic", "tanh", "relu"] = "relu"
    solver: Literal["lbfgs", "sgd", "adam"] = "adam"
    alpha: float = 0.0001
    batch_size: int | str = "auto"
    learning_rate: Literal["constant", "invscaling", "adaptive"] = "constant"
    learning_rate_init: float = 0.001
    power_t: float = 0.5
    max_iter: int = 200
    shuffle: bool = True
    tol: float = 1e-4
    verbose: bool = False
    warm_start: bool = False
    momentum: float = 0.9
    nesterovs_momentum: bool = True
    early_stopping: bool = False
    validation_fraction: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    n_iter_no_change: int = 10
    max_fun: int = 15000
    random_state: int | None = Field(42, description="Random seed for reproducibility.")


class GradientBoostingMixin(BaseModel):
    """Common parameters for Gradient Boosting models."""

    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 1.0
    criterion: Literal["friedman_mse", "squared_error"] = "friedman_mse"
    min_samples_split: int | float = 2
    min_samples_leaf: int | float = 1
    min_weight_fraction_leaf: float = 0.0
    max_depth: int = 3
    min_impurity_decrease: float = 0.0
    init: str | None = None
    max_features: str | int | float | None = None
    verbose: int = 0
    max_leaf_nodes: int | None = None
    warm_start: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 5
    tol: float = 1e-4
    ccp_alpha: float = 0.0
    random_state: int | None = Field(42, description="Random seed for reproducibility.")


# --- Classifiers ---


class LogisticRegressionConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.linear_model.LogisticRegression."""

    method: Literal["LogisticRegression"] = "LogisticRegression"
    penalty: Literal["l1", "l2", "elasticnet", None] = "l2"
    dual: bool = False
    tol: float = 1e-4
    C: float = Field(1.0, gt=0.0)
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    class_weight: dict | str | None = None
    solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = "lbfgs"
    max_iter: int = 100
    verbose: int = 0
    warm_start: bool = False
    n_jobs: int | None = None
    l1_ratio: float | None = None
    random_state: int | None = Field(42, description="Random seed for reproducibility.")


class RandomForestClassifierConfig(ClassicalEstimatorConfig, TreeMixin):
    """Configuration for sklearn.ensemble.RandomForestClassifier."""

    method: Literal["RandomForestClassifier"] = "RandomForestClassifier"
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    bootstrap: bool = True
    oob_score: bool = False
    class_weight: str | dict | list | None = None
    max_samples: int | float | None = None


class SVCConfig(ClassicalEstimatorConfig, SupportVectorMixin):
    """Configuration for sklearn.svm.SVC."""

    method: Literal["SVC"] = "SVC"
    probability: bool = True  # Default to True for metrics requiring proba
    class_weight: dict | str | None = None
    decision_function_shape: Literal["ovo", "ovr"] = "ovr"
    break_ties: bool = False
    random_state: int | None = Field(42, description="Random seed for reproducibility.")


class LinearSVCConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.svm.LinearSVC."""

    method: Literal["LinearSVC"] = "LinearSVC"
    penalty: Literal["l1", "l2"] = "l2"
    loss: Literal["hinge", "squared_hinge"] = "squared_hinge"
    dual: bool | Literal["auto"] = "auto"
    tol: float = 1e-4
    C: float = Field(1.0, gt=0.0)
    multi_class: Literal["ovr", "crammer_singer"] = "ovr"
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    class_weight: dict | str | None = None
    verbose: int = 0
    max_iter: int = 1000
    random_state: int | None = Field(42, description="Random seed for reproducibility.")


class KNeighborsClassifierConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.neighbors.KNeighborsClassifier."""

    method: Literal["KNeighborsClassifier"] = "KNeighborsClassifier"
    n_neighbors: int = Field(5, ge=1)
    weights: Literal["uniform", "distance"] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: str = "minkowski"
    metric_params: dict | None = None
    n_jobs: int | None = None


class GradientBoostingClassifierConfig(ClassicalEstimatorConfig, GradientBoostingMixin):
    """Configuration for sklearn.ensemble.GradientBoostingClassifier."""

    method: Literal["GradientBoostingClassifier"] = "GradientBoostingClassifier"
    loss: Literal["log_loss", "exponential"] = "log_loss"


class HistGradientBoostingClassifierConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.ensemble.HistGradientBoostingClassifier."""

    method: Literal["HistGradientBoostingClassifier"] = "HistGradientBoostingClassifier"
    learning_rate: float = 0.1
    max_iter: int = 100
    max_leaf_nodes: int = 31
    max_depth: int | None = None
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    max_bins: int = 255
    categorical_features: list[int] | list[str] | list[bool] | None = None
    monotonic_cst: Any | None = None
    interaction_cst: Any | None = None
    warm_start: bool = False
    early_stopping: bool | Literal["auto"] = "auto"
    scoring: str | None = "loss"
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    tol: float = 1e-7
    verbose: int = 0
    random_state: int | None = None


class SGDClassifierConfig(ClassicalEstimatorConfig, SGDMixin):
    """Configuration for sklearn.linear_model.SGDClassifier."""

    method: Literal["SGDClassifier"] = "SGDClassifier"
    class_weight: dict | str | None = None


class MLPClassifierConfig(ClassicalEstimatorConfig, MLPMixin):
    """Configuration for sklearn.neural_network.MLPClassifier."""

    method: Literal["MLPClassifier"] = "MLPClassifier"


class GaussianNBConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.naive_bayes.GaussianNB."""

    method: Literal["GaussianNB"] = "GaussianNB"
    priors: list[float] | None = None
    var_smoothing: float = 1e-9


class LDAConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.discriminant_analysis.LinearDiscriminantAnalysis."""

    method: Literal["LinearDiscriminantAnalysis"] = "LinearDiscriminantAnalysis"
    solver: Literal["svd", "lsqr", "eigen"] = "svd"
    shrinkage: str | float | None = None
    priors: list[float] | None = None
    n_components: int | None = None
    store_covariance: bool = False
    tol: float = 1e-4


class AdaBoostClassifierConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.ensemble.AdaBoostClassifier."""

    method: Literal["AdaBoostClassifier"] = "AdaBoostClassifier"
    n_estimators: int = 50
    learning_rate: float = 1.0
    random_state: int | None = Field(42, description="Random seed for reproducibility.")


class DummyClassifierConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.dummy.DummyClassifier."""

    method: Literal["DummyClassifier"] = "DummyClassifier"
    strategy: Literal["stratified", "most_frequent", "prior", "uniform"] = "prior"
    constant: Any | None = None
    random_state: int | None = Field(42, description="Random seed for reproducibility.")


# --- Deep Learning / Foundation Models ---


class LPFTConfig(BaseEstimatorConfig):
    """
    Configuration for Linear-Probe Fine-Tuning (LP-FT).
    Reference: Kumar et al. (2022).
    """

    method: Literal["LPFTClassifier"] = "LPFTClassifier"
    backbone_name: str = "gpt2"
    # LP Step
    lp_lr: float = 1e-3
    lp_epochs: int = 10
    # FT Step
    ft_lr: float = 1e-5
    ft_epochs: int = 5
    batch_size: int = 32
    max_length: int = 128
    device: str = "cpu"


class SkorchClassifierConfig(BaseEstimatorConfig):
    """Configuration for generic PyTorch wrappers via Skorch."""

    method: Literal["SkorchClassifier"] = "SkorchClassifier"
    module_name: str
    max_epochs: int = 10
    lr: float = 0.01
    batch_size: int = 64
    optimizer: str = "Adam"
    device: str = "cpu"


# --- Temporal Meta-Estimators (MNE) ---


class SlidingEstimatorConfig(BaseEstimatorConfig):
    """
    Configuration for MNE-style SlidingEstimator.
    Fits a separate estimator for each time point.
    """

    method: Literal["SlidingEstimator"] = "SlidingEstimator"
    base_estimator: EstimatorConfigType
    scoring: str | Callable | None = None
    n_jobs: int | None = 1
    position: float | None = 0
    allow_2d: bool = False
    verbose: bool | str | int | None = None


class GeneralizingEstimatorConfig(BaseEstimatorConfig):
    """
    Configuration for MNE-style GeneralizingEstimator.
    Fits an estimator on each time point and tests on all other time points.
    """

    method: Literal["GeneralizingEstimator"] = "GeneralizingEstimator"
    base_estimator: EstimatorConfigType
    scoring: str | Callable | None = None
    n_jobs: int | None = 1
    position: float | None = 0
    allow_2d: bool = False
    verbose: bool | str | int | None = None


# --- Regressors ---


class LinearRegressionConfig(ClassicalEstimatorConfig, LinearMixin):
    """Configuration for sklearn.linear_model.LinearRegression."""

    method: Literal["LinearRegression"] = "LinearRegression"
    positive: bool = False


class RidgeConfig(ClassicalEstimatorConfig, RegularizedLinearMixin):
    """Configuration for sklearn.linear_model.Ridge."""

    method: Literal["Ridge"] = "Ridge"
    alpha: float = 1.0
    fit_intercept: bool = True
    copy_X: bool = True
    solver: str = "auto"


class LassoConfig(ClassicalEstimatorConfig, RegularizedLinearMixin):
    """Configuration for sklearn.linear_model.Lasso."""

    method: Literal["Lasso"] = "Lasso"
    alpha: float = 1.0
    precompute: bool | list = False
    fit_intercept: bool = True
    copy_X: bool = True
    selection: Literal["cyclic", "random"] = "cyclic"
    warm_start: bool = False


class ElasticNetConfig(ClassicalEstimatorConfig, RegularizedLinearMixin):
    """Configuration for sklearn.linear_model.ElasticNet."""

    method: Literal["ElasticNet"] = "ElasticNet"
    alpha: float = 1.0
    l1_ratio: float = 0.5
    precompute: bool | list = False
    fit_intercept: bool = True
    copy_X: bool = True
    selection: Literal["cyclic", "random"] = "cyclic"
    warm_start: bool = False


class RandomForestRegressorConfig(ClassicalEstimatorConfig, TreeMixin):
    """Configuration for sklearn.ensemble.RandomForestRegressor."""

    method: Literal["RandomForestRegressor"] = "RandomForestRegressor"
    criterion: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"] = (
        "squared_error"
    )
    bootstrap: bool = True
    oob_score: bool = False
    max_samples: int | float | None = None


class SVRConfig(ClassicalEstimatorConfig, SupportVectorMixin):
    """Configuration for sklearn.svm.SVR."""

    method: Literal["SVR"] = "SVR"
    epsilon: float = 0.1


class GradientBoostingRegressorConfig(ClassicalEstimatorConfig, GradientBoostingMixin):
    """Configuration for sklearn.ensemble.GradientBoostingRegressor."""

    method: Literal["GradientBoostingRegressor"] = "GradientBoostingRegressor"
    loss: Literal["squared_error", "absolute_error", "huber", "quantile"] = (
        "squared_error"
    )
    alpha: float = 0.9


class SGDRegressorConfig(ClassicalEstimatorConfig, SGDMixin):
    """Configuration for sklearn.linear_model.SGDRegressor."""

    method: Literal["SGDRegressor"] = "SGDRegressor"
    loss: str = "squared_error"


class MLPRegressorConfig(ClassicalEstimatorConfig, MLPMixin):
    """Configuration for sklearn.neural_network.MLPRegressor."""

    method: Literal["MLPRegressor"] = "MLPRegressor"


class DummyRegressorConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.dummy.DummyRegressor."""

    method: Literal["DummyRegressor"] = "DummyRegressor"
    strategy: Literal["mean", "median", "quantile", "constant"] = "mean"
    constant: int | float | list | None = None
    quantile: float | None = None


class DecisionTreeRegressorConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.tree.DecisionTreeRegressor."""

    method: Literal["DecisionTreeRegressor"] = "DecisionTreeRegressor"
    criterion: Literal["squared_error", "friedman_mse", "absolute_error", "poisson"] = (
        "squared_error"
    )
    splitter: Literal["best", "random"] = "best"
    max_depth: int | None = None
    min_samples_split: int | float = 2
    min_samples_leaf: int | float = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: str | int | float | None = None
    random_state: int | None = None
    max_leaf_nodes: int | None = None
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0


class KNeighborsRegressorConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.neighbors.KNeighborsRegressor."""

    method: Literal["KNeighborsRegressor"] = "KNeighborsRegressor"
    n_neighbors: int = Field(5, ge=1)
    weights: Literal["uniform", "distance"] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: str = "minkowski"
    metric_params: dict | None = None
    n_jobs: int | None = None


class ExtraTreesRegressorConfig(ClassicalEstimatorConfig, TreeMixin):
    """Configuration for sklearn.ensemble.ExtraTreesRegressor."""

    method: Literal["ExtraTreesRegressor"] = "ExtraTreesRegressor"
    bootstrap: bool = False
    oob_score: bool = False
    max_samples: int | float | None = None


class HistGradientBoostingRegressorConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.ensemble.HistGradientBoostingRegressor."""

    method: Literal["HistGradientBoostingRegressor"] = "HistGradientBoostingRegressor"
    loss: Literal["squared_error", "absolute_error", "poisson", "quantile"] = (
        "squared_error"
    )
    learning_rate: float = 0.1
    max_iter: int = 100
    max_leaf_nodes: int = 31
    max_depth: int | None = None
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    max_bins: int = 255
    categorical_features: list[int] | list[str] | list[bool] | None = None
    monotonic_cst: Any | None = None
    interaction_cst: Any | None = None
    warm_start: bool = False
    early_stopping: bool | Literal["auto"] = "auto"
    scoring: str | None = "loss"
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    tol: float = 1e-7
    verbose: int = 0
    random_state: int | None = None


class AdaBoostRegressorConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.ensemble.AdaBoostRegressor."""

    method: Literal["AdaBoostRegressor"] = "AdaBoostRegressor"
    n_estimators: int = 50
    learning_rate: float = 1.0
    loss: Literal["linear", "square", "exponential"] = "linear"
    random_state: int | None = Field(42, description="Random seed for reproducibility.")


class BayesianRidgeConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.linear_model.BayesianRidge."""

    method: Literal["BayesianRidge"] = "BayesianRidge"
    max_iter: int = 300
    tol: float = 1e-3
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    alpha_init: float | None = None
    lambda_init: float | None = None
    compute_score: bool = False
    fit_intercept: bool = True
    copy_X: bool = True
    verbose: bool = False


class ARDRegressionConfig(ClassicalEstimatorConfig):
    """Configuration for sklearn.linear_model.ARDRegression."""

    method: Literal["ARDRegression"] = "ARDRegression"
    max_iter: int = 300
    tol: float = 1e-3
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    compute_score: bool = False
    threshold_lambda: float = 10000.0
    fit_intercept: bool = True
    copy_X: bool = True
    verbose: bool = False


# --- Recursive Unions Implementation ---


class ClassicalModelConfig(ClassicalEstimatorConfig):
    """Configuration for standard scikit-learn estimators."""

    method: Literal["ClassicalModel"] = "ClassicalModel"
    estimator: str
    params: dict[str, Any] = Field(default_factory=dict)
    input_kind: Literal["tabular", "embeddings"] = "tabular"


class FoundationEmbeddingModelConfig(BaseEstimatorConfig):
    """Configuration for pretrained feature extraction backbones."""

    kind: Literal["foundation_embedding"] = "foundation_embedding"
    model_key: str = "dummy"
    backend: str = "auto"
    n_outputs: int | None = None
    train_mode: Literal["frozen", "full", "lora", "qlora"] = "frozen"
    pooling: Literal["mean", "flatten"] = "mean"
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    sfreq: float | None = Field(None, gt=0)
    ch_names: list[str] | None = None
    backend_kwargs: dict[str, Any] = Field(default_factory=dict)


class LoRAConfig(BaseModel):
    """Low-Rank Adaptation (LoRA) configuration."""

    model_config = ConfigDict(extra="forbid")

    r: int = Field(16, ge=1)
    alpha: int = Field(32, ge=1)
    dropout: float = Field(0.0, ge=0.0, le=1.0)
    target_modules: str | list[str] = "all-linear"


class QuantizationConfig(BaseModel):
    """Model quantization settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    load_in_4bit: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"
    compute_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"


class DeviceConfig(BaseModel):
    """Compute device and precision policy."""

    model_config = ConfigDict(extra="forbid")

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"


class CheckpointConfig(BaseModel):
    """Neural training checkpoint policy."""

    model_config = ConfigDict(extra="forbid")

    save: Literal["none", "best", "last", "all"] = "best"
    monitor: str = "val_loss"
    output_dir: Path | None = None


class TrainerConfig(BaseModel):
    """Neural training loop configuration."""

    model_config = ConfigDict(extra="forbid")

    max_epochs: int = Field(10, ge=1)
    early_stopping_patience: int | None = Field(None, ge=1)
    batch_size: int = Field(32, ge=1)
    validation_fraction: float = Field(0.2, ge=0.0, lt=1.0)


class FrozenBackboneDecoderConfig(BaseEstimatorConfig):
    """Config for a frozen backbone followed by a classical decoding head."""

    kind: Literal["frozen_backbone"] = "frozen_backbone"
    backbone: FoundationEmbeddingModelConfig
    head: ClassicalModelConfig


class NeuralFineTuneConfig(BaseEstimatorConfig):
    """Configuration for end-to-end neural fine-tuning."""

    kind: Literal["neural_finetune"] = "neural_finetune"
    model_key: str = "dummy"
    backend: str = "auto"
    n_outputs: int | None = Field(None, ge=1)
    input_kind: Literal["temporal", "epoched", "tokens"] = "epoched"
    train_mode: Literal["full", "frozen", "linear_probe", "lora", "qlora"] = "full"
    optimizer: dict[str, Any] = Field(default_factory=lambda: {"name": "adamw"})
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)
    lora: LoRAConfig | None = None
    quantization: QuantizationConfig | None = None
    sfreq: float | None = Field(None, gt=0)
    ch_names: list[str] | None = None
    backend_kwargs: dict[str, Any] = Field(default_factory=dict)
    class_weight: str | dict[Any, float] | None = "balanced"


class TemporalDecoderConfig(BaseEstimatorConfig):
    """Config for MNE-style temporal meta-estimators."""

    kind: Literal["temporal"] = "temporal"
    wrapper: Literal["sliding", "generalizing"] = "sliding"
    base: ClassicalModelConfig
    scoring: str | Callable | None = None
    n_jobs: int | None = 1
    position: float | None = 0
    allow_2d: bool = False
    verbose: bool | str | int | None = None


AtomicEstimator = (
    LogisticRegressionConfig
    | RandomForestClassifierConfig
    | SVCConfig
    | LinearSVCConfig
    | KNeighborsClassifierConfig
    | GradientBoostingClassifierConfig
    | HistGradientBoostingClassifierConfig
    | SGDClassifierConfig
    | MLPClassifierConfig
    | GaussianNBConfig
    | LDAConfig
    | AdaBoostClassifierConfig
    | DummyClassifierConfig
    # Regressors
    | LinearRegressionConfig
    | RidgeConfig
    | LassoConfig
    | ElasticNetConfig
    | RandomForestRegressorConfig
    | SVRConfig
    | GradientBoostingRegressorConfig
    | SGDRegressorConfig
    | MLPRegressorConfig
    | DummyRegressorConfig
    | DecisionTreeRegressorConfig
    | KNeighborsRegressorConfig
    | ExtraTreesRegressorConfig
    | HistGradientBoostingRegressorConfig
    | AdaBoostRegressorConfig
    | BayesianRidgeConfig
    | ARDRegressionConfig
)

EstimatorConfigType = Annotated[
    AtomicEstimator | SlidingEstimatorConfig | GeneralizingEstimatorConfig,
    Field(discriminator="method"),
]

ClassicalModelType = Annotated[
    ClassicalModelConfig | AtomicEstimator, Field(discriminator="method")
]


# --- Experiment Config ---


class CVConfig(BaseModel):
    """
    Cross-validation configuration settings.

    Parameters
    ----------
    strategy : str, default="stratified"
        The splitting strategy. Note that 'stratified' strategies require
        classification tasks.
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    random_state : int, default=42
        Random seed for the splitter.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split for
        strategy='split'.
    stratify : bool, default=False
        Whether to use stratified sampling for strategy='split'.
    group_key : str, optional
        The column name in sample_metadata to use for group-aware strategies.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal[
        "stratified",
        "kfold",
        "group_kfold",
        "stratified_group_kfold",
        "leave_p_out",
        "leave_one_group_out",
        "timeseries",
        "split",
        "group_shuffle_split",
    ] = "stratified"
    n_splits: int = Field(5, ge=1)
    shuffle: bool = True
    random_state: int = 42
    test_size: float = Field(
        0.2, gt=0.0, lt=1.0, description="Holdout size for strategy='split'."
    )
    stratify: bool = Field(
        False, description="Whether strategy='split' should stratify by y."
    )
    group_key: str | None = Field(
        None, description="sample_metadata column used by grouped CV strategies."
    )
    auto_reduce_n_splits: bool = Field(
        True,
        description=(
            "Reduce grouped fold counts to the largest leakage-safe value supported "
            "by the observed groups and classes."
        ),
    )


class TuningConfig(BaseModel):
    """
    Hyperparameter Tuning Configuration.
    """

    enabled: bool = False
    search_type: Literal["grid", "random"] = "grid"
    n_iter: int = Field(10, ge=1, description="Number of iterations for random search")
    scoring: str | None = None  # Metric to optimize (defaults to first in list)
    n_jobs: int = -1
    random_state: int | None = Field(
        42, description="Random seed used by RandomizedSearchCV."
    )
    cv: CVConfig | None = Field(
        None,
        description=(
            "Inner CV used for model selection. Defaults to the outer CV family."
        ),
    )
    allow_nongroup_inner_cv: bool = Field(
        False,
        description=(
            "Allow a non-grouped tuning CV under grouped outer CV. This explicitly "
            "acknowledges the leakage/generalization trade-off."
        ),
    )


class FeatureSelectionConfig(BaseModel):
    """Feature selection settings."""

    enabled: bool = False
    method: Literal["k_best", "sfs"] = "sfs"
    n_features: int | None = Field(None, gt=0, description="Number of features.")
    direction: Literal["forward", "backward"] = "forward"
    tol: float | None = Field(
        None,
        description=(
            "SFS early-stopping tolerance. When set and n_features is None, the "
            "selector uses n_features_to_select='auto' and stops once the score "
            "stops improving by at least tol (plateau stop)."
        ),
    )
    cv: CVConfig | None = Field(
        None,
        description=(
            "Inner CV used by SequentialFeatureSelector. Defaults to tuning.cv "
            "when available, otherwise the outer CV family."
        ),
    )
    scoring: str | None = None
    allow_nongroup_inner_cv: bool = Field(
        False,
        description=(
            "Allow a non-grouped SFS CV under grouped outer CV. This explicitly "
            "acknowledges the leakage/generalization trade-off."
        ),
    )


class ReducerConfig(BaseModel):
    """Fold-local dimensionality reduction for classical decoding."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["pca"] = "pca"
    n_components: int | float | None = Field(
        None,
        description=(
            "PCA components. Integers select a fixed count; floats in (0, 1) "
            "select the explained-variance fraction."
        ),
    )
    whiten: bool = False
    svd_solver: Literal["auto", "full", "covariance_eigh", "arpack", "randomized"] = (
        "auto"
    )
    random_state: int | None = 42

    @model_validator(mode="after")
    def _validate_n_components(self) -> ReducerConfig:
        if isinstance(self.n_components, int) and self.n_components < 1:
            raise ValueError("Integer n_components must be at least 1.")
        if isinstance(self.n_components, float) and not 0 < self.n_components < 1:
            raise ValueError("Float n_components must be in the open interval (0, 1).")
        return self


class ErasureConfig(BaseModel):
    """Fold-local subject-nuisance removal for classical decoding."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["leace", "ea_coral", "ea_mean"] = "leace"
    params: dict[str, Any] = Field(default_factory=dict)


class CalibrationConfig(BaseModel):
    """Probability calibration settings for classification estimators."""

    enabled: bool = False
    method: Literal["sigmoid", "isotonic"] = "sigmoid"
    cv: CVConfig | None = Field(
        None,
        description=(
            "Inner CV used by CalibratedClassifierCV. Defaults to the outer CV family."
        ),
    )
    n_jobs: int | None = None
    allow_nongroup_inner_cv: bool = Field(
        False,
        description=("Allow a non-grouped calibration CV under grouped outer CV."),
    )


class ConfidenceIntervalConfig(BaseModel):
    """Analytical confidence interval settings."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    method: Literal["wilson", "clopper_pearson"] = "wilson"


class ChanceAssessmentConfig(BaseModel):
    """Null-hypothesis (chance level) assessment settings."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["permutation", "binomial", "auto"] = "permutation"
    n_permutations: int = Field(1000, ge=1)
    p0: float | Literal["auto"] | None = Field(
        "auto",
        description="Chance level for binomial test (e.g., 0.5 for binary).",
    )
    temporal_correction: Literal["max_stat", "fdr_bh", "none"] = Field(
        "max_stat", description="Method to correct for multiple comparisons."
    )
    store_null_distribution: bool = False


class StatisticalAssessmentConfig(BaseModel):
    """
    Settings for finite-sample statistical inference and uncertainty estimation.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    random_state: int | None = 42
    metrics: list[str] | None = Field(
        None, description="Subset of experiment metrics to run assessment for."
    )

    chance: ChanceAssessmentConfig = Field(default_factory=ChanceAssessmentConfig)
    confidence_intervals: ConfidenceIntervalConfig = Field(
        default_factory=ConfidenceIntervalConfig
    )

    unit_of_inference: (
        Literal["sample", "group_mean", "group_majority", "custom"] | None
    ) = Field(
        None,
        description="Independent unit for label permutation or binomial counts.",
    )
    custom_unit_column: str | None = None
    custom_aggregation: Literal["mean", "majority"] = "mean"


def grouped_chance_assessment(
    method: str = "permutation",
    *,
    metric: str = "accuracy",
    n_permutations: int = 100,
    store_null: bool = False,
    unit_column: str = "group_id",
) -> StatisticalAssessmentConfig:
    """Chance assessment whose independent unit is a group, not a sample.

    Keeps every sweep on an identical inferential contract: ``method="permutation"``
    builds an empirical null by shuffling labels *within groups* (via
    *unit_column*), so significance reflects the number of independent units
    (e.g. subjects) rather than the theoretical chance level; ``method="binomial"``
    falls back to the analytical chance level. Permutation refits the full pipeline
    per shuffle, so cost scales with *n_permutations*.
    """
    return StatisticalAssessmentConfig(
        enabled=True,
        metrics=[metric],
        chance=ChanceAssessmentConfig(
            method=method,
            n_permutations=n_permutations,
            temporal_correction="none",
            store_null_distribution=store_null,
        ),
        unit_of_inference="custom",
        custom_unit_column=unit_column,
    )


ModelConfigType = Annotated[
    ClassicalModelType
    | FoundationEmbeddingModelConfig
    | FrozenBackboneDecoderConfig
    | NeuralFineTuneConfig
    | TemporalDecoderConfig,
    Field(discriminator="kind"),
]


class ExperimentConfig(BaseModel):
    """
    Master configuration for a Decoding Experiment.

    This model serves as the single source of truth for an entire analysis,
    including data handling, model selection, hyperparameter tuning,
    feature selection, and statistical inference.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    task: MetricTask = "classification"
    output_dir: Path | None = None
    tag: str = "experiment"
    random_state: int | None = Field(
        None,
        description=(
            "Master random seed. If set, it is used to derive seeds for all "
            "components (CV, Tuning, Models, etc.) ensuring global reproducibility."
        ),
    )

    # Map of Friendly Name -> Polymorphic Config Object
    models: dict[str, ModelConfigType]

    # Map of Friendly Name -> Parameter Grid (Search Space)
    grids: dict[str, dict[str, list[Any]]] | None = None

    cv: CVConfig = Field(default_factory=CVConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig
    )
    erasure: ErasureConfig = Field(default_factory=ErasureConfig)
    reducer: ReducerConfig = Field(default_factory=ReducerConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    statistical_assessment: StatisticalAssessmentConfig = Field(
        default_factory=StatisticalAssessmentConfig,
        description=(
            "Statistical-assessment settings (chance-level tests, "
            "confidence intervals, bootstrap)."
        ),
    )

    metrics: list[str] = Field(
        default_factory=lambda: ["accuracy", "roc_auc"],
        description="List of metrics to compute.",
    )

    use_scaler: bool | str = Field(
        True, description="Whether to scalar normalize features upstream."
    )
    n_jobs: int = -1
    verbose: bool = True
    low_memory: bool = Field(
        False,
        description=(
            "If True, each model's cross-validation results are offloaded to a "
            "temporary directory on disk immediately after they are produced, "
            "and rehydrated only when the final ExperimentResult is assembled. "
            "This bounds peak host memory to roughly one model's results plus "
            "the currently-fitting model, rather than accumulating every "
            "model's fold predictions/splits in RAM for the whole sweep. "
            "Intended for memory-constrained runs (e.g. foundation models on "
            "SLURM); off by default so standard runs are unaffected."
        ),
    )

    def get_all_evaluation_metrics(self) -> list[str]:
        """Union of primary experiment metrics and stats-specific metrics."""
        primary = list(self.metrics)
        eval_metrics = self.statistical_assessment.metrics or []
        return sorted(set(primary + list(eval_metrics)))

    @model_validator(mode="after")
    def _validate_task_consistency(self) -> ExperimentConfig:
        """Ensure CV strategies and metrics match the selected task."""
        # 1. Validate Metrics
        from ._metrics import get_metric_names, get_metric_spec

        registered_metrics = get_metric_names()

        for metric in self.get_all_evaluation_metrics():
            if metric in registered_metrics:
                spec = get_metric_spec(metric)
                if spec.task != self.task:
                    raise ValueError(
                        f"Metric '{metric}' is for {spec.task} but experiment task "
                        f"is {self.task}."
                    )
            # Metrics not in the registry are assumed to be custom callables or
            # user-defined strings handled by the execution engine at runtime.

        # 2. Validate CV Strategy
        stratified_strategies = {"stratified", "stratified_group_kfold"}
        if self.task == "regression" and self.cv.strategy in stratified_strategies:
            raise ValueError(
                f"CV strategy '{self.cv.strategy}' is not valid for regression."
            )

        # 3. Validate Calibration task
        if self.calibration.enabled and self.task != "classification":
            raise ValueError("calibration is only available for classification.")

        # 4. Validate Tuning Metrics
        if (
            self.tuning.enabled
            and self.tuning.scoring
            and self.tuning.scoring in registered_metrics
        ):
            spec = get_metric_spec(self.tuning.scoring)
            if spec.task != self.task:
                raise ValueError(
                    f"Tuning metric '{self.tuning.scoring}' is for {spec.task} "
                    f"but task is {self.task}."
                )

        # 4. Validate Tuning CV
        if self.tuning.enabled and self.tuning.cv is None:
            if self.tuning.allow_nongroup_inner_cv:
                self.tuning.cv = self.cv
            else:
                raise ValueError(
                    "Tuning is enabled but tuning.cv is not defined. "
                    "You must explicitly define an inner CV strategy to ensure "
                    "scientific validity and acknowledge computational cost."
                )

        # 5. Validate FS CV (MANDATORY if SFS)
        if (
            self.feature_selection.enabled
            and self.feature_selection.method == "sfs"
            and self.feature_selection.cv is None
        ):
            if self.feature_selection.allow_nongroup_inner_cv:
                self.feature_selection.cv = self.cv
            else:
                raise ValueError(
                    "Sequential Feature Selection (SFS) is enabled but "
                    "feature_selection.cv is not defined."
                )

        # 6. Validate Calibration CV (MANDATORY)
        if self.calibration.enabled and self.calibration.cv is None:
            raise ValueError(
                "Calibration is enabled but calibration.cv is not defined."
            )

        return self
