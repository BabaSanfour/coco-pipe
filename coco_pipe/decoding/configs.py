"""
Decoding Configurations
=======================

Comprehensive Pydantic models for strict validation of Decoding/ML experiments.

Key Components:
- ModelConfigs: extensive hyperparameters for each estimator.
- ExperimentConfig: Top-level configuration for the entire analysis workflow.
"""

from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# --- Base Schemas ---


class BaseEstimatorConfig(BaseModel):
    """Base configuration for any estimator."""

    model_config = ConfigDict(extra="forbid")
    random_state: Optional[int] = Field(
        42, description="Random seed for reproducibility."
    )


# --- Mixins ---


class LinearMixin(BaseModel):
    """Common parameters for linear models."""

    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: Optional[int] = None


class RegularizedLinearMixin(LinearMixin):
    """Parameters for regularized linear models."""

    tol: float = 1e-3
    max_iter: Optional[int] = None
    solver: str = "auto"
    warm_start: bool = False
    positive: bool = False


class TreeMixin(BaseModel):
    """Common parameters for Tree-based models."""

    n_estimators: int = Field(100, ge=1)
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[str, int, float, None] = "sqrt"
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0
    n_jobs: Optional[int] = None
    verbose: int = 0
    warm_start: bool = False


class SupportVectorMixin(BaseModel):
    """Common parameters for Support Vector Machines."""

    C: float = Field(1.0, gt=0.0)
    kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf"
    degree: int = 3
    gamma: Union[str, float] = "scale"
    coef0: float = 0.0
    tol: float = 1e-3
    verbose: bool = False
    max_iter: int = -1
    shrinking: bool = True
    cache_size: float = 200


class SGDMixin(BaseModel):
    loss: str = "hinge"
    penalty: Literal["l2", "l1", "elasticnet", "null"] = "l2"
    alpha: float = 0.0001
    l1_ratio: float = 0.15
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float = 1e-3
    shuffle: bool = True
    verbose: int = 0
    epsilon: float = 0.1
    n_jobs: Optional[int] = None
    learning_rate: str = "optimal"
    eta0: float = 0.0
    power_t: float = 0.5
    early_stopping: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 5
    warm_start: bool = False
    average: bool = False


# --- Classifiers ---


class LogisticRegressionConfig(BaseEstimatorConfig):
    method: Literal["LogisticRegression"] = "LogisticRegression"
    penalty: Literal["l1", "l2", "elasticnet", "none", None] = "l2"
    dual: bool = False
    tol: float = 1e-4
    C: float = Field(1.0, gt=0.0)
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    class_weight: Optional[Union[Dict, str]] = None
    solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = "lbfgs"
    max_iter: int = 100
    multiclass: Literal["auto", "ovr", "multinomial"] = "auto"
    verbose: int = 0
    warm_start: bool = False
    n_jobs: Optional[int] = None
    l1_ratio: Optional[float] = None


class RandomForestClassifierConfig(BaseEstimatorConfig, TreeMixin):
    method: Literal["RandomForestClassifier"] = "RandomForestClassifier"
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    bootstrap: bool = True
    oob_score: bool = False
    class_weight: Optional[Union[str, Dict, List]] = None
    max_samples: Optional[Union[int, float]] = None


class SVCConfig(BaseEstimatorConfig, SupportVectorMixin):
    method: Literal["SVC"] = "SVC"
    probability: bool = True  # Default to True for metrics requiring proba
    class_weight: Optional[Union[Dict, str]] = None
    decision_function_shape: Literal["ovo", "ovr"] = "ovr"
    break_ties: bool = False


class KNeighborsClassifierConfig(BaseEstimatorConfig):
    method: Literal["KNeighborsClassifier"] = "KNeighborsClassifier"
    n_neighbors: int = Field(5, ge=1)
    weights: Literal["uniform", "distance"] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: str = "minkowski"
    metric_params: Optional[Dict] = None
    n_jobs: Optional[int] = None


class GradientBoostingClassifierConfig(BaseEstimatorConfig):
    method: Literal["GradientBoostingClassifier"] = "GradientBoostingClassifier"
    loss: Literal["log_loss", "exponential"] = "log_loss"
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 1.0
    criterion: Literal["friedman_mse", "squared_error"] = "friedman_mse"
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_depth: int = 3
    min_impurity_decrease: float = 0.0
    init: Optional[str] = None
    max_features: Union[str, int, float, None] = None
    verbose: int = 0
    max_leaf_nodes: Optional[int] = None
    warm_start: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: Optional[int] = None
    tol: float = 1e-4
    ccp_alpha: float = 0.0


class SGDClassifierConfig(BaseEstimatorConfig, SGDMixin):
    method: Literal["SGDClassifier"] = "SGDClassifier"
    class_weight: Optional[Union[Dict, str]] = None


class MLPClassifierConfig(BaseEstimatorConfig):
    method: Literal["MLPClassifier"] = "MLPClassifier"
    hidden_layer_sizes: tuple = (100,)
    activation: Literal["identity", "logistic", "tanh", "relu"] = "relu"
    solver: Literal["lbfgs", "sgd", "adam"] = "adam"
    alpha: float = 0.0001
    batch_size: Union[int, str] = "auto"
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


class GaussianNBConfig(BaseEstimatorConfig):
    method: Literal["GaussianNB"] = "GaussianNB"
    priors: Optional[List[float]] = None
    var_smoothing: float = 1e-9


class LDAConfig(BaseEstimatorConfig):
    method: Literal["LinearDiscriminantAnalysis"] = "LinearDiscriminantAnalysis"
    solver: Literal["svd", "lsqr", "eigen"] = "svd"
    shrinkage: Optional[Union[str, float]] = None
    priors: Optional[List[float]] = None
    n_components: Optional[int] = None
    store_covariance: bool = False
    tol: float = 1e-4


class AdaBoostClassifierConfig(BaseEstimatorConfig):
    method: Literal["AdaBoostClassifier"] = "AdaBoostClassifier"
    n_estimators: int = 50
    learning_rate: float = 1.0
    algorithm: Literal["SAMME", "SAMME.R"] = "SAMME.R"


class DummyClassifierConfig(BaseEstimatorConfig):
    method: Literal["DummyClassifier"] = "DummyClassifier"
    strategy: Literal["stratified", "most_frequent", "prior", "uniform"] = "prior"
    constant: Optional[Any] = None


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
    Configuration for MNE's SlidingEstimator.
    Fits a separate estimator for each time point.
    """

    method: Literal["SlidingEstimator"] = "SlidingEstimator"
    base_estimator: "EstimatorConfigType"
    scoring: Optional[Union[str, Callable]] = None
    n_jobs: Optional[int] = 1
    position: Optional[float] = 0
    allow_2d: bool = False
    verbose: Optional[Union[bool, str, int]] = None


class GeneralizingEstimatorConfig(BaseEstimatorConfig):
    """
    Configuration for MNE's GeneralizingEstimator.
    Fits an estimator on each time point and tests on all other time points.
    """

    method: Literal["GeneralizingEstimator"] = "GeneralizingEstimator"
    base_estimator: "EstimatorConfigType"
    scoring: Optional[Union[str, Callable]] = None
    n_jobs: Optional[int] = 1
    position: Optional[float] = 0
    allow_2d: bool = False
    verbose: Optional[Union[bool, str, int]] = None


# --- Regressors ---


class LinearRegressionConfig(BaseEstimatorConfig, LinearMixin):
    method: Literal["LinearRegression"] = "LinearRegression"
    positive: bool = False


class RidgeConfig(BaseEstimatorConfig, RegularizedLinearMixin):
    method: Literal["Ridge"] = "Ridge"
    alpha: float = 1.0
    fit_intercept: bool = True
    copy_X: bool = True


class LassoConfig(BaseEstimatorConfig, RegularizedLinearMixin):
    method: Literal["Lasso"] = "Lasso"
    alpha: float = 1.0
    precompute: Union[bool, List] = False
    fit_intercept: bool = True
    copy_X: bool = True
    selection: Literal["cyclic", "random"] = "cyclic"


class ElasticNetConfig(BaseEstimatorConfig, RegularizedLinearMixin):
    method: Literal["ElasticNet"] = "ElasticNet"
    alpha: float = 1.0
    l1_ratio: float = 0.5
    precompute: Union[bool, List] = False
    fit_intercept: bool = True
    copy_X: bool = True
    selection: Literal["cyclic", "random"] = "cyclic"


class RandomForestRegressorConfig(BaseEstimatorConfig, TreeMixin):
    method: Literal["RandomForestRegressor"] = "RandomForestRegressor"
    criterion: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"] = (
        "squared_error"
    )
    bootstrap: bool = True
    oob_score: bool = False
    max_samples: Optional[Union[int, float]] = None


class SVRConfig(BaseEstimatorConfig, SupportVectorMixin):
    method: Literal["SVR"] = "SVR"
    epsilon: float = 0.1


class GradientBoostingRegressorConfig(BaseEstimatorConfig):
    method: Literal["GradientBoostingRegressor"] = "GradientBoostingRegressor"
    loss: Literal["squared_error", "absolute_error", "huber", "quantile"] = (
        "squared_error"
    )
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 1.0
    criterion: Literal["friedman_mse", "squared_error"] = "friedman_mse"
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_depth: int = 3
    min_impurity_decrease: float = 0.0
    init: Optional[str] = None
    max_features: Union[str, int, float, None] = None
    alpha: float = 0.9
    verbose: int = 0
    max_leaf_nodes: Optional[int] = None
    warm_start: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: Optional[int] = None
    tol: float = 1e-4
    ccp_alpha: float = 0.0


class SGDRegressorConfig(BaseEstimatorConfig, SGDMixin):
    method: Literal["SGDRegressor"] = "SGDRegressor"
    loss: str = "squared_error"


class MLPRegressorConfig(BaseEstimatorConfig):
    method: Literal["MLPRegressor"] = "MLPRegressor"
    hidden_layer_sizes: tuple = (100,)
    activation: Literal["identity", "logistic", "tanh", "relu"] = "relu"
    alpha: float = 0.0001
    batch_size: Union[int, str] = "auto"
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


class DummyRegressorConfig(BaseEstimatorConfig):
    method: Literal["DummyRegressor"] = "DummyRegressor"
    strategy: Literal["mean", "median", "quantile", "constant"] = "mean"
    constant: Optional[Union[int, float, List]] = None
    quantile: Optional[float] = None


class DecisionTreeRegressorConfig(BaseEstimatorConfig):
    method: Literal["DecisionTreeRegressor"] = "DecisionTreeRegressor"
    criterion: Literal["squared_error", "friedman_mse", "absolute_error", "poisson"] = (
        "squared_error"
    )
    splitter: Literal["best", "random"] = "best"
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[str, int, float, None] = None
    random_state: Optional[int] = None
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0


class KNeighborsRegressorConfig(BaseEstimatorConfig):
    method: Literal["KNeighborsRegressor"] = "KNeighborsRegressor"
    n_neighbors: int = Field(5, ge=1)
    weights: Literal["uniform", "distance"] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: str = "minkowski"
    metric_params: Optional[Dict] = None
    n_jobs: Optional[int] = None


class ExtraTreesRegressorConfig(BaseEstimatorConfig, TreeMixin):
    method: Literal["ExtraTreesRegressor"] = "ExtraTreesRegressor"
    bootstrap: bool = False
    oob_score: bool = False
    max_samples: Optional[Union[int, float]] = None


class HistGradientBoostingRegressorConfig(BaseEstimatorConfig):
    method: Literal["HistGradientBoostingRegressor"] = "HistGradientBoostingRegressor"
    loss: Literal["squared_error", "absolute_error", "poisson", "quantile"] = (
        "squared_error"
    )
    learning_rate: float = 0.1
    max_iter: int = 100
    max_leaf_nodes: int = 31
    max_depth: Optional[int] = None
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    max_bins: int = 255
    categorical_features: Optional[Union[List[int], List[str], List[bool]]] = None
    monotonic_cst: Optional[Any] = None
    interaction_cst: Optional[Any] = None
    warm_start: bool = False
    early_stopping: str = "auto"
    scoring: Optional[str] = "loss"
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    tol: float = 1e-7
    verbose: int = 0
    random_state: Optional[int] = None


class AdaBoostRegressorConfig(BaseEstimatorConfig):
    method: Literal["AdaBoostRegressor"] = "AdaBoostRegressor"
    n_estimators: int = 50
    learning_rate: float = 1.0
    loss: Literal["linear", "square", "exponential"] = "linear"


class BayesianRidgeConfig(BaseEstimatorConfig):
    method: Literal["BayesianRidge"] = "BayesianRidge"
    n_iter: int = 300
    tol: float = 1e-3
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    alpha_init: Optional[float] = None
    lambda_init: Optional[float] = None
    compute_score: bool = False
    fit_intercept: bool = True
    copy_X: bool = True
    verbose: bool = False


class ARDRegressionConfig(BaseEstimatorConfig):
    method: Literal["ARDRegression"] = "ARDRegression"
    n_iter: int = 300
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


# 1. Define Atomic Unions (Terminal nodes)
AtomicEstimator = Union[
    LogisticRegressionConfig,
    RandomForestClassifierConfig,
    SVCConfig,
    KNeighborsClassifierConfig,
    GradientBoostingClassifierConfig,
    SGDClassifierConfig,
    MLPClassifierConfig,
    GaussianNBConfig,
    LDAConfig,
    AdaBoostClassifierConfig,
    DummyClassifierConfig,
    LPFTConfig,
    SkorchClassifierConfig,
    # Regressors
    LinearRegressionConfig,
    RidgeConfig,
    LassoConfig,
    ElasticNetConfig,
    RandomForestRegressorConfig,
    SVRConfig,
    GradientBoostingRegressorConfig,
    SGDRegressorConfig,
    MLPRegressorConfig,
    DummyRegressorConfig,
    DecisionTreeRegressorConfig,
    KNeighborsRegressorConfig,
    ExtraTreesRegressorConfig,
    HistGradientBoostingRegressorConfig,
    AdaBoostRegressorConfig,
    BayesianRidgeConfig,
    ARDRegressionConfig,
]

# 2. Define the Recursive Union using Annotated Discriminator
# This allows Pydantic to choose the correct class based on 'method' field
EstimatorConfigType = Annotated[
    Union[AtomicEstimator, SlidingEstimatorConfig, GeneralizingEstimatorConfig],
    Field(discriminator="method"),
]


# --- Experiment Config ---


class TemporalConfig(BaseModel):
    """Configuration for temporal decoding (Sliding/Generalizing)."""

    enabled: bool = False
    window_interaction: Literal["sliding", "generalizing"] = "sliding"


class CVConfig(BaseModel):
    """Cross-validation settings."""

    strategy: Literal[
        "stratified",
        "kfold",
        "group_kfold",
        "stratified_group_kfold",
        "leave_p_out",
        "leave_one_out",
        "timeseries",
        "split",
    ] = "stratified"
    n_splits: int = Field(5, ge=2)
    shuffle: bool = True
    random_state: int = 42


class TuningConfig(BaseModel):
    """
    Hyperparameter Tuning Configuration.
    Use this to define HOW to search (random vs grid).
    The WHAT (the grid itself) is passed in ExperimentConfig.grids.
    """

    enabled: bool = False
    search_type: Literal["grid", "random"] = "grid"
    n_iter: int = Field(10, description="Number of iterations for random search")
    scoring: Optional[str] = None  # Metric to optimize (defaults to first in list)
    n_jobs: int = -1


class FeatureSelectionConfig(BaseModel):
    """Configuration for Sequential Feature Selection."""

    enabled: bool = False
    method: Literal["k_best", "sfs"] = "sfs"
    n_features: Optional[int] = Field(None, description="Number of features to select.")
    direction: Literal["forward", "backward"] = "forward"
    cv: Optional[int] = Field(None, description="Inner CV splits. Required for SFS.")
    scoring: Optional[str] = None


class ExperimentConfig(BaseModel):
    """
    Master configuration for a Decoding Experiment.
    """

    task: Literal["classification", "regression"] = "classification"
    output_dir: Optional[Path] = None
    tag: str = "experiment"

    # Map of Friendly Name -> Polymorphic Config Object
    models: Dict[str, EstimatorConfigType]

    # Map of Friendly Name -> Parameter Grid (Search Space)
    grids: Optional[Dict[str, Dict[str, List[Any]]]] = None

    cv: CVConfig = Field(default_factory=CVConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig
    )

    metrics: List[str] = Field(
        default_factory=lambda: ["accuracy", "roc_auc"],
        description="List of metrics to compute.",
    )

    temporal: TemporalConfig = Field(default_factory=TemporalConfig)

    use_scaler: bool = Field(
        True, description="Whether to scalar normalize features upstream."
    )
    n_jobs: int = -1
    verbose: bool = True
