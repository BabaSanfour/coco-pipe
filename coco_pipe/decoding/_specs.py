"""
Estimator Specifications and Capability Metadata
===============================================

Internal module containing the static database of estimator metadata
and the dataclasses used to represent them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ._constants import (
    CalibrationSupport,
    DependencyGroup,
    EstimatorFamily,
    FeatureSelectionSupport,
    GroupedMetadata,
    ImportanceSupport,
    InputKind,
    InputRank,
    MetricTask,
    PredictionInterface,
    TemporalSupport,
)


@dataclass
class SignalMetadata:
    """Per-session signal properties bound at fit() time.

    Validated against model expectations in ``transform`` and ``predict``.
    """

    sfreq: float
    ch_names: list[str]


@dataclass(frozen=True)
class EstimatorCapabilities:
    """Machine-readable capabilities for a decoding estimator."""

    method: str
    tasks: tuple[MetricTask, ...]
    input_ranks: tuple[InputRank, ...] = ("2d",)
    prediction_interfaces: tuple[PredictionInterface, ...] = ("predict",)
    grouped_metadata: tuple[GroupedMetadata, ...] = ("none",)
    feature_selection: tuple[FeatureSelectionSupport, ...] = ("univariate", "sfs")
    calibration: CalibrationSupport = "eligible"
    importance: tuple[ImportanceSupport, ...] = ("unavailable",)
    temporal: TemporalSupport = "none"
    dependencies: tuple[DependencyGroup, ...] = ("core",)

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-friendly capability dictionary.

        This ensures that the capability metadata can be safely serialized
        for reporting or API responses.

        Returns
        -------
        caps : dict[str, Any]
            The capability dictionary.
        """
        return asdict(self)

    def supports_task(self, task: str) -> bool:
        """
        Check if the estimator supports the given task type.

        Parameters
        ----------
        task : str
            The task type to check (e.g., "classification").

        Returns
        -------
        available : bool
            True if the task is supported.
        """
        return task in self.tasks

    def has_response(self, response: str) -> bool:
        """
        Check if the estimator supports the given prediction interface.

        Parameters
        ----------
        response : str
            The interface to check (e.g., "predict_proba").

        Returns
        -------
        available : bool
            True if the response interface is available.
        """
        return response in self.prediction_interfaces


@dataclass(frozen=True)
class EstimatorSpec:
    """Typed registry entry for a decoding estimator."""

    name: str
    import_path: str
    family: EstimatorFamily
    task: tuple[MetricTask, ...]
    input_kinds: tuple[InputKind, ...] = ("tabular_2d",)
    supports_groups: bool = False
    supports_proba: bool = False
    supports_decision_function: bool = False
    supports_calibration: bool = True
    supports_feature_names: bool = True
    dependency_extra: DependencyGroup = "core"
    fit_smoke_required: bool = True
    default_search_space: dict[str, list[Any]] = field(default_factory=dict)
    feature_selection: tuple[FeatureSelectionSupport, ...] = ("univariate", "sfs")
    importance: tuple[ImportanceSupport, ...] = ("unavailable",)
    """Type of feature importance available (coefficients, importance)."""

    temporal: TemporalSupport = "none"
    """Temporal decoding wrapper type (sliding or generalizing)."""

    calibration: CalibrationSupport = "eligible"
    """Whether the model is eligible for probability calibration."""

    supports_random_state: bool = False
    """Whether the estimator class accepts a random_state parameter."""

    is_sparse_capable: bool = False
    """Whether the model can produce sparse coefficients (e.g. L1 regularization)."""

    @property
    def module_path(self) -> str:
        """
        Return the module part of the import path.

        Returns
        -------
        path : str
            The module path (e.g., "sklearn.linear_model").
        """
        return self.import_path.split(":")[0]

    @property
    def class_name(self) -> str:
        """
        Return the class name to be imported.

        Returns
        -------
        name : str
            The class name (e.g., "LogisticRegression").
        """
        if ":" in self.import_path:
            return self.import_path.split(":", 1)[1]
        return self.name

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-friendly spec dictionary.

        Returns
        -------
        spec : dict[str, Any]
            The full specification as a dictionary.
        """
        return asdict(self)

    def to_capabilities(self) -> EstimatorCapabilities:
        """
        Derive lightweight capability metadata from this spec.

        This conversion distills the full specification into a format
        used by the engine for runtime validation and capability-based
        routing.

        Returns
        -------
        caps : EstimatorCapabilities
            The derived capability metadata.

        See Also
        --------
        EstimatorCapabilities : The destination capability container.
        """
        responses = ["predict"]
        if self.supports_proba:
            responses.append("predict_proba")
        if self.supports_decision_function:
            responses.append("decision_function")

        input_ranks = []
        for kind in self.input_kinds:
            if kind in {"temporal_3d", "epoched"}:
                rank = "3d_temporal"
            elif kind == "tokens":
                rank = "tokens"
            else:
                rank = "2d"
            if rank not in input_ranks:
                input_ranks.append(rank)

        return EstimatorCapabilities(
            method=self.name,
            tasks=self.task,
            input_ranks=tuple(input_ranks),
            prediction_interfaces=tuple(responses),
            grouped_metadata=("search_cv",) if self.supports_groups else ("none",),
            feature_selection=self.feature_selection,
            calibration=self.calibration,
            importance=self.importance,
            temporal=self.temporal,
            dependencies=(self.dependency_extra,),
        )


@dataclass(frozen=True)
class SelectorCapabilities:
    """Machine-readable capabilities for a decoding feature selector."""

    method: str
    input_ranks: tuple[InputRank, ...]
    support: tuple[FeatureSelectionSupport, ...]
    grouped_metadata: tuple[GroupedMetadata, ...] = ("none",)

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-friendly dictionary representation.

        Returns
        -------
        caps : dict[str, Any]
            The selector capabilities dictionary.
        """
        return asdict(self)


# Shared Task Tuples
_CLASSIFICATION = ("classification",)
_REGRESSION = ("regression",)
_BOTH_TASKS = ("classification", "regression")

# Shared Importance Tuples
_COEF = ("coefficients",)
_TREE_IMPORTANCE = ("feature_importances",)


@dataclass(frozen=True, kw_only=True)
class FoundationModelSpec(EstimatorSpec):
    """Registry entry for an EEG/MEG foundation model estimator.

    Extends EstimatorSpec with model-loading metadata (hub repo, embedding
    dimensions, pretrained signal properties, backend routing).
    """

    hub_repo: str
    embedding_dim: int
    pretrained_sfreq: float
    preferred_backend: str
    display_name: str = ""
    checkpoint_revision: str | None = None
    checkpoint_filename: str | None = None
    requires_auth: bool = False
    pretrained_n_chans: int | None = None
    pretrained_n_times: int | None = None
    pretrained_ch_names: list[str] | None = None
    supports_channel_interpolation: bool = False
    supported_train_modes: tuple[str, ...] = ("frozen", "full", "lora")
    fallback_backends: tuple[str, ...] = ()
    paper_url: str | None = None
    model_notes: str = ""

    @property
    def pretrained_window_seconds(self) -> float | None:
        """Window length in seconds the checkpoint was pretrained on.

        ``None`` when ``pretrained_n_times`` is unknown. Single source of truth
        for "expected window duration" so preflight and loading agree.
        """
        if self.pretrained_n_times is None:
            return None
        return self.pretrained_n_times / self.pretrained_sfreq


_FM_DEFAULTS: dict[str, Any] = dict(
    import_path="coco_pipe.decoding.foundation_models:BackendBase",
    family="foundation",
    task=("classification", "regression"),
    input_kinds=("epoched",),
    supports_calibration=False,
    fit_smoke_required=False,
    feature_selection=("disabled",),
)


def _fm_spec(name: str, **kwargs: Any) -> FoundationModelSpec:
    return FoundationModelSpec(name=name, **_FM_DEFAULTS, **kwargs)


def _spec(
    name: str,
    import_path: str,
    family: EstimatorFamily,
    task: tuple[MetricTask, ...],
    **kwargs: Any,
) -> EstimatorSpec:
    return EstimatorSpec(
        name=name, import_path=import_path, family=family, task=task, **kwargs
    )


ESTIMATOR_SPECS: dict[str, EstimatorSpec] = {
    # --- Classifiers ---
    "LogisticRegression": _spec(
        "LogisticRegression",
        "sklearn.linear_model",
        "linear",
        _CLASSIFICATION,
        supports_proba=True,
        supports_decision_function=True,
        importance=_COEF,
        supports_random_state=True,
        is_sparse_capable=True,
        default_search_space={"C": [0.1, 1.0, 10.0]},
    ),
    "RandomForestClassifier": _spec(
        "RandomForestClassifier",
        "sklearn.ensemble",
        "ensemble",
        _CLASSIFICATION,
        supports_proba=True,
        importance=_TREE_IMPORTANCE,
        supports_random_state=True,
        default_search_space={
            "n_estimators": [100, 300],
            "max_depth": [None, 5, 10],
        },
    ),
    "ExtraTreesClassifier": _spec(
        "ExtraTreesClassifier",
        "sklearn.ensemble",
        "ensemble",
        _CLASSIFICATION,
        supports_proba=True,
        importance=_TREE_IMPORTANCE,
        supports_random_state=True,
        default_search_space={
            "n_estimators": [100, 300],
            "max_depth": [None, 5, 10],
        },
    ),
    "SVC": _spec(
        "SVC",
        "sklearn.svm",
        "svm",
        _CLASSIFICATION,
        supports_proba=True,
        supports_decision_function=True,
        supports_random_state=True,
        default_search_space={"C": [0.1, 1.0, 10.0]},
    ),
    "LinearSVC": _spec(
        "LinearSVC",
        "sklearn.svm",
        "svm",
        _CLASSIFICATION,
        supports_proba=False,
        supports_decision_function=True,
        importance=_COEF,
        supports_random_state=True,
        is_sparse_capable=True,
        default_search_space={"C": [0.1, 1.0, 10.0]},
    ),
    "KNeighborsClassifier": _spec(
        "KNeighborsClassifier",
        "sklearn.neighbors",
        "neighbors",
        _CLASSIFICATION,
        supports_proba=True,
        supports_random_state=False,
        default_search_space={"n_neighbors": [3, 5, 7]},
    ),
    "GradientBoostingClassifier": _spec(
        "GradientBoostingClassifier",
        "sklearn.ensemble",
        "ensemble",
        _CLASSIFICATION,
        supports_proba=True,
        importance=_TREE_IMPORTANCE,
        supports_random_state=True,
        default_search_space={"n_estimators": [100, 300], "learning_rate": [0.03, 0.1]},
    ),
    "HistGradientBoostingClassifier": _spec(
        "HistGradientBoostingClassifier",
        "sklearn.ensemble",
        "ensemble",
        _CLASSIFICATION,
        supports_proba=True,
        supports_random_state=True,
        default_search_space={"max_iter": [100, 300], "learning_rate": [0.03, 0.1]},
    ),
    "SGDClassifier": _spec(
        "SGDClassifier",
        "sklearn.linear_model",
        "linear",
        _CLASSIFICATION,
        supports_decision_function=True,
        importance=_COEF,
        supports_random_state=True,
        is_sparse_capable=True,
        default_search_space={"alpha": [0.0001, 0.001, 0.01]},
    ),
    "MLPClassifier": _spec(
        "MLPClassifier",
        "sklearn.neural_network",
        "neural",
        _CLASSIFICATION,
        supports_proba=True,
        supports_random_state=True,
        default_search_space={"alpha": [0.0001, 0.001]},
    ),
    "GaussianNB": _spec(
        "GaussianNB",
        "sklearn.naive_bayes",
        "bayes",
        _CLASSIFICATION,
        supports_proba=True,
        calibration="eligible",
        supports_random_state=False,
        # Note: GaussianNB is probabilistic but benefits from calibration
        # when priors are misspecified or features are non-independent.
        default_search_space={"var_smoothing": [1e-9, 1e-8, 1e-7]},
    ),
    "LinearDiscriminantAnalysis": _spec(
        "LinearDiscriminantAnalysis",
        "sklearn.discriminant_analysis",
        "linear",
        _CLASSIFICATION,
        supports_proba=True,
        importance=_COEF,
        supports_random_state=False,
    ),
    "AdaBoostClassifier": _spec(
        "AdaBoostClassifier",
        "sklearn.ensemble",
        "ensemble",
        _CLASSIFICATION,
        supports_proba=True,
        importance=_TREE_IMPORTANCE,
        supports_random_state=True,
        default_search_space={"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
    ),
    "DummyClassifier": _spec(
        "DummyClassifier",
        "sklearn.dummy",
        "dummy",
        _CLASSIFICATION,
        supports_proba=True,
        supports_calibration=False,
        calibration="unsupported",
    ),
    # --- Regressors ---
    "LinearRegression": _spec(
        "LinearRegression",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        supports_random_state=False,
    ),
    "Ridge": _spec(
        "Ridge",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        supports_random_state=True,
        default_search_space={"alpha": [0.1, 1.0, 10.0]},
    ),
    "Lasso": _spec(
        "Lasso",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        supports_random_state=True,
        is_sparse_capable=True,
        default_search_space={"alpha": [0.001, 0.01, 0.1, 1.0]},
    ),
    "ElasticNet": _spec(
        "ElasticNet",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        supports_random_state=True,
        is_sparse_capable=True,
        default_search_space={"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]},
    ),
    "RandomForestRegressor": _spec(
        "RandomForestRegressor",
        "sklearn.ensemble",
        "ensemble",
        _REGRESSION,
        importance=_TREE_IMPORTANCE,
        supports_random_state=True,
        default_search_space={
            "n_estimators": [100, 300],
            "max_depth": [None, 5, 10],
        },
    ),
    "ExtraTreesRegressor": _spec(
        "ExtraTreesRegressor",
        "sklearn.ensemble",
        "ensemble",
        _REGRESSION,
        importance=_TREE_IMPORTANCE,
        supports_random_state=True,
        default_search_space={
            "n_estimators": [100, 300],
            "max_depth": [None, 5, 10],
        },
    ),
    "SVR": _spec(
        "SVR",
        "sklearn.svm",
        "svm",
        _REGRESSION,
        supports_random_state=False,
        default_search_space={"C": [0.1, 1.0, 10.0]},
    ),
    "GradientBoostingRegressor": _spec(
        "GradientBoostingRegressor",
        "sklearn.ensemble",
        "ensemble",
        _REGRESSION,
        importance=_TREE_IMPORTANCE,
        supports_random_state=True,
        default_search_space={"n_estimators": [100, 300], "learning_rate": [0.03, 0.1]},
    ),
    "HistGradientBoostingRegressor": _spec(
        "HistGradientBoostingRegressor",
        "sklearn.ensemble",
        "ensemble",
        _REGRESSION,
        supports_random_state=True,
        default_search_space={"max_iter": [100, 300], "learning_rate": [0.03, 0.1]},
    ),
    "SGDRegressor": _spec(
        "SGDRegressor",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        supports_random_state=True,
        default_search_space={"alpha": [0.0001, 0.001, 0.01]},
    ),
    "MLPRegressor": _spec(
        "MLPRegressor",
        "sklearn.neural_network",
        "neural",
        _REGRESSION,
        supports_random_state=True,
        default_search_space={"alpha": [0.0001, 0.001]},
    ),
    "DummyRegressor": _spec(
        "DummyRegressor",
        "sklearn.dummy",
        "dummy",
        _REGRESSION,
        supports_calibration=False,
        calibration="unsupported",
    ),
    "DecisionTreeRegressor": _spec(
        "DecisionTreeRegressor",
        "sklearn.tree",
        "tree",
        _REGRESSION,
        importance=_TREE_IMPORTANCE,
        supports_random_state=True,
        default_search_space={"max_depth": [None, 5, 10]},
    ),
    "KNeighborsRegressor": _spec(
        "KNeighborsRegressor",
        "sklearn.neighbors",
        "neighbors",
        _REGRESSION,
        supports_random_state=False,
        default_search_space={"n_neighbors": [3, 5, 7]},
    ),
    "AdaBoostRegressor": _spec(
        "AdaBoostRegressor",
        "sklearn.ensemble",
        "ensemble",
        _REGRESSION,
        importance=_TREE_IMPORTANCE,
        supports_random_state=True,
        default_search_space={"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
    ),
    "BayesianRidge": _spec(
        "BayesianRidge",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        supports_random_state=False,
        default_search_space={"alpha_1": [1e-7, 1e-6]},
    ),
    "ARDRegression": _spec(
        "ARDRegression",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        supports_random_state=False,
        default_search_space={"alpha_1": [1e-7, 1e-6]},
    ),
    # --- Custom Wrappers ---
    "SlidingEstimator": _spec(
        "SlidingEstimator",
        "mne.decoding",
        "temporal",
        _BOTH_TASKS,
        input_kinds=("temporal_3d",),
        dependency_extra="mne",
        fit_smoke_required=False,
        feature_selection=("disabled",),
        temporal="sliding",
    ),
    "GeneralizingEstimator": _spec(
        "GeneralizingEstimator",
        "mne.decoding",
        "temporal",
        _BOTH_TASKS,
        input_kinds=("temporal_3d",),
        dependency_extra="mne",
        fit_smoke_required=False,
        feature_selection=("disabled",),
        temporal="generalizing",
    ),
    # --- Foundation Models ---
    "reve": _fm_spec(
        "reve",
        display_name="REVE",
        hub_repo="brain-bzh/reve-base",
        embedding_dim=512,
        pretrained_sfreq=200.0,
        checkpoint_revision="fa9a2163a4b7c0a42c8e28b56077ef9c368944dc",
        requires_auth=True,
        preferred_backend="hugging_face",
        supported_train_modes=("frozen", "full", "lora", "qlora"),
        dependency_extra="transformers",
        model_notes="Requires ch_names in SignalMetadata for positional encoding.",
    ),
    "cbramod": _fm_spec(
        "cbramod",
        display_name="CBraMod",
        hub_repo="braindecode/cbramod-pretrained",
        embedding_dim=200,
        pretrained_sfreq=200.0,
        checkpoint_revision="584cdc415913739a05d84bf0c1cb3db397764507",
        preferred_backend="braindecode",
        dependency_extra="braindecode",
        paper_url="https://arxiv.org/abs/2412.07236",
        model_notes=(
            "Pretrained at 200 Hz. Time dimension must be divisible by "
            "patch_size (default 200 samples = 1 s at 200 Hz)."
        ),
    ),
    "biot": _fm_spec(
        "biot",
        display_name="BIOT",
        hub_repo="braindecode/biot-pretrained-prest-16chs",
        embedding_dim=256,
        pretrained_sfreq=200.0,
        pretrained_n_chans=16,
        supports_channel_interpolation=True,
        preferred_backend="braindecode",
        dependency_extra="braindecode",
        model_notes=(
            "16-channel pretrained. "
            "Channel interpolation available with braindecode>=1.5."
        ),
    ),
    "labram": _fm_spec(
        "labram",
        display_name="LaBraM",
        hub_repo="braindecode/labram-pretrained",
        embedding_dim=200,
        pretrained_sfreq=200.0,
        checkpoint_revision="0563b6c626e7b40d9a36653b763715db94d945d7",
        pretrained_n_chans=128,
        pretrained_n_times=3000,
        supports_channel_interpolation=True,
        preferred_backend="braindecode",
        dependency_extra="braindecode",
        model_notes=(
            "Requires 128 channels in LABRAM_CHANNEL_ORDER. "
            "Use InterpolatedLaBraM (braindecode>=1.5) for arbitrary channel sets."
        ),
    ),
    "luna": _fm_spec(
        "luna",
        display_name="LUNA",
        hub_repo="PulpBio/LUNA",
        embedding_dim=256,
        pretrained_sfreq=200.0,
        checkpoint_revision="999c1af0fd6fbfb43ed47169c61fe9faa49cfe9f",
        checkpoint_filename="LUNA_base.safetensors",
        preferred_backend="braindecode",
        dependency_extra="braindecode",
        supported_train_modes=("frozen", "full", "lora"),
        paper_url="https://openreview.net/forum?id=uazfjnFL0G",
        model_notes=(
            "Verified base variant: embed_dim=64, num_queries=4, depth=8; "
            "pooled latent width is 256. Supports variable channel sets."
        ),
    ),
    "eegpt": _fm_spec(
        "eegpt",
        display_name="EEGPT",
        hub_repo="braindecode/eegpt-pretrained",
        embedding_dim=512,
        pretrained_sfreq=250.0,
        pretrained_n_chans=62,
        preferred_backend="braindecode",
        dependency_extra="braindecode",
        model_notes="Pretrained on 62-channel 250 Hz data.",
    ),
    "signaljepa": _fm_spec(
        "signaljepa",
        display_name="SignalJEPA",
        hub_repo="braindecode/signaljepa-pretrained",
        embedding_dim=256,
        pretrained_sfreq=200.0,
        pretrained_n_chans=62,
        supports_channel_interpolation=True,
        preferred_backend="braindecode",
        dependency_extra="braindecode",
        model_notes=(
            "62-channel pretrained. "
            "Channel interpolation available with braindecode>=1.5."
        ),
    ),
    "bendr": _fm_spec(
        "bendr",
        display_name="BENDR",
        hub_repo="braindecode/braindecode-bendr",
        embedding_dim=512,
        pretrained_sfreq=256.0,
        pretrained_n_chans=20,
        preferred_backend="braindecode",
        dependency_extra="braindecode",
        model_notes=(
            "20-channel 256 Hz pretrained. "
            "API changed in braindecode>=1.5; "
            "pin braindecode>=1.4,<1.6 if BENDR breaks."
        ),
    ),
}

SELECTOR_CAPABILITIES: dict[str, SelectorCapabilities] = {
    "k_best": SelectorCapabilities(
        "k_best", input_ranks=("2d",), support=("univariate",)
    ),
    "sfs": SelectorCapabilities(
        "sfs",
        input_ranks=("2d",),
        support=("sfs",),
        grouped_metadata=("sfs_metadata_routing",),
    ),
}


def canonical_estimator_name(name: str) -> str:
    """
    Map common model aliases to their canonical registry names.

    This ensures that user-friendly strings like 'lda' or 'logistic_regression'
    resolve correctly to the internal 'LinearDiscriminantAnalysis' and
    'LogisticRegression' specifications.

    Parameters
    ----------
    name : str
        The input name or alias (e.g., 'lda', 'logistic_regression').

    Returns
    -------
    str
        The canonical name used in the ESTIMATOR_SPECS registry.

    Examples
    --------
    >>> canonical_estimator_name("lda")
    'LinearDiscriminantAnalysis'

    See Also
    --------
    ESTIMATOR_SPECS : The registry of estimator specifications.
    """
    aliases = {
        "logistic_regression": "LogisticRegression",
        "random_forest_classifier": "RandomForestClassifier",
        "extra_trees_classifier": "ExtraTreesClassifier",
        "linear_svc": "LinearSVC",
        "lda": "LinearDiscriminantAnalysis",
        "dummy_classifier": "DummyClassifier",
        "ridge": "Ridge",
        "random_forest_regressor": "RandomForestRegressor",
        "extra_trees_regressor": "ExtraTreesRegressor",
        "hist_gradient_boosting_classifier": "HistGradientBoostingClassifier",
        "hist_gradient_boosting_regressor": "HistGradientBoostingRegressor",
    }
    return aliases.get(name, name)
