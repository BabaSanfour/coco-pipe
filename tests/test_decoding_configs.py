import pytest
from pydantic import ValidationError

from coco_pipe.decoding import Experiment
from coco_pipe.decoding.configs import (
    AdaBoostClassifierConfig,
    AdaBoostRegressorConfig,
    ARDRegressionConfig,
    BayesianRidgeConfig,
    CalibrationConfig,
    ConfidenceIntervalConfig,
    CVConfig,
    DecisionTreeRegressorConfig,
    DummyClassifierConfig,
    DummyRegressorConfig,
    ElasticNetConfig,
    ExperimentConfig,
    ExtraTreesRegressorConfig,
    FeatureSelectionConfig,
    GaussianNBConfig,
    GradientBoostingClassifierConfig,
    GradientBoostingRegressorConfig,
    HistGradientBoostingRegressorConfig,
    KNeighborsClassifierConfig,
    KNeighborsRegressorConfig,
    LassoConfig,
    LDAConfig,
    LinearRegressionConfig,
    LinearSVCConfig,
    LogisticRegressionConfig,
    MLPClassifierConfig,
    MLPRegressorConfig,
    RandomForestClassifierConfig,
    RandomForestRegressorConfig,
    RidgeConfig,
    SGDClassifierConfig,
    SGDRegressorConfig,
    StatisticalAssessmentConfig,
    SVCConfig,
    SVRConfig,
    TuningConfig,
)
from coco_pipe.decoding.registry import (
    EstimatorSpec,
    get_estimator_cls,
    register_estimator,
    register_estimator_spec,
)

ACTIVE_SKLEARN_CONFIGS = [
    LogisticRegressionConfig,
    RandomForestClassifierConfig,
    SVCConfig,
    LinearSVCConfig,
    KNeighborsClassifierConfig,
    GradientBoostingClassifierConfig,
    SGDClassifierConfig,
    MLPClassifierConfig,
    GaussianNBConfig,
    LDAConfig,
    AdaBoostClassifierConfig,
    DummyClassifierConfig,
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


def _experiment_for_instantiation():
    return Experiment(
        ExperimentConfig(
            task="classification",
            models={"lr": {"kind": "classical", "method": "LogisticRegression"}},
            metrics=["accuracy"],
            n_jobs=1,
            verbose=False,
        )
    )


def test_experiment_config_task_consistency():
    # Valid classification
    cfg = ExperimentConfig(
        task="classification",
        models={"lr": {"kind": "classical", "method": "LogisticRegression"}},
        metrics=["accuracy", "f1"],
    )
    assert cfg.task == "classification"

    # Invalid classification (regression metric)
    with pytest.raises(ValidationError, match="Metric 'r2' is for regression"):
        ExperimentConfig(
            task="classification",
            models={"lr": {"kind": "classical", "method": "LogisticRegression"}},
            metrics=["r2"],
        )

    # Invalid regression (stratified CV)
    with pytest.raises(
        ValidationError, match="CV strategy 'stratified' is not valid for regression"
    ):
        ExperimentConfig(
            task="regression",
            models={"ridge": {"kind": "classical", "method": "Ridge"}},
            cv=CVConfig(strategy="stratified"),
            metrics=["r2"],
        )


def test_model_discriminator():
    # Test that 'kind' and 'method' correctly resolve the subclass
    data = {
        "lr": {
            "kind": "classical",
            "method": "ClassicalModel",
            "estimator": "LogisticRegression",
            "params": {"C": 0.1},
        }
    }
    cfg = ExperimentConfig(task="classification", models=data)
    assert cfg.models["lr"].kind == "classical"
    # ClassicalModelConfig has .params
    assert cfg.models["lr"].params["C"] == 0.1


def test_scientific_defaults():
    cfg = ExperimentConfig(
        task="classification",
        models={"lr": {"kind": "classical", "method": "LogisticRegression"}},
    )
    assert cfg.cv.n_splits == 5
    assert cfg.evaluation.chance.n_permutations == 1000
    assert cfg.evaluation.confidence_intervals.alpha == 0.05


def test_field_constraints():
    # Negative splits
    with pytest.raises(ValidationError):
        CVConfig(n_splits=0)

    # Invalid alpha
    with pytest.raises(ValidationError):
        ConfidenceIntervalConfig(alpha=1.5)

    # Feature selection invalid features
    with pytest.raises(ValidationError):
        FeatureSelectionConfig(
            enabled=True, method="sfs", n_features=0, cv=CVConfig(strategy="stratified")
        )

    # Tuning iterations
    with pytest.raises(ValidationError):
        TuningConfig(enabled=True, n_iter=0, cv=CVConfig(strategy="stratified"))


def test_tuning_metric_consistency():
    # Invalid tuning metric (task mismatch)
    with pytest.raises(ValidationError, match="Tuning metric 'r2' is for regression"):
        ExperimentConfig(
            task="classification",
            models={"lr": {"kind": "classical", "method": "LogisticRegression"}},
            tuning={"enabled": True, "scoring": "r2", "cv": {"strategy": "stratified"}},
        )


def test_statistical_assessment_nesting():
    cfg = StatisticalAssessmentConfig(enabled=True)
    assert cfg.chance.n_permutations == 1000
    assert cfg.confidence_intervals.alpha == 0.05

    # Test custom unit column
    cfg_unit = StatisticalAssessmentConfig(
        unit_of_inference="custom", custom_unit_column="session"
    )
    assert cfg_unit.custom_unit_column == "session"


def test_every_active_sklearn_config_method_resolves():
    for config_cls in ACTIVE_SKLEARN_CONFIGS:
        config = config_cls()
        assert get_estimator_cls(config.method) is not None


def test_every_active_sklearn_default_config_instantiates():
    experiment = _experiment_for_instantiation()
    for config_cls in ACTIVE_SKLEARN_CONFIGS:
        config = config_cls()
        estimator = experiment._instantiate_model(config.method, config)
        assert estimator is not None


def test_experiment_config_forbids_extra_fields():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            task="classification",
            models={"lr": {"kind": "classical", "method": "LogisticRegression"}},
            unexpected=True,
        )


def test_removed_deprecated_config_fields_are_rejected():
    with pytest.raises(ValidationError):
        LogisticRegressionConfig(multiclass="ovr")

    with pytest.raises(ValidationError):
        AdaBoostClassifierConfig(algorithm="SAMME")

    with pytest.raises(ValidationError):
        BayesianRidgeConfig(n_iter=10)

    with pytest.raises(ValidationError):
        ARDRegressionConfig(n_iter=10)


def test_modern_iteration_fields_are_exposed():
    bayes = BayesianRidgeConfig(max_iter=12)
    ard = ARDRegressionConfig(max_iter=13)
    assert bayes.model_dump()["max_iter"] == 12
    assert ard.model_dump()["max_iter"] == 13
    assert "n_iter" not in bayes.model_dump()
    assert "n_iter" not in ard.model_dump()


def test_sgd_penalty_accepts_none_not_null_string():
    assert SGDClassifierConfig(penalty=None).penalty is None
    with pytest.raises(ValidationError):
        SGDClassifierConfig(penalty="null")


def test_invalid_constructor_params_are_not_silently_dropped():
    @register_estimator("StrictFakeEstimator")
    class StrictFakeEstimator:
        def __init__(self, known=1):
            self.known = known

    # Register spec for it
    spec = EstimatorSpec(
        name="StrictFakeEstimator",
        import_path="fake",
        family="linear",
        task=("classification",),
    )
    register_estimator_spec(spec)

    class FakeConfig:
        method = "StrictFakeEstimator"
        kind = "classical"

        def model_dump(self, exclude=None):
            data = {"method": self.method, "kind": self.kind, "unknown": 2}
            for key in exclude or set():
                data.pop(key, None)
            return data

    experiment = _experiment_for_instantiation()
    with pytest.raises(ValueError, match="Failed to instantiate model 'fake'"):
        experiment._instantiate_model("fake", FakeConfig())


def test_experiment_validation_hardening():
    # calibration for regression
    with pytest.raises(
        (ValueError, ValidationError),
        match="calibration is only available for classification",
    ):
        ExperimentConfig(
            task="regression",
            models={"lr": LogisticRegressionConfig()},
            metrics=["neg_mean_squared_error"],
            cv=CVConfig(strategy="kfold"),
            calibration=CalibrationConfig(enabled=True),
        )

    # stratified for regression
    with pytest.raises((ValueError, ValidationError), match="not valid for regression"):
        ExperimentConfig(
            task="regression",
            models={"lr": LogisticRegressionConfig()},
            metrics=["neg_mean_squared_error"],
            cv=CVConfig(strategy="stratified"),
        )

    # FS metric mismatch
    with pytest.raises(
        (ValueError, ValidationError), match="is not defined|incompatible with task"
    ):
        ExperimentConfig(
            task="classification",
            models={"lr": LogisticRegressionConfig()},
            feature_selection=FeatureSelectionConfig(
                enabled=True, scoring="neg_mean_squared_error"
            ),
        )
