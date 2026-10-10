import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import (
    AdaBoostClassifierConfig,
    AdaBoostRegressorConfig,
    ARDRegressionConfig,
    BayesianRidgeConfig,
    CalibrationConfig,
    ChanceAssessmentConfig,
    ClassicalModelConfig,
    CVConfig,
    DecisionTreeRegressorConfig,
    DummyClassifierConfig,
    DummyRegressorConfig,
    ElasticNetConfig,
    ExtraTreesRegressorConfig,
    FeatureSelectionConfig,
    GaussianNBConfig,
    GradientBoostingClassifierConfig,
    GradientBoostingRegressorConfig,
    HistGradientBoostingClassifierConfig,
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
    TemporalDecoderConfig,
    TuningConfig,
)
from coco_pipe.decoding.registry import (
    get_estimator_spec,
)

# =============================================================================
# FIXTURES & DATA GENERATORS
# =============================================================================


def _classification_data(n_samples=40, n_features=6):
    rng = np.random.default_rng(10)
    y = np.tile([0, 1], n_samples // 2)
    X = rng.normal(size=(len(y), n_features))
    X[:, 0] += y * 2.0
    X[:, 1] -= y * 1.0
    return X, y


def _regression_data(n_samples=28, n_features=5):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(n_samples, n_features))
    y = X[:, 0] * 1.5 - X[:, 1] * 0.5 + rng.normal(scale=0.05, size=X.shape[0])
    return X, y


def _temporal_data(n_samples=12, n_channels=3, n_times=4):
    rng = np.random.default_rng(42)
    y = np.array([0, 1] * (n_samples // 2))
    X = rng.normal(scale=0.1, size=(n_samples, n_channels, n_times))
    X[y == 1, 0, :] += 1.0
    X[y == 0, 0, :] -= 1.0
    return X, y


# =============================================================================
# SMOKE TEST CONFIGURATIONS
# =============================================================================

CLASSIFIER_SMOKE_CONFIGS = {
    "DummyClassifier": DummyClassifierConfig(strategy="prior"),
    "LogisticRegression": LogisticRegressionConfig(solver="liblinear", max_iter=200),
    "LinearSVC": LinearSVCConfig(max_iter=500),
    "LinearDiscriminantAnalysis": LDAConfig(),
    "RandomForestClassifier": RandomForestClassifierConfig(
        n_estimators=5, max_depth=3, n_jobs=1
    ),
    "SVC": SVCConfig(kernel="linear", probability=True, max_iter=500),
    "KNeighborsClassifier": KNeighborsClassifierConfig(n_neighbors=3),
    "GradientBoostingClassifier": GradientBoostingClassifierConfig(n_estimators=5),
    "SGDClassifier": SGDClassifierConfig(max_iter=500),
    "MLPClassifier": MLPClassifierConfig(hidden_layer_sizes=(4,), max_iter=50),
    "GaussianNB": GaussianNBConfig(),
    "AdaBoostClassifier": AdaBoostClassifierConfig(n_estimators=5),
    "ExtraTreesClassifier": RandomForestClassifierConfig(
        n_estimators=5, kind="classical"
    ),
    "HistGradientBoostingClassifier": HistGradientBoostingClassifierConfig(max_iter=5),
}

REGRESSOR_SMOKE_CONFIGS = {
    "DummyRegressor": DummyRegressorConfig(strategy="mean"),
    "Ridge": RidgeConfig(),
    "RandomForestRegressor": RandomForestRegressorConfig(
        n_estimators=5, max_depth=3, n_jobs=1
    ),
    "LinearRegression": LinearRegressionConfig(),
    "Lasso": LassoConfig(max_iter=500),
    "ElasticNet": ElasticNetConfig(max_iter=500),
    "SVR": SVRConfig(kernel="linear"),
    "GradientBoostingRegressor": GradientBoostingRegressorConfig(n_estimators=5),
    "SGDRegressor": SGDRegressorConfig(max_iter=500),
    "MLPRegressor": MLPRegressorConfig(hidden_layer_sizes=(4,), max_iter=50),
    "DecisionTreeRegressor": DecisionTreeRegressorConfig(max_depth=3),
    "KNeighborsRegressor": KNeighborsRegressorConfig(n_neighbors=3),
    "ExtraTreesRegressor": ExtraTreesRegressorConfig(
        n_estimators=5, max_depth=3, n_jobs=1
    ),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressorConfig(
        max_iter=5, min_samples_leaf=2
    ),
    "AdaBoostRegressor": AdaBoostRegressorConfig(n_estimators=5),
    "BayesianRidge": BayesianRidgeConfig(),
    "ARDRegression": ARDRegressionConfig(),
}

SMOKE_CONFIGS = {**CLASSIFIER_SMOKE_CONFIGS, **REGRESSOR_SMOKE_CONFIGS}


@pytest.mark.parametrize("method", sorted(SMOKE_CONFIGS))
def test_registered_estimators_run_and_give_their_declared_responses(method):
    spec = get_estimator_spec(method)
    is_clf = "classification" in spec.task
    X, y = _classification_data() if is_clf else _regression_data()
    config = ExperimentConfig(
        task="classification" if is_clf else "regression",
        models={method: SMOKE_CONFIGS[method]},
        metrics=["accuracy" if is_clf else "r2"],
        cv=CVConfig(strategy="stratified" if is_clf else "kfold", n_splits=2),
        verbose=False,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Experiment(config).run(X, y)
    assert result.raw[method]["status"] == "success"
    columns = set(result.get_predictions().columns)
    assert len(result.get_predictions()) == len(y)
    if is_clf and spec.supports_proba:
        assert {"y_proba_0", "y_proba_1"} <= columns
    elif is_clf and spec.supports_decision_function:
        assert "y_score" in columns


def test_grouped_cv_keeps_groups_apart():
    X, y = _classification_data(n_samples=20)
    groups = np.repeat(np.arange(5), 4)
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        cv=CVConfig(strategy="group_kfold", n_splits=5),
        verbose=False,
        n_jobs=1,
    )
    result = Experiment(config).run(X, y, groups=groups)
    for split in result.raw["lr"]["splits"]:
        assert set(groups[split["train_idx"]]).isdisjoint(groups[split["test_idx"]])
    # without sample_metadata, the groups become the subjects
    predictions = result.get_predictions()
    np.testing.assert_array_equal(
        predictions.sort_values("SampleIndex")["Subject"], groups
    )
    assert (predictions["Session"] == "01").all()


def test_temporal_decoders():
    X, y = _temporal_data()
    base = ClassicalModelConfig(estimator="LogisticRegression")
    config = ExperimentConfig(
        task="classification",
        models={
            "sliding": TemporalDecoderConfig(wrapper="sliding", base=base),
            "generalizing": TemporalDecoderConfig(wrapper="generalizing", base=base),
        },
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=2),
        n_jobs=1,
    )
    result = Experiment(config).run(X, y, time_axis=np.array([-0.1, 0.0, 0.1, 0.2]))
    assert "Time" in result.get_predictions().columns
    matrix = result.get_generalization_matrix("generalizing", metric="accuracy")
    assert matrix.shape == (4, 4)
    assert result.meta["time_axis"] == [-0.1, 0.0, 0.1, 0.2]


_TWO_FOLDS = CVConfig(strategy="stratified", n_splits=2)
_SFS = FeatureSelectionConfig(enabled=True, method="sfs", n_features=2, cv=_TWO_FOLDS)
_TUNING = TuningConfig(enabled=True, n_iter=2, cv=_TWO_FOLDS)
_CALIBRATION = CalibrationConfig(enabled=True, method="sigmoid", cv=_TWO_FOLDS)


@pytest.mark.parametrize(
    "steps",
    [
        {"tuning": _TUNING},
        {"calibration": _CALIBRATION},
        {"feature_selection": _SFS, "calibration": _CALIBRATION},
        {"tuning": _TUNING, "calibration": _CALIBRATION},
        {"feature_selection": _SFS, "tuning": _TUNING, "calibration": _CALIBRATION},
    ],
    ids=["tuning", "calibration", "sfs+calibration", "tuning+calibration", "all"],
)
def test_pipeline_steps_combine(steps):
    X, y = _classification_data(n_samples=30)
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        grids={"lr": {"clf__C": [0.1, 1.0]}},
        cv=_TWO_FOLDS,
        n_jobs=1,
        verbose=False,
        **steps,
    )
    result = Experiment(config).run(X, y)
    assert result.raw["lr"]["status"] == "success"
    assert "y_proba_1" in result.get_predictions().columns


def test_feature_selection_and_tuning_follow_the_groups():
    X, y = _classification_data(n_samples=24)
    groups = np.repeat(np.arange(6), 4)
    grouped = CVConfig(strategy="group_kfold", n_splits=2)
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        cv=grouped,
        feature_selection=FeatureSelectionConfig(
            enabled=True, method="sfs", n_features=2, cv=grouped
        ),
        tuning=TuningConfig(enabled=True, n_iter=2, cv=grouped),
        grids={"lr": {"clf__C": [0.1, 1.0]}},
        n_jobs=1,
        verbose=False,
    )
    result = Experiment(config).run(X, y, groups=groups)
    assert result.raw["lr"]["status"] == "success"
    assert len(result.get_selected_features()["FeatureName"].unique()) == 6
    selectors = result.meta["capabilities"]["feature_selectors"]
    assert "sfs" in selectors

    # a stratified inner CV under a grouped outer CV would leak subjects
    config.tuning = TuningConfig(enabled=True, cv=_TWO_FOLDS)
    with pytest.raises(ValueError, match=r"tuning\.cv strategy 'stratified' is not"):
        Experiment(config).run(X, y, groups=groups)


def test_result_metadata_and_statistics():
    X, y = _classification_data(n_samples=20)
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        cv=CVConfig(n_splits=2),
        statistical_assessment=StatisticalAssessmentConfig(
            enabled=True,
            metrics=["accuracy"],
            chance=ChanceAssessmentConfig(method="binomial", p0=0.5),
        ),
        tag="test_meta",
        verbose=False,
    )
    result = Experiment(config).run(X, y)
    assert isinstance(result.meta["coco_pipe_version"], str)
    assert "statistical_assessment" in result.meta
    assert "statistical_assessment" in result.raw["lr"]


def test_random_state_reaches_every_step():
    sliding = TemporalDecoderConfig(
        wrapper="sliding", base=ClassicalModelConfig(estimator="LogisticRegression")
    )
    config = ExperimentConfig(
        task="classification",
        random_state=42,
        models={"lr": LogisticRegressionConfig(random_state=None), "sliding": sliding},
        tuning=TuningConfig(enabled=True, cv=CVConfig(strategy="stratified")),
    )
    Experiment(config)
    assert config.cv.random_state == 42
    assert config.tuning.random_state == 44
    assert config.models["lr"].random_state is not None
    assert "random_state" in config.models["sliding"].base.params

    seedless = ExperimentConfig(
        task="classification",
        random_state=None,
        models={"lr": LogisticRegressionConfig()},
    )
    assert seedless.cv.random_state == 42


def _bare_config(**fields):
    """A config built without validation, to reach the checks Experiment makes."""
    defaults = {
        "task": "classification",
        "models": {"lr": LogisticRegressionConfig()},
        "metrics": ["accuracy"],
        "cv": CVConfig(),
        "tuning": TuningConfig(),
        "feature_selection": FeatureSelectionConfig(),
        "calibration": CalibrationConfig(),
        "statistical_assessment": StatisticalAssessmentConfig(),
        "verbose": False,
    }
    return ExperimentConfig.model_construct(**{**defaults, **fields})


def test_experiment_rejects_inconsistent_configs():
    regression = {"task": "regression", "models": {"dummy": DummyRegressorConfig()}}
    for fields, match in [
        ({**regression, "cv": CVConfig(strategy="kfold")}, "is for classification"),
        (
            {
                **regression,
                "metrics": ["r2"],
                "cv": CVConfig(strategy="kfold"),
                "calibration": CalibrationConfig(enabled=True),
            },
            "calibration is only available for classification",
        ),
        ({**regression, "metrics": ["r2"]}, "invalid for regression"),
    ]:
        with pytest.raises(ValueError, match=match):
            Experiment(_bare_config(**fields))


def test_run_checks_its_inputs():
    X, y = _classification_data(n_samples=10, n_features=2)
    plain = ExperimentConfig(
        task="classification", models={"lr": LogisticRegressionConfig()}, verbose=False
    )
    for kwargs, match in [
        ({"sample_metadata": pd.DataFrame({"a": [1]})}, "length mismatch"),
        (
            {"sample_metadata": pd.DataFrame({"a": range(10)})},
            "must include Subject and Session",
        ),
        ({"feature_names": ["only_one"]}, "feature_names length mismatch"),
        ({"observation_level": "invalid"}, "must be 'sample' or 'epoch'"),
    ]:
        with pytest.raises(ValueError, match=match):
            Experiment(plain).run(X, y, **kwargs)
    with pytest.raises(ValueError, match="time_axis length mismatch"):
        Experiment(plain).run(np.zeros((10, 2, 5)), y, time_axis=np.arange(10))

    grouped = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        cv=CVConfig(strategy="group_kfold", n_splits=2, group_key="missing"),
        verbose=False,
    )
    subjects = pd.DataFrame({"Subject": range(10), "Session": range(10)})
    with pytest.raises(ValueError, match="group_key 'missing' not found"):
        Experiment(grouped).run(X, y, sample_metadata=subjects)

    one_group = _bare_config(cv=CVConfig.model_construct(strategy="group_kfold"))
    with pytest.raises(ValueError, match="at least 2 unique groups"):
        Experiment(one_group).run(X, y, groups=np.zeros(10))

    tuned = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        tuning=TuningConfig(enabled=True, cv=CVConfig(strategy="stratified")),
        grids={"lr": {"invalid_key": [1, 2]}},
    )
    # a model whose setup fails is reported as failed, not raised
    result = Experiment(tuned).run(*_classification_data())
    assert result.raw["lr"]["status"] == "failed"
    assert "Invalid tuning keys" in result.raw["lr"]["error"]


def test_a_degenerate_fold_fails_its_model():
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = np.array([0] * 10 + [1] * 10)
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        cv=CVConfig(strategy="kfold", n_splits=2, shuffle=False),
        verbose=False,
        n_jobs=1,
    )
    result = Experiment(config).run(X, y)
    assert result.raw["lr"]["status"] == "failed"
    assert "Degenerate Train Fold" in result.raw["lr"]["error"]


def _low_memory_config(low_memory, **models):
    return ExperimentConfig(
        task="classification",
        models={
            "lr": LogisticRegressionConfig(),
            "rf": RandomForestClassifierConfig(n_estimators=10),
            **models,
        },
        cv=CVConfig(strategy="stratified", n_splits=2),
        metrics=["accuracy"],
        verbose=False,
        n_jobs=1,
        low_memory=low_memory,
    )


def test_low_memory_gives_the_same_results(tmp_path, monkeypatch):
    X, y = _classification_data(n_samples=24)
    baseline = Experiment(_low_memory_config(False)).run(X, y)

    # results spilled to disk come back unchanged, and the spill folder goes
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    # one model fails to fit, one fails to be built
    broken = ClassicalModelConfig(estimator="LogisticRegression", params={"C": -1.0})
    unbuildable = ClassicalModelConfig(estimator="LogisticRegression", params={"x": 1})
    offloaded = Experiment(
        _low_memory_config(True, broken=broken, unbuildable=unbuildable)
    ).run(X, y)
    pd.testing.assert_frame_equal(
        baseline.summary(),
        offloaded.summary().drop(index=["broken", "unbuildable"], errors="ignore"),
    )
    assert offloaded.raw["broken"]["status"] == "failed"
    assert "Failed to instantiate" in offloaded.raw["unbuildable"]["error"]
    assert list(tmp_path.iterdir()) == []
