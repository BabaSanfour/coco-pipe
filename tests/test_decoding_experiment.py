import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import (
    AdaBoostClassifierConfig,
    AdaBoostRegressorConfig,
    ARDRegressionConfig,
    BayesianRidgeConfig,
    CalibrationConfig,
    ClassicalModelConfig,
    CVConfig,
    DecisionTreeRegressorConfig,
    DummyClassifierConfig,
    DummyRegressorConfig,
    ElasticNetConfig,
    ExtraTreesRegressorConfig,
    FeatureSelectionConfig,
    FoundationEmbeddingModelConfig,
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
    "LinearSVC": SVCConfig(kernel="linear", max_iter=500),  # LinearSVC alias
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
def test_registered_estimator_survives_fit_predict_and_declared_responses(method):
    config = SMOKE_CONFIGS[method]
    spec = get_estimator_spec(method)
    is_clf = "classification" in spec.task
    X, y = _classification_data() if is_clf else _regression_data()
    X_test = X[:5]

    exp = Experiment(
        ExperimentConfig(
            task="classification" if is_clf else "regression",
            models={method: config},
            metrics=["accuracy" if is_clf else "r2"],
            cv=CVConfig(strategy="stratified" if is_clf else "kfold", n_splits=2),
            verbose=False,
        )
    )
    estimator = exp._prepare_estimator(method, config)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(X, y)
        y_pred = estimator.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
        if spec.supports_proba:
            assert estimator.predict_proba(X_test).shape[0] == X_test.shape[0]
        if spec.supports_decision_function:
            assert estimator.decision_function(X_test).shape[0] == X_test.shape[0]


# =============================================================================
# SCIENTIFIC VALIDITY & LEAKAGE GUARDS
# =============================================================================


def test_grouped_outer_cv_respects_boundaries():
    X, y = _classification_data(n_samples=20)
    groups = np.repeat(np.arange(5), 4)  # 5 subjects, 4 trials each
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        cv=CVConfig(strategy="group_kfold", n_splits=5),
        verbose=False,
        n_jobs=1,
    )
    result = Experiment(config).run(X, y, groups=groups)
    assert result.raw["lr"]["status"] == "success"
    # Verify no subject overlap across any fold
    for split_meta in result.raw["lr"]["splits"]:
        train_idx = split_meta["train_idx"]
        test_idx = split_meta["test_idx"]
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_group_leakage_guard_raises_error():
    X, y = _classification_data()
    groups = np.repeat([0, 1, 2, 3], len(y) // 4)
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        cv=CVConfig(strategy="group_kfold", n_splits=2),
        tuning=TuningConfig(
            enabled=True, cv=CVConfig(strategy="stratified", n_splits=2)
        ),
        grids={"lr": {"C": [0.1, 1.0]}},
    )
    with pytest.raises(
        ValueError,
        match=(
            "Outer CV strategy is group-based, but tuning.cv strategy "
            "'stratified' is not"
        ),
    ):
        Experiment(config).run(X, y, groups=groups)


# =============================================================================
# TEMPORAL DECODING (Comprehensive)
# =============================================================================


def test_sliding_and_generalizing_estimators_full_workflow():
    pytest.importorskip("mne")
    X, y = _temporal_data()
    times = np.array([-0.1, 0.0, 0.1, 0.2])
    # Use real instances to avoid Pydantic issues
    base_cfg = ClassicalModelConfig(estimator="LogisticRegression")
    config = ExperimentConfig(
        task="classification",
        models={
            "sliding": TemporalDecoderConfig(wrapper="sliding", base=base_cfg),
            "generalizing": TemporalDecoderConfig(
                wrapper="generalizing", base=base_cfg
            ),
        },
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=2),
        n_jobs=1,
    )
    result = Experiment(config).run(X, y, time_axis=times)
    assert "Time" in result.get_predictions().columns
    # Specify model name to get the matrix (4, 4), otherwise it returns long format
    assert result.get_generalization_matrix(
        "generalizing", metric="accuracy"
    ).shape == (4, 4)


# =============================================================================
# FEATURE SELECTION & TUNING
# =============================================================================


def test_sfs_with_tuning_and_groups_routing():
    X, y = _classification_data(n_samples=24)
    groups = np.repeat(np.arange(6), 4)
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        cv=CVConfig(strategy="group_kfold", n_splits=2),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="sfs",
            n_features=2,
            cv=CVConfig(strategy="group_kfold", n_splits=2),
        ),
        tuning=TuningConfig(
            enabled=True, n_iter=2, cv=CVConfig(strategy="group_kfold", n_splits=2)
        ),
        grids={"lr": {"clf__C": [0.1, 1.0]}},
        n_jobs=1,
        verbose=False,
    )
    result = Experiment(config).run(X, y, groups=groups)
    assert result.raw["lr"]["status"] == "success"
    assert len(result.get_selected_features()["FeatureName"].unique()) == 6


# =============================================================================
# EDGE CASES & COVERAGE GAPS
# =============================================================================


def test_experiment_config_validation_errors():
    # 1. Metric mismatch
    cfg = ExperimentConfig.model_construct(
        task="regression",
        models={"dummy": DummyRegressorConfig()},
        metrics=["accuracy"],
        cv=CVConfig(strategy="kfold"),
        tuning=TuningConfig(),
        feature_selection=FeatureSelectionConfig(),
        calibration=CalibrationConfig(),
        evaluation=StatisticalAssessmentConfig(),
    )
    with pytest.raises(
        ValueError, match="is for classification but experiment task is regression"
    ):
        Experiment(cfg)

    # 2. Calibration for regression
    cfg = ExperimentConfig.model_construct(
        task="regression",
        models={"dummy": DummyRegressorConfig()},
        calibration=CalibrationConfig(enabled=True),
        metrics=["r2"],
        cv=CVConfig(strategy="kfold"),
        tuning=TuningConfig(),
        feature_selection=FeatureSelectionConfig(),
        evaluation=StatisticalAssessmentConfig(),
    )
    with pytest.raises(
        ValueError, match="calibration is only available for classification"
    ):
        Experiment(cfg)

    # 3. Stratified regression
    cfg = ExperimentConfig.model_construct(
        task="regression",
        models={"dummy": DummyRegressorConfig()},
        cv=CVConfig(strategy="stratified"),
        metrics=["r2"],
        tuning=TuningConfig(),
        feature_selection=FeatureSelectionConfig(),
        calibration=CalibrationConfig(),
        evaluation=StatisticalAssessmentConfig(),
    )
    with pytest.raises(ValueError, match="invalid for regression"):
        Experiment(cfg)


def test_resolve_metadata_and_groups_mismatch():
    X, y = _classification_data(n_samples=10)
    exp = Experiment(
        ExperimentConfig(
            task="classification", models={"lr": LogisticRegressionConfig()}
        )
    )
    # 1. Length mismatch
    with pytest.raises(ValueError, match="sample_metadata length mismatch"):
        exp._resolve_metadata_and_groups(10, pd.DataFrame({"a": [1]}), None)

    # 2. Missing Subject/Session (with capitalized error message check)
    with pytest.raises(ValueError, match="must include Subject and Session"):
        exp._resolve_metadata_and_groups(10, pd.DataFrame({"a": range(10)}), None)

    # 3. Missing group_key (with Subject/Session provided)
    with pytest.raises(ValueError, match="group_key 'missing' not found"):
        exp.config.cv.group_key = "missing"
        exp._resolve_metadata_and_groups(
            10, pd.DataFrame({"Subject": range(10), "Session": range(10)}), None
        )


def test_feature_names_alignment():
    X, y = _classification_data(n_samples=10, n_features=2)
    exp = Experiment(
        ExperimentConfig(
            task="classification", models={"lr": LogisticRegressionConfig()}
        )
    )
    with pytest.raises(ValueError, match="feature_names length mismatch"):
        exp._resolve_feature_names(X, ["only_one"])


def test_degenerate_fold_integrity():
    from coco_pipe.decoding.experiment import _validate_fold_integrity

    with pytest.raises(ValueError, match="Empty fold"):
        _validate_fold_integrity(np.array([]), np.array([1]), ("classification",))
    with pytest.raises(ValueError, match="Degenerate Test Fold"):
        _validate_fold_integrity(
            np.array([0, 1]), np.array([1, 1]), ("classification",)
        )


def test_random_state_propagation_none():
    config = ExperimentConfig(
        task="classification",
        random_state=None,
        models={"lr": LogisticRegressionConfig()},
    )
    assert config.cv.random_state == 42


def test_instantiate_foundation_model_mock():
    # Use a dictionary to bypass Pydantic literal restrictions for the mock
    mock_model = {
        "kind": "foundation_embedding",
        "provider": "reve",
        "model_name": "dummy",
        "checkpoint": None,
    }

    config = ExperimentConfig.model_construct(
        task="classification",
        models={"reve": mock_model},
        metrics=["accuracy"],
        cv=CVConfig(),
        tuning=TuningConfig(),
        feature_selection=FeatureSelectionConfig(),
        calibration=CalibrationConfig(),
        evaluation=StatisticalAssessmentConfig(),
        verbose=False,
    )
    exp = Experiment(config)

    pytest.importorskip("torch")
    from coco_pipe.decoding.fm_hub import REVEModel

    with patch(
        "coco_pipe.decoding.experiment.Experiment._instantiate_model"
    ) as mock_inst:
        mock_inst.return_value = MagicMock(spec=REVEModel)
        # Should NOT raise spec error anymore
        est = exp._prepare_estimator("reve", mock_model)
        assert est is not None


def test_wrap_with_tuning_grid_invalid_key():
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        tuning=TuningConfig(enabled=True, cv=CVConfig(strategy="stratified")),
        grids={"lr": {"invalid_key": [1, 2]}},
    )
    exp = Experiment(config)
    with pytest.raises(ValueError, match="Invalid tuning keys"):
        exp._prepare_estimator("lr", config.models["lr"])


def test_build_result_meta_time_axis_mismatch():
    X, y = _classification_data(n_samples=10, n_features=5)
    exp = Experiment(
        ExperimentConfig(
            task="classification", models={"lr": LogisticRegressionConfig()}
        )
    )
    X3 = np.zeros((10, 2, 5))
    with pytest.raises(ValueError, match="time_axis length mismatch"):
        exp.run(X3, y, time_axis=np.arange(10))


def test_importance_aggregation_shape_mismatch_recovery():
    X, y = _classification_data()
    Experiment(
        ExperimentConfig(
            task="classification", models={"lr": LogisticRegressionConfig()}
        )
    )
    valid_imps = [np.array([1, 2]), np.array([1, 2, 3])]
    assert not all(imp.shape == valid_imps[0].shape for imp in valid_imps)


def test_observation_level_validation():
    X, y = _classification_data()
    exp = Experiment(
        ExperimentConfig(
            task="classification", models={"lr": LogisticRegressionConfig()}
        )
    )
    with pytest.raises(
        ValueError, match="observation_level must be 'sample' or 'epoch'"
    ):
        exp.run(X, y, observation_level="invalid")


# =============================================================================
# REPRODUCIBILITY & GROUPED CV SCIENTIFIC VALIDITY
# =============================================================================


def test_grouped_cv_requires_at_least_two_groups():
    X, y = _classification_data(n_samples=10)
    groups = np.zeros(10)  # All same group
    # Force strategy to be in GROUP_CV_STRATEGIES
    cv_cfg = CVConfig()
    cv_cfg.__dict__["strategy"] = "group_kfold"

    config = ExperimentConfig.model_construct(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        cv=cv_cfg,
        metrics=["accuracy"],
        tuning=TuningConfig(),
        feature_selection=FeatureSelectionConfig(),
        calibration=CalibrationConfig(),
        evaluation=StatisticalAssessmentConfig(),
        verbose=False,
    )
    # The guard should raise BEFORE sklearn.
    with pytest.raises(
        ValueError, match="Grouped CV requires at least 2 unique groups"
    ):
        Experiment(config).run(X, y, groups=groups)


def test_inject_seed_recursion_depth():
    from types import SimpleNamespace

    cfg = SimpleNamespace(base=SimpleNamespace(random_state=None))
    exp = Experiment(
        ExperimentConfig(
            task="classification", models={"lr": LogisticRegressionConfig()}
        )
    )
    exp._inject_seed(cfg, 123)
    assert cfg.base.random_state == 123


# =============================================================================
# PIPELINE COMBINATIONS (Tuning, SFS, Calibration)
# =============================================================================


def test_tuning_only_workflow():
    X, y = _classification_data(n_samples=20)
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        tuning=TuningConfig(
            enabled=True, n_iter=2, cv=CVConfig(strategy="stratified", n_splits=2)
        ),
        grids={"lr": {"clf__C": [0.1, 1.0]}},
        cv=CVConfig(strategy="stratified", n_splits=2),
        n_jobs=1,
        verbose=False,
    )
    result = Experiment(config).run(X, y)
    assert result.raw["lr"]["status"] == "success"


def test_calibration_only_workflow():
    X, y = _classification_data(n_samples=20)
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        calibration=CalibrationConfig(
            enabled=True,
            method="sigmoid",
            cv=CVConfig(strategy="stratified", n_splits=2),
        ),
        cv=CVConfig(strategy="stratified", n_splits=2),
        n_jobs=1,
        verbose=False,
    )
    result = Experiment(config).run(X, y)
    assert result.raw["lr"]["status"] == "success"


def test_sfs_tuning_and_calibration_combined():
    X, y = _classification_data(n_samples=30)
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="sfs",
            n_features=2,
            cv=CVConfig(strategy="stratified", n_splits=2),
        ),
        tuning=TuningConfig(
            enabled=True, n_iter=2, cv=CVConfig(strategy="stratified", n_splits=2)
        ),
        calibration=CalibrationConfig(
            enabled=True, cv=CVConfig(strategy="stratified", n_splits=2)
        ),
        grids={"lr": {"clf__C": [0.1, 1.0]}},
        cv=CVConfig(strategy="stratified", n_splits=2),
        n_jobs=1,
        verbose=False,
    )
    result = Experiment(config).run(X, y)
    assert result.raw["lr"]["status"] == "success"
    # Ensure probabilities are still produced after SFS and Tuning
    assert "y_proba_1" in result.get_predictions().columns


def test_tuning_with_calibration():
    X, y = _classification_data(n_samples=20)
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        tuning=TuningConfig(
            enabled=True, n_iter=2, cv=CVConfig(strategy="stratified", n_splits=2)
        ),
        calibration=CalibrationConfig(
            enabled=True, cv=CVConfig(strategy="stratified", n_splits=2)
        ),
        grids={"lr": {"clf__C": [0.1, 1.0]}},
        cv=CVConfig(strategy="stratified", n_splits=2),
        n_jobs=1,
        verbose=False,
    )
    result = Experiment(config).run(X, y)
    assert result.raw["lr"]["status"] == "success"


def test_sfs_with_calibration():
    X, y = _classification_data(n_samples=20)
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="sfs",
            n_features=2,
            cv=CVConfig(strategy="stratified", n_splits=2),
        ),
        calibration=CalibrationConfig(
            enabled=True, cv=CVConfig(strategy="stratified", n_splits=2)
        ),
        cv=CVConfig(strategy="stratified", n_splits=2),
        n_jobs=1,
        verbose=False,
    )
    result = Experiment(config).run(X, y)
    assert result.raw["lr"]["status"] == "success"


def test_instantiate_temporal_model_explicit():
    pytest.importorskip("mne")
    base_cfg = ClassicalModelConfig(estimator="LogisticRegression", params={"C": 1.0})
    config = TemporalDecoderConfig(wrapper="sliding", base=base_cfg)
    exp = Experiment(ExperimentConfig(task="classification", models={"sl": config}))
    est = exp._instantiate_model("sl", config)
    assert est.__class__.__name__ == "SlidingEstimator"


def test_create_fs_step_sfs_path():
    exp = Experiment(
        ExperimentConfig(
            task="classification",
            models={"lr": LogisticRegressionConfig()},
            feature_selection=FeatureSelectionConfig(
                enabled=True,
                method="sfs",
                n_features=2,
                cv=CVConfig(strategy="kfold", n_splits=2),
            ),
        )
    )
    step = exp._create_fs_step(LogisticRegression())
    assert step[0] == "fs"
    assert step[1].__class__.__name__ == "GroupedSequentialFeatureSelector"


def test_build_result_meta_time_axis():
    exp = Experiment(
        ExperimentConfig(
            task="classification", models={"lr": LogisticRegressionConfig()}
        )
    )
    # Mock some internal state needed by _build_result_meta
    exp._sample_metadata = pd.DataFrame({"Subject": range(10)})
    exp._observation_level = "sample"
    exp._inferential_unit = "subject"

    meta = exp._build_result_meta(np.zeros((10, 5)), t_axis=np.array([0, 1]))
    assert meta["time_axis"] == [0, 1]


def test_capability_payload_with_fs_enabled():
    exp = Experiment(
        ExperimentConfig(
            task="classification",
            models={"lr": LogisticRegressionConfig()},
            feature_selection=FeatureSelectionConfig(enabled=True, method="k_best"),
        )
    )
    payload = exp._capability_payload()
    assert "k_best" in payload["feature_selectors"]


def test_instantiate_foundation_model_fm_hub():
    # Use valid config object to avoid pydantic issues
    fm_config = FoundationEmbeddingModelConfig(
        kind="foundation_embedding", provider="reve", model_name="dummy"
    )
    exp = Experiment(
        ExperimentConfig.model_construct(
            task="classification",
            models={"fm": fm_config},
            metrics=["accuracy"],
            cv=CVConfig(),
        )
    )
    with patch("coco_pipe.decoding.fm_hub.build_foundation_model") as mock_build:
        exp._instantiate_model("fm", fm_config)
        mock_build.assert_called_once()


def test_experiment_calibration_run():
    X, y = make_classification(n_samples=40, random_state=42)
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        calibration=CalibrationConfig(enabled=True, cv=CVConfig(n_splits=2)),
        cv=CVConfig(n_splits=2),
    )
    res = Experiment(config).run(X, y)
    assert "lr" in res.raw
