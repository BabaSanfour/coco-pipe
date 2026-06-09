from types import SimpleNamespace

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding._engine import (
    compact_search_results,
    compute_metric_safe,
    extract_feature_importances,
    extract_metadata,
    fit_and_score_fold,
    fit_estimator,
    metadata_slice,
    warning_records_to_dict,
)
from coco_pipe.decoding.configs import (
    CalibrationConfig,
    CVConfig,
    LinearSVCConfig,
)
from coco_pipe.decoding.interfaces import NeuralTrainable

# --- Mock Objects ---


class MockConfig:
    def __init__(self, **kwargs):
        self.enabled = kwargs.get("enabled", False)
        self.method = kwargs.get("method", "none")
        self.cv = kwargs.get("cv", SimpleNamespace(strategy="group_kfold", n_splits=2))
        self.n_splits = kwargs.get("n_splits", 2)


class MockEstimator(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, **kwargs):
        self._estimator_type = "classifier"
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Defaults for coverage
        if not hasattr(self, "coef_"):
            self.coef_ = np.zeros(2)
        if not hasattr(self, "feature_importances_"):
            self.feature_importances_ = np.zeros(2)
        if not hasattr(self, "best_estimator_"):
            self.best_estimator_ = self
        if not hasattr(self, "best_params_"):
            self.best_params_ = {}
        if not hasattr(self, "best_score_"):
            self.best_score_ = 0.9
        if not hasattr(self, "best_index_"):
            self.best_index_ = 0
        if not hasattr(self, "cv_results_"):
            self.cv_results_ = {
                "params": [{}],
                "mean_test_score": [0.9],
                "rank_test_score": [1],
                "std_test_score": [0.1],
            }
        if not hasattr(self, "classes_"):
            self.classes_ = np.array([0, 1])

    def fit(self, X, y=None, **kwargs):
        self.fit_kwargs = kwargs
        return self

    def predict(self, X):
        return getattr(self, "y_pred_val", np.zeros(len(X)))

    def predict_proba(self, X):
        return getattr(self, "y_proba_val", np.zeros((len(X), 2)))

    def decision_function(self, X):
        return getattr(self, "y_score_val", np.zeros(len(X)))

    def get_support(self):
        return getattr(self, "support_", np.array([True, True]))


# --- Tests ---


def test_diagnostics_basics():
    meta = {"a": np.array([10, 20, 30])}
    assert metadata_slice(meta, np.array([0, 2])) == {"a": [10, 30]}
    assert metadata_slice(None, [0]) is None
    record = SimpleNamespace(category=UserWarning, message="test")
    assert len(warning_records_to_dict("fit", [record])) == 1


def test_importance_extraction_comprehensive():
    spec = SimpleNamespace(importance=("coefficients",), is_sparse_capable=True)
    clf = MockEstimator(coef_=np.array([1.0, 2.0]))
    assert np.allclose(extract_feature_importances(clf, spec), [1.0, 2.0])

    # Missing attribute (now safe thanks to hardening)
    assert extract_feature_importances(BaseEstimator(), spec) is None

    # Pipeline + FS
    fs = MockEstimator(support_=np.array([True, False]))
    pipe = Pipeline([("fs", fs), ("clf", MockEstimator(coef_=np.array([5.0])))])
    assert np.allclose(
        extract_feature_importances(pipe, spec, fs_enabled=True), [5.0, 0.0]
    )


def test_compute_metric_safe_variants():
    def scorer(yt, yp, **kw):
        return yp.mean()

    y_true = np.array([0, 1])
    assert np.isnan(compute_metric_safe(scorer, y_true, None, False))

    # Sliding
    y_sl = np.zeros((2, 5))
    assert compute_metric_safe(scorer, y_true, y_sl, False).shape == (5,)

    # Generalizing
    y_gen = np.zeros((2, 3, 3))
    assert compute_metric_safe(scorer, y_true, y_gen, False).shape == (3, 3)


def test_extract_metadata_exhaustive():
    # Search enabled
    search = MockEstimator()
    meta = extract_metadata(search, None, MockConfig(), search_enabled=True)
    assert "search_results" in meta

    # FS with ranking
    fs = MockEstimator(ranking_=np.array([1, 2]))
    pipe = Pipeline([("fs", fs), ("clf", MockEstimator())])
    meta_fs = extract_metadata(pipe, None, MockConfig(enabled=True, method="sfs"))
    assert "selection_order" in meta_fs


def test_fit_and_score_fold_response_logic():
    spec = SimpleNamespace(
        supports_proba=True,
        supports_decision_function=True,
        importance=("coefficients",),
        supports_groups=True,
        grouped_metadata="none",
        is_sparse_capable=False,
        family="linear",
    )
    X, y = np.zeros((4, 2)), np.array([0, 0, 1, 1])
    ids = np.array(["a", "b", "c", "d"])

    import coco_pipe.decoding._engine as engine
    from coco_pipe.decoding._metrics import MetricSpec

    old_get = engine.get_metric_spec

    try:
        # Use positional arguments for MetricSpec to be safe
        # MetricSpec(name, task, scorer, response_method)
        engine.get_metric_spec = lambda m: MetricSpec(
            m, "classification", lambda yt, yp: yp.mean(), "predict"
        )
        res1 = fit_and_score_fold(
            MockEstimator(),
            X,
            y,
            None,
            ids,
            None,
            train_idx=np.array([0, 2]),
            test_idx=np.array([1, 3]),
            metrics=["m1"],
            feature_selection_config=MockConfig(),
            calibration_config=MockConfig(),
            spec=spec,
        )
        assert "m1" in res1["scores"]

        # Proba missing path
        engine.get_metric_spec = lambda m: MetricSpec(
            m, "classification", lambda yt, yp: yp.mean(), "proba_or_score"
        )
        res2 = fit_and_score_fold(
            MockEstimator(y_score_val=np.zeros(2)),
            X,
            y,
            None,
            ids,
            None,
            train_idx=np.array([0, 2]),
            test_idx=np.array([1, 3]),
            metrics=["m2"],
            feature_selection_config=MockConfig(),
            calibration_config=MockConfig(),
            spec=SimpleNamespace(**{**spec.__dict__, "supports_proba": False}),
        )
        assert "y_score" in res2["preds"]
    finally:
        engine.get_metric_spec = old_get


def test_fit_estimator_complex():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(
        n_samples=20,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=42,
    )
    groups = np.repeat(np.arange(10), 2)
    fit_estimator(
        CalibratedClassifierCV(LogisticRegression(), cv=2),
        X,
        y,
        groups,
        MockConfig(),
        MockConfig(
            enabled=True, cv=SimpleNamespace(strategy="group_kfold", n_splits=2)
        ),
    )


def test_calibration_integration():
    config = ExperimentConfig(
        task="classification",
        models={"svm": LinearSVCConfig(max_iter=500, kind="classical")},
        metrics=["log_loss"],
        cv=CVConfig(strategy="stratified", n_splits=2),
        calibration=CalibrationConfig(
            enabled=True,
            method="sigmoid",
            cv=CVConfig(strategy="stratified", n_splits=2),
        ),
        n_jobs=1,
        verbose=False,
    )
    estimator = Experiment(config)._prepare_estimator("svm", config.models["svm"])
    assert estimator.__class__.__name__ == "CalibratedClassifierCV"


def test_fit_estimator_calibration_group_cv():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    X, y = make_classification(
        n_samples=10,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=42,
    )
    groups = np.repeat([0, 1], 5)

    cal_cfg = SimpleNamespace(cv=SimpleNamespace(strategy="group_kfold", n_splits=2))
    cal = CalibratedClassifierCV(LogisticRegression(), cv=GroupKFold(n_splits=2))

    fit_estimator(cal, X, y, groups, MockConfig(), cal_cfg)

    from coco_pipe.decoding._engine import _CVWithGroups

    assert isinstance(cal.cv, _CVWithGroups)


def test_importance_extraction_calibration_averaging():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(
        n_samples=10,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=42,
    )
    cal = CalibratedClassifierCV(LogisticRegression(), cv=2).fit(X, y)

    spec = SimpleNamespace(importance=("coefficients",), is_sparse_capable=True)

    # We need to ensure calibrated_classifiers_ is present
    assert hasattr(cal, "calibrated_classifiers_")

    # This should call the recursive path in extract_feature_importances
    imp = extract_feature_importances(cal, spec, calibration_enabled=True)
    assert imp.shape == (2,)


def test_compute_metric_safe_2d_generalizing():
    def scorer(yt, yp, **kw):
        return np.mean((yt - yp) ** 2)

    y_true = np.array([0, 1])
    y_gen = np.zeros((2, 2, 2))  # (n_samples, n_tr, n_te)
    y_gen[1, :, :] = 1.0  # Perfect predictions for y_true=1

    score = compute_metric_safe(scorer, y_true, y_gen, False, name="mse")
    assert score.shape == (2, 2)
    assert np.all(score == 0.0)


def test_extract_metadata_neural():
    class MockNeural(MockEstimator, NeuralTrainable):
        def get_artifact_metadata(self):
            return {"weight_norm": 1.0}

        def get_train_stage(self):
            return "final"

    est = MockNeural()
    meta = extract_metadata(est, None, MockConfig())
    assert meta["artifacts"] == {"weight_norm": 1.0}


def test_compact_search_results_missing_keys():
    est = SimpleNamespace(cv_results_={"params": [{"C": 1}]})
    res = compact_search_results(est)
    assert res == [{"candidate": 0, "params": {"C": 1}}]
