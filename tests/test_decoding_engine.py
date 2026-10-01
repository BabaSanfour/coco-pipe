from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding._engine import (
    GroupedSequentialFeatureSelector,
    _release_foundation_memory,
    _SafeSelectKBest,
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
    FeatureSelectionConfig,
    LinearSVCConfig,
)
from coco_pipe.decoding.interfaces import NeuralTrainable
from coco_pipe.decoding.registry import EstimatorSpec


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

    from coco_pipe.decoding import _engine as engine
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


# --- sample_weight tests ---


class _WeightCapturingClassifier(BaseEstimator, ClassifierMixin):
    """Records the sample_weight passed to fit(); predict always returns zeros."""

    _estimator_type = "classifier"
    classes_ = np.array([0, 1])

    def fit(self, X, y, sample_weight=None):
        self.recorded_weight_ = sample_weight
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


def _make_spec(**overrides):
    base = {
        "supports_proba": False,
        "supports_decision_function": False,
        "importance": ("unavailable",),
        "supports_groups": False,
        "grouped_metadata": "none",
        "is_sparse_capable": False,
        "family": "linear",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_fit_estimator_routes_sample_weight():
    """fit_estimator forwards sample_weight to clf step inside a Pipeline."""

    clf = _WeightCapturingClassifier()
    pipe = Pipeline([("clf", clf)])
    sw = np.array([1.0, 2.0, 3.0])

    fit_estimator(
        pipe,
        np.zeros((3, 2)),
        np.array([0, 1, 0]),
        None,
        MockConfig(),
        MockConfig(),
        sample_weight=sw,
    )

    assert np.allclose(clf.recorded_weight_, sw)


def test_fit_estimator_no_sample_weight_when_none():
    """fit_estimator passes None → clf.recorded_weight_ is None."""
    clf = _WeightCapturingClassifier()
    pipe = Pipeline([("clf", clf)])

    fit_estimator(
        pipe,
        np.zeros((3, 2)),
        np.array([0, 1, 0]),
        None,
        MockConfig(),
        MockConfig(),
        sample_weight=None,
    )

    assert clf.recorded_weight_ is None


def test_fit_estimator_skips_unsupported_clf():
    """fit_estimator does NOT crash when clf.fit lacks sample_weight param."""

    class NoWeightClf(BaseEstimator, ClassifierMixin):
        _estimator_type = "classifier"
        classes_ = np.array([0, 1])

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    pipe = Pipeline([("clf", NoWeightClf())])
    sw = np.array([1.0, 2.0, 3.0])
    # should not raise
    fit_estimator(
        pipe,
        np.zeros((3, 2)),
        np.array([0, 1, 0]),
        None,
        MockConfig(),
        MockConfig(),
        sample_weight=sw,
    )


def test_fit_and_score_fold_sample_weight_train_only():
    """Only the training-fold slice of sample_weight reaches the classifier."""
    from coco_pipe.decoding import _engine as engine
    from coco_pipe.decoding._metrics import MetricSpec

    clf = _WeightCapturingClassifier()
    pipe = Pipeline([("clf", clf)])
    X = np.zeros((6, 2))
    y = np.array([0, 0, 0, 1, 1, 1])
    ids = np.arange(6).astype(str)
    sw = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    train_idx = np.array([0, 1, 3, 4])
    test_idx = np.array([2, 5])

    old_get = engine.get_metric_spec
    try:
        engine.get_metric_spec = lambda m: MetricSpec(
            m, "classification", lambda yt, yp: float(yp.mean()), "predict"
        )
        fit_and_score_fold(
            pipe,
            X,
            y,
            None,
            ids,
            None,
            train_idx=train_idx,
            test_idx=test_idx,
            metrics=["acc"],
            feature_selection_config=MockConfig(),
            calibration_config=MockConfig(),
            spec=_make_spec(),
            sample_weight=sw,
        )
    finally:
        engine.get_metric_spec = old_get

    # Only train-fold weights should have been forwarded
    expected = sw[train_idx]
    assert np.allclose(clf.recorded_weight_, expected), (
        f"Expected {expected}, got {clf.recorded_weight_}"
    )


def test_experiment_run_rejects_length_mismatch():
    """Experiment.run raises ValueError when sample_weight length != len(X)."""
    import pytest

    from coco_pipe.decoding import Experiment, ExperimentConfig
    from coco_pipe.decoding.configs import CVConfig, LogisticRegressionConfig

    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig()},
        cv=CVConfig(strategy="stratified", n_splits=2),
        n_jobs=1,
        verbose=False,
    )
    X = np.zeros((10, 2))
    y = np.zeros(10, dtype=int)
    y[5:] = 1

    with pytest.raises(ValueError, match="sample_weight length"):
        Experiment(config).run(X, y, sample_weight=np.ones(5))


def test_grouped_sfs():
    from sklearn.model_selection import KFold

    sfs = GroupedSequentialFeatureSelector(
        LogisticRegression(), n_features_to_select=1, cv=KFold(2)
    )
    X = np.random.randn(10, 3)
    y = np.array([0, 1] * 5)
    groups = np.array([0] * 5 + [1] * 5)
    sfs.fit_transform(X, y, groups=groups)
    assert sfs.n_features_to_select_ == 1


def test_safe_select_k_best():
    skb = _SafeSelectKBest(k=10)
    X = np.random.randn(10, 3)
    y = np.array([0, 1] * 5)
    skb.fit(X, y)
    assert skb.effective_k_ == 3  # Should clamp to 3
    xt = skb.transform(X)
    assert xt.shape[1] == 3
    assert skb.get_support().sum() == 3


def test_compute_metric_safe_temporal_accuracy():
    y_true = np.array([0, 1])
    # Sliding
    y_est_sl = np.array([[0, 1], [1, 1]])  # (samples, times)
    acc_sl = compute_metric_safe(
        lambda y, p: None,
        y_true,
        y_est_sl,
        is_multiclass=False,
        is_proba=False,
        name="accuracy",
    )
    assert acc_sl.shape == (2,)
    # Generalizing
    y_est_gen = np.zeros((2, 2, 2))
    acc_gen = compute_metric_safe(
        lambda y, p: None,
        y_true,
        y_est_gen,
        is_multiclass=False,
        is_proba=False,
        name="accuracy",
    )
    assert acc_gen.shape == (2, 2)
    # Invalid
    with pytest.raises(ValueError):
        compute_metric_safe(lambda y, p: None, y_true, np.zeros((2, 2, 2, 2)), False)


def test_fit_estimator_pipeline_groups():
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("fs", _SafeSelectKBest(k=1)),
            ("clf", LogisticRegression()),
        ]
    )
    fs_cfg = FeatureSelectionConfig(enabled=True, method="k_best")
    X = np.random.randn(10, 3)
    y = np.array([0, 1] * 5)
    groups = np.array([0] * 5 + [1] * 5)
    sample_weight = np.ones(10)
    fit_estimator(
        pipe,
        X,
        y,
        groups,
        feature_selection_config=fs_cfg,
        calibration_config=None,
        sample_weight=sample_weight,
    )
    assert hasattr(pipe.named_steps["clf"], "coef_")


def test_extract_feature_importances_pipeline():
    pipe = Pipeline([("fs", _SafeSelectKBest(k=2)), ("clf", LogisticRegression())])
    X = np.random.randn(10, 3)
    y = np.array([0, 1] * 5)
    pipe.fit(X, y)

    from coco_pipe.decoding.registry import get_estimator_spec

    spec = get_estimator_spec("LogisticRegression")
    imp = extract_feature_importances(pipe, spec, fs_enabled=True)
    assert imp is not None
    assert len(imp) == 3


def test_fit_and_score_fold_needs_group_routing():
    from coco_pipe.decoding._engine import fit_and_score_fold
    from coco_pipe.decoding.configs import (
        FeatureSelectionConfig,
    )
    from coco_pipe.decoding.scalers import SubjectStandardScaler

    pipe = Pipeline(
        [("scaler", SubjectStandardScaler()), ("clf", LogisticRegression())]
    )
    from coco_pipe.decoding.registry import get_estimator_spec

    spec = get_estimator_spec("LogisticRegression")
    fs_cfg = FeatureSelectionConfig()
    cal_cfg = FeatureSelectionConfig(enabled=False)

    X = np.random.randn(10, 3)
    y = np.array([0, 1] * 5)
    groups = np.array([0] * 5 + [1] * 5)
    sample_ids = np.arange(10)
    train_idx = np.arange(6)
    test_idx = np.arange(6, 10)

    # We just need to mock warning context or ensure it passes without error
    res = fit_and_score_fold(
        estimator=pipe,
        X=X,
        y=y,
        groups=groups,
        sample_ids=sample_ids,
        sample_metadata=None,
        train_idx=train_idx,
        test_idx=test_idx,
        metrics=["accuracy"],
        feature_selection_config=fs_cfg,
        calibration_config=cal_cfg,
        spec=spec,
    )
    assert "accuracy" in res["scores"]


def test_extract_feature_importances_feature_importances():
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=2)
    X = np.random.randn(10, 3)
    y = np.array([0, 1] * 5)
    clf.fit(X, y)
    from coco_pipe.decoding.registry import get_estimator_spec

    spec = get_estimator_spec("RandomForestClassifier")
    imp = extract_feature_importances(clf, spec)
    assert imp is not None
    assert len(imp) == 3


def test_extract_feature_importances_calibrated():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    clf = CalibratedClassifierCV(LogisticRegression(), cv=2)
    X = np.random.randn(10, 3)
    y = np.array([0, 1] * 5)
    clf.fit(X, y)
    from coco_pipe.decoding.registry import get_estimator_spec

    spec = get_estimator_spec("LogisticRegression")
    imp = extract_feature_importances(clf, spec, calibration_enabled=True)
    assert imp is not None
    assert len(imp) == 3


def test_extract_metadata_fs():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from coco_pipe.decoding._engine import extract_metadata
    from coco_pipe.decoding.configs import FeatureSelectionConfig

    pipe = Pipeline([("fs", _SafeSelectKBest(k=2)), ("clf", LogisticRegression())])
    X = np.random.randn(10, 3)
    y = np.array([0, 1] * 5)
    pipe.fit(X, y)

    fs_cfg = FeatureSelectionConfig(enabled=True, method="k_best")
    from coco_pipe.decoding.registry import get_estimator_spec

    spec = get_estimator_spec("LogisticRegression")
    meta = extract_metadata(pipe, spec, fs_cfg)
    assert "feature_scores" in meta


def test_extract_feature_importances_edge_cases():
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    # search_enabled=True
    X = np.random.randn(10, 3)
    y = np.array([0, 1] * 5)
    search = GridSearchCV(LogisticRegression(), {"C": [1.0]})
    search.fit(X, y)

    from coco_pipe.decoding.registry import get_estimator_spec

    spec_lr = get_estimator_spec("LogisticRegression")
    imp1 = extract_feature_importances(search, spec_lr, search_enabled=True)
    assert imp1 is not None

    # _get_raw_importance returns None
    spec_svc = EstimatorSpec("SVC", ["feature_importances"], True, True, True)
    svc = SVC()
    svc.fit(X, y)
    imp2 = extract_feature_importances(svc, spec_svc)
    assert imp2 is None

    # fs_enabled but raw_imp is None
    pipe = Pipeline([("fs", _SafeSelectKBest(k=2)), ("clf", SVC())])
    pipe.fit(X, y)
    imp3 = extract_feature_importances(pipe, spec_svc, fs_enabled=True)
    assert imp3 is None


def test_release_foundation_memory_walks_wrappers():
    """release_memory is called on every component reachable through the
    standard sklearn container attributes (pipeline steps, search/calibration
    wrappers)."""
    step_hook = Mock()
    inner_hook = Mock()
    inner = MockEstimator(release_memory=inner_hook)
    step = MockEstimator(release_memory=step_hook, base_estimator=inner)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", step)])

    _release_foundation_memory(pipe)

    step_hook.assert_called_once()
    inner_hook.assert_called_once()


def test_release_foundation_memory_noop_on_classical():
    """An estimator without a release_memory hook is left untouched and does
    not raise (classical models must be unaffected)."""
    # No exception, returns None; nothing to assert beyond "does not blow up".
    _release_foundation_memory(MockEstimator())
    _release_foundation_memory(None)


def test_release_foundation_memory_swallows_hook_errors():
    """A component whose release_memory raises must not fail the fold that
    already produced its scores."""
    boom = Mock(side_effect=RuntimeError("gpu gone"))
    est = MockEstimator(release_memory=boom)

    # Should not propagate.
    _release_foundation_memory(est)
    boom.assert_called_once()


def test_release_foundation_memory_handles_cycles():
    """The seen-set prevents infinite loops when wrappers reference themselves
    (MockEstimator.best_estimator_ defaults to self)."""
    hook = Mock()
    est = MockEstimator(release_memory=hook)  # best_estimator_ is self
    _release_foundation_memory(est)
    hook.assert_called_once()


def _run_fold_with_family(family, estimator, monkeypatch):
    """Drive fit_and_score_fold with a minimal spec of the given family."""
    from coco_pipe.decoding import _engine as engine
    from coco_pipe.decoding._metrics import MetricSpec

    monkeypatch.setattr(
        engine,
        "get_metric_spec",
        lambda m: MetricSpec(m, "classification", lambda yt, yp: yp.mean(), "predict"),
    )
    spec = SimpleNamespace(
        supports_proba=False,
        supports_decision_function=False,
        importance=("unavailable",),
        supports_groups=True,
        grouped_metadata="none",
        is_sparse_capable=False,
        family=family,
    )
    X, y = np.zeros((4, 2)), np.array([0, 0, 1, 1])
    ids = np.array(["a", "b", "c", "d"])
    return fit_and_score_fold(
        estimator,
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


def test_fit_and_score_fold_releases_foundation_only(monkeypatch):
    """The teardown fires for the foundation family and is skipped for
    classical families so their folds pay no gc/torch overhead."""
    fired = {"count": 0}
    monkeypatch.setattr(
        "coco_pipe.decoding._engine._release_foundation_memory",
        lambda est: fired.__setitem__("count", fired["count"] + 1),
    )

    _run_fold_with_family("linear", MockEstimator(), monkeypatch)
    assert fired["count"] == 0

    _run_fold_with_family("foundation", MockEstimator(), monkeypatch)
    assert fired["count"] == 1
