import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from coco_pipe.ml.base import (
    CrossValidationStrategy,
    BasePipeline,
    PipelineError,
)


class DummyPipeline(BasePipeline):
    def run(self, estimator, metrics, metric_funcs, **cv_kwargs):
        cv_results = self.cross_validate(estimator, **cv_kwargs)
        metric_results = self.compute_metrics(
            cv_results["fold_predictions"], metrics, metric_funcs
        )
        cv_results.update(metric_results)
        return cv_results


def test_get_cv_splitter_stratified():
    splitter = CrossValidationStrategy.get_cv_splitter(
        "stratified", n_splits=3, shuffle=False, random_state=0
    )
    assert splitter.n_splits == 3
    assert splitter.shuffle is False
    assert splitter.random_state == 0


def test_get_cv_splitter_leave_p_out_and_group_kfold():
    # leave_p_out with n_groups>1
    cv1 = CrossValidationStrategy.get_cv_splitter("leave_p_out", n_groups=2)
    assert hasattr(cv1, "split")
    # leave_p_out with default (1) uses LeaveOneGroupOut
    cv2 = CrossValidationStrategy.get_cv_splitter("leave_p_out", n_groups=1)
    assert cv2.__class__.__name__ == "LeaveOneGroupOut"
    # group_kfold
    cv3 = CrossValidationStrategy.get_cv_splitter("group_kfold", n_splits=4)
    assert cv3.n_splits == 4


def test_get_cv_splitter_unknown():
    with pytest.raises(PipelineError):
        CrossValidationStrategy.get_cv_splitter("not_a_strategy")


def test_cross_validate_with_predictions_stratified():
    # simple binary balanced data
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 1] * 5)
    clf = DummyClassifier(strategy="most_frequent")
    res = CrossValidationStrategy.cross_validate_with_predictions(
        clf, X, y, cv_strategy="stratified", random_state=0
    )
    # should have 5 folds by default
    assert isinstance(res, dict)
    assert len(res["fold_predictions"]) == 5
    # final estimator trained
    assert hasattr(res["estimator"], "predict")
    # feature_importances should be None for DummyClassifier
    assert res["feature_importances"] is None


def test_cross_validate_requires_groups_for_group_strategies():
    X = np.zeros((6, 1))
    y = np.zeros(6)
    clf = DummyClassifier()
    with pytest.raises(PipelineError):
        CrossValidationStrategy.cross_validate_with_predictions(
            clf, X, y, cv_strategy="group_kfold"
        )

    with pytest.raises(PipelineError):
        CrossValidationStrategy.cross_validate_with_predictions(
            clf, X, y, cv_strategy="leave_p_out", n_groups=2
        )


def test_get_feature_importances_from_coef_and_importances():
    # test coef_ path
    X = np.random.randn(30, 4)
    y = np.random.randint(0, 2, size=30)
    lr = LogisticRegression().fit(X, y)
    coefs = CrossValidationStrategy.get_feature_importances(lr)
    assert coefs.shape == (4,)

    # test feature_importances_ path: use a tree-based model
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=10).fit(X, y)
    imps = CrossValidationStrategy.get_feature_importances(rf)
    assert imps.shape == (4,)


def test_compute_metrics_and_predictions():
    # two folds, perfect predictions
    fold_predictions = [
        {
            "y_true": np.array([0, 1]),
            "y_pred": np.array([0, 1]),
            "y_proba": np.array([[0.8, 0.2], [0.1, 0.9]]),
        },
        {
            "y_true": np.array([1, 0]),
            "y_pred": np.array([1, 0]),
            "y_proba": np.array([[0.6, 0.4], [0.7, 0.3]]),
        },
    ]
    metric_funcs = {"acc": lambda t, p: np.mean(t == p)}
    out = BasePipeline.compute_metrics(fold_predictions, ["acc"], metric_funcs)
    # mean accuracy 1.0
    assert out["metrics"]["acc"]["mean"] == pytest.approx(1.0)
    # concatenated length = 4
    assert len(out["predictions"]["y_true"]) == 4
    assert len(out["predictions"]["y_proba"]) == 4


def test_compute_metrics_missing_metric_func():
    fold_predictions = [
        {"y_true": np.array([0]), "y_pred": np.array([0])}
    ]
    # no 'f1' key in metric_funcs
    with pytest.raises(PipelineError):
        BasePipeline.compute_metrics(fold_predictions, ["f1"], {})


def test_basepipeline_validate_input_errors():
    # mismatched X/y lengths
    with pytest.raises(PipelineError):
        DummyPipeline(X=np.zeros((3, 2)), y=np.zeros(4))
    # wrong types
    with pytest.raises(PipelineError):
        DummyPipeline(X="not array", y=np.zeros(3))


def test_basepipeline_cross_validate_and_run_integration():
    X = np.arange(12).reshape(6, 2)
    y = np.array([0, 1] * 3)
    pipe = DummyPipeline(X, y, cv_strategy="stratified", random_state=0)
    clf = DummyClassifier(strategy="most_frequent")
    result = pipe.run(
        clf,
        metrics=["acc"],
        metric_funcs={"acc": lambda t, p: np.mean(t == p)},
    )
    assert "metrics" in result
    assert "predictions" in result
    assert isinstance(result["metrics"]["acc"]["scores"], list)
