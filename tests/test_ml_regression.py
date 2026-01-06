import pytest
import numpy as np
from sklearn.datasets import make_regression

from coco_pipe.ml.config import (
    REGRESSION_METRICS,
    REGRESSION_MODELS,
    MULTIOUTPUT_REG_MODELS,
    MULTIOUTPUT_REG_METRICS,
    DEFAULT_CV,
)
from coco_pipe.ml.regression import (
    SingleOutputRegressionPipeline,
    MultiOutputRegressionPipeline,
    RegressionPipeline,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper small datasets
# ─────────────────────────────────────────────────────────────────────────────
X_single = np.arange(40).reshape(20, 2)  # enough samples for 2-fold CV
y_single = X_single[:, 0] * 2.0 + 1.0

X_multi, y_multi = make_regression(
    n_samples=50,   # enough samples for 2-fold CV
    n_features=4,
    n_targets=3,
    noise=0.1,
    random_state=0
)


@pytest.fixture(autouse=True)
def tmp_working_dir(tmp_path, monkeypatch):
    """Use a clean working dir to avoid polluting the repo."""
    monkeypatch.chdir(tmp_path)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# RegressionPipeline wrapper
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "analysis_type, model_list, metrics, expected_task",
    [
        ("baseline", ["Linear Regression"], ["r2"], "singleoutput"),
        ("baseline", ["Random Forest"], ["r2"], "singleoutput"),
        ("baseline", ["Linear Regression"], ["mean_r2", "neg_mean_mse"], "multioutput"),
    ],
)
def test_pipeline_detect_and_run_baseline(
    analysis_type, model_list, metrics, expected_task, monkeypatch
):
    if expected_task == "singleoutput":
        X, y = X_single, y_single
    else:
        X, y = X_multi, y_multi

    saved = []
    monkeypatch.setattr(RegressionPipeline, "save", lambda self, name, res: saved.append(name))

    pipe = RegressionPipeline(
        X=X,
        y=y,
        analysis_type=analysis_type,
        models=model_list,
        metrics=metrics,
        random_state=0,
        cv_strategy="kfold",
        n_splits=2,
        n_jobs=1,
        save_intermediate=True,
        results_file="testres"
    )
    results = pipe.run()

    # basic structure
    assert isinstance(results, dict)
    assert set(results.keys()) == set(model_list)
    cls_name = type(pipe.pipeline).__name__.lower()
    assert expected_task in cls_name

    # each result has predictions and metric_scores
    for out in results.values():
        assert "predictions" in out
        assert "metric_scores" in out

    # save called once per model + final
    assert len(saved) == len(model_list) + 1
    assert any(n.startswith("testres") for n in saved)


def test_regression_pipeline_invalid_type():
    with pytest.raises(ValueError):
        RegressionPipeline(X=X_single, y=y_single, analysis_type="foo").run()


# ─────────────────────────────────────────────────────────────────────────────
# SingleOutputRegressionPipeline
# ─────────────────────────────────────────────────────────────────────────────
def test_single_output_metrics_correctness():
    # two folds with simple preds
    fold_preds = [
        {"y_true": np.array([1.0, 2.0]), "y_pred": np.array([1.0, 2.0])},
        {"y_true": np.array([3.0, 4.0]), "y_pred": np.array([2.5, 4.5])},
    ]
    # define fold-level scores manually
    fold_scores = {
        "r2": np.array([1.0, 0.5]),           # perfect then half explained
        "mse": np.array([0.0, 0.25])          # zero then (0.5^2 + 0.5^2)/2 = 0.25
    }
    fold_importances = {}

    pipe = SingleOutputRegressionPipeline(
        X=np.zeros((4, 2)),
        y=np.zeros(4),
        models="all",
        metrics=["r2", "mse"],
        cv_kwargs={**DEFAULT_CV, "n_splits":2, "cv_strategy":"kfold"},
        n_jobs=1
    )
    # BasePipeline._aggregate returns tuple
    predictions, metrics, feature_importances = pipe._aggregate(
        fold_preds, fold_scores, fold_importances
    )

    # check concatenated predictions
    assert np.array_equal(predictions["y_true"], np.concatenate([fp["y_true"] for fp in fold_preds]))
    assert "y_pred" in predictions

    # metric means
    assert pytest.approx(np.mean(fold_scores["r2"])) == metrics["r2"]["mean"]
    assert pytest.approx(np.mean(fold_scores["mse"])) == metrics["mse"]["mean"]

    # feature_importances empty
    assert feature_importances == {}


def test_baseline_all_models_run_single():
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=3,
        noise=0.1,
        random_state=0
    )
    metrics = ["r2", "mse", "mae"]
    seen = 0
    for name in REGRESSION_MODELS:
        pipe = SingleOutputRegressionPipeline(
            X=X,
            y=y,
            models=[name],
            metrics=metrics,
            random_state=0,
            n_jobs=1,
            cv_kwargs={**DEFAULT_CV, "n_splits":5, "cv_strategy":"kfold"}
        )
        out = pipe.baseline_evaluation(name)
        # essential keys
        for key in ("model_name", "params", "predictions", "metric_scores"):
            assert key in out
        # prediction length matches
        assert out["predictions"]["y_true"].shape[0] == y.shape[0]
        # each metric present
        for m in metrics:
            assert m in out["metric_scores"]
        seen += 1
    assert seen == len(REGRESSION_MODELS)


def test_target_validation_error_multioutput():
    X = np.zeros((10, 5))
    y = np.zeros(10)
    with pytest.raises(ValueError, match="Target must be 2D array"):
        MultiOutputRegressionPipeline(X=X, y=y)


def test_baseline_all_models_multioutput():
    X, y = make_regression(
        n_samples=100,
        n_features=6,
        n_targets=3,
        noise=0.1,
        random_state=0
    )
    metrics = ["mean_r2", "neg_mean_mse", "neg_mean_mae"]
    seen = 0
    for name in MULTIOUTPUT_REG_MODELS:
        pipe = MultiOutputRegressionPipeline(
            X=X,
            y=y,
            models=[name],
            metrics=metrics,
            random_state=42,
            n_jobs=1,
            cv_kwargs={**DEFAULT_CV, "n_splits":4, "cv_strategy":"kfold"}
        )
        out = pipe.baseline_evaluation(name)
        # essential keys
        for key in ("model_name", "params", "predictions", "metric_scores"):
            assert key in out
        # shape preserved
        assert out["predictions"]["y_true"].shape == y.shape
        # each metric present
        for m in metrics:
            assert m in out["metric_scores"]
        seen += 1
    assert seen == len(MULTIOUTPUT_REG_MODELS)