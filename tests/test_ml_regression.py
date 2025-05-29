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

# Helper small datasets
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
    monkeypatch.chdir(tmp_path)
    yield


@pytest.mark.parametrize(
    "analysis_type, model, metrics, expected_task",
    [
        ("baseline", ["Linear Regression"], ["r2"], "singleoutput"),
        ("baseline", ["Random Forest"], ["r2"], "singleoutput"),
        ("baseline", ["Linear Regression"], ["mean_r2", "neg_mean_mse"], "multioutput"),
    ],
)
def test_pipeline_detect_and_run_baseline(
    analysis_type, model, metrics, expected_task, monkeypatch
):
    # pick dataset
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
        models=model,
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
    cls_name = type(pipe.pipeline).__name__.lower()
    assert expected_task in cls_name

    # keys and contents
    assert set(results.keys()) == set(model)
    for out in results.values():
        assert "predictions" in out
        assert "metric_scores" in out

    # save called once per model + final
    assert len(saved) == len(model) + 1
    assert any(name.startswith("testres") for name in saved)


def test_regression_pipeline_invalid_type():
    with pytest.raises(ValueError):
        RegressionPipeline(X=X_single, y=y_single, analysis_type="foo").run()


def test_single_output_metrics_correctness():
    fold_preds = [
        {"y_true": np.array([1.0, 2.0]), "y_pred": np.array([1.0, 2.0])},
        {"y_true": np.array([3.0, 4.0, 5.0]), "y_pred": np.array([3.0, 4.0, 6.0])},
    ]
    # dummy pipeline with specific metrics including neg_mse
    pipe = SingleOutputRegressionPipeline(
        X=np.zeros((5,2)),
        y=np.arange(5, dtype=float),
        models="all",
        metrics=["r2", "mse"],  # Include both mse and neg_mse
        cv_kwargs={**DEFAULT_CV, "n_splits":2, "cv_strategy":"kfold"},
        n_jobs=1
    )
    agg = pipe._aggregate(fold_preds)

    assert "r2" in agg["metrics"]
    assert "mse" in agg["metrics"]
    
    # R2 should be between 0 and 1
    mean_r2 = agg["metrics"]["r2"]["mean"]
    assert 0.0 <= mean_r2 <= 1.0
    
    # MSE should be non-negative
    assert agg["metrics"]["mse"]["mean"] >= 0.0
    

def test_baseline_all_models_run_single():
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=3,
        noise=0.1,
        random_state=0
    )
    metrics = ["r2", "mse", "mae"]
    count = 0
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
        # baseline_evaluation returns 'metric_scores'
        assert all(k in out for k in ("model_name", "params", "predictions", "metric_scores"))
        # assert out["predictions"]["y_true"].shape[0] == y.shape[0]
        # check each metric is present
        # for m in metrics:
        #     assert m in out["metric_scores"]
        count += 1
    assert count == len(REGRESSION_MODELS)


def test_target_validation_error_multioutput():
    X = np.zeros((10,5))
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
    results = []
    for name in MULTIOUTPUT_REG_MODELS:
        pipe = MultiOutputRegressionPipeline(
            X=X,
            y=y,
            models=[name],
            metrics=metrics,
            random_state=42,
            n_jobs=1,
            cv_kwargs={**DEFAULT_CV, "n_splits":3, "cv_strategy":"kfold"}
        )
        out = pipe.baseline_evaluation(name)
        # should have shape preserved
        # assert out["predictions"]["y_true"].shape == y.shape
        # for m in metrics:
        #     assert m in out["metric_scores"]
        results.append(out)
    assert len(results) == len(MULTIOUTPUT_REG_MODELS)
