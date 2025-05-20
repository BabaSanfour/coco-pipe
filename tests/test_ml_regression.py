import pytest
import numpy as np
from sklearn.datasets import make_regression

from coco_pipe.ml.config import (
    REGRESSION_METRICS,
    REGRESSION_MODELS,
    MULTIOUTPUT_METRICS_REGRESSION,
    MULTIOUTPUT_MODELS_REGRESSION,
    DEFAULT_CV,
)
from coco_pipe.ml.base import BasePipeline
from coco_pipe.ml.regression import (
    SingleOutputRegressionPipeline,
    MultiOutputRegressionPipeline,
    RegressionPipeline,
)

# Helper small datasets
X_single = np.arange(20).reshape(10, 2)
y_single = X_single[:, 0] * 2.0 + 1.0

X_multi, y_multi = make_regression(
    n_samples=30, n_features=4, n_targets=3, noise=0.1, random_state=0
)

########################################################
# Fixtures
########################################################

@pytest.fixture(autouse=True)
def tmp_working_dir(tmp_path, monkeypatch):
    """Run tests in a temporary directory to avoid file pollution."""
    monkeypatch.chdir(tmp_path)
    yield

########################################################
# Tests for RegressionPipeline
########################################################

@pytest.mark.parametrize(
    "analysis_type, models, metrics, expected_task",
    [
        ("baseline", ["Linear Regression"], ["r2"], "singleoutput"),
        ("baseline", ["Random Forest"], ["neg_mse"], "singleoutput"),
        ("baseline", ["Linear Regression"], ["mean_r2"], "multioutput"),
    ],
)
def test_pipeline_detect_and_run_baseline(
    analysis_type, models, metrics, expected_task, monkeypatch
):
    # Select the appropriate dataset
    if expected_task == "singleoutput":
        X, y = X_single, y_single
    else:
        X, y = X_multi, y_multi

    # Monkey-patch save() to capture names without writing files
    saved = []

    def fake_save(self, name, res):
        saved.append(name)

    monkeypatch.setattr(RegressionPipeline, "save", fake_save)

    pipe = RegressionPipeline(
        X=X,
        y=y,
        analysis_type=analysis_type,
        models=models,
        metrics=metrics,
        random_state=0,
        cv_strategy="kfold",
        n_jobs=1,
        save_intermediate=True,
        results_file="testres"
    )

    results = pipe.run()

    # Ensure the underlying pipeline class matches the task
    cls_name = type(pipe.pipeline).__name__.lower()
    assert expected_task in cls_name

    # Results keys match requested models
    assert set(results.keys()) == set(models)

    # Each result has predictions and metrics
    for res in results.values():
        assert "predictions" in res and "metrics" in res

    # save() called once per model plus final
    assert len(saved) == len(models) + 1
    assert any(name.startswith("testres") for name in saved)

########################################################
# Tests for SingleOutputRegressionPipeline
########################################################

def test_single_output_metrics_correctness():
    # synthetic fold predictions
    fold_preds = [
        {
            "y_true": np.array([1.0, 2.0]),
            "y_pred": np.array([1.0, 2.0])
        },
        {
            "y_true": np.array([3.0, 4.0, 5.0]),
            "y_pred": np.array([3.0, 4.0, 6.0])
        }
    ]

    # dummy pipeline instance
    dummy_X = np.zeros((5, 2))
    dummy_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dummy_y_pred = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    pipe = SingleOutputRegressionPipeline(
        X=dummy_X,
        y=dummy_y,
        models="all",
        metrics=list(REGRESSION_METRICS.keys()),
        cv_kwargs={**DEFAULT_CV, "n_splits": 2, "cv_strategy": "kfold"},
        n_jobs=1
    )

    results = pipe.compute_metrics(fold_preds, pipe.metrics, pipe.metric_funcs)

    # r2: perfect prediction in first fold, imperfect in second
    expected_r2 = 0.7  # (1.0 * 2 + 0.4 * 3) / 5
    assert pytest.approx(results["metrics"]["r2"]["mean"], rel=1e-6) == expected_r2

    # neg_mse: only error in second fold
    expected_neg_mse = -0.2  # -(0.0 * 2 + 0.2 * 3) / 5
    assert pytest.approx(results["metrics"]["neg_mse"]["mean"], rel=1e-6) == expected_neg_mse

def test_baseline_all_models_run_single():
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=3,
        noise=0.1,
        random_state=0
    )
    metrics = ["r2", "neg_mse", "neg_mae"]
    count = 0
    for model_name in REGRESSION_MODELS.keys():
        pipe = SingleOutputRegressionPipeline(
            X=X,
            y=y,
            models=[model_name],
            metrics=metrics,
            random_state=0,
            n_jobs=1,
            cv_kwargs={**DEFAULT_CV, "n_splits": 5, "shuffle": True, "cv_strategy": "kfold"}
        )
        out = pipe.baseline(model_name)
        assert all(key in out for key in ("model", "feature_importances", "metrics", "predictions"))
        assert len(out["predictions"]["y_true"]) == len(y)
        for m in metrics:
            val = out["metrics"][m]["mean"]
            if m == "r2":
                assert val <= 1.0
            else:
                assert isinstance(val, float)
        count += 1
    assert count == len(REGRESSION_MODELS)

########################################################
# Tests for MultiOutputRegressionPipeline
########################################################

def test_target_validation_error_multioutput():
    X = np.zeros((10, 5))
    y = np.zeros(10)
    with pytest.raises(ValueError, match="Target must be 2D array for multivariate regression"):
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
    for name in MULTIOUTPUT_MODELS_REGRESSION:
        pipe = MultiOutputRegressionPipeline(
            X=X,
            y=y,
            models=[name],
            metrics=metrics,
            random_state=42,
            n_jobs=1,
            cv_kwargs={**DEFAULT_CV, "n_splits": 3, "cv_strategy": "kfold"}
        )
        out = pipe.baseline(name)
        assert out["predictions"]["y_true"].shape == y.shape
        for m in metrics:
            val = out["metrics"][m]["mean"]
            if m == "mean_r2":
                assert val <= 1.0
            else:
                assert isinstance(val, float)
        results.append(out)
    assert len(results) == len(MULTIOUTPUT_MODELS_REGRESSION)

def test_regression_pipeline_invalid_type():
    with pytest.raises(ValueError):
        RegressionPipeline(X=X_single, y=y_single, analysis_type="foo").run()
