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

# --- Fixtures: tiny datasets ---
X1 = np.arange(20).reshape(10, 2)
y1 = X1[:, 0] * 2.0 + 1.0

X2, y2 = make_regression(
    n_samples=30, n_features=4, n_targets=3, noise=0.1, random_state=0
)

@pytest.fixture(autouse=True)
def tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield

# -------------------------------------------------------------------
# 1) compute_metrics for single‐target via BasePipeline
# -------------------------------------------------------------------
def test_compute_metrics_single():
    fold_preds = [
        {"y_true": np.array([1.0, 2.0]), "y_pred": np.array([1.0, 2.0])},
        {"y_true": np.array([3.0, 4.0, 5.0]), "y_pred": np.array([3.0, 4.0, 6.0])},
    ]
    metrics = ["r2", "neg_mse"]
    funcs = {m: REGRESSION_METRICS[m] for m in metrics}
    out = BasePipeline.compute_metrics(fold_preds, metrics, funcs)
    # Weighted means as before
    assert pytest.approx(out["metrics"]["r2"]["mean"], rel=1e-6) == 0.7
    assert pytest.approx(out["metrics"]["neg_mse"]["mean"], rel=1e-6) == -0.2
    assert len(out["predictions"]["y_true"]) == 5

# -------------------------------------------------------------------
# 2) Parametrized baseline for single‐target
# -------------------------------------------------------------------
@pytest.mark.parametrize("model_name", list(REGRESSION_MODELS.keys()))
@pytest.mark.parametrize("metrics", [
    ["r2"],
    ["neg_mse"],
    ["r2", "neg_mse", "neg_mae"],
])
def test_baseline_single_model_metrics(model_name, metrics):
    pipe = SingleOutputRegressionPipeline(
        X=X1,
        y=y1,
        models=[model_name],
        metrics=metrics,
        random_state=0,
        cv_kwargs={**DEFAULT_CV, "n_splits": 3, "cv_strategy": "kfold"},
        n_jobs=1
    )
    out = pipe.baseline(model_name)
    # Structure
    assert "predictions" in out and "metrics" in out
    # Only requested metrics present
    assert set(out["metrics"].keys()) == set(metrics)
    # values within expected ranges
    for m in metrics:
        val = out["metrics"][m]["mean"]
        if m == "r2":
            assert val <= 1.0
        else:
            assert isinstance(val, float)

# -------------------------------------------------------------------
# 3) Parametrized baseline for multioutput
# -------------------------------------------------------------------
@pytest.mark.parametrize("model_name", list(MULTIOUTPUT_MODELS_REGRESSION.keys()))
@pytest.mark.parametrize("metrics", [
    ["mean_r2"],
    ["neg_mean_mse"],
    ["mean_r2", "neg_mean_mae"]
])
def test_baseline_multioutput_model_metrics(model_name, metrics):
    pipe = MultiOutputRegressionPipeline(
        X=X2,
        y=y2,
        models=[model_name],
        metrics=metrics,
        random_state=0,
        cv_kwargs={**DEFAULT_CV, "n_splits": 3, "cv_strategy": "kfold"},
        n_jobs=1
    )
    out = pipe.baseline(model_name)
    assert "predictions" in out and "metrics" in out
    assert set(out["metrics"].keys()) == set(metrics)
    # shape of predictions
    pred = out["predictions"]["y_true"]
    assert pred.shape == y2.shape
    # check values
    for m in metrics:
        val = out["metrics"][m]["mean"]
        assert isinstance(val, float)

# -------------------------------------------------------------------
# 4) RegressionPipeline wrapper (baseline only)
# -------------------------------------------------------------------
def test_regression_pipeline_wrapper_baseline_all(monkeypatch):
    saved = []
    def fake_save(self, name, res):
        saved.append(name)
    monkeypatch.setattr(RegressionPipeline, "save", fake_save)

    # test both single‐output and multi‐output
    for X, y, MODELS, prefix in [
        (X1, y1, list(REGRESSION_MODELS.keys()), "singleoutput"),
        (X2, y2, list(MULTIOUTPUT_MODELS_REGRESSION.keys()), "multioutput")
    ]:
        cp = RegressionPipeline(
            X=X,
            y=y,
            analysis_type="baseline", 
            models=MODELS,
            random_state=0,

            # force at least 2 folds and use k‐fold CV
            cv_strategy="kfold",
            n_splits=2,

            n_jobs=1,
            save_intermediate=True,
            results_file="results_test",
            results_dir="results"
        )
        results = cp.run()

        # at least one save() call (final metadata)
        assert saved, "Expected at least one save call, but none were made"
        # every filename should include the correct task prefix
        assert all(prefix in name for name in saved), (
            f"Expected prefix {prefix!r} in all saved filenames, got: {saved}"
        )

        # verify results for each model
        assert set(results.keys()) == set(MODELS)
        for m in MODELS:
            assert "metrics" in results[m] and "predictions" in results[m]

        # expect one save per model + one final metadata save
        assert len(saved) == len(MODELS) + 1
        saved.clear()

def test_regression_pipeline_invalid_type():
    with pytest.raises(ValueError):
        RegressionPipeline(X=X1, y=y1, analysis_type="foo").run()
