import pytest
import numpy as np
from pytest import approx
from sklearn.datasets import make_regression
from coco_pipe.ml.config import REGRESSION_METRICS, REGRESSION_MODELS, DEFAULT_CV
from coco_pipe.ml.single_target_regression import SingleOutputRegressionPipeline

########################################################
# Test compute_metrics for neg_mean_squared_error
########################################################
def test_compute_metrics_neg_mse_weighted():
    # Fold 1: y_true = [0,0,0], y_pred = [1,1,1] → MSE=1 → neg_mse=-1
    # Fold 2: y_true = [0,0],   y_pred = [0,0]   → MSE=0 → neg_mse=0
    fold_preds = [
        {"y_true": np.zeros(3), "y_pred": np.ones(3)},
        {"y_true": np.zeros(2), "y_pred": np.zeros(2)},
    ]
    # Dummy pipeline just to access compute_metrics
    dummy_X = np.zeros((5, 1))
    dummy_y = np.zeros(5)
    pipe = SingleOutputRegressionPipeline(
        X=dummy_X,
        y=dummy_y,
        models="Linear Regression",
        metrics=["neg_mse"],
        random_state=0,
        n_jobs=1,
        cv_kwargs={**DEFAULT_CV, "n_splits": 2, "strategy": "kfold"}
    )

    results = pipe.compute_metrics(
        fold_preds,
        pipe.metrics,
        pipe.metric_funcs
    )

    # Weighted mean = (-1*3 + 0*2) / 5 = -0.6
    expected_mean = -0.6
    # Std = sqrt( [3*(-1+0.6)^2 + 2*(0+0.6)^2] / 5 ) = sqrt((0.48+0.72)/5)=sqrt(1.2/5)=sqrt(0.24)
    expected_std = np.sqrt(0.24)
    out = results["metrics"]["neg_mse"]
    assert out["mean"] == approx(expected_mean, rel=1e-6)
    assert out["std"] == approx(expected_std, rel=1e-6)

    # Predictions concatenated
    y_true_concat = np.concatenate([f["y_true"] for f in fold_preds])
    y_pred_concat = np.concatenate([f["y_pred"] for f in fold_preds])
    assert np.all(results["predictions"]["y_true"] == y_true_concat)
    assert np.all(results["predictions"]["y_pred"] == y_pred_concat)
    # No probabilities for regression
    assert results["predictions"]["y_proba"] is None


########################################################
# Tests for SingleTargetRegressionPipeline.baseline
########################################################
@pytest.mark.parametrize("model_name", list(REGRESSION_MODELS.keys()))
def test_baseline_all_models_single_target(model_name):
    # Create a simple regression dataset
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=0
    )
    metrics = ["r2", "neg_mse"]
    pipe = SingleOutputRegressionPipeline(
        X=X,
        y=y,
        models=[model_name],
        metrics=metrics,
        random_state=0,
        n_jobs=1,
        cv_kwargs={**DEFAULT_CV, "n_splits": 5, "strategy": "kfold"}
    )
    result = pipe.baseline(model_name)

    # Check structure
    assert "model" in result
    assert "predictions" in result and "metrics" in result
    preds = result["predictions"]
    assert preds["y_true"].shape[0] == y.shape[0]
    assert preds["y_pred"].shape[0] == y.shape[0]

    # Metric sanity checks
    r2 = result["metrics"]["r2"]["mean"]
    neg_mse = result["metrics"]["neg_mse"]["mean"]
    # R^2 should be ≤1
    assert r2 <= 1.0
    # neg_mse should be ≤0
    assert neg_mse <= 0.0


########################################################
# Test target validation
########################################################
def test_target_validation_raises_on_2d_y():
    X = np.zeros((10, 3))
    y_2d = np.zeros((10, 1))
    with pytest.raises(ValueError):
        SingleOutputRegressionPipeline(
            X=X,
            y=y_2d,
            models="Linear Regression"
        )

