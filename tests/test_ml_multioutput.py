import pytest
import numpy as np
from sklearn.datasets import make_multilabel_classification

from coco_pipe.ml.multivariate_classification import MultiOutputClassificationPipeline
from coco_pipe.ml.config import BINARY_MODELS, DEFAULT_CV

def test_target_validation_error():
    X = np.zeros((10, 5))
    y = np.zeros(10)  # 1D target
    with pytest.raises(ValueError):
        MultiOutputClassificationPipeline(X=X, y=y)

def test_compute_metrics_per_output():
    # Create synthetic fold preds for 2 outputs
    fold_preds = [
        {
            "y_true": np.array([[1, 0], [0, 1], [1, 1]]),
            "y_pred": np.array([[1, 0], [1, 0], [1, 1]]),
        },
        {
            "y_true": np.array([[0, 1], [1, 0]]),
            "y_pred": np.array([[0, 1], [0, 0]]),
        }
    ]
    # Use sample-based precision/recall/f1
    metrics = ["precision_samples", "recall_samples", "f1_samples", "subset_accuracy"]
    pipe = MultiOutputClassificationPipeline(
        X=np.zeros((5, 2)), y=np.zeros((5, 2)),
        models="all", metrics=metrics,
        cv_kwargs={**DEFAULT_CV, "n_splits": 2, "strategy": "kfold"},
        n_jobs=1
    )
    results = pipe.compute_metrics(fold_preds, pipe.metrics, pipe.metric_funcs)

    # Check top-level metrics present
    for m in metrics:
        assert m in results["metrics"]

    # Check per-output metrics exist
    pom = results.get("per_output_metrics", {})
    # Two outputs => keys 0 and 1
    assert set(pom.keys()) == {0, 1}
    for out_idx, stats in pom.items():
        # each stats dict has some of the metrics
        assert "precision" in stats and "recall" in stats and "f1" in stats

def test_baseline_all_models_multioutput():
    X, y = make_multilabel_classification(
        n_samples=100,
        n_features=6,
        n_classes=3,  # Reduced from 4 to ensure consistent output size
        n_labels=2,
        random_state=0,
        allow_unlabeled=False  # Ensure consistent number of labels
    )
    metrics = ["subset_accuracy", "hamming_loss"]
    results = []
    for name in BINARY_MODELS:
        pipe = MultiOutputClassificationPipeline(
            X=X, y=y,
            models=[name], metrics=metrics,
            random_state=42, n_jobs=1,
            cv_kwargs={**DEFAULT_CV, "n_splits": 3, "strategy": "kfold"}
        )
        out = pipe.baseline(name)
        assert "predictions" in out and "metrics" in out
        # predictions shape matches y
        pred = out["predictions"]["y_true"]
        assert pred.shape == y.shape
        for m in metrics:
            val = out["metrics"][m]["mean"]
            # subset_accuracy and hamming_loss in [0,1]
            assert 0.0 <= val <= 1.0
        results.append(out)
    assert len(results) == len(BINARY_MODELS)
