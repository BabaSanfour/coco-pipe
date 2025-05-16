import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from coco_pipe.ml.multiclass_classification import MultiClassPipeline
from coco_pipe.ml.config import MULTICLASS_MODELS, DEFAULT_CV

def test_multiclass_compute_metrics():
    # Synthetic fold predictions for 3 classes
    fold_preds = [
        {
            "y_true": np.array([0, 1, 2]),
            "y_pred": np.array([0, 2, 1]),
            "y_proba": np.array([
                [0.7, 0.2, 0.1],
                [0.1, 0.1, 0.8],
                [0.2, 0.6, 0.2]
            ])
        },
        {
            "y_true": np.array([1, 0]),
            "y_pred": np.array([1, 0]),
            "y_proba": np.array([
                [0.1, 0.8, 0.1],
                [0.9, 0.05, 0.05]
            ])
        }
    ]
    # Instantiate pipeline (X, y unused here)
    dummy_X = np.zeros((5,4))
    dummy_y = np.array([0,1,2,1,0])
    pipe = MultiClassPipeline(
        X=dummy_X,
        y=dummy_y,
        models="all",
        metrics=["accuracy", "roc_auc_ovr"],
        per_class=True,
        cv_kwargs={**DEFAULT_CV, "n_splits":2},
        n_jobs=1,
    )

    results = pipe.compute_metrics(fold_preds, pipe.metrics, pipe.metric_funcs)

    # Check accuracy
    y_true = np.concatenate([fp["y_true"] for fp in fold_preds])
    y_pred = np.concatenate([fp["y_pred"] for fp in fold_preds])
    acc = np.mean(y_true == y_pred)

    assert pytest.approx(acc, rel=1e-6) == results["metrics"]["accuracy"]["mean"]

    # Check roc_auc_ovr
    y_proba = np.concatenate([fp["y_proba"] for fp in fold_preds], axis=0)
    # binarize and compute
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=[0,1,2])
    expected_auc = roc_auc_score(y_true_bin, y_proba, average="weighted", multi_class="ovr")
    assert pytest.approx(results["metrics"]["roc_auc_ovr"]["mean"], rel=1e-6) == expected_auc

    # Check per-class metrics present
    pcm = results.get("per_class_metrics", {})
    assert set(pcm.keys()) == set(pipe.classes_)
    # each per-class entry has precision, recall, f1
    for cls, metrics_dict in pcm.items():
        assert all(k in metrics_dict for k in ("precision", "recall", "f1"))

def test_baseline_all_multiclass_models():
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=4,
        n_classes=3,
        n_redundant=0,
        random_state=1
    )
    metrics = ["accuracy", "roc_auc_ovr"]
    results = []
    for name in MULTICLASS_MODELS:
        pipe = MultiClassPipeline(
            X=X,
            y=y,
            models=[name],
            metrics=metrics,
            per_class=False,
            random_state=0,
            n_jobs=1,
            cv_kwargs={**DEFAULT_CV, "n_splits":5},
        )
        res = pipe.baseline(name)
        # essential keys
        assert "model" in res
        assert "predictions" in res and len(res["predictions"]["y_true"]) == len(y)
        assert "metrics" in res
        # metric values in [0,1]
        for m in metrics:
            val = res["metrics"][m]["mean"]
            assert 0.0 <= val <= 1.0
        results.append(res)
    assert len(results) == len(MULTICLASS_MODELS)

def test_per_class_option_effect():
    X, y = make_classification(
        n_samples=100,
        n_features=6,
        n_informative=3,
        n_classes=4,
        random_state=2
    )
    pipe = MultiClassPipeline(
        X=X,
        y=y,
        models="all",
        metrics=["accuracy"],
        per_class=True,
        random_state=0,
        n_jobs=1,
        cv_kwargs={**DEFAULT_CV, "n_splits":3},
    )
    res = pipe.baseline(next(iter(MULTICLASS_MODELS)))
    assert "per_class_metrics" in res
    pcm = res["per_class_metrics"]
    # Should have one entry per class
    assert set(pcm.keys()) == set(pipe.classes_)
    # Values between 0 and 1
    for cls_stats in pcm.values():
        for metric_name, metric_val in cls_stats.items():
            assert 0.0 <= metric_val <= 1.0
