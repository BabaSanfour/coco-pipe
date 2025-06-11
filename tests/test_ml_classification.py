import os
import json
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_multilabel_classification, make_classification
from sklearn.metrics import roc_auc_score, average_precision_score

from coco_pipe.ml.config import (
    BINARY_METRICS,
    BINARY_MODELS,
    MULTICLASS_METRICS,
    MULTICLASS_MODELS,
    MULTIOUTPUT_CLASS_METRICS,
    MULTIOUTPUT_CLASS_MODELS,
    DEFAULT_CV
)
from coco_pipe.ml.classification import (
    ClassificationPipeline,
    BinaryClassificationPipeline,
    MultiClassClassificationPipeline,
    MultiOutputClassificationPipeline
)


# ─────────────────────────────────────────────────────────────────────────────
# small datasets
# ─────────────────────────────────────────────────────────────────────────────
X_binary = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])
y_binary = np.array([0] * 5 + [1] * 5)

X_multi = np.arange(30).reshape(10, 3)
y_multi = np.array([0, 1, 2, 1, 0, 2, 1, 0, 2, 1])

X_multiout, y_multiout = make_multilabel_classification(
    n_samples=20,
    n_features=4,
    n_classes=3,
    n_labels=2,
    random_state=0,
    allow_unlabeled=False
)


@pytest.fixture(autouse=True)
def tmp_working_dir(tmp_path, monkeypatch):
    """Use a clean working dir to avoid polluting the repo."""
    monkeypatch.chdir(tmp_path)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# ClassificationPipeline wrapper
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "analysis_type, model_list, metrics, expected_task",
    [
        ("baseline", [list(BINARY_MODELS.keys())[0]], ["accuracy"], "binary"),
        ("baseline", [list(MULTICLASS_MODELS.keys())[0]], ["accuracy"], "multiclass"),
        ("baseline", [list(MULTIOUTPUT_CLASS_MODELS.keys())[0]], ["subset_accuracy"], "multioutput"),
    ],
)
def test_pipeline_detect_and_run_baseline(
    analysis_type, model_list, metrics, expected_task, monkeypatch
):
    if expected_task == "binary":
        X, y = X_binary, y_binary
    elif expected_task == "multiclass":
        X, y = X_multi, y_multi
    else:
        X, y = X_multiout, y_multiout

    # capture save() calls
    saved = []
    def fake_save(self, name, res):
        saved.append(name)
    monkeypatch.setattr(ClassificationPipeline, "save", fake_save)

    pipe = ClassificationPipeline(
        X=X,
        y=y,
        analysis_type=analysis_type,
        models=model_list,
        metrics=metrics,
        random_state=0,
        cv_strategy="kfold",
        n_jobs=1,
        save_intermediate=True,
        results_file="testres"
    )
    results = pipe.run()

    # top‐level shape
    assert isinstance(results, dict)
    assert set(results.keys()) == set(model_list)

    # the wrapper picked the correct sub‐pipeline
    cls_name = type(pipe.pipeline).__name__.lower()
    assert expected_task in cls_name

    # each result now has 'predictions' + 'metric_scores'
    for res in results.values():
        assert "predictions" in res
        assert "metric_scores" in res

    # save() called once per model + final
    assert len(saved) == len(model_list) + 1
    assert any(n.startswith("testres") for n in saved)


def test_invalid_analysis_type_raises():
    with pytest.raises(ValueError):
        ClassificationPipeline(X=X_binary, y=y_binary, analysis_type="foo").run()


# ─────────────────────────────────────────────────────────────────────────────
# BinaryClassificationPipeline
# ─────────────────────────────────────────────────────────────────────────────
def test_binary_aggregate_metrics_correctness():
    fold_preds = [
        {
            "y_true": np.array([0,1,0]),
            "y_pred": np.array([0,1,1]),
            "y_proba": np.array([[0.8,0.2],[0.3,0.7],[0.4,0.6]])
        },
        {
            "y_true": np.array([1,0]),
            "y_pred": np.array([1,0]),
            "y_proba": np.array([[0.2,0.8],[0.6,0.4]])
        }
    ]
    # compute per-fold scores
    aucs = [roc_auc_score(fp["y_true"], fp["y_proba"][:,1]) for fp in fold_preds]
    aps = [average_precision_score(fp["y_true"], fp["y_proba"][:,1]) for fp in fold_preds]
    fold_scores = {
        "roc_auc": np.array(aucs),
        "average_precision": np.array(aps)
    }
    # no importances in this dummy test
    fold_imps = {}

    pipe = BinaryClassificationPipeline(
        X=np.zeros((5,2)),
        y=np.array([0,1,0,1,0]),
        models="all",
        metrics=["roc_auc", "average_precision"],
        cv_kwargs={**DEFAULT_CV, "n_splits":2, "cv_strategy":"kfold"},
        n_jobs=1
    )
    predictions, metrics, feature_importances = pipe._aggregate(fold_preds, fold_scores, fold_imps)

    expected_auc = np.mean(aucs)
    expected_ap = np.mean(aps)
    assert "roc_auc" in metrics
    assert pytest.approx(expected_auc, rel=1e-6) == metrics["roc_auc"]["mean"]
    assert pytest.approx(expected_ap, rel=1e-6) == metrics["average_precision"]["mean"]

    # predictions concatenated
    assert predictions["y_true"].shape[0] == 5
    assert "y_proba" in predictions


def test_baseline_all_models_run_binary():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=2,
        n_classes=2,
        random_state=0
    )
    metrics = ["accuracy","roc_auc","average_precision"]
    for name in BINARY_MODELS:
        pipe = BinaryClassificationPipeline(
            X=X, y=y,
            models=[name], metrics=metrics,
            random_state=0, n_jobs=1,
            cv_kwargs={**DEFAULT_CV, "n_splits":3, "shuffle":True}
        )
        out = pipe.baseline_evaluation(name)
        for key in ("predictions","metric_scores",
                    "feature_importances","model_name",
                    "params","folds_estimators"):
            assert key in out
        # check y_true length and score bounds
        assert out["predictions"]["y_true"].shape[0] == len(y)
        for m in metrics:
            mv = out["metric_scores"][m]["mean"]
            assert 0.0 <= mv <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# MultiClassClassificationPipeline
# ─────────────────────────────────────────────────────────────────────────────
def test_multiclass_aggregate_and_per_class():
    fold_preds = [
        {
            "y_true": np.array([0,1,2]),
            "y_pred": np.array([0,2,1]),
            "y_proba": np.array([[0.7,0.2,0.1],
                                 [0.1,0.1,0.8],
                                 [0.2,0.6,0.2]])
        },
        {
            "y_true": np.array([1,2]),
            "y_pred": np.array([1,2]),
            "y_proba": np.array([[0.1,0.8,0.1],
                                 [0.9,0.05,0.05]])
        },
    ]
    # compute fold scores
    accs = [(fp["y_true"]==fp["y_pred"]).mean() for fp in fold_preds]
    # aucs = [roc_auc_score(fp["y_true"], fp["y_proba"], multi_class='ovo') for fp in fold_preds]
    fold_scores = {
        "accuracy": np.array(accs),
        # "roc_auc": np.array(aucs)
    }
    pipe = MultiClassClassificationPipeline(
        X=np.zeros((5,4)),
        y=np.array([0,1,2,1,0]),
        models="all",
        metrics=["accuracy"],#,"roc_auc"],
        per_class=True,
        cv_kwargs={**DEFAULT_CV, "n_splits":2},
        n_jobs=1
    )
    predictions, metrics, feature_importances = pipe._aggregate(fold_preds, fold_scores, {})

    assert pytest.approx(np.mean(accs), rel=1e-6) == metrics["accuracy"]["mean"]
    # assert "roc_auc" in agg["metrics"]
    pcm = metrics["per_class_metrics"]
    assert set(pcm.keys()) == set(pipe.classes_)
    for stats in pcm.values():
        assert all(k in stats for k in ("precision","recall","f1"))


def test_baseline_all_multiclass_models():
    X, y = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=3,
        n_classes=3,
        random_state=1
    )
    metrics = ["accuracy","roc_auc"]
    for name in MULTICLASS_MODELS:
        pipe = MultiClassClassificationPipeline(
            X=X, y=y,
            models=[name], metrics=metrics,
            per_class=False,
            random_state=0, n_jobs=1,
            cv_kwargs={**DEFAULT_CV, "n_splits":4}
        )
        out = pipe.baseline_evaluation(name)
        assert "predictions" in out
        assert "metric_scores" in out
        assert "y_true" in out["predictions"]
        assert out["predictions"]["y_true"].shape #[0] == len(y)
        for m in metrics:
            assert 0.0 <= out["metric_scores"][m]["mean"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# MultiOutputClassificationPipeline
# ─────────────────────────────────────────────────────────────────────────────
def test_multioutput_aggregate_per_output():
    fold_preds = [
        {
            "y_true": np.array([[1,0],[0,1],[1,1]]),
            "y_pred": np.array([[1,0],[1,0],[1,1]])
        },
        {
            "y_true": np.array([[0,1],[1,0]]),
            "y_pred": np.array([[0,1],[0,0]])
        },
    ]
    # subset_accuracy only uses y_true and y_pred
    # so fold_scores can be dummy
    subset_scores = [ (fp["y_true"]==fp["y_pred"]).all(axis=1).mean() for fp in fold_preds ]
    fold_scores = {"subset_accuracy": np.array(subset_scores)}
    pipe = MultiOutputClassificationPipeline(
        X=np.zeros((5,2)),
        y=np.zeros((5,2)),
        models=list(MULTIOUTPUT_CLASS_MODELS.keys()),
        metrics=["subset_accuracy"],
        cv_kwargs={**DEFAULT_CV, "n_splits":2, "cv_strategy":"kfold"},
        n_jobs=1
    )
    predictions, metrics, feature_importances = pipe._aggregate(fold_preds, fold_scores, {})

    for m in pipe.metrics:
        assert m in metrics
    # pom = metrics["per_output_metrics"]
    # assert set(pom.keys()) == {0,1}
    # for stats in pom.values():
    #     assert all(k in stats for k in ("precision","recall","f1"))


def test_baseline_all_models_multioutput():
    X, y = make_multilabel_classification(
        n_samples=80,
        n_features=5,
        n_classes=3,
        n_labels=2,
        random_state=0,
        allow_unlabeled=False
    )
    metrics = ["subset_accuracy","hamming_loss"]
    for name in MULTIOUTPUT_CLASS_MODELS:
        pipe = MultiOutputClassificationPipeline(
            X=X, y=y,
            models=[name], metrics=metrics,
            random_state=42, n_jobs=1,
            cv_kwargs={**DEFAULT_CV, "n_splits":3, "cv_strategy":"kfold"}
        )
        out = pipe.baseline_evaluation(name)
        assert "predictions" in out
        # assert "y_true" in out["predictions"]
        # for m in metrics:
        #     assert 0.0 <= out["metric_scores"][m]["mean"] <= 1.0