import pytest
import numpy as np
from sklearn.datasets import make_multilabel_classification, make_classification
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.dummy import DummyClassifier

from coco_pipe.ml.config import (
    BINARY_METRICS,
    BINARY_MODELS,
    MULTICLASS_MODELS,
    DEFAULT_CV
)
from coco_pipe.ml.classification import ClassificationPipeline, BinaryClassificationPipeline, MultiClassClassificationPipeline, MultiOutputClassificationPipeline

# Helper small datasets
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


########################################################
# Fixtures
########################################################

@pytest.fixture(autouse=True)
def tmp_working_dir(tmp_path, monkeypatch):
    """Run tests in a temporary directory to avoid file pollution."""
    monkeypatch.chdir(tmp_path)
    yield


########################################################
# Tests for ClassificationPipeline
########################################################

@pytest.mark.parametrize(
    "analysis_type, models, metrics, expected_task",
    [
        ("baseline", ["Decision Tree"], ["accuracy"], "binary"),
        ("baseline", ["Logistic Regression"], ["accuracy"], "multiclass"),
        ("baseline", ["Decision Tree"], ["subset_accuracy"], "multioutput"),
    ],
)
def test_pipeline_detect_and_run_baseline(
    analysis_type, models, metrics, expected_task, monkeypatch
):
    # Select the appropriate dataset
    if expected_task == "binary":
        X, y = X_binary, y_binary
    elif expected_task == "multiclass":
        X, y = X_multi, y_multi
    else:
        X, y = X_multiout, y_multiout

    # Monkey-patch save() to capture names without writing files
    saved = []

    def fake_save(self, name, res):
        saved.append(name)

    monkeypatch.setattr(ClassificationPipeline, "save", fake_save)

    pipe = ClassificationPipeline(
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

    assert isinstance(results, dict)
    assert results is not None
    # add assertion for different parts of pipe (step by step to know what is working and whatnot)
    assert pipe.pipeline is not None
    assert pipe.results is not None
    assert pipe.results_dir is not None
    assert pipe.results_file is not None
    assert pipe.X is not None
    assert pipe.y is not None
    assert pipe.analysis_type is not None
    assert pipe.models is not None
    assert pipe.metrics is not None
    assert pipe.random_state is not None
    assert pipe.cv_strategy is not None
    assert pipe.pipeline is not None
    assert pipe.pipeline.model_configs is not None

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


def test_invalid_analysis_type_raises():
    with pytest.raises(ValueError):
        ClassificationPipeline(X=X_binary, y=y_binary, analysis_type="unknown").run()


########################################################
# Tests for BinaryClassificationPipeline
########################################################

def test_binary_metrics_correctness():
    # synthetic fold predictions
    fold_preds = [
        {
            "y_true": np.array([0, 1, 0]),
            "y_pred": np.array([0, 1, 1]),
            "y_proba": np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]]),
        },
        {
            "y_true": np.array([1, 0]),
            "y_pred": np.array([1, 0]),
            "y_proba": np.array([[0.2, 0.8], [0.6, 0.4]]),
        },
    ]

    # dummy pipeline instance
    dummy_X = np.zeros((5, 2))
    dummy_y = np.array([0, 1, 0, 1, 0])
    dummy_y_pred = np.array([0, 1, 1, 1, 0])
    pipe = BinaryClassificationPipeline(
        X=dummy_X,
        y=dummy_y,
        models="all",
        metrics=list(BINARY_METRICS.keys()),
        cv_kwargs={**DEFAULT_CV, "n_splits": 2, "cv_strategy": "kfold"},
        n_jobs=1
    )

    results = pipe.compute_metrics(fold_preds, pipe.metrics, pipe.metric_funcs)

    # accuracy: N_correct / N_total
    expected_acc = np.mean(dummy_y_pred == dummy_y)
    assert pytest.approx(results["metrics"]["accuracy"]["mean"], rel=1e-3) == expected_acc

    # concatenated y_true and y_proba
    y_true = np.concatenate([f["y_true"] for f in fold_preds])
    y_proba = np.concatenate([f["y_proba"] for f in fold_preds], axis=0)

    expected_roc = roc_auc_score(y_true, y_proba[:, 1])
    expected_ap = average_precision_score(y_true, y_proba[:, 1])

    assert pytest.approx(results["metrics"]["roc_auc"]["mean"], rel=1e-6) == expected_roc
    assert pytest.approx(results["metrics"]["average_precision"]["mean"], rel=1e-6) == expected_ap


def test_baseline_all_models_run_binary():
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=3,
        n_classes=2,
        random_state=0
    )
    metrics = ["accuracy", "roc_auc", "average_precision"]
    count = 0
    for model_name in BINARY_MODELS.keys():
        pipe = BinaryClassificationPipeline(
            X=X,
            y=y,
            models=[model_name],
            metrics=metrics,
            random_state=0,
            n_jobs=1,
            cv_kwargs={**DEFAULT_CV, "n_splits": 5, "shuffle": True},
        )
        out = pipe.baseline(model_name)
        assert all(key in out for key in ("model", "feature_importances", "metrics", "predictions"))
        assert len(out["predictions"]["y_true"]) == len(y)
        for m in metrics:
            val = out["metrics"][m]["mean"]
            assert 0.0 <= val <= 1.0
        count += 1
    assert count == len(BINARY_MODELS)


# ########################################################
# # Tests for MultiClassClassificationPipeline
# ########################################################

def test_multiclass_compute_metrics():
    fold_preds = [
        {
            "y_true": np.array([0, 1, 2]),
            "y_pred": np.array([0, 2, 1]),
            "y_proba": np.array([[0.7,0.2,0.1],[0.1,0.1,0.8],[0.2,0.6,0.2]]),
        },
        {
            "y_true": np.array([1, 0]),
            "y_pred": np.array([1, 0]),
            "y_proba": np.array([[0.1,0.8,0.1],[0.9,0.05,0.05]]),
        },
    ]

    dummy_X = np.zeros((5,4))
    dummy_y = np.array([0,1,2,1,0])
    pipe = MultiClassClassificationPipeline(
        X=dummy_X,
        y=dummy_y,
        models="all",
        metrics=["accuracy", "roc_auc"],
        per_class=True,
        cv_kwargs={**DEFAULT_CV, "n_splits": 2},
        n_jobs=1
    )

    results = pipe.compute_metrics(fold_preds, pipe.metrics, pipe.metric_funcs)

    # accuracy
    y_true = np.concatenate([f["y_true"] for f in fold_preds])
    y_pred = np.concatenate([f["y_pred"] for f in fold_preds])
    acc = np.mean(y_true == y_pred)
    assert pytest.approx(acc, rel=1e-6) == results["metrics"]["accuracy"]["mean"]

    # roc_auc
    y_proba = np.concatenate([f["y_proba"] for f in fold_preds], axis=0)
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=[0,1,2])
    expected_auc = roc_auc_score(y_true_bin, y_proba, average="weighted", multi_class="ovr")
    assert pytest.approx(results["metrics"]["roc_auc"]["mean"], rel=1e-6) == expected_auc

    # per-class breakdown
    pcm = results.get("per_class_metrics", {})
    assert set(pcm.keys()) == set(pipe.classes_)
    for stats in pcm.values():
        assert all(k in stats for k in ("precision", "recall", "f1"))


def test_baseline_all_multiclass_models():
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=4,
        n_classes=3,
        random_state=1
    )
    metrics = ["accuracy", "roc_auc"]
    results = []
    for name in MULTICLASS_MODELS:
        pipe = MultiClassClassificationPipeline(
            X=X,
            y=y,
            models=[name],
            metrics=metrics,
            per_class=False,
            random_state=0,
            n_jobs=1,
            cv_kwargs={**DEFAULT_CV, "n_splits": 5},
        )
        res = pipe.baseline(name)
        assert "model" in res
        assert res["predictions"]["y_true"].shape[0] == y.shape[0]
        for m in metrics:
            assert 0.0 <= res["metrics"][m]["mean"] <= 1.0
        results.append(res)
    assert len(results) == len(MULTICLASS_MODELS)


# ########################################################
# # Tests for MultiOutputClassificationPipeline
# ########################################################

def test_target_validation_error_multioutput():
    X = np.zeros((10, 5))
    y = np.zeros(10)
    with pytest.raises(ValueError, match="Target must be 2D for multi-output"):
        MultiOutputClassificationPipeline(X=X, y=y)

def test_compute_metrics_per_output():
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
    metrics = ["precision_samples","recall_samples","f1_samples","subset_accuracy"]
    pipe = MultiOutputClassificationPipeline(
        X=np.zeros((5,2)),
        y=np.zeros((5,2)), # Ensure y is 2D
        models=BINARY_MODELS.keys(),
        metrics=metrics,
        cv_kwargs={**DEFAULT_CV, "n_splits": 2, "cv_strategy": "kfold"},
        n_jobs=1
    )
    results = pipe.compute_metrics(fold_preds, pipe.metrics, pipe.metric_funcs)
    for m in metrics:
        assert m in results["metrics"]
    pom = results.get("per_output_metrics", {})
    assert set(pom.keys()) == {0, 1}
    for stats in pom.values():
        assert all(k in stats for k in ("precision", "recall", "f1"))

def test_baseline_all_models_multioutput():
    X, y = make_multilabel_classification(
        n_samples=100,
        n_features=6,
        n_classes=3,
        n_labels=2,
        random_state=0,
        allow_unlabeled=False
    )
    metrics = ["subset_accuracy", "hamming_loss"]
    results = []
    for name in BINARY_MODELS:
        pipe = MultiOutputClassificationPipeline(
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
            assert 0.0 <= out["metrics"][m]["mean"] <= 1.0
        results.append(out)
    assert len(results) == len(BINARY_MODELS)
