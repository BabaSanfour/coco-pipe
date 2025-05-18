import pytest
import numpy as np

from coco_pipe.ml.pipeline import MLPipeline
from coco_pipe.ml.classification import ClassificationPipeline
from coco_pipe.ml.regression    import RegressionPipeline

@pytest.fixture(autouse=True)
def dummy_X_y():
    # simple dummy data for both tasks
    X = np.arange(20).reshape(10, 2)
    y_class = np.array([0,1] * 5)
    y_regr  = np.arange(10).astype(float)
    return X, y_class, y_regr

def test_invalid_task_raises(dummy_X_y):
    X, y_class, y_regr = dummy_X_y
    cfg = {"task": "not_a_task"}
    p = MLPipeline(X, y_class, cfg)
    with pytest.raises(ValueError) as exc:
        p.run()
    assert "Invalid task" in str(exc.value)

@pytest.mark.parametrize("task, y_key, ExpectedPipeline", [
    ("classification", 1, ClassificationPipeline),
    ("regression",     2, RegressionPipeline),
])
def test_correct_pipeline_class_selected(dummy_X_y, task, y_key, ExpectedPipeline):
    X, y_class, y_regr = dummy_X_y
    y = dummy_X_y[y_key]
    cfg = {
        "task": task,
        # make sure execute gets called with these
        "type": "hp_search",
        "models": ["A","B"],
        "metrics": ["m1","m2"],
        "random_state": 123,
        "cv_kwargs": {"strategy":"stratified","n_splits":3},
        "n_jobs": 7,
        "n_features": 4,
        "direction": "backward",
        "search_type": "random",
        "n_iter": 5,
        "scoring": "dummy_scorer",
    }

    # intercept the inner pipeline execute
    recorded = {}
    def fake_execute(self, **kwargs):
        # capture the self type and forwarding kwargs
        recorded['pipeline_type'] = type(self)
        recorded['kwargs'] = kwargs
        return {"dummy": "result"}

    # patch the execute method on both pipeline classes
    setattr(ExpectedPipeline, "execute", fake_execute)

    mlp = MLPipeline(X, y, cfg)
    out = mlp.run()

    # ensure the correct inner class was instantiated
    assert recorded['pipeline_type'] is ExpectedPipeline

    # ensure run() returned whatever fake_execute returned
    assert out == {"dummy": "result"}

    # ensure execute got the kwargs forwarded exactly
    assert recorded['kwargs'] == {
        "type":        cfg["type"],
        "n_features":  cfg["n_features"],
        "direction":   cfg["direction"],
        "search_type": cfg["search_type"],
        "n_iter":      cfg["n_iter"],
        "scoring":     cfg["scoring"],
    }

def test_defaults_are_applied(dummy_X_y):
    X, y_class, _ = dummy_X_y
    # omit many keys from config to trigger defaults
    cfg = {"task": "classification"}
    recorded = {}
    def fake_execute(self, **kwargs):
        recorded.update(kwargs)
        return {}

    setattr(ClassificationPipeline, "execute", fake_execute)

    mlp = MLPipeline(X, y_class, cfg)
    mlp.run()

    # default type
    assert recorded["type"] == "baseline"
    # default other kwargs should be present (None if missing)
    for name in ("n_features","direction","search_type","n_iter","scoring"):
        assert name in recorded

