import pytest
import numpy as np

from coco_pipe.ml.pipeline import MLPipeline
from coco_pipe.ml.classification import ClassificationPipeline
from coco_pipe.ml.regression    import RegressionPipeline

@pytest.fixture(autouse=True)
def dummy_X_y():
    # simple dummy data for both tasks
    X = np.arange(20).reshape(10, 2)
    y_class = np.array([0, 1] * 5)
    y_regr = np.arange(10).astype(float)
    return X, y_class, y_regr

def test_invalid_task_raises(dummy_X_y):
    X, y_class, _ = dummy_X_y
    cfg = {"task": "not_a_task"}
    with pytest.raises(ValueError) as exc:
        # error is raised in __init__
        _ = MLPipeline(X, y_class, cfg)
    assert "Invalid task" in str(exc.value)

@pytest.mark.parametrize("task, y_key, ExpectedPipeline, metric", [
    ("classification", 1, ClassificationPipeline, "accuracy"),
    ("regression",     2, RegressionPipeline, "r2"),
])
def test_correct_pipeline_class_selected(dummy_X_y, task, y_key, ExpectedPipeline, metric, monkeypatch):
    X, y_class, y_regr = dummy_X_y
    y = dummy_X_y[y_key]
    cfg = {
        "task": task,
        "analysis_type": "hp_search",
        "models": ["Random Forest"],
        "metrics": [metric],
        "random_state": 123,
        "cv_strategy": "stratified",
        "n_splits": 3,
        "cv_kwargs": {},
        "n_jobs": 7,
        "n_features": 4,
        "direction": "backward",
        "search_type": "random",
        "n_iter": 5,
        "scoring": metric,
        "save_intermediate": False,
        "results_dir": "results",
        "results_file": "results"
    }

    recorded = {}
    def fake_run(self):
        recorded['pipeline_type'] = type(self)
        return {"dummy": "result"}

    # apply the patch just for this test
    monkeypatch.setattr(ExpectedPipeline, "run", fake_run, raising=True)

    mlp = MLPipeline(X, y, cfg)
    out = mlp.run()

    assert recorded['pipeline_type'] is ExpectedPipeline
    assert out == {"dummy": "result"}

def test_defaults_are_applied(dummy_X_y):
    X, y_class, _ = dummy_X_y
    cfg = {"task": "classification"}  # use only required field

    recorded = {}
    def fake_run(self):
        recorded.update({
            "analysis_type": self.analysis_type,
            "models": self.models,
            "metrics": self.metrics,
            "random_state": self.random_state,
            "cv_strategy": self.cv_strategy,
            "n_splits": self.n_splits,
            "n_features": self.n_features,
            "direction": self.direction,
            "search_type": self.search_type,
            "n_iter": self.n_iter,
            "scoring": self.scoring,
            "n_jobs": self.n_jobs,
            "save_intermediate": self.save_intermediate,
            "results_dir": self.results_dir,
            "results_file": self.results_file
        })
        return {}

    setattr(ClassificationPipeline, "run", fake_run)

    mlp = MLPipeline(X, y_class, cfg)
    mlp.run()

    # verify defaults
    assert recorded["analysis_type"] == "baseline"
    assert recorded["models"] == "all"
    assert recorded["metrics"] is None
    assert recorded["random_state"] == 42
    assert recorded["cv_strategy"] == "stratified"
    assert recorded["n_splits"] == 5
    assert recorded["n_features"] is None
    assert recorded["direction"] == "forward"
    assert recorded["search_type"] == "grid"
    assert recorded["n_iter"] == 100
    assert recorded["scoring"] is None
    assert recorded["n_jobs"] == -1
    assert recorded["save_intermediate"] is False
    assert recorded["results_dir"] == "results"
    assert recorded["results_file"] == "results"
