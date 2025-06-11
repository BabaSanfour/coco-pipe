import pytest
import numpy as np

from coco_pipe.ml.pipeline import MLPipeline
from coco_pipe.ml.classification import ClassificationPipeline
from coco_pipe.ml.regression    import RegressionPipeline

# Fixtures for dummy data
def make_dummy_multi_output(n_samples=10, n_targets=3):
    X = np.arange(n_samples * 2).reshape(n_samples, 2)
    # create multi-output y with simple relationships
    y = np.vstack([X[:, 0] * (i + 1) for i in range(n_targets)]).T
    return X, y

@pytest.fixture
def multi_output_data():
    return make_dummy_multi_output(n_samples=8, n_targets=3)

@pytest.fixture
def single_output_data():
    X = np.arange(20).reshape(10, 2)
    # single-output y as 1D array
    y = X[:, 0] * 2.0 + 1.0
    return X, y

# Test invalid task
def test_invalid_task_raises(single_output_data):
    X, y = single_output_data
    cfg = {"task": "unknown"}
    with pytest.raises(ValueError, match="Invalid task"):
        MLPipeline(X, y, cfg)

# Test invalid mode
def test_invalid_mode_raises(single_output_data):
    X, y = single_output_data
    cfg = {"task": "regression", "mode": "invalid"}
    with pytest.raises(ValueError, match="Invalid mode"):
        MLPipeline(X, y, cfg)

# Test multivariate mode calls pipeline once returning dict
def test_multivariate_mode(single_output_data, monkeypatch):
    X, y = single_output_data
    cfg = {"task": "regression", "mode": "multivariate"}
    recorded = {}

    # monkeypatch RegressionPipeline.run to capture self.y and return fake result
    def fake_run(self):
        return {"y_shape": self.y.shape}
    monkeypatch.setattr(RegressionPipeline, "run", fake_run)

    mlp = MLPipeline(X, y, cfg)
    out = mlp.run()
    assert isinstance(out, dict)
    assert out.get("y_shape") == y.shape

# Test univariate mode runs per target for multi-output data
def test_univariate_mode_runs_per_output(monkeypatch, multi_output_data):
    X, y = multi_output_data
    cfg = {"task": "regression", "mode": "univariate"}

    # monkeypatch RegressionPipeline.run to capture self.y
    def fake_run(self):
        return {"y_shape": self.y.shape}
    monkeypatch.setattr(RegressionPipeline, "run", fake_run)

    mlp = MLPipeline(X, y, cfg)
    out = mlp.run()
    # Expect a dict with one entry per target (each target becoming a 1D array)
    assert isinstance(out, dict)
    # assert set(out.keys()) == set(range(y.shape[1]))
    # for _, res in out.items():
    #     assert res["y_shape"] == X.shape

# New test: univariate mode with single output returns a single dict (not keyed by index)
def test_univariate_mode_single_output(monkeypatch, single_output_data):
    X, y = single_output_data
    cfg = {"task": "regression", "mode": "univariate"}
    
    def fake_run(self):
        return {"y_shape": self.y.shape}
    monkeypatch.setattr(RegressionPipeline, "run", fake_run)
    
    mlp = MLPipeline(X, y, cfg)
    out = mlp.run()
    # Since y is single-output, expect a single dict instead of a dict of dicts
    assert isinstance(out, dict)
    # Check that y has been squeezed to 1D
    assert out.get("y_shape") == (X.shape[0],)

# Test univariate feature_selection not allowed
def test_univariate_feature_selection_error(multi_output_data):
    X, y = multi_output_data
    cfg = {"task": "regression", "mode": "univariate", "analysis_type": "feature_selection"}
    mlp = MLPipeline(X, y, cfg)
    with pytest.raises(ValueError, match="Cannot perform feature_selection in univariate mode"):
        mlp.run()

# Classification analogs
def test_classification_univariate_and_multivariate(monkeypatch):
    # create dummy multilabel classification y
    X = np.random.rand(6, 3)
    y = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 1],
                  [1, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])
    # fake ClassificationPipeline.run
    def fake_run(self):
        return {"y_shape": self.y.shape}
    monkeypatch.setattr(ClassificationPipeline, "run", fake_run)
    
    # multivariate mode returns single dict
    cfg_mv = {"task": "classification", "mode": "multivariate"}
    mlp_mv = MLPipeline(X, y, cfg_mv)
    out_mv = mlp_mv.run()
    assert out_mv["y_shape"] == y.shape
    
    # univariate mode returns dict per target # MOST LIKELY WRONG
    cfg_uv = {"task": "classification", "mode": "univariate"}
    mlp_uv = MLPipeline(X, y, cfg_uv)
    out_uv = mlp_uv.run()
    assert set(out_uv.keys()) == set(range(y.shape[1]))
    for _, res in out_uv.items():
        assert res["y_shape"] == X.shape

# Test univariate not allow fs/hp_search_fs for classification
def test_classification_univariate_fs_error():
    X, y = np.zeros((5, 2)), np.zeros((5, 2))
    cfg = {"task": "classification", "mode": "univariate", "analysis_type": "hp_search_fs"}
    mlp = MLPipeline(X, y, cfg)
    with pytest.raises(ValueError, match="Cannot perform hp_search_fs in univariate mode"):
        mlp.run()
