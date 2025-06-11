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

# Test invalid task now raises when run() is invoked
def test_invalid_task_raises(single_output_data):
    X, y = single_output_data
    cfg = {"task": "unknown"}
    with pytest.raises(ValueError, match="Invalid task"):
        MLPipeline(X, y, None, cfg)


# Test invalid mode now raises when run() is invoked
def test_invalid_mode_raises(single_output_data):
    X, y = single_output_data
    cfg = {"task": "regression", "mode": "invalid"}
    with pytest.raises(ValueError, match="Invalid mode"):
        MLPipeline(X, y, None, cfg)

# Test multivariate mode calls pipeline once returning dict
def test_multivariate_mode(single_output_data, monkeypatch):
    X, y = single_output_data
    cfg = {"task": "regression", "mode": "multivariate"}
    def fake_run(self):
        return {"y_shape": self.y.shape}
    monkeypatch.setattr(RegressionPipeline, "run", fake_run)

    mlp = MLPipeline(X, y, None, cfg)
    out = mlp.run()
    assert isinstance(out, dict)
    assert out["y_shape"] == y.shape

# Test univariate mode runs per feature for multi-output data
def test_univariate_mode_runs_per_feature(monkeypatch, multi_output_data):
    X, y = multi_output_data
    cfg = {"task": "regression", "mode": "univariate"}
    def fake_run(self):
        return {"y_shape": self.y.shape}
    monkeypatch.setattr(RegressionPipeline, "run", fake_run)

    mlp = MLPipeline(X, y, None, cfg)
    out = mlp.run()
    assert isinstance(out, dict)
    expected_keys = set(range(X.shape[1]))
    assert set(out.keys()) == expected_keys
    for res in out.values():
        assert res["y_shape"] == y.shape

# Univariate mode with single-output still loops over features
def test_univariate_mode_single_output(monkeypatch, single_output_data):
    X, y = single_output_data
    cfg = {"task": "regression", "mode": "univariate"}
    def fake_run(self):
        return {"y_shape": self.y.shape}
    monkeypatch.setattr(RegressionPipeline, "run", fake_run)

    mlp = MLPipeline(X, y, None, cfg)
    out = mlp.run()
    assert isinstance(out, dict)
    expected_keys = set(range(X.shape[1]))
    assert set(out.keys()) == expected_keys
    for res in out.values():
        assert res["y_shape"] == y.shape

# Feature-selection not allowed in univariate
def test_univariate_feature_selection_error(multi_output_data):
    X, y = multi_output_data
    cfg = {
        "task": "regression",
        "mode": "univariate",
        "analysis_type": "feature_selection"
    }
    mlp = MLPipeline(X, y, None, cfg)
    with pytest.raises(ValueError, match="Cannot perform feature_selection in univariate mode"):
        mlp.run()

# Classification analogs
def test_classification_modes(monkeypatch):
    X = np.random.rand(6, 3)
    y = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])
    def fake_run(self):
        return {"y_shape": self.y.shape}
    monkeypatch.setattr(ClassificationPipeline, "run", fake_run)

    # Multivariate
    cfg_mv = {"task": "classification", "mode": "multivariate"}
    mlp_mv = MLPipeline(X, y, None, cfg_mv)
    out_mv = mlp_mv.run()
    assert out_mv["y_shape"] == y.shape

    # Univariate
    cfg_uv = {"task": "classification", "mode": "univariate"}
    mlp_uv = MLPipeline(X, y, None, cfg_uv)
    out_uv = mlp_uv.run()
    expected_keys = set(range(X.shape[1]))
    assert set(out_uv.keys()) == expected_keys
    for res in out_uv.values():
        assert res["y_shape"] == y.shape

# Classification FS+HP search not allowed in univariate
def test_classification_univariate_fs_error():
    X, y = np.zeros((5, 2)), np.zeros((5, 2))
    cfg = {
        "task": "classification",
        "mode": "univariate",
        "analysis_type": "hp_search_fs"
    }
    mlp = MLPipeline(X, y, None, cfg)
    with pytest.raises(ValueError, match="Cannot perform hp_search_fs in univariate mode"):
        mlp.run()
