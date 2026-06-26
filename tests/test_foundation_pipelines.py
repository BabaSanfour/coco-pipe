# tests/fm/test_foundation_pipelines.py

import numpy as np
import pytest
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

from coco_pipe.fm.pipeline import (
    FoundationRegressor,
    FoundationRegressionPipeline,
    FoundationClassifier,
    FoundationClassificationPipeline,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper dummy embedder
# ─────────────────────────────────────────────────────────────────────────────

class DummyEmbedder:
    """
    Simple deterministic embedder used for testing.

    Maps any X (shape [n_samples, ...]) to 2D embeddings of shape [n_samples, 4].
    """
    def __call__(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        return np.tile(np.array([[1.0, 2.0, 3.0, 4.0]]), (n, 1))


@pytest.fixture
def dummy_embedder():
    return DummyEmbedder()


# ─────────────────────────────────────────────────────────────────────────────
# FoundationRegressor tests
# ─────────────────────────────────────────────────────────────────────────────

def test_foundation_regressor_fit_and_predict(dummy_embedder):
    X = np.arange(20).reshape(10, 2)
    y = X[:, 0] * 2.0 + 1.0

    reg = FoundationRegressor(
        embed_fn=dummy_embedder,
        base_regressor=Ridge(random_state=0),
    )
    reg.fit(X, y)
    y_pred = reg.predict(X)

    assert y_pred.shape == y.shape


class BadEmbedder1D:
    def __call__(self, X):
        X = np.asarray(X)
        # 1D output is invalid (should be [n_samples, d])
        return X[:, 0]


def test_foundation_regressor_raises_on_non_2d_embed():
    X = np.arange(20).reshape(10, 2)
    y = X[:, 0]

    reg = FoundationRegressor(embed_fn=BadEmbedder1D())
    with pytest.raises(ValueError, match="Expected 2D embeddings"):
        reg.fit(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# FoundationClassifier tests
# ─────────────────────────────────────────────────────────────────────────────

def test_foundation_classifier_fit_and_predict(dummy_embedder):
    X = np.random.rand(12, 3)
    y = (X[:, 0] > 0.5).astype(int)

    clf = FoundationClassifier(
        embed_fn=dummy_embedder,
        base_classifier=LogisticRegression(max_iter=500),
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert y_pred.shape == y.shape


# ─────────────────────────────────────────────────────────────────────────────
# FoundationRegressionPipeline tests
# ─────────────────────────────────────────────────────────────────────────────

def test_foundation_regression_pipeline_run_baseline(monkeypatch, dummy_embedder):
    X = np.arange(20).reshape(10, 2)
    y = X[:, 0] * 0.5

    pipe = FoundationRegressionPipeline(
        X=X,
        y=y,
        embed_fn=dummy_embedder,
        base_regressor=Ridge(random_state=0),
        metrics=["r2"],
        use_scaler=False,
    )

    def fake_baseline(self, model_name):
        # Ensure correct dispatch
        assert model_name == "FoundationModel"
        return {"model_name": model_name, "ok": True}

    monkeypatch.setattr(type(pipe), "baseline_evaluation", fake_baseline)

    out = pipe.run(analysis_type="baseline")
    assert isinstance(out, dict)
    assert out["model_name"] == "FoundationModel"
    assert out["ok"] is True


def test_foundation_regression_pipeline_multioutput_uses_multioutput_regressor(dummy_embedder):
    X = np.arange(30).reshape(10, 3)
    # multi-output y: shape (10, 2)
    y = np.vstack([X[:, 0], X[:, 1]]).T

    pipe = FoundationRegressionPipeline(
        X=X,
        y=y,
        embed_fn=dummy_embedder,
        base_regressor=Ridge(random_state=0),
        metrics=["r2"],
    )

    foundation_cfg = pipe.model_configs["FoundationModel"]
    foundation_estimator = foundation_cfg.estimator
    assert hasattr(foundation_estimator, "_estimator")
    assert isinstance(foundation_estimator._estimator, MultiOutputRegressor)


# ─────────────────────────────────────────────────────────────────────────────
# FoundationClassificationPipeline tests
# ─────────────────────────────────────────────────────────────────────────────

def test_foundation_classification_pipeline_run_baseline(monkeypatch, dummy_embedder):
    X = np.random.rand(10, 3)
    y = (X[:, 0] > 0.5).astype(int)

    pipe = FoundationClassificationPipeline(
        X=X,
        y=y,
        embed_fn=dummy_embedder,
        base_classifier=LogisticRegression(max_iter=500),
        metrics=["accuracy"],
        use_scaler=False,
    )

    def fake_baseline(self, model_name):
        assert model_name == "FoundationModel"
        return {"model_name": model_name, "task": "classification"}

    monkeypatch.setattr(type(pipe), "baseline_evaluation", fake_baseline)

    out = pipe.run(analysis_type="baseline")
    assert isinstance(out, dict)
    assert out["model_name"] == "FoundationModel"
    assert out["task"] == "classification"


def test_foundation_classification_pipeline_multioutput_uses_multioutput_classifier(dummy_embedder):
    X = np.random.rand(8, 2)
    # multi-label: shape (8, 2)
    y = (X > 0.5).astype(int)

    pipe = FoundationClassificationPipeline(
        X=X,
        y=y,
        embed_fn=dummy_embedder,
        base_classifier=LogisticRegression(max_iter=200),
        metrics=["accuracy"],
    )

    foundation_cfg = pipe.model_configs["FoundationModel"]
    foundation_estimator = foundation_cfg.estimator
    assert hasattr(foundation_estimator, "_estimator")
    assert isinstance(foundation_estimator._estimator, MultiOutputClassifier)
