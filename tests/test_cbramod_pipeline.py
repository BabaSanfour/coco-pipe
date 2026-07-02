# tests/fm/test_cbramod_pipeline.py

import os
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from coco_pipe.fm import CBRAModEmbedder
from coco_pipe.fm.cbramod.embedder import CBraMod
from coco_pipe.fm.pipeline import FoundationRegressionPipeline

# TODO: adjust this path to wherever your CBraMod weights live
WEIGHTS_PATH = "/home/mat/CBraMod/pretrained_weights/pretrained_weights.pth"

# Skip if weights or CBraMod code are missing
CBRAMOD_READY = CBraMod is not None and os.path.exists(WEIGHTS_PATH)
SKIP_REASON = "CBRaMod weights or model code not found; skipping CBRaMod tests."


@pytest.mark.skipif(not CBRAMOD_READY, reason=SKIP_REASON)
def test_cbramod_embedder_output_is_2d_numpy():
    """Integration-style test: CBRAModEmbedder returns 2D numpy embeddings."""
    embedder = CBRAModEmbedder(weights_path=WEIGHTS_PATH, device="cpu")

    # fake EEG with shapes compatible with CBraMod (N, C, S, P) where P=200
    # and C*S = 16*10 matches conv reshaping inside the model.
    X = np.zeros((2, 16, 10, 200), dtype="float32")

    emb = embedder(X)

    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 2
    assert emb.shape[0] == X.shape[0]


@pytest.mark.skipif(not CBRAMOD_READY, reason=SKIP_REASON)
def test_cbramod_with_foundation_regression_pipeline_smoke():
    """
    Simple smoke test: run CBRAModEmbedder + FoundationRegressionPipeline
    end-to-end on tiny fake data and ensure it returns a dict.
    """
    # tiny fake EEG data (N, C, S, P) with valid CBraMod shapes
    X = np.zeros((4, 16, 10, 200), dtype="float32")
    # simple continuous target
    y = np.linspace(0.0, 1.0, num=X.shape[0])

    embedder = CBRAModEmbedder(weights_path=WEIGHTS_PATH, device="cpu")

    pipe = FoundationRegressionPipeline(
        X=X,
        y=y,
        embed_fn=embedder,
        base_regressor=Ridge(random_state=0),
        metrics=["r2"],
        use_scaler=False,
        # keep CV light: 2 folds
        cv_kwargs={"n_splits": 2, "cv_strategy": "kfold"},
        n_jobs=1,
    )

    out = pipe.run(analysis_type="baseline")
    assert isinstance(out, dict)
    assert "metric_scores" in out or "predictions" in out  # depending on BasePipeline format
