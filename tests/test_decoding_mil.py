import builtins

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from coco_pipe.decoding import (
    ABMILClassifierConfig,
    ClassicalModelConfig,
    CVConfig,
    Experiment,
    ExperimentConfig,
    ReducerConfig,
    get_capabilities,
)
from coco_pipe.decoding.mil import ABMILClassifier
from coco_pipe.io import DataContainer


def _make_bag_data(n_bags: int = 36, seed: int = 0):
    """Variable-length bags where positive bags hold informative instances."""
    rng = np.random.default_rng(seed)
    X_parts, y_parts, g_parts = [], [], []
    for i in range(n_bags):
        label = i % 2
        n_inst = int(rng.integers(3, 7))
        emb = rng.normal(size=(n_inst, 6)).astype(np.float32)
        if label == 1:
            # Make the first two instances carry the positive signal
            emb[:2, 0] += 3.5
            emb[:2, 1] -= 2.5
        cond = (np.arange(n_inst) % 2).astype(np.float32)[:, None]
        X_parts.append(np.hstack([emb, cond]))
        y_parts.append(np.full(n_inst, label, dtype=int))
        g_parts.append(np.full(n_inst, f"bag_{i:02d}", dtype=object))
    return (
        np.vstack(X_parts),
        np.concatenate(y_parts),
        np.concatenate(g_parts),
    )


def test_abmil_runs_in_experiment_with_per_block_pca_and_attention_artifacts():
    X, y, groups = _make_bag_data(n_bags=60, seed=0)
    container = DataContainer(
        X=X,
        dims=("obs", "feature"),
        coords={
            "feature": np.array([f"f{j}" for j in range(7)], dtype=object),
            "feature_block": np.array(["emb"] * 6 + ["cond"], dtype=object),
        },
        ids=np.arange(len(y)),
    )
    cfg = ExperimentConfig(
        models={
            "mil": ABMILClassifierConfig(
                att_dim=16,
                hidden_dim=16,
                gated=True,
                max_epochs=60,
                patience=15,
                batch_size=16,
                lr=5e-3,
                n_restarts=1,
                random_state=0,
            )
        },
        cv=CVConfig(strategy="stratified_group_kfold", n_splits=3),
        use_scaler=True,
        reducer=ReducerConfig(enabled=True, n_components={"emb": 4, "cond": None}),
        metrics=["accuracy", "roc_auc"],
        n_jobs=1,
        verbose=False,
    )
    res = Experiment(cfg).run(container, y, groups=groups)
    preds = res.get_predictions()
    bag_preds = preds.groupby("Group", sort=False).agg(
        y_true=("y_true", "first"),
        y_proba_1=("y_proba_1", "first"),
        n_unique_proba=("y_proba_1", "nunique"),
    )
    assert (bag_preds["n_unique_proba"] == 1).all()
    assert roc_auc_score(bag_preds["y_true"], bag_preds["y_proba_1"]) > 0.85

    # Per-instance attention weights align with test indices and sum to 1 per bag
    raw = res.raw["mil"]
    for test_idx, meta in zip(raw["indices"], raw["metadata"], strict=True):
        att = meta["artifacts"]["attention"]
        assert att.shape == (len(test_idx),)
        assert np.all(att >= 0.0)
        bag_sums = pd.Series(att).groupby(groups[test_idx]).sum()
        np.testing.assert_allclose(bag_sums.to_numpy(), 1.0, atol=1e-5)


def test_abmil_options_alias_and_reproducibility():
    X, y, groups = _make_bag_data(n_bags=20, seed=1)
    assert get_capabilities("ABMILClassifier").has_response("predict_proba")

    clf_a = ABMILClassifier(
        att_dim=8,
        hidden_dim=None,
        gated=False,
        max_instances=3,
        max_epochs=10,
        val_fraction=0.0,
        n_restarts=2,
        device="auto",
        random_state=7,
    )
    clf_b = ABMILClassifier(
        att_dim=8,
        hidden_dim=None,
        gated=False,
        max_instances=3,
        max_epochs=10,
        val_fraction=0.0,
        n_restarts=2,
        device="auto",
        random_state=7,
    )
    p_a = clf_a.fit(X, y, groups=groups).predict_proba(X, groups=groups)
    p_b = clf_b.fit(X, y, groups=groups).predict_proba(X, groups=groups)
    np.testing.assert_allclose(p_a, p_b)
    assert set(clf_a.predict(X, groups=groups)).issubset({0, 1})

    # Canonical alias "abmil" resolves cleanly via ClassicalModelConfig
    cfg = ExperimentConfig(
        models={
            "mil": ClassicalModelConfig(
                estimator="abmil",
                params={
                    "att_dim": 8,
                    "hidden_dim": 8,
                    "max_epochs": 5,
                    "val_fraction": 0.0,
                },
            )
        },
        cv=CVConfig(strategy="stratified_group_kfold", n_splits=2),
        use_scaler=False,
        metrics=["accuracy"],
        n_jobs=1,
        verbose=False,
    )
    res = Experiment(cfg).run(X, y, groups=groups)
    assert not res.summary().empty


def test_abmil_validation_and_missing_dependency(monkeypatch):
    X, y, groups = _make_bag_data(n_bags=8, seed=2)
    clf = ABMILClassifier(att_dim=8, hidden_dim=8, max_epochs=3, val_fraction=0.0)

    with pytest.raises(ValueError, match="requires bag identifiers"):
        clf.fit(X, y)
    with pytest.raises(ValueError, match="Expected 2D"):
        clf.fit(X[:, :, None], y, groups=groups)
    with pytest.raises(ValueError, match="groups length"):
        clf.fit(X, y, groups=groups[:-1])
    with pytest.raises(ValueError, match="binary targets"):
        clf.fit(X, np.zeros_like(y), groups=groups)

    mixed_y = y.copy()
    mixed_y[0] = 1 - mixed_y[0]
    with pytest.raises(ValueError, match="constant within each bag"):
        clf.fit(X, mixed_y, groups=groups)

    clf.fit(X, y, groups=groups)
    with pytest.raises(ValueError, match="requires bag identifiers"):
        clf.predict_proba(X)
    with pytest.raises(ValueError, match="groups length"):
        clf.predict_proba(X, groups=groups[:-1])

    real_import = builtins.__import__

    def _block_torchmil(name, *args, **kwargs):
        if name.startswith("torchmil"):
            raise ImportError("no torchmil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_torchmil)
    with pytest.raises(ImportError, match="coco-pipe\\[mil\\]"):
        ABMILClassifier().fit(X, y, groups=groups)
