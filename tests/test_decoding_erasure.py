"""Fold-local erasure integration with the decoding engine."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from coco_pipe.decoding import (
    CVConfig,
    ErasureConfig,
    Experiment,
    ExperimentConfig,
    TuningConfig,
)
from coco_pipe.decoding.configs import LogisticRegressionConfig


def _within_subject_data(seed=0):
    rng = np.random.default_rng(seed)
    n_subjects, per, d = 8, 20, 12
    axis = rng.standard_normal(d)
    axis /= np.linalg.norm(axis)
    groups = np.repeat(np.arange(n_subjects), per)
    labels = np.tile(np.arange(per) % 2, n_subjects)
    x = np.empty((len(groups), d))
    for sid in range(n_subjects):
        rows = groups == sid
        x[rows] = (
            rng.standard_normal((per, d))
            + rng.standard_normal(d) * 6.0
            + np.outer(labels[rows], axis) * 3.0
        )
    return x, labels, groups


def test_experiment_places_erasure_first_and_routes_groups():
    x, labels, groups = _within_subject_data()
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig(max_iter=1000)},
        metrics=["balanced_accuracy"],
        cv=CVConfig(strategy="group_kfold", n_splits=4),
        erasure=ErasureConfig(enabled=True, method="leace"),
        n_jobs=1,
        verbose=False,
    )
    experiment = Experiment(config)
    estimator = experiment._prepare_estimator("lr", config.models["lr"])
    assert list(estimator.named_steps)[:2] == ["erasure", "scaler"]

    result = experiment.run(x, labels, groups=groups)
    assert result.raw["lr"]["status"] == "success"
    predictions = result.get_predictions()
    assert len(predictions) == len(x)


def test_euclidean_alignment_handles_unseen_fold_subjects():
    x, labels, groups = _within_subject_data()
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig(max_iter=1000)},
        metrics=["balanced_accuracy"],
        cv=CVConfig(strategy="group_kfold", n_splits=4),
        erasure=ErasureConfig(enabled=True, method="ea_mean"),
        n_jobs=1,
        verbose=False,
    )
    result = Experiment(config).run(x, labels, groups=groups)
    assert result.raw["lr"]["status"] == "success"


def test_tuned_euclidean_alignment_routes_groups_through_search_wrapper():
    x, labels, groups = _within_subject_data()
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig(max_iter=1000)},
        grids={"lr": {"clf__C": [0.5, 1.0]}},
        metrics=["balanced_accuracy"],
        cv=CVConfig(strategy="group_kfold", n_splits=4),
        tuning=TuningConfig(
            enabled=True,
            cv=CVConfig(strategy="group_kfold", n_splits=2),
            n_jobs=1,
        ),
        erasure=ErasureConfig(enabled=True, method="ea_mean"),
        n_jobs=1,
        verbose=False,
    )
    result = Experiment(config).run(x, labels, groups=groups)
    assert result.raw["lr"]["status"] == "success"


def _decoding_ba(
    x,
    target,
    groups,
    *,
    erasure_method: str | None,
    cv: CVConfig,
) -> float:
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig(max_iter=2000)},
        metrics=["balanced_accuracy"],
        cv=cv,
        erasure=ErasureConfig(
            enabled=erasure_method is not None,
            method=erasure_method or "leace",
        ),
        n_jobs=1,
        verbose=False,
    )
    predictions = Experiment(config).run(x, target, groups=groups).get_predictions()
    return float(balanced_accuracy_score(predictions["y_true"], predictions["y_pred"]))


@pytest.mark.parametrize("erasure_method", ["leace", "ea_mean", "ea_coral"])
def test_fold_local_erasure_removes_subject_probe_and_preserves_label_control(
    erasure_method,
):
    """V3 gate: identity collapses while a shared within-subject label survives."""
    rng = np.random.default_rng(22)
    n_subjects, per_subject, n_features = 8, 40, 20
    subject = np.repeat(np.arange(n_subjects), per_subject)
    label = np.tile(np.arange(per_subject) % 2, n_subjects)
    label_axis = rng.standard_normal(n_features)
    label_axis /= np.linalg.norm(label_axis)
    centers = rng.standard_normal((n_subjects, n_features)) * 7.0
    x = np.empty((len(subject), n_features))
    for sid in range(n_subjects):
        rows = subject == sid
        x[rows] = (
            rng.standard_normal((per_subject, n_features))
            + centers[sid]
            + np.outer(label[rows] * 2 - 1, label_axis) * 8.0
        )

    label_cv = CVConfig(strategy="group_kfold", n_splits=4)
    label_before = _decoding_ba(x, label, subject, erasure_method=None, cv=label_cv)
    label_after = _decoding_ba(
        x, label, subject, erasure_method=erasure_method, cv=label_cv
    )

    subject_cv = CVConfig(strategy="stratified", n_splits=4)
    subject_before = _decoding_ba(
        x, subject, subject, erasure_method=None, cv=subject_cv
    )
    subject_after = _decoding_ba(
        x,
        subject,
        subject,
        erasure_method=erasure_method,
        cv=subject_cv,
    )

    chance = 1.0 / n_subjects
    assert subject_before > 0.95
    assert subject_after <= chance + 0.05
    assert label_before >= 0.70
    assert label_after >= label_before - 0.08


def _erasure_experiment(method, strategy):
    x, labels, groups = _within_subject_data()
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig(max_iter=1000)},
        metrics=["balanced_accuracy"],
        cv=CVConfig(strategy=strategy, n_splits=4),
        erasure=ErasureConfig(enabled=True, method=method),
        use_scaler=False,
        n_jobs=1,
        verbose=False,
    )
    return Experiment(config), x, labels, groups


@pytest.mark.parametrize("method", ["leace", "ea_mean"])
@pytest.mark.parametrize("strategy", ["group_kfold", "stratified"])
def test_erasure_is_always_a_fold_local_pipeline_step(method, strategy):
    """Experiment fits erasure inside each fold, regardless of method or CV."""
    exp, x, labels, groups = _erasure_experiment(method, strategy)
    result = exp.run(x, labels, groups=groups)
    assert result.raw["lr"]["status"] == "success"
    estimator = exp._prepare_estimator("lr", exp.config.models["lr"])
    assert next(iter(estimator.named_steps)) == "erasure"


def test_precomputed_erasure_reused_matches_inline_fold_local():
    """Erase once and run with erasure disabled → identical to inline fold-local.

    Under subject-disjoint (group) CV, a per-subject eraser fit once globally
    equals the per-fold fit, so precomputing it is a leakage-free compute win.
    """
    from coco_pipe.transforms import make_subject_transform

    exp, x, labels, groups = _erasure_experiment("ea_mean", "group_kfold")
    inline = exp.run(x, labels, groups=groups)

    eraser = make_subject_transform("ea_mean")
    assert eraser.fold_local is False
    x_erased = eraser.fit_transform(x, groups=groups)
    exp2, *_ = _erasure_experiment("ea_mean", "group_kfold")
    exp2.config.erasure.enabled = False
    reused = exp2.run(x_erased, labels, groups=groups)

    np.testing.assert_allclose(
        inline.raw["lr"]["metrics"]["balanced_accuracy"]["folds"],
        reused.raw["lr"]["metrics"]["balanced_accuracy"]["folds"],
        rtol=1e-9,
        atol=1e-9,
    )
