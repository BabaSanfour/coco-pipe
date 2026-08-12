"""Temporal Procrustes alignment and decoding integration."""

from __future__ import annotations

import numpy as np
import pytest

from coco_pipe.decoding import (
    CVConfig,
    Experiment,
    ExperimentConfig,
    TemporalAlignmentConfig,
    TemporalDecoderConfig,
)
from coco_pipe.decoding.configs import ClassicalModelConfig
from coco_pipe.transforms import TemporalProcrustesAlignment


def _rotated_trajectories(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_trials, n_latent, n_features, n_times = 20, 4, 8, 15
    latent = rng.normal(size=(n_trials, n_latent, n_times))
    latent *= np.arange(n_latent, 0, -1)[None, :, None]
    arrays = []
    groups = []
    for subject in ("a", "b", "c"):
        basis = np.linalg.qr(rng.normal(size=(n_features, n_latent)))[0]
        arrays.append(np.einsum("fl,nlt->nft", basis, latent))
        groups.extend([subject] * n_trials)
    return np.concatenate(arrays), np.asarray(groups)


def test_temporal_alignment_maps_an_unseen_participant_to_the_training_template():
    X, groups = _rotated_trajectories()
    train = groups != "c"
    aligner = TemporalProcrustesAlignment(n_components=4, random_state=42)

    aligned_train = aligner.fit_transform(X[train], groups=groups[train])
    aligned_test = aligner.transform(X[~train], groups=groups[~train])

    first_training_subject = groups[train] == "a"
    np.testing.assert_allclose(
        aligned_train[first_training_subject], aligned_test, atol=1e-8
    )
    assert "c" not in aligner.training_groups_


def test_temporal_alignment_requires_3d_grouped_data():
    aligner = TemporalProcrustesAlignment(n_components=2)
    with pytest.raises(ValueError, match="expects"):
        aligner.fit(np.zeros((10, 4)), groups=np.repeat([0, 1], 5))
    with pytest.raises(ValueError, match="requires participant groups"):
        aligner.fit(np.zeros((10, 4, 3)))


def test_rotate_false_keeps_the_subject_pca_but_skips_the_procrustes_step():
    X, groups = _rotated_trajectories()
    train = groups != "c"
    unrotated = TemporalProcrustesAlignment(n_components=4, rotate=False, random_state=42)
    rotated = TemporalProcrustesAlignment(n_components=4, random_state=42)

    unrotated.fit(X[train], groups=groups[train])
    rotated.fit(X[train], groups=groups[train])

    for subject in unrotated.rotations_:
        np.testing.assert_allclose(unrotated.rotations_[subject], np.eye(4), atol=1e-12)
    # Same per-subject bases, different output: only the rotation differs.
    np.testing.assert_allclose(
        unrotated.subject_pcas_["a"].components_,
        rotated.subject_pcas_["a"].components_,
        atol=1e-10,
    )
    assert not np.allclose(
        unrotated.transform(X[~train], groups=groups[~train]),
        rotated.transform(X[~train], groups=groups[~train]),
    )


def test_alignment_diagnostics_report_the_rotation_geometry():
    X, groups = _rotated_trajectories()
    train = groups != "c"
    aligner = TemporalProcrustesAlignment(n_components=4, random_state=42)
    aligner.fit(X[train], groups=groups[train])
    aligner.transform(X[~train], groups=groups[~train])

    assert set(aligner.alignment_diagnostics_) == {"a", "b", "c"}
    assert aligner.alignment_diagnostics_["a"]["seen_in_training"]
    assert not aligner.alignment_diagnostics_["c"]["seen_in_training"]
    for record in aligner.alignment_diagnostics_.values():
        # Rotating onto the template can only improve shape agreement.
        assert record["similarity_gain"] >= -1e-9
        assert -1.0 <= record["template_similarity_rotated"] <= 1.0 + 1e-9
        assert 0.0 <= record["rotation_angle_deg"] <= 180.0


def test_calibration_adaptation_never_uses_a_trial_for_its_own_mapping():
    X, groups = _rotated_trajectories()
    train = groups != "c"
    aligner = TemporalProcrustesAlignment(
        n_components=2, adaptation="calibration", random_state=0
    )
    aligner.fit(X[train], groups=groups[train])

    aligned = aligner.transform(X[~train], groups=groups[~train])
    assert aligned.shape == (int((~train).sum()), 2, X.shape[2])
    assert np.isfinite(aligned).all()
    assert not aligner.alignment_diagnostics_["c"]["seen_in_training"]

    single_trial = np.flatnonzero(~train)[:1]
    with pytest.raises(ValueError, match="at least two trials"):
        aligner.transform(X[single_trial], groups=groups[single_trial])


def test_experiment_applies_temporal_alignment_inside_each_loso_fold():
    rng = np.random.default_rng(12)
    groups = np.repeat(["01", "02", "03"], 12)
    labels = np.tile(np.repeat([0, 1], 6), 3)
    X = rng.normal(scale=0.7, size=(len(groups), 6, 5))
    X[labels == 1, 0, 2:] += 2.5
    decoder = TemporalDecoderConfig(
        wrapper="sliding",
        base=ClassicalModelConfig(
            estimator="LogisticRegression",
            params={"class_weight": "balanced", "max_iter": 1000},
        ),
        n_jobs=1,
        verbose=False,
    )
    config = ExperimentConfig(
        task="classification",
        models={"aligned": decoder},
        metrics=["balanced_accuracy"],
        cv=CVConfig(strategy="leave_one_group_out", shuffle=False),
        temporal_alignment=TemporalAlignmentConfig(
            enabled=True,
            n_components=3,
            adaptation="transductive",
        ),
        random_state=42,
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, labels, groups=groups, time_axis=np.arange(5))

    folds = result.raw["aligned"]["metrics"]["balanced_accuracy"]["folds"]
    assert np.asarray(folds).shape == (3, 5)
    assert all(
        fold["alignment_time"] >= 0 for fold in result.raw["aligned"]["diagnostics"]
    )
