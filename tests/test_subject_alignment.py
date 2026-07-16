"""Subject-alignment transforms and diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from coco_pipe.diagnostics.variance import (
    crossed_ss_fractions,
    nested_ss,
    null_control,
    omega_squared_from_ss,
    subject_probe,
    variance_decomposition_report,
)
from coco_pipe.io import DataContainer
from coco_pipe.transforms.subject_alignment import (
    EuclideanAlign,
    LeaceEraser,
    RiemannAlign,
    make_subject_transform,
    tokens_to_covariances,
)


def _subject_data(n_subjects=8, per=30, d=12, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_subjects, d)) * 7.0
    groups = np.repeat([f"s{i}" for i in range(n_subjects)], per)
    labels = np.repeat(np.arange(n_subjects) % 2, per)
    x = np.vstack(
        [centers[index] + rng.standard_normal((per, d)) for index in range(n_subjects)]
    )
    return x, groups, labels


def test_leace_erases_linear_subject_axis_and_flags_dimension_limit():
    x, groups, _ = _subject_data()
    pre, _ = subject_probe(x, groups)
    eraser = LeaceEraser().fit(x, groups=groups)
    post, _ = subject_probe(eraser.transform(x, groups=groups), groups)
    assert pre is not None and post is not None and post < pre - 0.2
    assert not eraser.degenerate_

    small_x, small_groups, _ = _subject_data(n_subjects=10, per=6, d=4)
    assert LeaceEraser().fit(small_x, groups=small_groups).degenerate_


@pytest.mark.parametrize(
    ("kind", "method"),
    [("linear", "LogisticRegression"), ("mlp", "MLPClassifier")],
)
def test_subject_probe_delegates_both_models_to_experiment(monkeypatch, kind, method):
    calls = []

    def fake_run(experiment, x, y):
        calls.append((experiment.config, x, y))
        return SimpleNamespace(
            raw={"subject_probe": {"metrics": {"balanced_accuracy": {"mean": 0.625}}}}
        )

    monkeypatch.setattr("coco_pipe.diagnostics.variance.Experiment.run", fake_run)
    x, groups, _ = _subject_data(n_subjects=3, per=5)

    score, n_subjects = subject_probe(x, groups, kind=kind, n_splits=5)

    assert score == pytest.approx(0.625)
    assert n_subjects == 3
    assert len(calls) == 1
    config, delegated_x, delegated_y = calls[0]
    assert config.models["subject_probe"].method == method
    assert config.metrics == ["balanced_accuracy"]
    assert config.cv.strategy == "stratified"
    assert config.use_scaler is True
    np.testing.assert_array_equal(delegated_x, x)
    np.testing.assert_array_equal(delegated_y, np.repeat(np.arange(3), 5))


def test_euclidean_alignment_centers_and_whitens_each_subject():
    x, groups, _ = _subject_data(per=60)
    centered = EuclideanAlign("mean").fit_transform(x, groups=groups)
    whitened = EuclideanAlign("coral", shrinkage=False).fit_transform(x, groups=groups)
    for sid in np.unique(groups):
        rows = groups == sid
        np.testing.assert_allclose(centered[rows].mean(0), 0.0, atol=1e-9)
        np.testing.assert_allclose(
            np.cov(whitened[rows], rowvar=False), np.eye(x.shape[1]), atol=0.1
        )


def test_riemann_alignment_returns_finite_tangent_vectors():
    pytest.importorskip("pyriemann")
    rng = np.random.default_rng(2)
    tokens = rng.standard_normal((24, 15, 6))
    groups = np.repeat(np.arange(4), 6)
    covariances = tokens_to_covariances(tokens)
    assert np.linalg.eigvalsh(covariances).min() > 0
    output = RiemannAlign().fit_transform(tokens, groups=groups)
    assert output.shape == (24, 21)
    assert RiemannAlign().fit(tokens, groups=groups).n_output_features_ == 21
    assert np.isfinite(output).all()


def test_registry_and_fingerprint_are_stable():
    for name in ("leace", "ea_coral", "ea_mean", "ra"):
        transform = make_subject_transform(name)
        assert transform.name == name
        assert transform.fingerprint() == transform.fingerprint()
    assert (
        make_subject_transform("leace", shrinkage=True).fingerprint()
        != make_subject_transform("leace", shrinkage=False).fingerprint()
    )
    with pytest.raises(ValueError, match="Unknown subject transform"):
        make_subject_transform("missing")


def test_only_leace_is_fold_local():
    assert make_subject_transform("leace").fold_local is True
    for name in ("ea_coral", "ea_mean", "ra"):
        assert make_subject_transform(name).fold_local is False


def test_variance_diagnostics_match_null_and_nested_partition():
    x, groups, labels = _subject_data()
    fractions = crossed_ss_fractions(x, groups, labels)
    assert fractions["subject_frac"] > 0.9
    ss = nested_ss(x, groups, labels)
    np.testing.assert_allclose(
        ss["label"] + ss["subject_within_label"] + ss["residual"], ss["total"]
    )
    omega = omega_squared_from_ss(ss)
    assert omega["omega2_label_subject_level"] >= 0
    assert omega["partial_omega2_subject_within_label"] >= 0

    rng = np.random.default_rng(10)
    iid = rng.standard_normal(x.shape)
    control = null_control(iid, groups, labels, n_null_seeds=100, rng=rng)
    expected = (len(np.unique(groups)) - 1) / (len(groups) - 1)
    assert control["null_subject_frac"]["mean"] == pytest.approx(expected, abs=0.01)
    assert control["method"] == "hierarchy_preserving_permutation"
    report = variance_decomposition_report(x, groups, labels, n_null_seeds=2)
    assert isinstance(report, pd.DataFrame)
    assert set(report["metric"]) >= {
        "subject_within_label_fraction",
        "label_fraction",
        "between_subject_excess_over_null",
        "subject_probe_linear_balanced_accuracy",
        "partial_omega2_subject_within_label",
    }


def test_variance_report_accepts_data_container_coordinates():
    x, groups, labels = _subject_data(n_subjects=4, per=10)
    container = DataContainer(
        X=x,
        dims=("obs", "feature"),
        coords={"study_id": groups, "diagnosis": labels},
    )
    report = variance_decomposition_report(
        container,
        subject="study_id",
        label="diagnosis",
        n_null_seeds=2,
    )
    assert isinstance(report, pd.DataFrame)
    assert "subject_within_label_fraction" in set(report["metric"])


def test_variance_report_uses_container_y_as_default_label():
    x, groups, labels = _subject_data(n_subjects=4, per=10)
    container = DataContainer(
        X=x,
        dims=("obs", "feature"),
        coords={"study_id": groups},
        y=labels,
    )

    report = variance_decomposition_report(
        container,
        subject="study_id",
        n_null_seeds=2,
    )

    assert "label_fraction" in set(report["metric"])


def test_variance_report_does_not_guess_coordinate_roles():
    x, groups, labels = _subject_data(n_subjects=4, per=10)
    container = DataContainer(
        X=x,
        dims=("obs", "feature"),
        coords={"subject": groups, "label": labels},
    )

    with pytest.raises(ValueError, match="subject must be supplied"):
        variance_decomposition_report(container, label="label", n_null_seeds=2)
    with pytest.raises(ValueError, match="label must be supplied"):
        variance_decomposition_report(container, subject="subject", n_null_seeds=2)


def test_variance_report_moves_container_observations_to_rows():
    x, groups, labels = _subject_data(n_subjects=4, per=10)
    container = DataContainer(
        X=x.T,
        dims=("feature", "obs"),
        coords={"study_id": groups, "diagnosis": labels},
    )

    report = variance_decomposition_report(
        container,
        subject="study_id",
        label="diagnosis",
        n_null_seeds=2,
    )

    assert "subject_within_label_fraction" in set(report["metric"])


def _metric(report, name):
    return float(report.loc[report["metric"] == name, "value"].iloc[0])


def test_nested_report_partition_is_additive_despite_marginal_overlap():
    x = np.asarray([[0.0], [0.0], [1.0], [1.0], [10.0], [10.0], [11.0], [11.0]])
    groups = np.repeat(["s0", "s1", "s2", "s3"], 2)
    labels = np.repeat([0, 0, 1, 1], 2)

    report = variance_decomposition_report(
        x, groups, labels, feature_scaling="none", n_null_seeds=10
    )

    partition = sum(
        _metric(report, name)
        for name in (
            "label_fraction",
            "subject_within_label_fraction",
            "residual_fraction",
        )
    )
    assert partition == pytest.approx(1.0)
    assert (
        _metric(report, "marginal_label_eta2") + _metric(report, "between_subject_eta2")
        > 1.0
    )
    assert _metric(report, "omega2_label_subject_level") == pytest.approx(99.5 / 101.5)


def test_crossed_report_uses_adjusted_additive_partition():
    rng = np.random.default_rng(12)
    n_subjects, repeats, d = 6, 4, 5
    groups = np.repeat(np.arange(n_subjects), 2 * repeats)
    labels = np.tile(np.repeat([0, 1], repeats), n_subjects)
    subject_effect = rng.normal(scale=2.0, size=(n_subjects, d))
    label_effect = np.asarray([-2.0, 2.0])[labels, None]
    x = (
        subject_effect[groups]
        + label_effect
        + rng.normal(scale=0.2, size=(len(groups), d))
    )

    report = variance_decomposition_report(x, groups, labels, n_null_seeds=10)

    assert set(report["design"]) == {"crossed_adjusted_fixed_effects"}
    partition = sum(
        _metric(report, name)
        for name in (
            "unique_label_fraction",
            "unique_subject_fraction",
            "shared_or_confounding_fraction",
            "residual_fraction",
        )
    )
    assert partition == pytest.approx(1.0)
    assert _metric(report, "unique_label_fraction") > 0.1
    assert _metric(report, "unique_subject_fraction") > 0.1


def test_zscore_report_is_invariant_to_feature_units():
    x, groups, labels = _subject_data(n_subjects=6, per=10)
    rescaled = x * np.geomspace(0.01, 100.0, x.shape[1])
    first = variance_decomposition_report(x, groups, labels, n_null_seeds=2)
    second = variance_decomposition_report(rescaled, groups, labels, n_null_seeds=2)

    for name in (
        "label_fraction",
        "subject_within_label_fraction",
        "residual_fraction",
        "marginal_label_eta2",
        "between_subject_eta2",
    ):
        assert _metric(first, name) == pytest.approx(_metric(second, name))


def test_report_tracks_constant_features_and_rejects_nonfinite_values():
    x, groups, labels = _subject_data(n_subjects=4, per=10)
    with_constant = np.column_stack((x, np.ones(len(x))))
    report = variance_decomposition_report(
        with_constant, groups, labels, n_null_seeds=2
    )
    assert set(report["n_constant_features"]) == {1}

    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        variance_decomposition_report(x, groups, labels, n_null_seeds=2)


def test_subject_probe_can_hold_out_complete_blocks(monkeypatch):
    calls = []

    def fake_run(experiment, x, y, groups=None):
        calls.append((experiment.config, groups))
        return SimpleNamespace(
            raw={"subject_probe": {"metrics": {"balanced_accuracy": {"mean": 0.5}}}}
        )

    monkeypatch.setattr("coco_pipe.diagnostics.variance.Experiment.run", fake_run)
    rng = np.random.default_rng(3)
    subject = np.repeat(np.arange(3), 6)
    blocks = np.tile(np.repeat(np.arange(3), 2), 3)
    x = rng.standard_normal((len(subject), 4))

    subject_probe(x, subject, blocks=blocks, n_splits=3)

    config, delegated_groups = calls[0]
    assert config.cv.strategy == "stratified_group_kfold"
    assert len(np.unique(delegated_groups)) == 9
