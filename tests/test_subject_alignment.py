"""Subject-alignment transforms and diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from coco_pipe.diagnostics.variance import (
    crossed_ss_fractions,
    direction_traits,
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


def _condition_data(n_subjects=60, per=30, d=12, seed=0):
    """Subjects differing in 3 directions, a shared condition shift in a 4th,
    and each subject with its own share of the condition."""
    rng = np.random.default_rng(seed)
    subject = np.repeat(np.arange(n_subjects), per)
    share = rng.uniform(0.0, 1.0, n_subjects)
    condition = (rng.random(len(subject)) < share[subject]).astype(int)
    x = rng.standard_normal((len(subject), d))
    x[:, :3] += rng.standard_normal((n_subjects, 3))[subject] * 4.0
    x[:, 3] += 3.0 * condition
    return x, subject, condition


def test_leace_erases_the_subject_directions_and_keeps_the_condition():
    x, subject, condition = _condition_data()

    def shift(values):
        return values[condition == 1, 3].mean() - values[condition == 0, 3].mean()

    eraser = LeaceEraser().fit(x, groups=subject, condition=condition)
    erased = eraser.transform(x)
    assert eraser.rank_ == 3
    assert crossed_ss_fractions(x, subject, condition)["subject_frac"] > 0.7
    assert crossed_ss_fractions(erased, subject, condition)["subject_frac"] < 0.15
    assert shift(erased) == pytest.approx(shift(x), abs=0.1)

    # Without the conditions, a subject's share of them looks like identity.
    blind = LeaceEraser(eta2=0.2).fit(x, groups=subject)
    assert blind.rank_ == 4 and abs(shift(blind.transform(x))) < 0.5
    assert LeaceEraser(eta2=0.2).fit(x, groups=subject, condition=condition).rank_ == 3

    # The projector carries over to subjects it was not fitted on.
    seen = subject < 40
    unseen = LeaceEraser().fit(x[seen], groups=subject[seen], condition=condition[seen])
    assert (
        crossed_ss_fractions(
            unseen.transform(x[~seen]), subject[~seen], condition[~seen]
        )["subject_frac"]
        < 0.15
    )


def test_leace_refuses_to_erase_almost_everything():
    x, subject, _ = _condition_data()
    with pytest.raises(ValueError, match="would erase 12 of 12"):
        LeaceEraser(eta2=0.0).fit(x, groups=subject)
    # max_rank bounds the directions erased; with eta2=0 it is their number.
    assert LeaceEraser(eta2=0.0, max_rank=5).fit(x, groups=subject).rank_ == 5
    assert LeaceEraser(max_rank=2).fit(x, groups=subject).rank_ == 2


def test_direction_traits_names_what_the_subject_directions_carry():
    rng = np.random.default_rng(0)
    n_subjects = 80
    subject = np.repeat(np.arange(n_subjects), 20)
    age = rng.uniform(5.0, 60.0, n_subjects)
    sex = rng.choice(["F", "M"], n_subjects)
    x = rng.standard_normal((len(subject), 8))
    x[:, 0] += 0.2 * (age - age.mean())[subject]
    x[:, 1] += np.where(sex == "F", -2.0, 2.0)[subject]
    traits = pd.DataFrame(
        {"age": age, "sex": sex, "noise": rng.standard_normal(n_subjects)}
    )
    traits.loc[:9, "age"] = np.nan

    eraser = LeaceEraser(eta2=0.0, max_rank=2).fit(x, groups=subject)
    table = direction_traits(eraser, x, subject, traits, n_directions=3)
    r2 = table.pivot(index="trait", columns="direction", values="r2")
    assert r2.loc["age", 0] > 0.9 and r2.loc["sex", 1] > 0.9
    assert r2.loc["noise"].max() < 0.1 and r2[2].max() < 0.1
    assert table.groupby("trait")["n_subjects"].first().to_dict() == {
        "age": 70,
        "noise": 80,
        "sex": 80,
    }
    assert table.loc[table["erased"], "direction"].unique().tolist() == [0, 1]


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

    # With shrinkage, the output size does not follow a subject's sample count.
    rng = np.random.default_rng(3)
    counts = np.array([5, 20, 200])
    groups = np.repeat(np.arange(3), counts)
    x = rng.standard_normal((counts.sum(), 30))
    whitened = EuclideanAlign("coral").fit_transform(x, groups=groups)
    for sid in range(3):
        norms = np.sum(whitened[groups == sid] ** 2, axis=1)
        assert norms.mean() == pytest.approx(30.0)


def _subject_tokens(n_subjects=4, per=20, axes=(3, 5), d=6, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"s{i}" for i in range(n_subjects)], per)
    tokens = np.concatenate(
        [
            rng.standard_normal(d) * 5
            + rng.standard_normal((per, *axes, d)) @ rng.standard_normal((d, d))
            for _ in range(n_subjects)
        ]
    )
    return tokens.astype(np.float32), groups


def test_euclidean_alignment_of_tokens_takes_all_their_tokens():
    tokens, groups = _subject_tokens()
    d = tokens.shape[-1]
    whitened = EuclideanAlign("coral", shrinkage=False).fit_transform(
        tokens, groups=groups
    )
    assert whitened.shape == tokens.shape and whitened.dtype == np.float32
    for sid in np.unique(groups):
        rows = whitened[groups == sid].reshape(-1, d).astype(np.float64)
        np.testing.assert_allclose(rows.mean(0), 0.0, atol=1e-5)
        np.testing.assert_allclose(rows.T @ rows / len(rows), np.eye(d), atol=1e-4)

    exact = tokens.astype(np.float64)
    constrained = exact - exact.mean(-1, keepdims=True)
    whitened = EuclideanAlign("coral", shrinkage=False).fit_transform(
        constrained[groups == "s0"], groups=groups[groups == "s0"]
    )
    rows = whitened.reshape(-1, d)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(rows.T @ rows / len(rows)), [0] + [1] * (d - 1), atol=1e-9
    )

    np.testing.assert_allclose(
        EuclideanAlign("mean").fit_transform(exact, groups=groups).mean(axis=(1, 2)),
        EuclideanAlign("mean").fit_transform(exact.mean(axis=(1, 2)), groups=groups),
        atol=1e-12,
    )

    seen = groups != "s0"
    fitted = EuclideanAlign("coral").fit(tokens[seen], groups=groups[seen])
    np.testing.assert_allclose(
        fitted.transform(tokens, groups=groups)[~seen],
        EuclideanAlign("coral").fit_transform(tokens[~seen], groups=groups[~seen]),
    )
    with pytest.raises(ValueError, match="expects"):
        EuclideanAlign("mean").fit(tokens[0, 0, 0], groups=groups[:d])


def test_euclidean_alignment_can_center_each_channel():
    tokens, groups = _subject_tokens()
    offsets = np.random.default_rng(1).standard_normal((4, 3, 1, tokens.shape[-1]))
    tokens = tokens.astype(np.float64) + np.repeat(offsets * 3, 20, axis=0)
    d = tokens.shape[-1]

    whitened = EuclideanAlign("coral", shrinkage=False, channel_axis=1).fit_transform(
        tokens, groups=groups
    )
    for sid in np.unique(groups):
        subject = whitened[groups == sid]
        np.testing.assert_allclose(subject.mean(axis=(0, 2)), 0.0, atol=1e-9)
        rows = subject.reshape(-1, d)
        np.testing.assert_allclose(rows.T @ rows / len(rows), np.eye(d), atol=1e-9)

    per_channel = EuclideanAlign("mean", channel_axis=1).fit_transform(
        tokens, groups=groups
    )
    plain = EuclideanAlign("mean").fit_transform(tokens, groups=groups)
    for axes in ((1, 2), 1):
        np.testing.assert_allclose(
            per_channel.mean(axis=axes), plain.mean(axis=axes), atol=1e-12
        )
    channels = per_channel.mean(axis=2)
    assert not np.allclose(channels, plain.mean(axis=2))
    for sid in np.unique(groups):
        np.testing.assert_allclose(channels[groups == sid].mean(0), 0.0, atol=1e-12)

    for axis, X in ((0, tokens), (3, tokens), (True, tokens), (1, tokens[:, 0, 0])):
        with pytest.raises(ValueError, match="channel_axis"):
            EuclideanAlign("mean", channel_axis=axis).fit(X, groups=groups)
    assert (
        EuclideanAlign("mean", channel_axis=1).fingerprint()
        != EuclideanAlign("mean").fingerprint()
    )


def test_riemann_alignment_recenters_each_subject_on_token_covariances():
    pytest.importorskip("pyriemann")
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(4), 6)
    scales = rng.uniform(0.5, 3.0, (4, 6))
    tokens = rng.standard_normal((24, 15, 6)) * scales[groups][:, None, :]
    output = RiemannAlign().fit_transform(tokens, groups=groups)
    assert output.shape == (24, 21)

    # Each subject's tangent vectors average to zero at its own reference.
    for subject in range(4):
        np.testing.assert_allclose(output[groups == subject].mean(0), 0.0, atol=1e-6)
    # Where a window's tokens sit does not matter, only how they co-vary.
    shifted = tokens + rng.standard_normal((24, 1, 6)) * 10.0
    np.testing.assert_allclose(
        RiemannAlign().fit_transform(shifted, groups=groups), output, atol=1e-8
    )
    with pytest.raises(ValueError, match="at least two tokens"):
        tokens_to_covariances(tokens[:, :1])
    assert np.linalg.eigvalsh(tokens_to_covariances(tokens[:, :3])).min() > 0
    with pytest.raises(ValueError, match="identical tokens"):
        tokens_to_covariances(np.ones((2, 5, 6)))


def test_registry_and_fingerprint_are_stable():
    for name in ("leace", "ea_coral", "ea_mean", "ra"):
        transform = make_subject_transform(name)
        assert transform.name == name
        assert transform.fingerprint() == transform.fingerprint()
    assert (
        make_subject_transform("leace", eta2=0.5).fingerprint()
        != make_subject_transform("leace", eta2=0.3).fingerprint()
    )
    with pytest.raises(ValueError, match="Unknown subject transform"):
        make_subject_transform("missing")


def test_only_leace_transforms_subjects_without_their_groups():
    x, subject, _ = _condition_data()
    leace = make_subject_transform("leace").fit(x, groups=subject)
    assert leace.transform(x).shape == x.shape
    for name in ("ea_mean", "ea_coral"):
        aligner = make_subject_transform(name).fit(x, groups=subject)
        with pytest.raises(ValueError, match="requires groups"):
            aligner.transform(x)


def test_ea_reference_moves_with_the_condition_share_unless_taken_per_condition():
    x, subject, condition = _condition_data()
    cell = np.char.add(np.char.add(subject.astype(str), "|"), condition.astype(str))
    mixed = EuclideanAlign("mean").fit_transform(x, groups=subject)
    per_condition = EuclideanAlign("mean").fit_transform(x, groups=cell)

    # Within condition 0, a subject's windows sit where its share of condition 1 puts
    # them on the condition axis (3), unless the reference is taken per condition.
    kept = [s for s in np.unique(subject) if ((subject == s) & (condition == 0)).any()]
    share = np.array([condition[subject == s].mean() for s in kept])

    def position(aligned):
        rows = [(subject == s) & (condition == 0) for s in kept]
        return np.array([aligned[r, 3].mean() for r in rows])

    assert np.corrcoef(share, position(mixed))[0, 1] < -0.9
    np.testing.assert_allclose(position(per_condition), 0.0, atol=1e-9)


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
    control = null_control(iid, groups, labels, n_null_permutations=100, rng=rng)
    expected = (len(np.unique(groups)) - 1) / (len(groups) - 1)
    assert control["null_subject_frac"]["mean"] == pytest.approx(expected, abs=0.01)
    assert control["method"] == "hierarchy_preserving_permutation"
    report = variance_decomposition_report(x, groups, labels, n_null_permutations=2)
    assert isinstance(report, pd.DataFrame)
    assert set(report["metric"]) >= {
        "total_sample_variance",
        "variance_participation_ratio",
        "variance_participation_ratio_fraction",
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
        n_null_permutations=2,
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
        n_null_permutations=2,
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
        variance_decomposition_report(container, label="label", n_null_permutations=2)
    with pytest.raises(ValueError, match="label must be supplied"):
        variance_decomposition_report(
            container, subject="subject", n_null_permutations=2
        )


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
        n_null_permutations=2,
    )

    assert "subject_within_label_fraction" in set(report["metric"])


def _metric(report, name):
    return float(report.loc[report["metric"] == name, "value"].iloc[0])


def test_nested_report_partition_is_additive_despite_marginal_overlap():
    x = np.asarray([[0.0], [0.0], [1.0], [1.0], [10.0], [10.0], [11.0], [11.0]])
    groups = np.repeat(["s0", "s1", "s2", "s3"], 2)
    labels = np.repeat([0, 0, 1, 1], 2)

    report = variance_decomposition_report(
        x, groups, labels, feature_scaling="none", n_null_permutations=10
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

    report = variance_decomposition_report(x, groups, labels, n_null_permutations=10)

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
    first = variance_decomposition_report(x, groups, labels, n_null_permutations=2)
    second = variance_decomposition_report(
        rescaled, groups, labels, n_null_permutations=2
    )

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
        with_constant, groups, labels, n_null_permutations=2
    )
    assert set(report["n_constant_features"]) == {1}

    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        variance_decomposition_report(x, groups, labels, n_null_permutations=2)


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
