from __future__ import annotations

import numpy as np
import pytest

from coco_pipe.diagnostics import (
    streamed_subject_probe,
    streamed_variance_decomposition_report,
    variance_decomposition_report,
)


class _Batches:
    def __init__(self, features: np.ndarray, stops=(3, 11)):
        self.features = features
        self.stops = stops

    def __iter__(self):
        start = 0
        for stop in (*self.stops, len(self.features)):
            if start < stop:
                yield np.arange(start, stop), self.features[start:stop]
            start = stop


def _ok_metrics(report):
    return {
        row.metric: float(row.value)
        for row in report.itertuples()
        if row.status == "ok" and not row.metric.startswith("subject_probe")
    }


@pytest.mark.parametrize("pure_labels", [True, False])
def test_streamed_report_matches_dense_statistics_and_seeded_nulls(pure_labels):
    rng = np.random.default_rng(12)
    subjects = np.repeat(np.arange(6), 8 if not pure_labels else 4)
    if pure_labels:
        labels = np.repeat(np.arange(6) % 2, 4)
    else:
        labels = np.tile(np.repeat([0, 1], 4), 6)
    subject_effect = rng.normal(scale=2.0, size=(6, 7))
    features = (
        subject_effect[subjects]
        + np.asarray([-1.5, 1.5])[labels, None]
        + rng.normal(scale=0.3, size=(len(subjects), 7))
    )

    dense = variance_decomposition_report(
        features,
        subjects,
        labels,
        n_null_permutations=4,
        probe_n_splits=9,
    )
    streamed = streamed_variance_decomposition_report(
        _Batches(features),
        subjects,
        labels,
        n_null_permutations=4,
        probe_n_splits=9,
    )

    assert set(_ok_metrics(streamed)) == set(_ok_metrics(dense))
    for metric, expected in _ok_metrics(dense).items():
        assert _ok_metrics(streamed)[metric] == pytest.approx(expected, abs=1e-10)


def test_participation_ratio_guard_skips_only_rank_metrics():
    rng = np.random.default_rng(2)
    subjects = np.repeat(np.arange(4), 5)
    labels = np.repeat(np.arange(4) % 2, 5)
    report = streamed_variance_decomposition_report(
        _Batches(rng.normal(size=(20, 4))),
        subjects,
        labels,
        n_null_permutations=2,
        probe_n_splits=6,
        participation_ratio_max_gram_bytes=8,
    )

    rank = report[report.metric.str.startswith("variance_participation_ratio")]
    assert set(rank.status) == {"skipped"}
    assert rank.reason.str.contains("128 bytes").all()
    assert set(report.loc[report.status == "ok", "metric"]) >= {
        "total_sample_variance",
        "between_subject_eta2",
        "label_fraction",
    }


def test_streamed_subject_probe_is_deterministic_and_detects_separation(monkeypatch):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    counts = np.asarray([20, 25, 30])
    subjects = np.repeat(np.arange(3), counts)
    centers = np.eye(3, 6) * 12.0
    features = centers[subjects] + rng.normal(scale=0.1, size=(len(subjects), 6))
    batches = _Batches(features, stops=())

    real_linear = torch.nn.Linear
    real_cross_entropy = torch.nn.CrossEntropyLoss
    standardized_training_batches = []
    class_weights = []

    class _TrackingLinear(real_linear):
        def forward(self, values):
            if self.training:
                standardized_training_batches.append(values.detach().numpy().copy())
            return super().forward(values)

    def tracking_cross_entropy(*args, **kwargs):
        class_weights.append(kwargs["weight"].detach().numpy().copy())
        return real_cross_entropy(*args, **kwargs)

    monkeypatch.setattr(torch.nn, "Linear", _TrackingLinear)
    monkeypatch.setattr(torch.nn, "CrossEntropyLoss", tracking_cross_entropy)

    first = streamed_subject_probe(batches, subjects, seed=9)
    second = streamed_subject_probe(batches, subjects, seed=9)

    assert first == second
    assert first[1] == 3
    assert first[0] is not None and first[0] > 0.95
    assert standardized_training_batches
    for training_batch in standardized_training_batches:
        np.testing.assert_allclose(training_batch.mean(axis=0), 0.0, atol=1e-5)
    assert class_weights
    assert all(np.ptp(weights) > 0 for weights in class_weights)


def test_streamed_report_rejects_duplicate_or_missing_rows():
    subjects = np.repeat(np.arange(2), 3)
    labels = subjects.copy()
    features = np.arange(18, dtype=float).reshape(6, 3)

    with pytest.raises(ValueError, match="duplicate"):
        streamed_variance_decomposition_report(
            [(np.asarray([0, 0, 1, 2, 3, 4, 5]), features[[0, 0, 1, 2, 3, 4, 5]])],
            subjects,
            labels,
            n_null_permutations=1,
        )
    with pytest.raises(ValueError, match="missing"):
        streamed_variance_decomposition_report(
            [(np.arange(5), features[:5])],
            subjects,
            labels,
            n_null_permutations=1,
        )
