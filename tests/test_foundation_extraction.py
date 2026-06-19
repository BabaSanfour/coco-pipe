from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pytest import MonkeyPatch

from coco_pipe.decoding._specs import SignalMetadata
from coco_pipe.decoding.foundation_models import (
    FoundationEmbeddingExtractor,
    check_capability,
    normalize_channel_names,
)
from coco_pipe.decoding.foundation_models._braindecode import BrainDecodeBackend
from coco_pipe.decoding.registry import get_foundation_model_spec


class FakeFoundationModel:
    def bind_signal_metadata(self, metadata):
        self.metadata = metadata
        return self

    def transform(self, X):
        return X.mean(axis=-1)

    def get_embedding_info(self):
        return get_foundation_model_spec("cbramod")


def test_extractor_normalizes_channels_pools_and_records_provenance():
    X = np.arange(4 * 2 * 10, dtype=np.float32).reshape(4, 2, 10)
    extractor = FoundationEmbeddingExtractor(
        "cbramod",
        model=FakeFoundationModel(),
        normalize_embeddings=False,
        recording_pooling="mean",
        resample=False,
    )
    result = extractor.extract(
        X,
        signal_metadata=SignalMetadata(sfreq=200.0, ch_names=["T3", "T4"]),
        metadata={"recording_id": "r1"},
    )
    assert result.window_embeddings.shape == (4, 2)
    np.testing.assert_allclose(
        result.recording_embedding, result.window_embeddings.mean(axis=0)
    )
    assert result.metadata["model_channels"] == ["T7", "T8"]
    assert result.metadata["recording_id"] == "r1"
    assert result.metadata["input_sfreq"] == 200.0
    assert result.metadata["model_sfreq"] == 200.0
    assert normalize_channel_names(["T5", "T6"]) == ["P7", "P8"]


def test_extractor_caches_window_embeddings_without_changing_output():
    X = np.arange(4 * 2 * 10, dtype=np.float32).reshape(4, 2, 10)
    meta = SignalMetadata(sfreq=200.0, ch_names=["T3", "T4"])

    plain = FoundationEmbeddingExtractor(
        "cbramod",
        model=FakeFoundationModel(),
        normalize_embeddings=False,
        resample=False,
    )
    expected = plain.extract(X, signal_metadata=meta).window_embeddings

    model = FakeFoundationModel()
    rows_seen = {"n": 0}
    real_transform = model.transform

    def counting_transform(batch):
        rows_seen["n"] += len(batch)
        return real_transform(batch)

    model.transform = counting_transform
    cached = FoundationEmbeddingExtractor(
        "cbramod",
        model=model,
        normalize_embeddings=False,
        resample=False,
        cache_embeddings=True,
    )
    first = cached.extract(X, signal_metadata=meta).window_embeddings
    assert rows_seen["n"] == len(X)
    np.testing.assert_allclose(first, expected, rtol=1e-6)

    second = cached.extract(X, signal_metadata=meta).window_embeddings
    assert rows_seen["n"] == len(X)  # second pass fully served from cache
    np.testing.assert_allclose(second, expected, rtol=1e-6)

    cached.clear_cache()
    cached.extract(X, signal_metadata=meta)
    assert rows_seen["n"] == 2 * len(X)


def test_labram_channel_adaptation_reports_real_19_to_128_mapping():
    source = [f"src-{index}" for index in range(19)]
    target = [f"target-{index}" for index in range(128)]
    layer = SimpleNamespace(
        src_chs_info=[{"ch_name": value} for value in source],
        tgt_chs_info=[{"ch_name": value} for value in target],
        mode="always",
        method="spline",
        matrix=torch.zeros(128, 19),
    )
    model = SimpleNamespace(interpolation_layer=layer)
    backend = BrainDecodeBackend(
        metadata=get_foundation_model_spec("labram"),
        model=model,
        feat_dim=200,
        n_outputs=None,
        device="cpu",
        train_mode="frozen",
        task="classification",
    )
    backend.signal_metadata_ = SignalMetadata(sfreq=200, ch_names=source)
    backend._uses_interpolation = True
    adaptation = backend.get_channel_adaptation()
    assert len(adaptation["interpolated_channels"]) == 128
    assert adaptation["interpolation_matrix_shape"] == [128, 19]
    assert adaptation["zero_filled_channels"] == []


def test_labram_capability_requires_explicit_interpolation():
    channels = [f"ch-{index}" for index in range(19)]
    rejected = check_capability(
        "labram",
        train_mode="frozen",
        sfreq=200,
        ch_names=channels,
        n_times=3000,
    )
    assert rejected.status == "incompatible_channels"
    accepted = check_capability(
        "labram",
        train_mode="frozen",
        sfreq=200,
        ch_names=channels,
        n_times=3000,
        backend_kwargs={"interpolate_channels": True},
    )
    assert accepted.status == "available"


def test_reve_capability_reports_missing_gated_checkpoint_authentication():
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    result = check_capability(
        "reve",
        train_mode="frozen",
        sfreq=200,
        ch_names=["C3", "C4"],
        backend_kwargs={},
    )
    monkeypatch.undo()
    assert result.status == "authentication_required"


def test_reve_capability_accepts_explicit_token():
    result = check_capability(
        "reve",
        train_mode="frozen",
        sfreq=200,
        ch_names=["C3", "C4"],
        backend_kwargs={"token": "test-token"},
    )
    assert result.status == "available"


def test_prepare_backend_and_adapt_paths():
    from coco_pipe.decoding.foundation_models._prepare import prepare_backend

    # 1. ValueError on invalid ndim
    with pytest.raises(ValueError, match="must be \\(sample, channel, time\\)"):
        prepare_backend("cbramod", np.zeros((10, 10)))

    # 2. Resampling path in PreparedBackend.adapt()
    X = np.zeros((2, 19, 100), dtype=np.float32)
    fake_model = FakeFoundationModel()
    prepared = prepare_backend(
        "cbramod",
        X,
        sfreq=100.0,
        model=fake_model,
    )
    assert prepared.source_sfreq == 100.0
    assert prepared.target_sfreq == 200.0

    adapted = prepared.adapt(X)
    # 100 samples at 100 Hz resampled to 200 Hz -> 200 samples
    assert adapted.shape[-1] == 200

    # 3. ValueError in PreparedBackend.adapt() when length doesn't match spec
    prepared_mismatch = prepare_backend(
        "cbramod",
        np.zeros((2, 19, 50), dtype=np.float32),
        sfreq=200.0,
        model=fake_model,
    )
    import dataclasses

    prepared_mismatch = dataclasses.replace(
        prepared_mismatch,
        spec=dataclasses.replace(prepared_mismatch.spec, pretrained_n_times=200),
    )
    with pytest.raises(ValueError, match="checkpoint requires 200 samples"):
        prepared_mismatch.adapt(np.zeros((2, 19, 50), dtype=np.float32))

    # 4. Fallback channel_adaptation when backend doesn't have get_channel_adaptation
    # FakeFoundationModel has no get_channel_adaptation
    adapt_dict = prepared.channel_adaptation()
    assert adapt_dict["source_channels"] == prepared.ch_names
    assert adapt_dict["interpolated_channels"] == []


def test_prepare_backend_fallback_binding():
    from coco_pipe.decoding.foundation_models._prepare import prepare_backend

    # Model with no bind_signal_metadata nor signal_metadata_ originally
    class BareModel:
        pass

    bare = BareModel()
    prepared = prepare_backend("cbramod", np.zeros((2, 19, 200)), model=bare)
    assert hasattr(prepared.backend, "signal_metadata_")


def test_check_capability_missing_dependency():
    from unittest.mock import patch

    # Mock preferred backend is_available to return False
    with patch(
        "coco_pipe.decoding.foundation_models._braindecode.BrainDecodeBackend.is_available",
        return_value=False,
    ):
        res = check_capability("cbramod", "frozen", 200, ["Fp1"])
        assert res.status == "missing_dependency"

    # Test checking raising exception
    with patch(
        "coco_pipe.decoding.foundation_models.extraction._BACKEND_MAP"
    ) as mock_map:
        mock_map.get.side_effect = Exception("failed to get class")
        res2 = check_capability("cbramod", "frozen", 200, ["Fp1"])
        assert res2.status == "missing_dependency"
        assert "failed to get class" in res2.reason


def test_embedding_extractor_3d_pooling_and_normalization():
    class Fake3DModel:
        def bind_signal_metadata(self, metadata):
            return self

        def transform(self, X):
            # Shape (4, 10, 2)
            return np.ones((4, 10, 2))

        def get_embedding_info(self):
            return get_foundation_model_spec("cbramod")

    # 1. pooling="mean"
    extractor_mean = FoundationEmbeddingExtractor(
        "cbramod",
        model=Fake3DModel(),
        normalize_embeddings=True,
        pooling="mean",
        resample=False,
    )
    res_mean = extractor_mean.extract(
        np.zeros((4, 19, 200)),
        signal_metadata=SignalMetadata(
            sfreq=200.0, ch_names=[f"ch{i}" for i in range(19)]
        ),
    )
    # Normalization divides by norm, making all entries equal norm
    assert res_mean.window_embeddings.shape == (4, 2)
    # All rows should be normalized (norm == 1.0)
    np.testing.assert_allclose(np.linalg.norm(res_mean.window_embeddings, axis=1), 1.0)

    # 2. pooling="flatten"
    extractor_flat = FoundationEmbeddingExtractor(
        "cbramod",
        model=Fake3DModel(),
        normalize_embeddings=True,
        pooling="flatten",
        resample=False,
    )
    res_flat = extractor_flat.extract(
        np.zeros((4, 19, 200)),
        signal_metadata=SignalMetadata(
            sfreq=200.0, ch_names=[f"ch{i}" for i in range(19)]
        ),
    )
    # Shape is flattened to (4, 20)
    assert res_flat.window_embeddings.shape == (4, 20)
    np.testing.assert_allclose(np.linalg.norm(res_flat.window_embeddings, axis=1), 1.0)


def _window_container(n_times, sfreq=200.0):
    from coco_pipe.io import DataContainer

    X = np.zeros((2, 3, n_times), dtype=np.float32)
    return DataContainer(
        X=X,
        dims=("obs", "channel", "time"),
        coords={
            "obs": np.array(["a", "b"], dtype=object),
            "channel": np.array(["Fz", "Cz", "Pz"], dtype=object),
            "time": np.arange(n_times),
        },
        ids=np.array(["a", "b"], dtype=object),
        meta={"sfreq": sfreq},
    )


def test_normalize_inclusive_endpoint_passes_correct_length_through():
    from coco_pipe.decoding.foundation_models import normalize_inclusive_endpoint

    container = _window_container(200)
    out, reason = normalize_inclusive_endpoint(
        container, segment_duration=1.0, expected_sfreq=200.0, model_key="m"
    )
    assert reason is None
    assert out is container
    assert out.X.shape[-1] == 200


def test_normalize_inclusive_endpoint_trims_extra_sample():
    from coco_pipe.decoding.foundation_models import normalize_inclusive_endpoint

    out, reason = normalize_inclusive_endpoint(
        _window_container(201),
        segment_duration=1.0,
        expected_sfreq=200.0,
        model_key="m",
    )
    assert reason is None
    assert out.X.shape[-1] == 200
    assert out.meta["inclusive_endpoint_removed"] is True
    assert out.meta["original_n_times"] == 201
    assert out.meta["normalized_n_times"] == 200


def test_normalize_inclusive_endpoint_skip_vs_error_on_bad_length():
    from coco_pipe.decoding.foundation_models import normalize_inclusive_endpoint

    out, reason = normalize_inclusive_endpoint(
        _window_container(205),
        segment_duration=1.0,
        expected_sfreq=200.0,
        model_key="m",
        on_mismatch="skip",
    )
    assert out is None
    assert "205" in reason

    with pytest.raises(ValueError, match="205"):
        normalize_inclusive_endpoint(
            _window_container(205),
            segment_duration=1.0,
            expected_sfreq=200.0,
            model_key="m",
            on_mismatch="error",
        )


def test_spec_pretrained_window_seconds_matches_formula():
    # labram declares a fixed window (3000 samples @ 200 Hz = 15 s)
    labram = get_foundation_model_spec("labram")
    assert labram.pretrained_window_seconds == (
        labram.pretrained_n_times / labram.pretrained_sfreq
    )
    assert labram.pretrained_window_seconds == 15.0
    # cbramod leaves pretrained_n_times unset → no fixed window
    assert get_foundation_model_spec("cbramod").pretrained_window_seconds is None
