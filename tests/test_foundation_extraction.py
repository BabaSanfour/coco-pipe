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
