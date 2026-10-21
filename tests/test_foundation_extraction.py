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
from tests.fixtures.embeddings import TokenModel, extract


class FakeFoundationModel:
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
        resample=False,
    )
    result = extractor.extract(
        X,
        signal_metadata=SignalMetadata(sfreq=200.0, ch_names=["T3", "T4"]),
        metadata={"recording_id": "r1"},
    )
    np.testing.assert_allclose(result.window_embeddings, X.mean(axis=-1))
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


def test_labram_capability_needs_known_channels_or_interpolation():
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
    # its own pretraining channels need no interpolation
    known = check_capability(
        "labram", train_mode="frozen", sfreq=200, ch_names=["Fp1", "Cz"], n_times=3000
    )
    assert known.status == "available"


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

    # 1b. ch_names is required — no implicit generic-name fallback
    with pytest.raises(ValueError, match="requires ch_names"):
        prepare_backend("cbramod", np.zeros((2, 19, 200), dtype=np.float32))

    # 1c. so is sfreq — the rate of the data is not assumed to be the model's
    with pytest.raises(ValueError, match="requires sfreq"):
        prepare_backend(
            "cbramod",
            np.zeros((2, 19, 200), dtype=np.float32),
            ch_names=[f"ch{i}" for i in range(19)],
        )

    # 2. Resampling path in PreparedBackend.adapt()
    X = np.zeros((2, 19, 100), dtype=np.float32)
    ch19 = [f"ch{i}" for i in range(19)]
    fake_model = FakeFoundationModel()
    prepared = prepare_backend(
        "cbramod",
        X,
        sfreq=100.0,
        ch_names=ch19,
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
        ch_names=ch19,
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

    # a model given by the caller gets the metadata of the input
    class BareModel:
        pass

    bare = BareModel()
    prepared = prepare_backend(
        "cbramod",
        np.zeros((2, 19, 200)),
        sfreq=200.0,
        ch_names=[f"ch{i}" for i in range(19)],
        model=bare,
    )
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


def test_embedding_extractor_normalizes_backend_pooled_rows():
    class FakePooledModel:
        def transform(self, X):
            return np.ones((len(X), 2))

        def get_embedding_info(self):
            return get_foundation_model_spec("cbramod")

    extractor_mean = FoundationEmbeddingExtractor(
        "cbramod",
        model=FakePooledModel(),
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
    assert res_mean.window_embeddings.shape == (4, 2)
    np.testing.assert_allclose(np.linalg.norm(res_mean.window_embeddings, axis=1), 1.0)


def test_normalization_keeps_a_nan_embedding_nan():
    rows = np.array([[3.0, 4.0], [0.0, 0.0], [np.nan, 1.0]])
    np.testing.assert_array_equal(
        FoundationEmbeddingExtractor._l2_normalize(rows),
        [[0.6, 0.8], [0.0, 0.0], [np.nan, np.nan]],
    )


def test_embedding_extractor_store_tokens_returns_native_tensor():
    raw = np.arange(4 * 10 * 2, dtype=float).reshape(4, 10, 2)

    class Fake3DModel:
        def transform(self, X, *, return_tokens=False):
            pooled = raw[: len(X)].mean(axis=1)
            return (pooled, raw[: len(X)]) if return_tokens else pooled

        def get_token_output_metadata(self):
            return {
                "token_source": "test_native_output",
                "token_axes": ["window", "token", "feature"],
                "token_observation_axes": ["token"],
                "token_feature_axis": "feature",
            }

        def get_embedding_info(self):
            return get_foundation_model_spec("cbramod")

    extractor = FoundationEmbeddingExtractor(
        "cbramod",
        model=Fake3DModel(),
        normalize_embeddings=True,
        pooling="mean",
        resample=False,
        store_tokens=True,
    )
    res = extractor.extract(
        np.zeros((4, 19, 200)),
        signal_metadata=SignalMetadata(
            sfreq=200.0, ch_names=[f"ch{i}" for i in range(19)]
        ),
    )
    # Tokens are the unnormalized pre-pool tensor; window embeddings are the
    # normalized mean over the token axis.
    assert res.token_embeddings.shape == (4, 10, 2)
    np.testing.assert_allclose(res.token_embeddings, raw)
    np.testing.assert_allclose(np.linalg.norm(res.window_embeddings, axis=1), 1.0)
    assert res.metadata["token_shape"] == [4, 10, 2]


def test_store_tokens_preserves_native_rank_values_and_dtype():
    raw = np.arange(4 * 3 * 5 * 2, dtype=np.float64).reshape(4, 3, 5, 2)

    class Fake3DModel:
        def transform(self, X, *, return_tokens=False):
            pooled = raw[: len(X)].mean(axis=(1, 2))
            return (pooled, raw[: len(X)]) if return_tokens else pooled

        def get_token_output_metadata(self):
            return {
                "token_source": "test_native_output",
                "token_axes": ["window", "channel", "patch", "feature"],
                "token_observation_axes": ["channel", "patch"],
                "token_feature_axis": "feature",
            }

        def get_embedding_info(self):
            return get_foundation_model_spec("cbramod")

    extractor = FoundationEmbeddingExtractor(
        "cbramod",
        model=Fake3DModel(),
        normalize_embeddings=True,
        pooling="mean",
        resample=False,
        store_tokens=True,
    )
    res = extractor.extract(
        np.zeros((4, 19, 200)),
        signal_metadata=SignalMetadata(
            sfreq=200.0, ch_names=[f"ch{i}" for i in range(19)]
        ),
    )
    assert res.token_embeddings.shape == raw.shape
    assert res.token_embeddings.dtype == raw.dtype
    np.testing.assert_array_equal(res.token_embeddings, raw)


def test_backend_pooling_is_kept_separate_from_native_tokens():
    raw = np.arange(3 * 6 * 4, dtype=float).reshape(3, 6, 4)

    class Fake3DModel:
        def transform(self, X, *, return_tokens=False):
            pooled = raw[: len(X)].reshape(len(X), -1)
            return (pooled, raw[: len(X)]) if return_tokens else pooled

        def get_token_output_metadata(self):
            return {
                "token_source": "test_native_output",
                "token_axes": ["window", "token", "feature"],
                "token_observation_axes": ["token"],
                "token_feature_axis": "feature",
            }

        def get_embedding_info(self):
            return get_foundation_model_spec("cbramod")

    ex = FoundationEmbeddingExtractor(
        "cbramod",
        model=Fake3DModel(),
        normalize_embeddings=False,
        resample=False,
        store_tokens=True,
    )
    res = ex.extract(
        np.zeros((3, 3, 10)),
        signal_metadata=SignalMetadata(sfreq=200.0, ch_names=["C3", "C4", "Cz"]),
    )
    np.testing.assert_allclose(res.window_embeddings, raw.reshape(3, -1))
    np.testing.assert_array_equal(res.token_embeddings, raw)
    # one token axis: nothing to average for a channel or patch view
    assert res.channel_embeddings is None
    assert res.patch_embeddings is None


def test_channel_and_patch_views_average_the_tokens():
    full = extract([0, 1, 2])
    tokens = full.token_embeddings  # (window, channel, time_patch, feature)
    np.testing.assert_allclose(full.channel_embeddings, tokens.mean(axis=2))
    np.testing.assert_allclose(full.patch_embeddings, tokens.mean(axis=1))
    # the same without keeping the tokens or in chunks, and never normalized
    for lean in (
        extract([0, 1, 2], store_tokens=False),
        extract([0, 1, 2], store_tokens=False, batch_size=2),
    ):
        assert lean.token_embeddings is None
        assert lean.metadata["normalize_embeddings"]
        np.testing.assert_array_equal(lean.channel_embeddings, full.channel_embeddings)
        np.testing.assert_array_equal(lean.patch_embeddings, full.patch_embeddings)


def test_extra_poolings_pool_the_same_tokens():
    full = extract([0, 1, 2], extra_poolings="max")
    maxima = full.token_embeddings.max(axis=(1, 2))
    np.testing.assert_allclose(
        full.extra_window_embeddings["max"],
        maxima / np.linalg.norm(maxima, axis=1, keepdims=True),
    )
    assert list(full.extra_window_embeddings) == ["max"]
    np.testing.assert_array_equal(
        full.window_embeddings, extract([0, 1, 2]).window_embeddings
    )
    assert extract([0]).extra_window_embeddings == {}
    raw = extract([0, 1, 2], extra_poolings=["max"], normalize_embeddings=False)
    np.testing.assert_array_equal(raw.extra_window_embeddings["max"], maxima)
    # the same without keeping the tokens or in chunks
    for lean in (
        extract([0, 1, 2], extra_poolings=["max"], store_tokens=False),
        extract([0, 1, 2], extra_poolings=["max"], store_tokens=False, batch_size=2),
    ):
        np.testing.assert_array_equal(
            lean.extra_window_embeddings["max"], full.extra_window_embeddings["max"]
        )


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"extra_poolings": ["mean"]}, "other than 'mean'"),
        ({"extra_poolings": ["max", "max"]}, "once each"),
        ({"extra_poolings": ["max"], "cache_embeddings": True}, "incompatible"),
    ],
)
def test_extra_poolings_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        FoundationEmbeddingExtractor("cbramod", **settings)


def test_extractor_records_the_weights_of_its_model():
    class RandomTokenModel(TokenModel):
        weights_seed = 3

    def extract_with(model, **weights):
        return FoundationEmbeddingExtractor(
            "cbramod", model=model, resample=False, **weights
        ).extract(
            np.zeros((2, 2, 10), dtype=np.float32),
            signal_metadata=SignalMetadata(sfreq=200.0, ch_names=["T7", "Cz"]),
        )

    metadata = extract_with(TokenModel()).metadata
    assert metadata["weights"] == "pretrained" and "seed" not in metadata
    metadata = extract_with(RandomTokenModel(), pretrained=False, seed=3).metadata
    assert (metadata["weights"], metadata["seed"]) == ("random", 3)
    with pytest.raises(ValueError, match="holds random weights of seed 3, but"):
        extract_with(RandomTokenModel())
    with pytest.raises(ValueError, match="asks for random weights of seed 0"):
        extract_with(TokenModel(), pretrained=False, seed=0)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"pretrained": False}, "need an integer seed"),
        ({"seed": 0}, "seed is for random weights"),
        ({"backend_kwargs": {"seed": 0}}, "not backend_kwargs"),
    ],
)
def test_random_weights_settings_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        FoundationEmbeddingExtractor("cbramod", **settings)


def test_store_tokens_incompatible_with_cache():
    with pytest.raises(ValueError, match="store_tokens is incompatible"):
        FoundationEmbeddingExtractor(
            "cbramod", store_tokens=True, cache_embeddings=True
        )


def test_batch_size_chunks_forward_without_changing_output():
    class CountingModel:
        def __init__(self):
            self.batch_sizes = []

        def transform(self, X):
            self.batch_sizes.append(len(X))
            # Content-based (chunk-invariant) embedding: per-window mean broadcast
            # to 4 features, so chunking must not change the result.
            return X.mean(axis=(1, 2))[:, None] * np.ones((1, 4), dtype=np.float32)

        def get_embedding_info(self):
            return get_foundation_model_spec("cbramod")

    n_windows = 10
    epochs = np.arange(n_windows * 19 * 200, dtype=np.float32).reshape(
        n_windows, 19, 200
    )
    meta = SignalMetadata(sfreq=200.0, ch_names=[f"ch{i}" for i in range(19)])

    single = FoundationEmbeddingExtractor(
        "cbramod", model=CountingModel(), normalize_embeddings=False, resample=False
    )
    single_out = single.extract(epochs, signal_metadata=meta).window_embeddings

    counting = CountingModel()
    batched = FoundationEmbeddingExtractor(
        "cbramod",
        model=counting,
        normalize_embeddings=False,
        resample=False,
        batch_size=4,
    )
    batched_out = batched.extract(epochs, signal_metadata=meta).window_embeddings

    # Chunked into 4 + 4 + 2, capped at batch_size, and numerically identical.
    assert counting.batch_sizes == [4, 4, 2]
    np.testing.assert_array_equal(single_out, batched_out)


def test_batch_size_rejects_non_positive():
    with pytest.raises(ValueError, match="batch_size"):
        FoundationEmbeddingExtractor("cbramod", batch_size=0)


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
