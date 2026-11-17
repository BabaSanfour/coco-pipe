"""Tests for BrainDecodeBackend on real braindecode models with random weights.

Only the download is replaced: each model is built from its class instead of
being fetched from the Hugging Face Hub.
"""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

from coco_pipe.decoding._specs import SignalMetadata
from coco_pipe.decoding.foundation_models import (
    FoundationClassifier,
    FoundationEmbeddingExtractor,
    check_capability,
)
from coco_pipe.decoding.foundation_models._braindecode import BrainDecodeBackend
from coco_pipe.decoding.foundation_models.validation import DEFAULT_CHANNELS
from coco_pipe.decoding.registry import get_estimator_spec
from tests.fixtures.hub import LATEST

pytest.importorskip("braindecode")
torch = pytest.importorskip("torch")

BACKBONES = ["bendr", "biot", "cbramod", "eegpt", "labram", "luna", "signaljepa"]
# LUNA's features and BIOT's tokens are in braindecode main, not in 2.0
LUNA = importlib.import_module("braindecode.models").LUNA
BRAINDECODE_MAIN = "return_features" in inspect.signature(LUNA.forward).parameters


def _skip_before_main(model_key):
    if model_key in {"biot", "luna"} and not BRAINDECODE_MAIN:
        pytest.skip(f"{model_key} needs braindecode main")


def _load(
    model_key, train_mode="frozen", n_outputs=2, channels=DEFAULT_CHANNELS, **kwargs
):
    _skip_before_main(model_key)
    spec = get_estimator_spec(model_key)
    if model_key == "luna":
        # LUNA cuts its input into patches of 40 samples: 2.5 s at 256 Hz
        kwargs.setdefault("n_times", 640)
    backend = BrainDecodeBackend.load(
        model_key,
        spec,
        n_outputs=n_outputs,
        device="cpu",
        train_mode=train_mode,
        electrode_names=list(channels),
        sfreq=spec.pretrained_sfreq,
        **kwargs,
    )
    backend.signal_metadata_ = SignalMetadata(
        sfreq=spec.pretrained_sfreq, ch_names=list(channels)
    )
    n_times = kwargs.get("n_times") or round(spec.pretrained_sfreq * 2)
    X = np.random.default_rng(0).normal(size=(4, len(channels), n_times))
    return backend, X.astype(np.float32)


def _trainable(backend):
    return {
        name
        for component in backend.checkpoint_components().values()
        for name, parameter in component.named_parameters()
        if parameter.requires_grad
    }


@pytest.mark.parametrize("model_key", BACKBONES)
def test_backbones_embed_and_expose_their_tokens(offline, model_key):
    backend, X = _load(model_key)
    spec = get_estimator_spec(model_key)

    embeddings = backend.transform(X)
    assert embeddings.shape == (4, spec.embedding_dim)
    assert np.isfinite(embeddings).all()

    pooled, tokens = backend.transform(X, return_tokens=True)
    np.testing.assert_allclose(pooled, embeddings, rtol=1e-5, atol=1e-6)
    layout = backend.get_token_output_metadata()
    assert len(layout["token_axes"]) == tokens.ndim
    assert tokens.shape[0] == 4
    feature_axis = layout["token_axes"].index(layout["token_feature_axis"])
    assert tokens.shape[feature_axis] == spec.embedding_dim
    assert "model" in backend.checkpoint_components()
    with pytest.raises(NotImplementedError, match="no pooling but its own"):
        backend.pool_tokens(tokens, "attention")


def _weights(backend):
    return {
        f"{name}.{key}": value
        for name, component in backend.checkpoint_components().items()
        for key, value in component.state_dict().items()
    }


@pytest.mark.parametrize("model_key", BACKBONES)
def test_random_weights_keep_the_architecture(offline, model_key):
    loaded, X = _load(model_key, n_outputs=None)
    random, _ = _load(model_key, n_outputs=None, pretrained=False, seed=0)
    assert not any({"pretrained", "seed"} & set(call) for call in offline)
    again = _weights(_load(model_key, n_outputs=None, pretrained=False, seed=0)[0])
    other = _weights(_load(model_key, n_outputs=None, pretrained=False, seed=1)[0])
    pretrained, weights = _weights(loaded), _weights(random)
    assert list(weights) == list(pretrained)
    assert all(weights[key].shape == pretrained[key].shape for key in weights)
    assert all(torch.equal(weights[key], again[key]) for key in weights)
    drawn = [key for key in weights if not torch.equal(weights[key], other[key])]
    assert drawn
    assert not any(torch.equal(weights[key], pretrained[key]) for key in drawn)
    assert np.isfinite(random.transform(X)).all()


@pytest.mark.parametrize(
    "model_key, train_mode",
    [*((key, "frozen") for key in BACKBONES), ("cbramod", "lora"), ("bendr", "full")],
)
def test_random_weights_draw_the_head_from_the_seed(offline, model_key, train_mode):
    def weights(global_seed, **settings):
        torch.manual_seed(global_seed)
        return _weights(_load(model_key, train_mode, **settings)[0])

    first = weights(1, pretrained=False, seed=0)
    again = weights(2, pretrained=False, seed=0)
    assert all(torch.equal(first[key], again[key]) for key in first)
    other = weights(1, pretrained=False, seed=1)
    head = [key for key in first if not torch.equal(first[key], other[key])]
    assert any(".final_layer." in key or ".classifier." in key for key in head)
    pretrained = weights(1), weights(1), weights(2)
    assert all(torch.equal(pretrained[0][key], pretrained[1][key]) for key in first)
    assert any(not torch.equal(pretrained[0][key], pretrained[2][key]) for key in first)


@pytest.mark.parametrize(
    "weights, message",
    [
        ({"pretrained": False}, "need an integer seed"),
        ({"pretrained": False, "seed": True}, "need an integer seed"),
        ({"seed": 0}, "seed is for random weights"),
        ({"pretrained": "no", "seed": 0}, "True or False"),
    ],
)
def test_random_weights_need_a_seed(offline, weights, message):
    with pytest.raises(ValueError, match=message):
        _load("cbramod", **weights)


@pytest.mark.parametrize("model_key", ["cbramod", "bendr"])
def test_train_modes_choose_what_trains(offline, model_key):
    frozen, _ = _load(model_key, "frozen")
    head = _trainable(frozen)
    assert head and all("lora" not in name for name in head)

    full, _ = _load(model_key, "full")
    assert len(_trainable(full)) > len(head)

    lora, _ = _load(model_key, "lora", lora_r=4, lora_alpha=8)
    trained = _trainable(lora)
    assert any("lora_" in name for name in trained)
    assert len(trained) < len(_trainable(full))


def test_fit_predict_and_reset_head(offline):
    backend, X = _load("cbramod", "full")
    y = np.array([0, 1, 0, 1])
    backend.fit(X, y, max_epochs=1, batch_size=2, validation_fraction=0.0)
    # no validation split: the epoch trains on all four windows
    assert "valid_loss" not in backend.training_history()[0]
    # dropout is off once training is over
    np.testing.assert_array_equal(backend.transform(X), backend.transform(X))
    # without groups, the fraction is held out by sample
    backend.fit(np.tile(X, (2, 1, 1)), np.tile(y, 2), max_epochs=1, batch_size=2)
    assert "valid_loss" in backend.training_history()[0]
    assert backend.predict(X).shape == (4,)
    assert backend.predict_proba(X).shape == (4, 2)

    backend.reset_head(3)
    backend.fit(X, np.array([0, 1, 2, 0]), max_epochs=1, validation_fraction=0.0)
    assert backend.predict_proba(X).shape == (4, 3)


@pytest.mark.parametrize("model_key", ["cbramod", "ctnet"])
def test_dropout_and_head_weight_decay_reach_the_model(offline, model_key):
    # CBraMod has torch's attention, CTNet braindecode's
    weights = {"pretrained": False, "seed": 0} if model_key == "ctnet" else {}
    backend, X = _load(model_key, "full", **weights)
    rates = {"backbone_dropout": 0.1, "head_dropout": 0.3, "head_weight_decay": 0.5}
    for _ in range(2):  # a second fit puts no second dropout before the head
        backend.fit(
            X, np.array([0, 1, 0, 1]), max_epochs=1, validation_fraction=0, **rates
        )

    head = backend.head_module()
    assert isinstance(head[0], torch.nn.Dropout) and head[0].p == 0.3
    assert sum(isinstance(module, torch.nn.Dropout) for module in head.modules()) == 1
    in_head = {id(module) for module in head.modules()}
    backbone = [
        getattr(module, name)
        for module in backend._model.modules()
        if id(module) not in in_head
        for name in ("p", "dropout", "att_drop")
        if isinstance(getattr(module, name, None), float)
    ]
    assert len(backbone) > 2 and set(backbone) == {0.1}
    groups = backend._net_.optimizer_.param_groups
    assert [group["weight_decay"] for group in groups] == [0.01, 0.5]
    assert {id(p) for p in groups[1]["params"]} == {id(p) for p in head.parameters()}


def test_experiment_saves_the_fine_tuned_features_of_each_fold(tmp_path):
    from pydantic import ValidationError

    from coco_pipe.decoding import (
        CVConfig,
        Experiment,
        ExperimentConfig,
        NeuralFineTuneConfig,
    )
    from coco_pipe.decoding.configs import CheckpointConfig

    with pytest.raises(ValidationError, match="output_dir"):
        CheckpointConfig(save_features=True)
    X = np.random.default_rng(0).normal(scale=20, size=(16, 19, 256))
    y = np.repeat([0, 1], 8)
    groups = np.repeat([f"s{index}" for index in range(8)], 2)
    ids = [f"w{index:02d}" for index in range(16)]
    model = NeuralFineTuneConfig(
        model_key="eegnet",
        sfreq=128,
        ch_names=DEFAULT_CHANNELS,
        device={"device": "cpu"},
        backend_kwargs={"pretrained": False, "seed": 0},
        trainer={"max_epochs": 1, "batch_size": 4},
        checkpoints={"output_dir": tmp_path, "save_features": True},
    )
    config = ExperimentConfig(
        models={"eegnet": model},
        cv=CVConfig(strategy="stratified_group_kfold", n_splits=2, group_key="Subject"),
        metrics=["accuracy"],
        n_jobs=1,
    )
    result = Experiment(config).run(X, y, groups=groups, sample_ids=ids)

    artifacts = result.get_model_artifacts()
    paths = artifacts[artifacts["Key"] == "features_path"]["Value"]
    assert sorted(path.name for path in tmp_path.glob("*_features.npz")) == sorted(
        Path(path).name for path in paths
    )
    for fold, path in enumerate(paths):
        saved = np.load(path)
        assert int(saved["fold"]) == fold
        assert saved["features"].dtype == np.float32
        assert saved["features"].shape[0] == 16
        assert sorted(saved["sample_id"]) == ids
        group_of = dict(zip(ids, groups, strict=True))
        assert saved["group"].tolist() == [group_of[i] for i in saved["sample_id"]]
        assert saved["y"].tolist() == [y[ids.index(i)] for i in saved["sample_id"]]
        # training rows first; the groups held out for early stopping are marked
        test = saved["split"] == "test"
        assert test.tolist() == sorted(test.tolist())
        assert set(saved["split"][~test]) == {"train", "valid"}
        assert not set(saved["group"][test]) & set(saved["group"][~test])


def _zscore(X):
    return (X - X.mean(-1, keepdims=True)) / X.std(-1, ddof=1, keepdims=True)


# What each model's authors feed it, from microvolt channels in its own montage.
AUTHORS_INPUT = {
    "cbramod": lambda X: X / 100,
    "labram": lambda X: X / 100,
    "biot": lambda X: X / np.quantile(np.abs(X), 0.95, axis=-1, keepdims=True),
    "luna": _zscore,
    "eegpt": lambda X: (X - X.mean(1, keepdims=True)) / 1000,
    "steegformer_largev2": _zscore,
}


@pytest.mark.parametrize("model_key", AUTHORS_INPUT)
def test_models_receive_the_input_of_their_authors(offline, model_key):
    backend, X = _load(model_key, "full")
    seen = []
    backend._model.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().numpy())
    )

    backend.transform(X)
    backend.predict(X)
    backend.fit(
        X, np.array([0, 1, 0, 1]), max_epochs=1, batch_size=4, validation_fraction=0.0
    )
    backend.predict_proba(X)

    adapter = backend._channel_adapter
    expected = AUTHORS_INPUT[model_key](adapter(X) if adapter else X)
    # embedding, prediction, training and probabilities all see that input
    assert len(seen) == 4
    for batch in (seen[0], seen[1], seen[3]):
        np.testing.assert_allclose(batch, expected, rtol=1e-4, atol=1e-6)


def test_bendr_head_reads_the_embedding_not_the_start_token(offline):
    backend, X = _load("bendr", "full")
    backend.fit(X, np.array([0, 1, 0, 1]), max_epochs=1, validation_fraction=0.0)
    probabilities = backend.predict_proba(X)
    with torch.no_grad():
        logits = backend._model.final_layer(torch.from_numpy(backend.transform(X)))
    np.testing.assert_allclose(
        probabilities, logits.softmax(-1).numpy(), rtol=1e-4, atol=1e-6
    )
    np.testing.assert_array_equal(backend.predict(X), logits.argmax(-1).numpy())


def test_volt_scale_input_is_rejected(offline):
    backend, X = _load("cbramod", "full")
    volts = X * 30e-6
    y = np.array([0, 1, 0, 1])

    for call in (
        lambda: backend.fit(volts, y, max_epochs=1),
        lambda: backend.transform(volts),
        lambda: backend.predict(volts),
        lambda: backend.predict_proba(volts),
    ):
        with pytest.raises(ValueError, match="expects EEG in microvolts"):
            call()
    # all-zero windows are not a units mistake
    assert backend.transform(np.zeros_like(X)).shape[0] == 4


def test_windows_with_nan_or_inf_are_rejected(offline):
    backend, X = _load("cbramod", "full")
    X[1, 0, 5] = np.nan
    X[3, 2, 0] = np.inf
    y = np.array([0, 1, 0, 1])

    for call in (
        lambda: backend.fit(X, y, max_epochs=1),
        lambda: backend.transform(X),
        lambda: backend.predict(X),
        lambda: backend.predict_proba(X),
        lambda: FoundationEmbeddingExtractor("cbramod").extract(
            X, signal_metadata=SignalMetadata(200.0, DEFAULT_CHANNELS)
        ),
    ):
        with pytest.raises(ValueError, match=r"2 of 4 windows .* \[1, 3\]"):
            call()


def test_load_takes_the_pinned_checkpoint_unless_given_a_revision(offline):
    pinned = get_estimator_spec("cbramod").checkpoint_revision
    # the commit is resolved first, so the one loaded is the one recorded
    backend, X = _load("cbramod")
    assert offline[-1]["revision"] == pinned
    assert backend.checkpoint_revisions == {"braindecode/cbramod-pretrained": pinned}
    backend, _ = _load("cbramod", revision="abc123")
    assert offline[-1]["revision"] == "abc123"
    assert backend.checkpoint_revisions == {"braindecode/cbramod-pretrained": "abc123"}
    _load("cbramod", revision=None)
    assert offline[-1]["revision"] == LATEST

    # extracted embeddings say which commit made them
    meta = SignalMetadata(sfreq=200.0, ch_names=DEFAULT_CHANNELS)
    result = FoundationEmbeddingExtractor("cbramod", device="cpu").extract(
        X, signal_metadata=meta
    )
    assert result.metadata["checkpoint_revisions"] == {
        "braindecode/cbramod-pretrained": pinned
    }
    result = FoundationEmbeddingExtractor(
        "cbramod", device="cpu", backend_kwargs={"revision": "abc123"}
    ).extract(X, signal_metadata=meta)
    assert result.metadata["checkpoint_revisions"] == {
        "braindecode/cbramod-pretrained": "abc123"
    }


def test_extractor_draws_random_weights_from_its_seed(offline):
    _, X = _load("cbramod")
    meta = SignalMetadata(sfreq=200.0, ch_names=DEFAULT_CHANNELS)
    extractor = FoundationEmbeddingExtractor(
        "cbramod", device="cpu", pretrained=False, seed=0
    )
    first = extractor.extract(X, signal_metadata=meta)
    assert (first.metadata["weights"], first.metadata["seed"]) == ("random", 0)
    assert not any({"pretrained", "seed"} & set(call) for call in offline)
    again = FoundationEmbeddingExtractor(
        "cbramod", device="cpu", pretrained=False, seed=0
    ).extract(X, signal_metadata=meta)
    np.testing.assert_array_equal(again.window_embeddings, first.window_embeddings)
    extractor.seed = 1
    other = extractor.extract(X, signal_metadata=meta)
    assert other.metadata["seed"] == 1
    assert not np.allclose(other.window_embeddings, first.window_embeddings)


def test_labram_takes_its_channels_by_name(offline):
    # the 19 channels go in as they are, found by name
    plain, X = _load("labram")
    # frozen: the pretrained final norm, not the untrained readout norm
    assert offline[-1]["use_mean_pooling"] is False
    assert not plain.get_channel_adaptation().get("interpolated_channels")
    _, tokens = plain.transform(X, return_tokens=True)
    assert tokens.shape == (4, len(DEFAULT_CHANNELS), 2, 200)
    assert plain.get_token_output_metadata()["token_axes"] == [
        "window",
        "channel",
        "time_patch",
        "feature",
    ]


def test_luna_refuses_a_channel_without_a_position(offline):
    with pytest.raises(ValueError, match="X1"):
        _load("luna", channels=[*DEFAULT_CHANNELS[:-1], "X1"])


def test_luna_finds_positions_whatever_the_case_of_the_names(offline):
    def embed(channels):
        backend, X = _load("luna", channels=channels, pretrained=False, seed=0)
        return backend.transform(X)

    upper = [name.upper() for name in DEFAULT_CHANNELS]
    np.testing.assert_array_equal(embed(upper), embed(DEFAULT_CHANNELS))


def test_steegformer_finds_the_slot_of_each_channel_by_name(offline):
    from braindecode.models.steegformer import STEEGFORMER_CHANNEL_ORDER

    backend, _ = _load("steegformer_largev2", pretrained=False, seed=0)
    slots = [STEEGFORMER_CHANNEL_ORDER.index(name) for name in DEFAULT_CHANNELS]
    assert backend._model.channel_indices.tolist() == slots
    with pytest.raises(ValueError, match="X1"):
        _load("steegformer_largev2", channels=[*DEFAULT_CHANNELS[:-1], "X1"])


@pytest.mark.parametrize(
    "model_key", ["ctnet", "deep4net", "eegnet", "shallowfbcspnet"]
)
def test_networks_without_checkpoint_train_from_their_seed(model_key):
    sfreq = get_estimator_spec(model_key).pretrained_sfreq
    X = np.random.default_rng(0).normal(scale=20, size=(8, 19, round(sfreq * 10)))
    y = np.repeat([0, 1], 4)

    def fit(seed):
        return FoundationClassifier(
            model_key,
            device="cpu",
            sfreq=sfreq,
            ch_names=DEFAULT_CHANNELS,
            trainer={"max_epochs": 1, "batch_size": 4},
            backend_kwargs={"pretrained": False, "seed": seed},
        ).fit(X, y, groups=np.repeat(np.arange(4), 2))

    first, again, other = fit(0), fit(0), fit(1)
    assert first.backend_.checkpoint_revisions == {}
    np.testing.assert_allclose(first.predict_proba(X), again.predict_proba(X))
    assert not np.allclose(first.predict_proba(X), other.predict_proba(X))


@pytest.mark.parametrize(
    "train_mode, weights, message",
    [
        ("full", {}, "no pretrained weights"),
        ("linear_probe", {"pretrained": False, "seed": 0}, "supports train_mode"),
    ],
)
def test_networks_without_checkpoint_train_only_in_full(train_mode, weights, message):
    signal = {"sfreq": 128.0, "ch_names": DEFAULT_CHANNELS}
    capability = check_capability(
        "eegnet", train_mode, backend_kwargs=weights, **signal
    )
    assert capability.status == "unsupported"
    classifier = FoundationClassifier(
        "eegnet", train_mode=train_mode, device="cpu", backend_kwargs=weights, **signal
    )
    with pytest.raises(ValueError, match=message):
        classifier.fit(np.ones((4, 19, 1280)), np.array([0, 1, 0, 1]))


@pytest.mark.parametrize(("model_key", "n_times"), [("luna", 512), ("labram", 2100)])
def test_models_refuse_windows_they_cannot_patch(offline, model_key, n_times):
    with pytest.raises(ValueError, match="not divisible by patch_size"):
        _load(model_key, n_times=n_times)


def test_labram_tokens_follow_their_channel(offline):
    plain, X = _load("labram")
    order = np.random.default_rng(0).permutation(len(DEFAULT_CHANNELS))
    moved, _ = _load(
        "labram",
        channels=[DEFAULT_CHANNELS[index] for index in order],
    )
    moved._model.load_state_dict(plain._model.state_dict())
    _, tokens = plain.transform(X, return_tokens=True)
    _, moved_tokens = moved.transform(X[:, order], return_tokens=True)
    np.testing.assert_allclose(moved_tokens, tokens[:, order], rtol=1e-4, atol=1e-4)


def test_load_errors(offline, monkeypatch):
    spec = get_estimator_spec("cbramod")
    with pytest.raises(NotImplementedError, match="QLoRA"):
        BrainDecodeBackend.load(
            "cbramod", spec, n_outputs=None, device="cpu", train_mode="qlora"
        )
    with pytest.raises(ValueError, match="support pooling='mean'"):
        BrainDecodeBackend.load(
            "cbramod",
            spec,
            n_outputs=None,
            device="cpu",
            train_mode="frozen",
            pooling="flatten",
        )
    with pytest.raises(ValueError, match="BrainDecodeBackend does not support"):
        BrainDecodeBackend.load(
            "reve",
            get_estimator_spec("reve"),
            n_outputs=None,
            device="cpu",
            train_mode="frozen",
        )

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "braindecode" else real_find_spec(name, *a),
    )
    assert not BrainDecodeBackend.is_available()
    with pytest.raises(ImportError, match="requires braindecode"):
        BrainDecodeBackend.load(
            "cbramod", spec, n_outputs=2, device="cpu", train_mode="frozen"
        )


def test_transform_checks_its_input(offline):
    backend, X = _load("cbramod")
    unfitted = BrainDecodeBackend(
        metadata=get_estimator_spec("cbramod"),
        model=backend._model,
        n_outputs=None,
        device="cpu",
        train_mode="frozen",
        task="classification",
    )
    with pytest.raises(RuntimeError, match="fitted"):
        unfitted.transform(X)

    backend.signal_metadata_ = SignalMetadata(
        sfreq=100.0, ch_names=list(DEFAULT_CHANNELS)
    )
    with pytest.warns(UserWarning, match="sfreq"):
        backend.transform(X)


def test_extractor_loads_again_for_another_montage(offline):
    _skip_before_main("biot")
    X = np.random.default_rng(0).normal(scale=20, size=(3, 19, 2000))
    extractor = FoundationEmbeddingExtractor(
        "biot", device="cpu", normalize_embeddings=False
    )

    def extract(X, ch_names):
        torch.manual_seed(0)  # a model loaded again gets the same random weights
        result = extractor.extract(
            X, signal_metadata=SignalMetadata(sfreq=200.0, ch_names=ch_names)
        )
        return result.window_embeddings

    first = extract(X, DEFAULT_CHANNELS)
    extract(X, DEFAULT_CHANNELS)
    assert len(offline) == 1  # same montage: the loaded model is kept
    # the same signals with the channels listed in another order embed the same
    again = extract(X[:, ::-1], DEFAULT_CHANNELS[::-1])
    assert len(offline) == 2
    np.testing.assert_allclose(again, first, rtol=1e-5, atol=1e-5)


def test_weight_sets_extract_a_recording_as_each_extractor_alone(offline, monkeypatch):
    import mne

    _skip_before_main("luna")

    from coco_pipe.decoding.foundation_models import extract_weight_sets
    from coco_pipe.io import DataContainer

    # LUNA at 256 Hz, on windows of 2.5 s recorded at 200 Hz
    conditions = ["EO", "EC", "EO", "EC"]
    recording = DataContainer(
        X=np.random.default_rng(0).normal(scale=20, size=(4, 19, 500)),
        dims=("obs", "channel", "time"),
        coords={
            "channel": np.array(DEFAULT_CHANNELS, dtype=object),
            "time": np.arange(500) / 200.0,
            "condition": np.array(conditions, dtype=object),
            "epoch": np.arange(4),
        },
        ids=np.array([f"epoch-{epoch}" for epoch in range(4)]),
        meta={"sfreq": 200.0},
    )
    extractors = [
        FoundationEmbeddingExtractor(
            "luna", device="cpu", pretrained=seed is None, seed=seed, store_tokens=True
        )
        for seed in (None, 0, 1)
    ]
    resampled = []
    resample = mne.filter.resample

    def counted(*args, **kwargs):
        resampled.append(args[0].shape)
        return resample(*args, **kwargs)

    monkeypatch.setattr(mne.filter, "resample", counted)

    alone = [extractor.extract_recording(recording) for extractor in extractors]
    assert len(resampled) == 6
    together = extract_weight_sets(extractors, recording)
    assert len(resampled) == 6 + 2  # once per condition

    for one, other in zip(alone, together, strict=True):
        assert one.keys() == other.keys() == {"EO", "EC"}
        for condition, result in one.items():
            shared = other[condition]
            for name in ("window_embeddings", "token_embeddings", "window_index"):
                np.testing.assert_array_equal(
                    getattr(shared, name), getattr(result, name)
                )
            assert shared.metadata == result.metadata
    assert not np.allclose(
        alone[0]["EC"].window_embeddings, alone[1]["EC"].window_embeddings
    )
