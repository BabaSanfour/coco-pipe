"""Tests for BrainDecodeBackend on real braindecode models with random weights.

Only the download is replaced: each model is built from its class instead of
being fetched from the Hugging Face Hub.
"""

import importlib.util

import numpy as np
import pytest

from coco_pipe.decoding._specs import SignalMetadata
from coco_pipe.decoding.foundation_models._braindecode import BrainDecodeBackend
from coco_pipe.decoding.foundation_models.validation import DEFAULT_CHANNELS
from coco_pipe.decoding.registry import get_estimator_spec

pytest.importorskip("braindecode")
torch = pytest.importorskip("torch")

BACKBONES = ["bendr", "biot", "cbramod", "eegpt", "labram", "luna", "signaljepa"]
# Settings a checkpoint's config would otherwise provide.
CHECKPOINT_DEFAULTS = {
    "LaBraM": {"n_times": 3000},
    "InterpolatedLaBraM": {"n_times": 3000},
    "BIOT": {"n_times": 2000, "sfreq": 200.0},
}


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Build models from their classes; return the kwargs each load received."""
    import braindecode.models as bm
    import huggingface_hub
    import mne

    calls = []

    def build(cls, repo_id, **kwargs):
        calls.append(kwargs)
        params = {
            key: value
            for key, value in kwargs.items()
            if key not in {"revision", "filename", "token"}
        }
        for key, value in CHECKPOINT_DEFAULTS.get(cls.__name__, {}).items():
            params.setdefault(key, value)
        return cls(**params)

    classes = ("BENDR", "BIOT", "CBraMod", "LaBraM", "InterpolatedLaBraM", "LUNA")
    for name in (*classes, "SignalJEPA_Contextual"):
        if hasattr(bm, name):
            monkeypatch.setattr(
                getattr(bm, name), "from_pretrained", classmethod(build)
            )

    # EEGPT reads its weights file itself: give it a randomly initialised one.
    info = mne.create_info(list(DEFAULT_CHANNELS), 250.0, "eeg")
    info.set_montage("colin27_1020")
    eegpt = bm.EEGPT(
        chs_info=info["chs"],
        n_chans=len(DEFAULT_CHANNELS),
        n_times=500,
        sfreq=250.0,
        n_outputs=1,
        chan_proj_type="none",
    )
    weights = tmp_path / "eegpt.pt"
    torch.save(eegpt.state_dict(), weights)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: str(weights)
    )
    return calls


def _load(model_key, train_mode="frozen", n_outputs=2, **kwargs):
    spec = get_estimator_spec(model_key)
    if model_key == "labram":
        kwargs.setdefault("interpolate_channels", True)
    if model_key == "luna":
        # LUNA cuts its input into patches of 40 samples: 2.5 s at 256 Hz
        kwargs.setdefault("n_times", 640)
    backend = BrainDecodeBackend.load(
        model_key,
        spec,
        n_outputs=n_outputs,
        device="cpu",
        train_mode=train_mode,
        electrode_names=list(DEFAULT_CHANNELS),
        sfreq=spec.pretrained_sfreq,
        **kwargs,
    )
    backend.signal_metadata_ = SignalMetadata(
        sfreq=spec.pretrained_sfreq, ch_names=list(DEFAULT_CHANNELS)
    )
    n_times = (
        spec.pretrained_n_times
        or kwargs.get("n_times")
        or round(spec.pretrained_sfreq * 2)
    )
    X = np.random.default_rng(0).normal(size=(4, len(DEFAULT_CHANNELS), n_times))
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
    # without groups, the fraction is held out by sample
    backend.fit(np.tile(X, (2, 1, 1)), np.tile(y, 2), max_epochs=1, batch_size=2)
    assert "valid_loss" in backend.training_history()[0]
    assert backend.predict(X).shape == (4,)
    assert backend.predict_proba(X).shape == (4, 2)

    backend.reset_head(3)
    backend.fit(X, np.array([0, 1, 2, 0]), max_epochs=1, validation_fraction=0.0)
    assert backend.predict_proba(X).shape == (4, 3)


def _zscore(X):
    return (X - X.mean(-1, keepdims=True)) / X.std(-1, ddof=1, keepdims=True)


# What each model's authors feed it, from microvolt channels in its own montage.
AUTHORS_INPUT = {
    "cbramod": lambda X: X / 100,
    "labram": lambda X: X / 100,
    "biot": lambda X: X / np.quantile(np.abs(X), 0.95, axis=-1, keepdims=True),
    "luna": _zscore,
    "eegpt": lambda X: (X - X.mean(1, keepdims=True)) / 1000,
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


def test_load_takes_the_latest_checkpoint_unless_given_a_revision(offline):
    _load("cbramod")
    assert "revision" not in offline[-1]
    _load("cbramod", revision="abc123")
    assert offline[-1]["revision"] == "abc123"


def test_labram_interpolates_missing_channels(offline):
    backend, _ = _load("labram")
    adaptation = backend.get_channel_adaptation()
    names = {name.upper() for name in DEFAULT_CHANNELS}
    assert {name.upper() for name in adaptation["source_channels"]} == names
    assert {name.upper() for name in adaptation["direct_channels"]} <= names
    assert adaptation["interpolated_channels"]
    assert adaptation["interpolation_method"]

    plain, _ = _load("cbramod")
    assert not plain.get_channel_adaptation().get("interpolated_channels")


def test_eegpt_refuses_a_partly_loaded_checkpoint(offline, tmp_path, monkeypatch):
    import huggingface_hub

    torch.save({}, tmp_path / "empty.pt")
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: str(tmp_path / "empty.pt")
    )
    with pytest.raises(RuntimeError, match="left weights unloaded"):
        _load("eegpt")


def test_load_errors(offline, monkeypatch):
    spec = get_estimator_spec("cbramod")
    with pytest.raises(NotImplementedError, match="QLoRA"):
        BrainDecodeBackend.load(
            "cbramod", spec, n_outputs=None, device="cpu", train_mode="qlora"
        )
    with pytest.raises(ValueError, match="BrainDecodeBackend does not support"):
        BrainDecodeBackend.load(
            "reve",
            get_estimator_spec("reve"),
            n_outputs=None,
            device="cpu",
            train_mode="frozen",
        )

    import braindecode.models as bm

    monkeypatch.delattr(bm, "InterpolatedLaBraM", raising=False)
    with pytest.raises(ImportError, match="requires InterpolatedLaBraM"):
        _load("labram")

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
        feat_dim=200,
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


def test_token_hook_is_removed_when_forward_fails():
    class _Transformer(torch.nn.Module):
        def forward(self, x):
            return x.transpose(1, 2)

    class _BrokenBIOT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.transformer = _Transformer()

        def forward(self, x, return_features=False):
            self.encoder.transformer(x)
            raise RuntimeError("forward failed")

    metadata = get_estimator_spec("biot")
    model = _BrokenBIOT()
    backend = BrainDecodeBackend(
        metadata=metadata,
        model=model,
        feat_dim=metadata.embedding_dim,
        n_outputs=2,
        device="cpu",
        train_mode="frozen",
        task="classification",
    )
    backend.signal_metadata_ = SignalMetadata(
        sfreq=metadata.pretrained_sfreq, ch_names=[f"ch{i}" for i in range(19)]
    )
    with pytest.raises(RuntimeError, match="forward failed"):
        backend.transform(np.zeros((2, 19, 400), dtype=np.float32), return_tokens=True)
    assert not model.encoder.transformer._forward_hooks
