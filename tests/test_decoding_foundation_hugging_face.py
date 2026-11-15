"""Tests for HuggingFaceBackend (REVE) on a toy model with REVE's interface.

Only the Hugging Face Hub is replaced: ``AutoModel.from_pretrained`` returns a
small local model shaped like REVE and its position bank, and each repository
resolves to a commit of its own without the network.
"""

import hashlib
import sys

import numpy as np
import pytest

from coco_pipe.decoding._specs import SignalMetadata
from coco_pipe.decoding.foundation_models._hugging_face import HuggingFaceBackend
from coco_pipe.decoding.registry import get_estimator_spec

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

SPEC = get_estimator_spec("reve")
POSITIONS = "brain-bzh/reve-positions"
CHANNELS = ["Fp1", "Fp2", "Cz", "Oz"]


def _latest(repo_id):
    return hashlib.sha1(repo_id.encode()).hexdigest()


class TinyReveConfig(transformers.PretrainedConfig):
    model_type = "tiny-reve"

    def __init__(self, hidden_size=8, patch_size=50, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.patch_size = patch_size


class TinyReve(transformers.PreTrainedModel):
    """REVE at toy size: one token per channel and time patch."""

    config_class = TinyReveConfig

    def __init__(self, config):
        super().__init__(config)
        self.to_patch_embedding = torch.nn.Linear(config.patch_size, config.hidden_size)
        self.to_position = torch.nn.Linear(3, config.hidden_size)
        self.cls_query_token = torch.nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.post_init()

    def _init_weights(self, module):
        """Keep torch's random initialisation of each layer."""

    def forward(self, eeg, pos):
        size = self.config.patch_size
        patches = eeg.float().unfold(2, size, size)
        return self.to_patch_embedding(patches) + self.to_position(pos)[:, :, None]

    def attention_pooling(self, x):
        # REVE's single-query read-out over all channel and time-patch tokens
        b, c, s, e = x.shape
        x = x.reshape(b, c * s, e)
        scores = self.cls_query_token.expand(b, -1, -1) @ x.transpose(-1, -2)
        return (torch.softmax(scores / e**0.5, dim=-1) @ x).squeeze(1)


class TinyPositionBank(torch.nn.Module):
    """REVE's position bank: the 3D position of each channel, by name."""

    def __init__(self, names):
        super().__init__()
        self.mapping = {name: index for index, name in enumerate(names)}
        generator = torch.Generator().manual_seed(0)
        self.register_buffer(
            "embedding", torch.randn(len(names), 3, generator=generator)
        )

    def forward(self, channel_names):
        return self.embedding[[self.mapping[name] for name in channel_names]]


@pytest.fixture
def hub(monkeypatch):
    """Serve REVE's two repositories from toy models; return each load."""
    import huggingface_hub

    loads = []

    def from_pretrained(repo_id, **kwargs):
        loads.append((repo_id, {k: v for k, v in kwargs.items() if k != "token"}))
        if repo_id == POSITIONS:
            return TinyPositionBank(CHANNELS)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)  # the checkpoint's weights
            model = TinyReve(TinyReveConfig())
        if "quantization_config" in kwargs:
            model.is_loaded_in_4bit = True  # as transformers marks it
        return model

    def download(repo_id, filename, revision=None, **kwargs):
        # a cached snapshot folder is named by its commit
        return f"/snapshots/{revision or _latest(repo_id)}/{filename}"

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", from_pretrained)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return loads


def _load(train_mode="frozen", n_outputs=2, **kwargs):
    backend = HuggingFaceBackend.load(
        "reve",
        SPEC,
        n_outputs=n_outputs,
        device="cpu",
        train_mode=train_mode,
        electrode_names=CHANNELS,
        **kwargs,
    )
    backend.signal_metadata_ = SignalMetadata(
        sfreq=SPEC.pretrained_sfreq, ch_names=CHANNELS
    )
    return backend


def _windows(n_windows=3):
    rng = np.random.default_rng(0)
    return rng.normal(scale=30.0, size=(n_windows, len(CHANNELS), 400))


def _trainable(backend):
    return {
        f"{name}.{key}"
        for name, component in backend.checkpoint_components().items()
        for key, parameter in component.named_parameters()
        if parameter.requires_grad
    }


def _state(module):
    return {key: value.clone() for key, value in module.state_dict().items()}


def test_hugging_face_needs_transformers(monkeypatch):
    assert HuggingFaceBackend.is_available()
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert not HuggingFaceBackend.is_available()


def test_each_repository_loads_the_commit_resolved_for_it(hub):
    # the backbone loads the spec's pinned commit, the position bank its own pin
    bank = "befa5b57a455b77cf302daf610c2e9ed8140bace"
    backend = _load()
    assert all(kwargs["trust_remote_code"] for _, kwargs in hub)
    assert (
        {repo: kwargs["revision"] for repo, kwargs in hub}
        == backend.checkpoint_revisions
        == {SPEC.hub_repo: SPEC.checkpoint_revision, POSITIONS: bank}
    )
    # a given revision replaces the backbone's pin; None follows the Hub's latest
    pinned = _load(revision="abc123")
    assert pinned.checkpoint_revisions == {SPEC.hub_repo: "abc123", POSITIONS: bank}
    assert pinned.get_embedding_info() == SPEC
    latest = _load(revision=None)
    assert latest.checkpoint_revisions[SPEC.hub_repo] == _latest(SPEC.hub_repo)


def test_train_modes_choose_what_trains(hub):
    frozen = _load("frozen")
    assert _trainable(frozen) == {"head.weight", "head.bias"}
    components = frozen.checkpoint_components()
    assert set(components) == {"backbone", "position_bank", "head"}
    assert not components["backbone"].training
    assert not components["position_bank"].training

    backbone = {
        f"backbone.{key}" for key, _ in TinyReve(TinyReveConfig()).named_parameters()
    }
    assert _trainable(_load("full")) == backbone | {"head.weight", "head.bias"}

    # REVE names its layers to_qkv, to_out...: every linear layer gets an adapter
    lora = {name for name in _trainable(_load("lora")) if name.startswith("backbone.")}
    assert lora and all("lora_" in name for name in lora)
    assert {name.split(".")[3] for name in lora} == {
        "to_patch_embedding",
        "to_position",
    }


def test_qlora_quantizes_the_backbone_without_gradient_checkpointing(hub):
    # the toy model, like REVE, refuses gradient checkpointing: peft only
    # loads it when coco-pipe turns it off
    backend = _load("qlora")
    (_, backbone_kwargs), (_, bank_kwargs) = hub
    config = backbone_kwargs["quantization_config"]
    assert config.load_in_4bit
    assert config.bnb_4bit_quant_type == "nf4"
    assert "quantization_config" not in bank_kwargs
    trained = {name for name in _trainable(backend) if name.startswith("backbone.")}
    assert trained and all("lora_" in name for name in trained)


def test_random_weights_rebuild_the_backbone_and_head_from_the_seed(hub):
    def weights(global_seed=0, **kwargs):
        if global_seed is not None:
            torch.manual_seed(global_seed)
        components = _load(**kwargs).checkpoint_components()
        return {
            f"{name}.{key}": value
            for name in ("backbone", "head")
            for key, value in components[name].state_dict().items()
        }

    torch.manual_seed(0)
    state = torch.random.get_rng_state()
    random = weights(global_seed=None, pretrained=False, seed=0)
    assert torch.equal(state, torch.random.get_rng_state())
    assert not any({"pretrained", "seed"} & set(kwargs) for _, kwargs in hub)
    pretrained = weights()
    again = weights(global_seed=1, pretrained=False, seed=0)
    other = weights(pretrained=False, seed=1)
    for key, value in random.items():
        assert torch.equal(value, again[key])
        assert not torch.equal(value, other[key])
        if key.startswith("backbone."):
            assert not torch.equal(value, pretrained[key])
    assert torch.equal(weights()["head.weight"], pretrained["head.weight"])
    assert not torch.equal(
        weights(global_seed=1)["head.weight"], pretrained["head.weight"]
    )


@pytest.mark.parametrize(
    "weights, message",
    [
        ({"pretrained": False}, "need an integer seed"),
        ({"seed": 0}, "seed is for random weights"),
        ({"pretrained": False, "seed": 0, "train_mode": "qlora"}, "QLoRA quantizes"),
    ],
)
def test_random_weights_refused(weights, message):
    with pytest.raises(ValueError, match=message):
        HuggingFaceBackend.load(
            "reve",
            SPEC,
            n_outputs=None,
            device="cpu",
            **{"train_mode": "frozen", **weights},
        )


def test_load_refuses_other_models_and_pooling():
    with pytest.raises(ValueError, match="HuggingFaceBackend does not support"):
        HuggingFaceBackend.load(
            "cbramod",
            get_estimator_spec("cbramod"),
            n_outputs=None,
            device="cpu",
            train_mode="frozen",
        )
    with pytest.raises(ValueError, match="pooling='mean' or 'attention'"):
        HuggingFaceBackend.load(
            "reve",
            SPEC,
            n_outputs=None,
            device="cpu",
            train_mode="frozen",
            pooling="flatten",
        )


@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_transform_pools_the_tokens_of_one_forward_pass(hub, pooling):
    backend = _load(pooling=pooling)
    X = _windows()

    pooled, tokens = backend.transform(X, return_tokens=True)

    # REVE's native output: window, channel, time patch, feature
    assert tokens.shape == (3, len(CHANNELS), 8, 8)
    assert backend.get_token_output_metadata()["token_axes"] == [
        "window",
        "channel",
        "time_patch",
        "feature",
    ]
    if pooling == "mean":
        expected = tokens.mean(axis=(1, 2))
    else:
        backbone = backend.checkpoint_components()["backbone"]
        expected = backbone.attention_pooling(torch.from_numpy(tokens)).detach()
    np.testing.assert_allclose(pooled, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(backend.transform(X), pooled, rtol=1e-6)
    np.testing.assert_allclose(backend.pool_tokens(tokens, pooling), pooled, rtol=1e-6)
    with pytest.raises(ValueError, match="pooling='mean' or 'attention'"):
        backend.pool_tokens(tokens, "max")


def test_reve_input_depends_on_whether_the_backbone_is_trained(hub):
    X = _windows()
    X[0, 0, 0] = 5000.0  # an artefact, far above 15 standard deviations
    seen = []

    def record(module, args):
        seen.append(args[0].numpy().copy())

    # frozen backbone: each channel z-scored and clipped, as in pretraining
    frozen = _load("frozen")
    frozen.checkpoint_components()["backbone"].register_forward_pre_hook(record)
    frozen.transform(X)
    assert seen[-1].max() == pytest.approx(15.0)
    np.testing.assert_allclose(seen[-1][1].std(axis=-1, ddof=1), 1.0, rtol=1e-4)

    # trained backbone: units of 100 microvolts, as in the authors' fine-tunes
    full = _load("full")
    full.checkpoint_components()["backbone"].register_forward_pre_hook(record)
    full.predict(X)
    np.testing.assert_allclose(seen[-1], X / 100, rtol=1e-6)


def test_transform_checks_its_input(hub):
    backend = _load()
    backend.signal_metadata_ = None
    with pytest.raises(RuntimeError, match="fitted"):
        backend.transform(_windows())

    backend.signal_metadata_ = SignalMetadata(sfreq=100.0, ch_names=CHANNELS)
    with pytest.warns(UserWarning, match="sfreq"):
        backend.transform(_windows())


def test_predict_reads_the_head_on_the_pooled_embedding(hub):
    X = _windows(6)

    def head_output(backend):
        head = backend.checkpoint_components()["head"]
        return head(torch.from_numpy(backend.transform(X))).detach().numpy()

    classifier = _load(n_outputs=3)
    np.testing.assert_array_equal(
        classifier.predict(X), head_output(classifier).argmax(axis=1)
    )
    regressor = _load(n_outputs=1, task="regression")
    np.testing.assert_allclose(regressor.predict(X), head_output(regressor)[:, 0])


def test_reset_head_keeps_the_backbone(hub):
    backend = _load(n_outputs=2)
    X = _windows()
    embeddings = backend.transform(X)
    backbone = _state(backend.checkpoint_components()["backbone"])

    backend.reset_head(5)

    assert backend.checkpoint_components()["head"].out_features == 5
    np.testing.assert_array_equal(backend.transform(X), embeddings)
    after = backend.checkpoint_components()["backbone"].state_dict()
    assert all(torch.equal(value, after[key]) for key, value in backbone.items())


def test_frozen_fit_trains_the_head_alone(hub):
    X = _windows(8)
    y = np.repeat([0, 1], 4)

    def fit(train_mode):
        torch.manual_seed(0)
        backend = _load(train_mode)
        components = backend.checkpoint_components()
        before = {name: _state(components[name]) for name in ("backbone", "head")}
        backend.fit(X, y, max_epochs=2, batch_size=4, lr=0.1, validation_fraction=0.0)
        changed = {
            name: any(
                not torch.equal(value, components[name].state_dict()[key])
                for key, value in before[name].items()
            )
            for name in before
        }
        return backend, changed

    frozen, changed = fit("frozen")
    assert changed == {"backbone": False, "head": True}
    # training leaves the frozen backbone and the position bank in eval mode
    assert not frozen.checkpoint_components()["backbone"].training
    assert not frozen.checkpoint_components()["position_bank"].training
    assert frozen.predict_proba(X).shape == (8, 2)

    _, changed = fit("full")
    assert changed == {"backbone": True, "head": True}


def test_configure_peft_adds_adapters_to_a_loaded_backbone(hub):
    backend = _load("frozen")
    backend.configure_peft({"r": 2, "target_modules": "all-linear"})
    trained = {name for name in _trainable(backend) if name.startswith("backbone.")}
    assert trained and all("lora_" in name for name in trained)
    assert backend.transform(_windows()).shape == (3, 8)
