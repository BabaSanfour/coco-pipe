"""Continued pretraining of real braindecode models with random weights, and
of a small REVE-shaped model, without the network (see tests/fixtures/hub.py)."""

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from coco_pipe.decoding.foundation_models import (
    FoundationClassifier,
    FrozenBackboneTransformer,
    continue_pretraining,
)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
reve = pytest.importorskip("braindecode.models.reve")
FourierEmb4D, TransformerBackbone = reve.FourierEmb4D, reve.TransformerBackbone
# LUNA's features are in braindecode main, not in 2.0
LUNA_FEATURES = (
    "return_features"
    in inspect.signature(
        __import__("braindecode.models", fromlist=["LUNA"]).LUNA.forward
    ).parameters
)

CHANNELS = ["Fp1", "Fp2", "Cz", "O1"]


class TinyReve(torch.nn.Module):
    """REVE at toy size, from braindecode's layers: one token per channel and
    overlapping time patch."""

    embed_dim, patch_size, overlap_size = 16, 200, 20
    config = SimpleNamespace(hidden_size=16)

    def __init__(self):
        super().__init__()
        self.to_patch_embedding = torch.nn.Linear(self.patch_size, self.embed_dim)
        self.fourier4d = FourierEmb4D(self.embed_dim, freqs=2)
        self.mlp4d = torch.nn.Sequential(
            torch.nn.Linear(4, self.embed_dim, bias=False),
            torch.nn.GELU(),
            torch.nn.LayerNorm(self.embed_dim),
        )
        self.ln = torch.nn.LayerNorm(self.embed_dim)
        self.transformer = TransformerBackbone(self.embed_dim, 1, 2, 8, 32, True)

    def forward(self, eeg, pos):
        step = self.patch_size - self.overlap_size
        patches = eeg.unfold(2, self.patch_size, step)
        n_windows, n_channels, n_patches, _ = patches.shape
        pos = FourierEmb4D.add_time_patch(pos, n_patches)
        pos_embed = self.ln(self.fourier4d(pos) + self.mlp4d(pos))
        tokens = self.to_patch_embedding(patches).flatten(1, 2) + pos_embed
        return self.transformer(tokens).view(n_windows, n_channels, n_patches, -1)


class TinyPositionBank(torch.nn.Module):
    """REVE's position bank: the 3D position of each channel, by name."""

    def __init__(self):
        super().__init__()
        self.register_buffer("positions", torch.rand(len(CHANNELS), 3))

    def forward(self, names):
        return self.positions[[CHANNELS.index(name) for name in names]]


@pytest.fixture
def reve_hub(monkeypatch):
    """Serve REVE and its position bank from toy models."""

    def from_pretrained(repo_id, **kwargs):
        if repo_id == "brain-bzh/reve-positions":
            return TinyPositionBank()
        return TinyReve()

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", from_pretrained)


@pytest.mark.parametrize(
    "model_key",
    [
        "cbramod",
        pytest.param(
            "luna",
            marks=pytest.mark.skipif(
                not LUNA_FEATURES, reason="luna needs braindecode main"
            ),
        ),
        "reve",
    ],
)
def test_continued_weights_reach_embedding_and_fine_tuning(
    offline, reve_hub, tmp_path, model_key
):
    # 10 s windows at 200 Hz, a 10 Hz rhythm in noise
    rng = np.random.default_rng(0)
    t = np.arange(2000) / 200
    X = 20 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 5, (12, len(CHANNELS), 2000))
    info = continue_pretraining(
        model_key,
        X.astype(np.float32),
        CHANNELS,
        200.0,
        output_dir=tmp_path,
        epochs=2,
        batch_size=4,
        held_out=0.25,
        groups=np.repeat(np.arange(4), 3),
        device="cpu",
    )

    assert info == json.loads((tmp_path / "pretraining.json").read_text())
    assert info["epochs"] == 2 and info["n_windows"] == {"train": 9, "held_out": 3}
    assert np.isfinite(info["train_loss"] + info["held_out_loss"]).all()
    saved = torch.load(tmp_path / "encoder.pt", weights_only=True)
    kwargs = {"sfreq": 200.0, "ch_names": CHANNELS, "device": "cpu"}
    backend_kwargs = {"checkpoint_dir": str(tmp_path)}
    frozen = FrozenBackboneTransformer(
        model_key, backend_kwargs=backend_kwargs, **kwargs
    )
    frozen.fit(X)
    loaded = {
        key: value
        for component in frozen.backend_.checkpoint_components().values()
        for key, value in component.state_dict().items()
    }
    shared = saved.keys() & loaded.keys()
    assert len(shared) > len(saved) / 2
    for key in shared:
        torch.testing.assert_close(loaded[key], saved[key])
    assert str(tmp_path) in frozen.backend_.checkpoint_revisions
    assert np.isfinite(frozen.transform(X)).all()

    model = FoundationClassifier(
        model_key,
        train_mode="full",
        trainer={"max_epochs": 1, "validation_fraction": 0},
        backend_kwargs=backend_kwargs,
        **kwargs,
    ).fit(X, np.tile([0, 1], 6))
    assert str(tmp_path) in model.get_model_card_info()["checkpoint_revisions"]
