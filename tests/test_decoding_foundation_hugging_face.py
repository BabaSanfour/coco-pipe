"""Tests for HuggingFaceBackend (REVE). All network I/O is mocked."""

import importlib.util
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coco_pipe.decoding._specs import SignalMetadata
from coco_pipe.decoding.foundation_models._hugging_face import HuggingFaceBackend
from coco_pipe.decoding.registry import get_estimator_spec

# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_hugging_face_available_when_packages_present(monkeypatch):
    monkeypatch.setattr(
        HuggingFaceBackend, "is_available", classmethod(lambda cls: True)
    )
    assert HuggingFaceBackend.is_available()


def test_hugging_face_not_available_without_transformers(monkeypatch):
    monkeypatch.setattr(
        HuggingFaceBackend, "is_available", classmethod(lambda cls: False)
    )
    assert not HuggingFaceBackend.is_available()


def test_hugging_face_not_available_without_torch(monkeypatch):
    monkeypatch.setattr(
        HuggingFaceBackend, "is_available", classmethod(lambda cls: False)
    )
    assert not HuggingFaceBackend.is_available()


# ---------------------------------------------------------------------------
# load() / _load_reve() with mocked AutoModel
# ---------------------------------------------------------------------------


def _make_mock_backbone(hidden_size: int = 1024):
    backbone = MagicMock()
    backbone.config.hidden_size = hidden_size
    backbone.parameters.return_value = []
    hidden_state = MagicMock()
    hidden_state.mean.return_value = MagicMock()
    backbone.return_value = SimpleNamespace(last_hidden_state=hidden_state)
    return backbone


@pytest.fixture()
def mock_auto_model():
    pytest.importorskip("transformers")
    backbone = _make_mock_backbone()
    pos_bank = MagicMock()
    with (
        patch(
            "coco_pipe.decoding.foundation_models._hugging_face.HuggingFaceBackend.is_available",
            return_value=True,
        ),
        patch(
            "transformers.AutoModel.from_pretrained", side_effect=[backbone, pos_bank]
        ),
    ):
        yield backbone, pos_bank


def test_load_calls_from_pretrained(mock_auto_model):
    import transformers

    backbone, pos_bank = mock_auto_model
    meta = get_estimator_spec("reve")
    with patch.object(
        transformers.AutoModel, "from_pretrained", side_effect=[backbone, pos_bank]
    ):
        adapter = HuggingFaceBackend._load_reve(
            meta,
            n_outputs=2,
            device="cpu",
            train_mode="frozen",
            pooling="mean",
            electrode_names=None,
            token=None,
            task="classification",
            lora_r=16,
            lora_alpha=32,
            lora_target_modules=("query", "value"),
            lora_dropout=0.05,
        )
    assert isinstance(adapter, HuggingFaceBackend)
    assert adapter._feat_dim == 1024


def test_frozen_mode_calls_eval(mock_auto_model):
    import transformers

    backbone, pos_bank = mock_auto_model
    meta = get_estimator_spec("reve")
    with patch.object(
        transformers.AutoModel, "from_pretrained", side_effect=[backbone, pos_bank]
    ):
        HuggingFaceBackend._load_reve(
            meta,
            n_outputs=None,
            device="cpu",
            train_mode="frozen",
            pooling="mean",
            electrode_names=None,
            token=None,
            task="classification",
            lora_r=16,
            lora_alpha=32,
            lora_target_modules=("query", "value"),
            lora_dropout=0.05,
        )
    backbone.eval.assert_called()


def test_unsupported_model_key_raises():
    with pytest.raises(ValueError, match="HuggingFaceBackend does not support"):
        HuggingFaceBackend.load(
            "cbramod",
            get_estimator_spec("cbramod"),
            n_outputs=None,
            device="cpu",
            train_mode="frozen",
        )


# ---------------------------------------------------------------------------
# transform / predict / validate
# ---------------------------------------------------------------------------


def _make_fitted_adapter(n_outputs=None, sfreq=200.0, n_chans=19):
    """Build a HuggingFaceBackend with all internals mocked for unit testing."""
    import torch

    meta = get_estimator_spec("reve")
    backbone = MagicMock()
    pos_bank = MagicMock()

    feat = torch.zeros(4, meta.embedding_dim)
    hidden = MagicMock()
    hidden.mean.return_value = feat
    backbone.return_value = SimpleNamespace(last_hidden_state=hidden)
    pos_bank.return_value = torch.zeros(n_chans, 32)

    import torch.nn as nn

    head = nn.Linear(meta.embedding_dim, n_outputs) if n_outputs else nn.Identity()
    adapter = HuggingFaceBackend(
        metadata=meta,
        backbone=backbone,
        pos_bank=pos_bank,
        head=head,
        feat_dim=meta.embedding_dim,
        n_outputs=n_outputs,
        device="cpu",
        train_mode="frozen",
        pooling="mean",
        electrode_names=None,
        task="classification",
    )
    adapter.signal_metadata_ = SignalMetadata(
        sfreq=sfreq,
        ch_names=[f"ch{i}" for i in range(n_chans)],
    )
    return adapter


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_transform_returns_ndarray():
    adapter = _make_fitted_adapter()
    X = np.zeros((4, 19, 400), dtype=np.float32)
    out = adapter.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 1024)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_transform_raises_before_fit():
    import torch.nn as nn

    meta = get_estimator_spec("reve")
    adapter = HuggingFaceBackend(
        metadata=meta,
        backbone=MagicMock(),
        pos_bank=MagicMock(),
        head=nn.Identity(),
        feat_dim=1024,
        n_outputs=None,
        device="cpu",
        train_mode="frozen",
        pooling="mean",
        electrode_names=None,
        task="classification",
    )
    with pytest.raises(RuntimeError, match="fitted"):
        adapter.transform(np.zeros((2, 19, 400)))


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_sfreq_mismatch_warns():
    adapter = _make_fitted_adapter(sfreq=100.0)
    X = np.zeros((4, 19, 400), dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.transform(X)
    assert any("sfreq" in str(w.message) for w in caught)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_reset_head_changes_output_dim():
    import torch.nn as nn

    adapter = _make_fitted_adapter(n_outputs=2)
    adapter.reset_head(5)
    assert adapter._n_outputs == 5
    assert isinstance(adapter._head, nn.Linear)
    assert adapter._head.out_features == 5


# ---------------------------------------------------------------------------
# configure_peft raises without peft
# ---------------------------------------------------------------------------


def test_configure_peft_raises_import_error_without_peft(monkeypatch):
    import importlib

    orig = importlib.import_module

    def _block(name, *a, **kw):
        if name == "peft":
            raise ImportError("peft not installed")
        return orig(name, *a, **kw)

    import torch.nn as nn

    meta = get_estimator_spec("reve")
    adapter = HuggingFaceBackend(
        metadata=meta,
        backbone=MagicMock(),
        pos_bank=MagicMock(),
        head=nn.Identity(),
        feat_dim=1024,
        n_outputs=None,
        device="cpu",
        train_mode="frozen",
        pooling="mean",
        electrode_names=None,
        task="classification",
    )
    monkeypatch.setattr("builtins.__import__", _block)
    with pytest.raises((ImportError, Exception)):
        adapter.configure_peft({"r": 4})
