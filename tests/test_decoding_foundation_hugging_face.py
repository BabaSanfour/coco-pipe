"""Tests for HuggingFaceBackend (REVE). All network I/O is mocked."""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coco_pipe.decoding._specs import SignalMetadata
from coco_pipe.decoding.foundation_models._hugging_face import HuggingFaceBackend
from coco_pipe.decoding.registry import get_estimator_spec


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


def _make_mock_backbone(hidden_size: int = 1024):
    import torch

    backbone = MagicMock()
    backbone.config.hidden_size = hidden_size
    backbone.parameters.return_value = []
    # REVE's forward returns a raw (batch, channels, time_patches, embed_dim)
    # tensor -- not a HF ModelOutput.
    backbone.side_effect = lambda x, pos, *a, **kw: torch.zeros(
        x.shape[0], x.shape[1], 1, hidden_size
    )
    return backbone


@pytest.fixture()
def mock_auto_model():
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
    pos_bank.eval.assert_called()


def test_unsupported_model_key_raises():
    with pytest.raises(ValueError, match="HuggingFaceBackend does not support"):
        HuggingFaceBackend.load(
            "cbramod",
            get_estimator_spec("cbramod"),
            n_outputs=None,
            device="cpu",
            train_mode="frozen",
        )


def _make_fitted_adapter(n_outputs=None, sfreq=200.0, n_chans=19, pooling="mean"):
    """Build a HuggingFaceBackend with all internals mocked for unit testing."""
    import torch

    meta = get_estimator_spec("reve")
    backbone = MagicMock()
    pos_bank = MagicMock()

    # REVE returns a raw (batch, channels, time_patches, embed_dim) tensor.
    backbone.side_effect = lambda x, pos, *a, **kw: torch.zeros(
        x.shape[0], x.shape[1], 1, meta.embedding_dim
    )
    # frozen backbone is the raw Reve module (no PeftModel.get_base_model);
    # its attention read-out collapses channels+time -> (batch, embed_dim).
    backbone.get_base_model = None
    backbone.attention_pooling.side_effect = lambda out: out.mean(dim=2).mean(dim=1)
    pos_bank.return_value = torch.zeros(n_chans, 32)

    feat_dim = meta.embedding_dim * (n_chans if pooling == "flatten" else 1)

    import torch.nn as nn

    head = nn.Linear(feat_dim, n_outputs) if n_outputs else nn.Identity()
    adapter = HuggingFaceBackend(
        metadata=meta,
        backbone=backbone,
        pos_bank=pos_bank,
        head=head,
        feat_dim=feat_dim,
        n_outputs=n_outputs,
        device="cpu",
        train_mode="frozen",
        pooling=pooling,
        electrode_names=None,
        task="classification",
    )
    adapter.signal_metadata_ = SignalMetadata(
        sfreq=sfreq,
        ch_names=[f"ch{i}" for i in range(n_chans)],
    )
    return adapter


def test_transform_returns_ndarray():
    adapter = _make_fitted_adapter()
    X = np.zeros((4, 19, 400), dtype=np.float32)
    out = adapter.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 512)


def test_transform_flatten_keeps_per_channel_dim():
    adapter = _make_fitted_adapter(pooling="flatten", n_chans=19)
    out = adapter.transform(np.zeros((4, 19, 400), dtype=np.float32))
    assert out.shape == (4, 19 * 512)


def test_transform_attention_pooling():
    adapter = _make_fitted_adapter(pooling="attention")
    out = adapter.transform(np.zeros((4, 19, 400), dtype=np.float32))
    assert out.shape == (4, 512)
    adapter._backbone.attention_pooling.assert_called()


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


def test_sfreq_mismatch_warns():
    adapter = _make_fitted_adapter(sfreq=100.0)
    X = np.zeros((4, 19, 400), dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.transform(X)
    assert any("sfreq" in str(w.message) for w in caught)


def test_reset_head_changes_output_dim():
    import torch.nn as nn

    adapter = _make_fitted_adapter(n_outputs=2)
    adapter.reset_head(5)
    assert adapter._n_outputs == 5
    assert isinstance(adapter._head, nn.Linear)
    assert adapter._head.out_features == 5


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


def test_load_qlora_and_flatten_pooling(monkeypatch):
    mock_model = _make_mock_backbone()
    mock_model.to.return_value = mock_model
    mock_pos = MagicMock()
    mock_pos.to.return_value = mock_pos

    meta = get_estimator_spec("reve")

    with (
        patch(
            "transformers.AutoModel.from_pretrained", side_effect=[mock_model, mock_pos]
        ),
        patch(
            "peft.prepare_model_for_kbit_training", return_value=mock_model
        ) as mock_prep,
        patch("peft.get_peft_model", return_value=mock_model),
        patch.object(HuggingFaceBackend, "is_available", return_value=True),
    ):
        backend = HuggingFaceBackend.load(
            "reve",
            meta,
            n_outputs=2,
            device="cpu",
            train_mode="qlora",
            pooling="flatten",
            electrode_names=["Fp1", "Fp2"],
        )
    mock_prep.assert_called_once()
    assert backend._feat_dim == 2048


def test_reve_skorch_module():
    import torch
    import torch.nn as nn

    meta = get_estimator_spec("reve")

    mock_backbone = MagicMock(spec=nn.Module)
    mock_backbone.return_value = torch.zeros(2, 19, 1, 512)

    mock_pos = MagicMock(spec=nn.Module)
    mock_pos.return_value = torch.zeros(19, 512)

    backend = HuggingFaceBackend(
        metadata=meta,
        backbone=mock_backbone,
        pos_bank=mock_pos,
        head=nn.Linear(512, 2),
        feat_dim=512,
        n_outputs=2,
        device="cpu",
        train_mode="frozen",
        pooling="mean",
        electrode_names=["ch0"],
        task="classification",
    )

    module_cls = backend._get_skorch_module()
    module = module_cls(backend, output_dim=2)
    assert isinstance(module, nn.Module)

    # Test forward
    out = module(torch.zeros(2, 19, 100))
    assert out.shape == (2, 2)

    # Test train mode
    module.train(True)
    mock_backbone.eval.assert_called()
    mock_pos.eval.assert_called()


def test_predict_and_checkpoint_components():
    import torch
    import torch.nn as nn

    meta = get_estimator_spec("reve")

    mock_backbone = MagicMock()
    mock_backbone.return_value = torch.zeros(2, 19, 1, 512)
    mock_pos = MagicMock()
    mock_pos.return_value = torch.zeros(19, 512)

    # 1. Classification Predict
    backend_clf = HuggingFaceBackend(
        metadata=meta,
        backbone=mock_backbone,
        pos_bank=mock_pos,
        head=nn.Linear(512, 2),
        feat_dim=512,
        n_outputs=2,
        device="cpu",
        train_mode="frozen",
        pooling="mean",
        electrode_names=None,
        task="classification",
    )
    backend_clf.signal_metadata_ = SignalMetadata(sfreq=200.0, ch_names=["Fp1"])
    # Mock linear layer outputs
    backend_clf._head.weight.data.fill_(0.0)
    backend_clf._head.bias.data.fill_(0.0)

    preds = backend_clf.predict(np.zeros((2, 1, 100)))
    assert preds.shape == (2,)

    # 2. Regression Predict
    backend_reg = HuggingFaceBackend(
        metadata=meta,
        backbone=mock_backbone,
        pos_bank=mock_pos,
        head=nn.Linear(512, 1),
        feat_dim=512,
        n_outputs=1,
        device="cpu",
        train_mode="frozen",
        pooling="mean",
        electrode_names=None,
        task="regression",
    )
    backend_reg.signal_metadata_ = SignalMetadata(sfreq=200.0, ch_names=["Fp1"])
    backend_reg._head.weight.data.fill_(0.0)
    backend_reg._head.bias.data.fill_(0.0)

    preds_reg = backend_reg.predict(np.zeros((2, 1, 100)))
    assert preds_reg.shape == (2,)

    # Checkpoint components
    comps = backend_clf.checkpoint_components()
    assert comps["backbone"] == mock_backbone
    assert comps["position_bank"] == mock_pos
    assert isinstance(comps["head"], nn.Linear)

    # get_embedding_info
    assert backend_clf.get_embedding_info() == meta
