"""Tests for BrainDecodeBackend. All network I/O is mocked."""

import importlib.util
import os
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coco_pipe.decoding._specs import SignalMetadata
from coco_pipe.decoding.foundation_models._braindecode import (
    _BD_MODEL_MAP,
    BrainDecodeBackend,
)
from coco_pipe.decoding.registry import get_estimator_spec

# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_braindecode_available_when_packages_present(monkeypatch):
    monkeypatch.setattr(
        BrainDecodeBackend, "is_available", classmethod(lambda cls: True)
    )
    assert BrainDecodeBackend.is_available()


def test_braindecode_not_available_without_braindecode(monkeypatch):
    monkeypatch.setattr(
        BrainDecodeBackend, "is_available", classmethod(lambda cls: False)
    )
    assert not BrainDecodeBackend.is_available()


# ---------------------------------------------------------------------------
# Model map completeness
# ---------------------------------------------------------------------------


def test_bd_model_map_covers_expected_models():
    expected = {"cbramod", "biot", "labram", "eegpt", "signaljepa", "bendr"}
    assert expected <= set(_BD_MODEL_MAP)


def test_qlora_raises_not_implemented_for_braindecode(monkeypatch):
    monkeypatch.setattr(
        BrainDecodeBackend, "is_available", classmethod(lambda cls: True)
    )
    with pytest.raises(NotImplementedError, match="QLoRA"):
        BrainDecodeBackend.load(
            "cbramod",
            get_estimator_spec("cbramod"),
            n_outputs=None,
            device="cpu",
            train_mode="qlora",
        )


def test_unsupported_model_key_raises():
    with patch.object(BrainDecodeBackend, "is_available", return_value=True):
        with pytest.raises(ValueError, match="BrainDecodeBackend does not support"):
            BrainDecodeBackend.load(
                "reve",
                get_estimator_spec("reve"),
                n_outputs=None,
                device="cpu",
                train_mode="frozen",
            )


# ---------------------------------------------------------------------------
# load() with mocked braindecode
# ---------------------------------------------------------------------------


def _make_mock_bd_model(feat_dim: int = 200, n_outputs: int = 2):
    """Return a MagicMock that mimics a BrainDecode EEGModuleMixin model."""
    import torch

    model = MagicMock()
    model.to.return_value = model  # nn.Module.to() returns self
    model.parameters.return_value = []
    feats = torch.zeros(1, feat_dim)
    model.return_value = {"features": feats, "logits": torch.zeros(1, n_outputs)}
    model.side_effect = None
    return model


@pytest.fixture()
def mock_bd_model():
    return _make_mock_bd_model()


def _load_with_mock(model_key: str, mock_model, train_mode: str = "frozen"):
    meta = get_estimator_spec(model_key)
    class_name, module_path = _BD_MODEL_MAP[model_key]

    import importlib

    orig = importlib.import_module

    def _fake_import(name, *a, **kw):
        if name == module_path:
            mod = MagicMock()
            setattr(
                mod,
                class_name,
                MagicMock(from_pretrained=MagicMock(return_value=mock_model)),
            )
            return mod
        return orig(name, *a, **kw)

    with (
        patch("importlib.import_module", side_effect=_fake_import),
        patch.object(BrainDecodeBackend, "is_available", return_value=True),
        patch.object(BrainDecodeBackend, "_probe_feat_dim", return_value=200),
    ):
        return BrainDecodeBackend.load(
            model_key,
            meta,
            n_outputs=2,
            device="cpu",
            train_mode=train_mode,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_load_cbramod_returns_braindecode_backend(mock_bd_model):
    adapter = _load_with_mock("cbramod", mock_bd_model)
    assert isinstance(adapter, BrainDecodeBackend)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_frozen_mode_calls_eval(mock_bd_model):
    _load_with_mock("cbramod", mock_bd_model, train_mode="frozen")
    mock_bd_model.eval.assert_called()


# ---------------------------------------------------------------------------
# transform / predict / validate
# ---------------------------------------------------------------------------


def _make_fitted_adapter(model_key: str = "cbramod", sfreq: float = 200.0):
    import torch

    meta = get_estimator_spec(model_key)
    n_ch = meta.pretrained_n_chans or 19
    feat_dim = meta.embedding_dim

    model = MagicMock()
    feats = torch.zeros(4, feat_dim)
    logits = torch.zeros(4, 2)
    model.return_value = {"features": feats, "logits": logits}
    model.parameters.return_value = []

    adapter = BrainDecodeBackend(
        metadata=meta,
        model=model,
        feat_dim=feat_dim,
        n_outputs=2,
        device="cpu",
        train_mode="frozen",
        task="classification",
    )
    adapter.signal_metadata_ = SignalMetadata(
        sfreq=sfreq,
        ch_names=[f"ch{i}" for i in range(n_ch)],
    )
    return adapter, model, n_ch


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_transform_returns_ndarray_not_dict():
    adapter, _, n_ch = _make_fitted_adapter()
    out = adapter.transform(np.zeros((4, n_ch, 400), dtype=np.float32))
    assert isinstance(out, np.ndarray)
    assert not isinstance(out, dict)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_transform_shape_matches_embedding_dim():
    adapter, _, n_ch = _make_fitted_adapter()
    meta = get_estimator_spec("cbramod")
    out = adapter.transform(np.zeros((4, n_ch, 400), dtype=np.float32))
    assert out.shape == (4, meta.embedding_dim)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_transform_raises_before_fit():
    meta = get_estimator_spec("cbramod")
    adapter = BrainDecodeBackend(
        metadata=meta,
        model=MagicMock(),
        feat_dim=200,
        n_outputs=None,
        device="cpu",
        train_mode="frozen",
        task="classification",
    )
    with pytest.raises(RuntimeError, match="fitted"):
        adapter.transform(np.zeros((2, 19, 400)))


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_sfreq_mismatch_warns_for_eegpt():
    adapter, _, n_ch = _make_fitted_adapter("eegpt", sfreq=200.0)
    X = np.zeros((4, n_ch, 400), dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.transform(X)
    assert any("sfreq" in str(w.message) for w in caught)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_channel_mismatch_raises_for_labram():
    meta = get_estimator_spec("labram")

    adapter = BrainDecodeBackend(
        metadata=meta,
        model=MagicMock(),
        feat_dim=200,
        n_outputs=None,
        device="cpu",
        train_mode="frozen",
        task="classification",
    )
    adapter.signal_metadata_ = SignalMetadata(
        sfreq=200.0,
        ch_names=[f"ch{i}" for i in range(128)],
    )
    with pytest.raises(ValueError, match="128 channels"):
        adapter.transform(np.zeros((2, 64, 400)))


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_reset_head_delegates_to_bd_model():
    adapter, model, _ = _make_fitted_adapter()
    adapter.reset_head(10)
    model.reset_head.assert_called_once_with(10)
    assert adapter._n_outputs == 10


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_configure_peft_applies_lora(monkeypatch):
    pytest.importorskip("peft")
    """configure_peft() wraps the model with peft LoRA."""
    meta = get_estimator_spec("cbramod")
    mock_model = MagicMock()
    adapter = BrainDecodeBackend(
        metadata=meta,
        model=mock_model,
        feat_dim=200,
        n_outputs=2,
        device="cpu",
        train_mode="frozen",
        task="classification",
    )
    wrapped = MagicMock()
    with patch("peft.get_peft_model", return_value=wrapped) as mock_peft:
        adapter.configure_peft({"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"]})
    mock_peft.assert_called_once()
    assert adapter._model is wrapped
    assert adapter._train_mode == "lora"


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_cbramod_src_directory_does_not_exist():
    assert not os.path.exists(
        "coco_pipe/decoding/foundation_models/cbramod_src"
    ), "cbramod_src/ must be deleted — BrainDecode provides the implementation."
