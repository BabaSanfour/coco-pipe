import importlib.util
import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from coco_pipe.decoding._specs import SignalMetadata
from coco_pipe.decoding.foundation_models._base import BackendBase
from coco_pipe.decoding.foundation_models._braindecode import BrainDecodeBackend
from coco_pipe.decoding.foundation_models._hugging_face import HuggingFaceBackend
from coco_pipe.decoding.foundation_models._loader import load
from coco_pipe.decoding.registry import get_estimator_spec

# ---------------------------------------------------------------------------
# BackendBase ABC
# ---------------------------------------------------------------------------


def test_backend_base_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BackendBase()  # type: ignore[abstract]


def test_configure_peft_raises_not_implemented_by_default():
    """A backend that doesn't override configure_peft raises NotImplementedError."""

    class _Bare(BackendBase):
        def reset_head(self, n): ...
        def get_embedding_info(self):
            return get_estimator_spec("cbramod")

        @classmethod
        def is_available(cls):
            return True

        @classmethod
        def load(cls, *a, **kw):
            return cls()

        def transform(self, X):
            return X

        def predict(self, X):
            return X

    with pytest.raises(NotImplementedError, match="LoRA"):
        _Bare().configure_peft({})


# ---------------------------------------------------------------------------
# BackendBase sklearn contract (via BrainDecodeBackend with mocked model)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_backend_is_sklearn_estimator():
    meta = get_estimator_spec("cbramod")
    adapter = BrainDecodeBackend(
        metadata=meta,
        model=MagicMock(),
        feat_dim=200,
        n_outputs=2,
        device="cpu",
        train_mode="frozen",
        task="classification",
    )
    assert isinstance(adapter, BaseEstimator)
    assert isinstance(adapter, TransformerMixin)
    for attr in ("fit", "transform", "predict", "get_params", "set_params"):
        assert hasattr(adapter, attr)


# ---------------------------------------------------------------------------
# Validation (_validate) — uses BrainDecodeBackend with mocked model
# ---------------------------------------------------------------------------


def _make_bd_adapter(model_key="cbramod", sfreq=200.0):
    import torch

    meta = get_estimator_spec(model_key)
    n_ch = meta.pretrained_n_chans or 19
    feat_dim = meta.embedding_dim
    model = MagicMock()
    model.return_value = {
        "features": torch.zeros(4, feat_dim),
        "logits": torch.zeros(4, 2),
    }
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
        sfreq=sfreq, ch_names=[f"ch{i}" for i in range(n_ch)]
    )
    return adapter, n_ch


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_validate_raises_before_fit():
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
def test_validate_warns_on_sfreq_mismatch():
    adapter, n_ch = _make_bd_adapter("eegpt", sfreq=200.0)
    X = np.zeros((4, n_ch, 400), dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.transform(X)
    assert any("sfreq" in str(w.message) for w in caught)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_validate_raises_on_channel_count_mismatch():
    adapter, _ = _make_bd_adapter("biot")
    with pytest.raises(ValueError, match="16 channels"):
        adapter.transform(np.zeros((2, 32, 400)))


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
def test_signal_metadata_stored_after_transform():
    adapter, n_ch = _make_bd_adapter()
    assert adapter.signal_metadata_ is not None
    assert len(adapter.signal_metadata_.ch_names) == n_ch


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def test_load_fm_unknown_key_raises_key_error():
    with pytest.raises(KeyError, match="Available"):
        load("not_a_model")


def test_load_fm_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unknown backend"):
        load("reve", backend="nonexistent")


def test_load_fm_auto_raises_import_error_when_no_backend_available(monkeypatch):
    monkeypatch.setattr(
        BrainDecodeBackend, "is_available", classmethod(lambda cls: False)
    )
    monkeypatch.setattr(
        HuggingFaceBackend, "is_available", classmethod(lambda cls: False)
    )
    with pytest.raises(ImportError, match="pip install"):
        load("reve", backend="auto")
