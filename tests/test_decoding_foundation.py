import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from coco_pipe.decoding._specs import SignalMetadata
from coco_pipe.decoding.foundation_models._base import BackendBase
from coco_pipe.decoding.foundation_models._braindecode import BrainDecodeBackend
from coco_pipe.decoding.foundation_models._hugging_face import HuggingFaceBackend
from coco_pipe.decoding.foundation_models._loader import load
from coco_pipe.decoding.registry import get_estimator_spec


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


def test_validate_warns_on_sfreq_mismatch():
    adapter, n_ch = _make_bd_adapter("eegpt", sfreq=200.0)
    X = np.zeros((4, n_ch, 400), dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.transform(X)
    assert any("sfreq" in str(w.message) for w in caught)


def test_validate_raises_on_channel_count_mismatch():
    adapter, _ = _make_bd_adapter("biot")
    with pytest.raises(ValueError, match="16 channels"):
        adapter.transform(np.zeros((2, 32, 400)))


def test_signal_metadata_stored_after_transform():
    adapter, n_ch = _make_bd_adapter()
    assert adapter.signal_metadata_ is not None
    assert len(adapter.signal_metadata_.ch_names) == n_ch


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


def test_register_backend_errors():
    from coco_pipe.decoding.foundation_models._loader import (
        register_backend,
    )

    with pytest.raises(KeyError, match="already registered"):
        register_backend("hugging_face", HuggingFaceBackend)

    class FakeBackend:
        pass

    with pytest.raises(TypeError, match="must inherit BackendBase"):
        register_backend("fake", FakeBackend)  # type: ignore[arg-type]


def test_unregister_backend_errors():
    from coco_pipe.decoding.foundation_models._loader import unregister_backend

    with pytest.raises(ValueError, match="cannot be unregistered"):
        unregister_backend("hugging_face")


def test_resolve_device_override():
    from coco_pipe.decoding.foundation_models._loader import _resolve_device

    assert _resolve_device("cuda") == "cuda"
    assert _resolve_device("cpu") == "cpu"
    # Test auto device resolution under different environments
    with patch("torch.cuda.is_available", return_value=True):
        assert _resolve_device("auto") == "cuda"


def test_resolve_hf_token(monkeypatch):
    from coco_pipe.decoding.foundation_models.validation import _resolve_hf_token

    # Test env variable resolution
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert _resolve_hf_token(None) == "env-token"

    # Test direct arg override
    assert _resolve_hf_token("direct-token") == "direct-token"


@patch("coco_pipe.decoding.foundation_models.validation.FoundationEmbeddingExtractor")
def test_validate_real_checkpoints(mock_extractor_cls):
    from coco_pipe.decoding.foundation_models.validation import (
        validate_real_checkpoints,
    )

    # Mock extractor instance and output
    mock_extractor = MagicMock()
    mock_extractor_cls.return_value = mock_extractor

    mock_result = MagicMock()
    mock_result.window_embeddings = MagicMock(shape=(1, 256))
    mock_result.metadata = {"channel_adaptation": "none"}
    mock_extractor.extract.return_value = mock_result

    records = validate_real_checkpoints(model_keys=["cbramod"], token="dummy-token")
    assert len(records) == 1
    assert records[0]["model_key"] == "cbramod"
    assert records[0]["status"] in ("verified", "dimension_mismatch")


@patch("coco_pipe.decoding.foundation_models.validation.FoundationClassifier")
def test_validate_real_training(mock_classifier_cls):
    from coco_pipe.decoding.foundation_models.validation import validate_real_training

    mock_classifier = MagicMock()
    mock_classifier_cls.return_value = mock_classifier
    mock_classifier.fit.return_value = mock_classifier
    mock_classifier.predict_proba.return_value = np.zeros((2, 2))

    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_classifier.checkpoint_path_ = mock_path

    mock_classifier.get_training_history.return_value = [1]

    mock_classifier._checkpoint_components.return_value = {}

    records = validate_real_training(
        model_keys=["cbramod"],
        train_modes=["linear_probe"],
        token="dummy-token",
    )
    assert len(records) == 1
    assert records[0]["model_key"] == "cbramod"
    assert records[0]["status"] == "verified"


@patch("coco_pipe.decoding.foundation_models.validation.validate_real_checkpoints")
@patch("coco_pipe.decoding.foundation_models.validation.validate_real_training")
def test_validation_main(mock_training, mock_checkpoints):
    import tempfile

    from coco_pipe.decoding.foundation_models.validation import main

    mock_checkpoints.return_value = [{"model_key": "cbramod", "status": "verified"}]
    mock_training.return_value = [{"model_key": "cbramod", "status": "verified"}]

    with patch("sys.argv", ["validation.py", "--models", "cbramod"]):
        main()

    with patch("sys.argv", ["validation.py", "--models", "cbramod", "--training"]):
        main()

    # Output file and failed exit code path
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "out.json"
        mock_checkpoints.return_value = [{"model_key": "cbramod", "status": "failed"}]
        with patch(
            "sys.argv",
            ["validation.py", "--models", "cbramod", "--output", str(out_file)],
        ):
            with pytest.raises(SystemExit):
                main()
        assert out_file.exists()


def test_validation_edge_cases():
    from coco_pipe.decoding.foundation_models.validation import (
        _resolve_hf_token,
        validate_real_checkpoints,
    )

    # 1. get_token raises exception
    with patch("huggingface_hub.get_token", side_effect=Exception("error")):
        assert _resolve_hf_token(None) is None

    # 2. validate_real_checkpoints requires auth
    # REVE spec requires auth
    with patch(
        "coco_pipe.decoding.foundation_models.validation._resolve_hf_token",
        return_value=None,
    ):
        records = validate_real_checkpoints(model_keys=["reve"], token=None)
        assert records[0]["status"] == "authentication_required"

    # 3. validate_real_checkpoints with labram (triggers interpolate_channels)
    with patch(
        "coco_pipe.decoding.foundation_models.validation.FoundationEmbeddingExtractor"
    ) as mock_extractor_cls:
        mock_extractor = MagicMock()
        mock_extractor_cls.return_value = mock_extractor
        mock_result = MagicMock()
        mock_result.window_embeddings = MagicMock(shape=(1, 200))
        mock_result.metadata = {"channel_adaptation": "none"}
        mock_extractor.extract.return_value = mock_result

        records = validate_real_checkpoints(model_keys=["labram"], token="dummy-token")
        assert len(records) == 1

        # 4. validate_real_checkpoints raises exception
        mock_extractor_cls.side_effect = Exception("extractor failed")
        records = validate_real_checkpoints(model_keys=["cbramod"], token="dummy-token")
        assert records[0]["status"] == "failed"
