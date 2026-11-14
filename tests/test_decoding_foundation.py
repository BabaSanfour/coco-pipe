import json
import sys

import numpy as np
import pytest

from coco_pipe.decoding._specs import SignalMetadata
from coco_pipe.decoding.foundation_models import validation
from coco_pipe.decoding.foundation_models._base import BackendBase
from coco_pipe.decoding.foundation_models._hugging_face import HuggingFaceBackend
from coco_pipe.decoding.foundation_models._loader import (
    load,
    register_backend,
    unregister_backend,
)
from coco_pipe.decoding.foundation_models.validation import DEFAULT_CHANNELS
from coco_pipe.decoding.registry import get_estimator_spec
from tests.fixtures.hub import LATEST


def test_backend_base_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BackendBase()  # type: ignore[abstract]


def test_configure_peft_raises_not_implemented_by_default():
    """A backend that doesn't override configure_peft raises NotImplementedError."""

    class _Bare(BackendBase):
        def reset_head(self, n):
            return None

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


def test_loaded_model_takes_the_channels_it_was_loaded_for(offline):
    channels = SignalMetadata(sfreq=200.0, ch_names=list(DEFAULT_CHANNELS))
    backend = load(
        "labram",
        device="cpu",
        signal_metadata=channels,
        electrode_names=list(DEFAULT_CHANNELS),
        sfreq=200.0,
    )
    X = np.random.default_rng(0).normal(scale=20.0, size=(2, 19, 400))

    assert backend.transform(X).shape == (2, get_estimator_spec("labram").embedding_dim)
    with pytest.raises(ValueError, match="expects 19 channels, got 18"):
        backend.transform(X[:, :18])


def test_load_fm_unknown_key_raises_key_error():
    with pytest.raises(KeyError, match="Available"):
        load("not_a_model")


def test_load_fm_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unknown backend"):
        load("reve", backend="nonexistent")


def test_load_fm_raises_import_error_when_the_backend_is_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="pip install"):
        load("reve", backend="auto")


def test_register_backend_errors():
    with pytest.raises(KeyError, match="already registered"):
        register_backend("hugging_face", HuggingFaceBackend)

    class FakeBackend:
        pass

    with pytest.raises(TypeError, match="must inherit BackendBase"):
        register_backend("fake", FakeBackend)  # type: ignore[arg-type]


def test_unregister_backend_errors():
    with pytest.raises(ValueError, match="cannot be unregistered"):
        unregister_backend("hugging_face")


def test_resolve_device(monkeypatch):
    torch = pytest.importorskip("torch")
    from coco_pipe.utils import resolve_device

    assert resolve_device("cpu") == "cpu"
    # no GPU in CI: CUDA is reported present or absent
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


def test_validate_real_checkpoints_embeds_different_windows(offline):
    spec = get_estimator_spec("cbramod")

    (record,) = validation.validate_real_checkpoints(["cbramod"], token="hf-token")

    assert record["status"] == "verified"
    assert record["observed_embedding_dim"] == spec.embedding_dim
    assert record["relative_spread"] > validation.DEGENERATE_SPREAD
    assert record["checkpoint_revisions"] == {spec.hub_repo: LATEST}
    assert record["channel_adaptation"]["source_channels"] == DEFAULT_CHANNELS
    assert offline[-1]["token"] == "hf-token"


def test_validate_real_training_fits_and_reloads_a_checkpoint(offline):
    linear_probe, qlora = validation.validate_real_training(
        ["cbramod"], train_modes=["linear_probe", "qlora"]
    )

    assert linear_probe["status"] == "verified"
    assert linear_probe["checkpoint_reload"]
    assert linear_probe["probability_shape"] == [2, 2]
    assert linear_probe["training_history_rows"] == 1
    assert linear_probe["trainable_parameters"] > 0
    # braindecode models do not quantize
    assert qlora == {
        "model_key": "cbramod",
        "train_mode": "qlora",
        "status": "unsupported",
    }


def test_validation_cli_writes_the_records_and_fails_on_a_failed_model(
    offline, monkeypatch, tmp_path, capsys
):
    import braindecode.models as bm

    def run(*args):
        output = tmp_path / "records.json"
        monkeypatch.setattr(sys, "argv", ["validation", *args, "--output", str(output)])
        validation.main()
        records = json.loads(output.read_text())
        assert json.loads(capsys.readouterr().out) == records
        return records

    (record,) = run("--models", "cbramod")
    assert record["status"] == "verified"
    (record,) = run("--models", "cbramod", "--training", "--train-modes", "qlora")
    assert record["status"] == "unsupported"

    def unreachable(cls, repo_id, **kwargs):
        raise OSError("the Hub cannot be reached")

    monkeypatch.setattr(bm.CBraMod, "from_pretrained", classmethod(unreachable))
    with pytest.raises(SystemExit) as exit_:
        run("--models", "cbramod")
    assert exit_.value.code == 1
    (record,) = json.loads((tmp_path / "records.json").read_text())
    assert record["status"] == "failed"
    assert record["reason"] == "OSError: the Hub cannot be reached"
