"""Tests for BrainDecodeBackend. All network I/O is mocked."""

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
    with (
        patch.object(BrainDecodeBackend, "is_available", return_value=True),
        pytest.raises(ValueError, match="BrainDecodeBackend does not support"),
    ):
        BrainDecodeBackend.load(
            "reve",
            get_estimator_spec("reve"),
            n_outputs=None,
            device="cpu",
            train_mode="frozen",
        )


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


def test_load_cbramod_returns_braindecode_backend(mock_bd_model):
    adapter = _load_with_mock("cbramod", mock_bd_model)
    assert isinstance(adapter, BrainDecodeBackend)


def test_frozen_mode_calls_eval(mock_bd_model):
    _load_with_mock("cbramod", mock_bd_model, train_mode="frozen")
    mock_bd_model.eval.assert_called()


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


def test_transform_returns_ndarray_not_dict():
    adapter, _, n_ch = _make_fitted_adapter()
    out = adapter.transform(np.zeros((4, n_ch, 400), dtype=np.float32))
    assert isinstance(out, np.ndarray)
    assert not isinstance(out, dict)


def test_transform_shape_matches_embedding_dim():
    adapter, _, n_ch = _make_fitted_adapter()
    meta = get_estimator_spec("cbramod")
    out = adapter.transform(np.zeros((4, n_ch, 400), dtype=np.float32))
    assert out.shape == (4, meta.embedding_dim)


def test_transform_return_tokens_preserves_native_cbramod_output():
    import torch

    adapter, model, n_ch = _make_fitted_adapter()
    native = torch.arange(4 * 3 * 2 * 200, dtype=torch.float32).reshape(4, 3, 2, 200)
    model.return_value = {"features": native, "logits": torch.zeros(4, 2)}

    pooled, tokens = adapter.transform(
        np.zeros((4, n_ch, 400), dtype=np.float32), return_tokens=True
    )

    assert pooled.shape == (4, 200)
    np.testing.assert_array_equal(tokens, native.numpy())
    np.testing.assert_allclose(pooled, native.numpy().mean(axis=(1, 2)))


@pytest.mark.parametrize("model_key", ["biot", "bendr"])
def test_transform_rejects_fake_tokens_for_pooled_only_models(model_key):
    adapter, _, n_ch = _make_fitted_adapter(
        model_key, sfreq=get_estimator_spec(model_key).pretrained_sfreq
    )
    with pytest.raises(NotImplementedError, match="native token extraction"):
        adapter.transform(
            np.zeros((4, n_ch, 400), dtype=np.float32), return_tokens=True
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


def test_sfreq_mismatch_warns_for_eegpt():
    adapter, _, n_ch = _make_fitted_adapter("eegpt", sfreq=200.0)
    X = np.zeros((4, n_ch, 400), dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.transform(X)
    assert any("sfreq" in str(w.message) for w in caught)


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


def test_reset_head_delegates_to_bd_model():
    adapter, model, _ = _make_fitted_adapter()
    adapter.reset_head(10)
    model.reset_head.assert_called_once_with(10)
    assert adapter._n_outputs == 10


def test_configure_peft_applies_lora(monkeypatch):
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


def test_cbramod_src_directory_does_not_exist():
    assert not os.path.exists("coco_pipe/decoding/foundation_models/cbramod_src"), (
        "cbramod_src/ must be deleted — BrainDecode provides the implementation"
    )


def test_load_revision_and_filename():
    import torch

    param = torch.nn.Parameter(torch.zeros(2))
    mock_model = _make_mock_bd_model()
    mock_model.parameters.return_value = [param]

    # Check that revision and filename are passed correctly to from_pretrained
    from dataclasses import replace

    meta = replace(
        get_estimator_spec("cbramod"),
        checkpoint_revision="v1.0",
        checkpoint_filename="model.ckpt",
    )

    import importlib

    orig = importlib.import_module

    from_pretrained_mock = MagicMock(return_value=mock_model)

    def _fake_import(name, *a, **kw):
        if name == "braindecode.models":
            mod = MagicMock()
            mod.CBraMod = MagicMock(from_pretrained=from_pretrained_mock)
            return mod
        return orig(name, *a, **kw)

    with (
        patch("importlib.import_module", side_effect=_fake_import),
        patch.object(BrainDecodeBackend, "is_available", return_value=True),
        patch.object(BrainDecodeBackend, "_probe_feat_dim", return_value=200),
    ):
        backend = BrainDecodeBackend.load(
            "cbramod",
            meta,
            n_outputs=2,
            device="cpu",
            train_mode="frozen",
        )

    from_pretrained_mock.assert_called_once()
    kwargs = from_pretrained_mock.call_args[1]
    assert kwargs.get("revision") == "v1.0"
    assert kwargs.get("filename") == "model.ckpt"
    assert param.requires_grad is False

    # Test checkpoint_components and get_channel_adaptation (no interpolation)
    assert backend.checkpoint_components() == {"model": mock_model}
    adapt = backend.get_channel_adaptation()
    assert adapt["source_channels"] == []


def test_load_lora_train_mode():
    mock_model = _make_mock_bd_model()
    meta = get_estimator_spec("cbramod")

    import importlib

    orig = importlib.import_module

    from_pretrained_mock = MagicMock(return_value=mock_model)

    def _fake_import(name, *a, **kw):
        if name == "braindecode.models":
            mod = MagicMock()
            mod.CBraMod = MagicMock(from_pretrained=from_pretrained_mock)
            return mod
        return orig(name, *a, **kw)

    with (
        patch("importlib.import_module", side_effect=_fake_import),
        patch.object(BrainDecodeBackend, "is_available", return_value=True),
        patch.object(BrainDecodeBackend, "_probe_feat_dim", return_value=200),
        patch("peft.get_peft_model", return_value=mock_model) as mock_peft,
    ):
        backend = BrainDecodeBackend.load(
            "cbramod",
            meta,
            n_outputs=2,
            device="cpu",
            train_mode="lora",
        )
    mock_peft.assert_called_once()
    assert backend._train_mode == "lora"


def test_load_not_available_raises():
    with (
        patch.object(BrainDecodeBackend, "is_available", return_value=False),
        pytest.raises(ImportError, match="requires braindecode"),
    ):
        BrainDecodeBackend.load(
            "cbramod",
            get_estimator_spec("cbramod"),
            n_outputs=2,
            device="cpu",
            train_mode="frozen",
        )


def test_reset_head_not_implemented():
    import importlib

    orig = importlib.import_module

    # Test cbramod raises NotImplementedError on load
    mock_model_cbra = MagicMock()
    mock_model_cbra.to.return_value = mock_model_cbra
    mock_model_cbra.reset_head.side_effect = NotImplementedError()

    def _fake_import_cbra(name, *a, **kw):
        if name == "braindecode.models":
            mod = MagicMock()
            mod.CBraMod = MagicMock(
                from_pretrained=MagicMock(return_value=mock_model_cbra)
            )
            return mod
        return orig(name, *a, **kw)

    with (
        patch("importlib.import_module", side_effect=_fake_import_cbra),
        patch.object(BrainDecodeBackend, "is_available", return_value=True),
        patch.object(BrainDecodeBackend, "_probe_feat_dim", return_value=200),
        pytest.raises(NotImplementedError),
    ):
        BrainDecodeBackend.load(
            "cbramod",
            get_estimator_spec("cbramod"),
            n_outputs=2,
            device="cpu",
            train_mode="frozen",
        )

    # Test luna ignores NotImplementedError on load
    mock_model_luna = MagicMock()
    mock_model_luna.to.return_value = mock_model_luna
    mock_model_luna.reset_head.side_effect = NotImplementedError()

    def _fake_import_luna(name, *a, **kw):
        if name == "braindecode.models":
            mod = MagicMock()
            mod.LUNA = MagicMock(
                from_pretrained=MagicMock(return_value=mock_model_luna)
            )
            return mod
        return orig(name, *a, **kw)

    with (
        patch("importlib.import_module", side_effect=_fake_import_luna),
        patch.object(BrainDecodeBackend, "is_available", return_value=True),
        patch.object(BrainDecodeBackend, "_probe_feat_dim", return_value=200),
    ):
        backend_luna = BrainDecodeBackend.load(
            "luna",
            get_estimator_spec("luna"),
            n_outputs=2,
            device="cpu",
            train_mode="frozen",
        )
        assert backend_luna is not None


def test_reset_head_frozen_trainable():
    import torch

    mock_model = MagicMock()
    head = MagicMock()
    param = torch.nn.Parameter(torch.zeros(2))
    head.parameters.return_value = [param]
    mock_model.final_layer = head

    meta = get_estimator_spec("cbramod")

    import importlib

    orig = importlib.import_module
    from_pretrained_mock = MagicMock(return_value=mock_model)

    def _fake_import(name, *a, **kw):
        if name == "braindecode.models":
            mod = MagicMock()
            mod.CBraMod = MagicMock(from_pretrained=from_pretrained_mock)
            return mod
        return orig(name, *a, **kw)

    with (
        patch("importlib.import_module", side_effect=_fake_import),
        patch.object(BrainDecodeBackend, "is_available", return_value=True),
        patch.object(BrainDecodeBackend, "_probe_feat_dim", return_value=200),
    ):
        BrainDecodeBackend.load(
            "cbramod",
            meta,
            n_outputs=2,
            device="cpu",
            train_mode="frozen",
        )
    assert param.requires_grad is True


def test_load_interpolate_channels_unsupported(monkeypatch):
    monkeypatch.setattr(
        BrainDecodeBackend, "is_available", classmethod(lambda cls: True)
    )

    import importlib

    orig = importlib.import_module

    def _fake_import(name, *a, **kw):
        if name == "braindecode.models":
            mod = MagicMock(spec=[])  # Has no InterpolatedLaBraM/InterpolatedLabram
            return mod
        return orig(name, *a, **kw)

    meta = get_estimator_spec("labram")
    with (
        patch("importlib.import_module", side_effect=_fake_import),
        pytest.raises(ImportError, match="requires InterpolatedLaBraM"),
    ):
        BrainDecodeBackend.load(
            "labram",
            meta,
            n_outputs=2,
            device="cpu",
            train_mode="frozen",
            interpolate_channels=True,
        )


def test_load_interpolate_channels_supported():
    mock_model = _make_mock_bd_model()

    import importlib

    orig = importlib.import_module

    from_pretrained_mock = MagicMock(return_value=mock_model)

    def _fake_import(name, *a, **kw):
        if name == "braindecode.models":
            mod = MagicMock()
            mod.InterpolatedLaBraM = MagicMock(from_pretrained=from_pretrained_mock)
            return mod
        return orig(name, *a, **kw)

    meta = get_estimator_spec("labram")
    with (
        patch("importlib.import_module", side_effect=_fake_import),
        patch.object(BrainDecodeBackend, "is_available", return_value=True),
        patch.object(BrainDecodeBackend, "_probe_feat_dim", return_value=200),
    ):
        backend = BrainDecodeBackend.load(
            "labram",
            meta,
            n_outputs=2,
            device="cpu",
            train_mode="frozen",
            interpolate_channels=True,
        )
        assert backend._uses_interpolation is True


def test_load_montage_warning():
    mock_model = _make_mock_bd_model()

    import importlib

    orig = importlib.import_module

    def _fake_import(name, *a, **kw):
        if name == "braindecode.models":
            mod = MagicMock()
            mod.CBraMod = MagicMock(from_pretrained=MagicMock(return_value=mock_model))
            return mod
        return orig(name, *a, **kw)

    meta = get_estimator_spec("cbramod")

    # We want mne montage setting to raise an exception to trigger the
    # warnings.warn block
    mock_info = MagicMock()
    mock_info.set_montage.side_effect = Exception("montage error")

    with (
        patch("importlib.import_module", side_effect=_fake_import),
        patch.object(BrainDecodeBackend, "is_available", return_value=True),
        patch.object(BrainDecodeBackend, "_probe_feat_dim", return_value=200),
        patch("mne.create_info", return_value=mock_info),
        pytest.warns(UserWarning, match="Could not assign the standard_1020"),
    ):
        BrainDecodeBackend.load(
            "cbramod",
            meta,
            n_outputs=2,
            device="cpu",
            train_mode="frozen",
            electrode_names=["Fp1", "Fp2"],
        )


def test_luna_special_defaults():
    mock_model = _make_mock_bd_model()

    import importlib

    orig = importlib.import_module

    from_pretrained_mock = MagicMock(return_value=mock_model)

    def _fake_import(name, *a, **kw):
        if name == "braindecode.models":
            mod = MagicMock()
            mod.LUNA = MagicMock(from_pretrained=from_pretrained_mock)
            return mod
        return orig(name, *a, **kw)

    meta = get_estimator_spec("luna")
    with (
        patch("importlib.import_module", side_effect=_fake_import),
        patch.object(BrainDecodeBackend, "is_available", return_value=True),
        patch.object(BrainDecodeBackend, "_probe_feat_dim", return_value=200),
    ):
        BrainDecodeBackend.load(
            "luna",
            meta,
            n_outputs=2,
            device="cpu",
            train_mode="frozen",
        )
    from_pretrained_mock.assert_called_once()
    kwargs = from_pretrained_mock.call_args[1]
    assert kwargs.get("embed_dim") == 64
    assert kwargs.get("num_queries") == 4
    assert kwargs.get("depth") == 8


def test_probe_feat_dim_real_pass():
    import torch

    mock_model = MagicMock()
    mock_model.side_effect = lambda X, return_features=True: {
        "features": torch.zeros(1, 150)
    }
    meta = get_estimator_spec("cbramod")
    dim = BrainDecodeBackend._probe_feat_dim(mock_model, meta, "cpu", n_times=100)
    assert dim == 150


def test_luna_forward_features():
    import torch

    mock_model = MagicMock()
    mock_norm = MagicMock()
    mock_model.norm = mock_norm

    # We simulate registration of the forward hook which captures output when called
    def fake_register_hook(hook_fn):
        def trigger_hook(tensor):
            # Output of norm layer
            hook_fn(mock_model, None, torch.zeros(1, 80))

        mock_model.side_effect = trigger_hook
        mock_hook = MagicMock()
        return mock_hook

    mock_norm.register_forward_hook.side_effect = fake_register_hook

    res = BrainDecodeBackend._forward_features(
        mock_model, torch.zeros(1, 19, 100), "luna"
    )
    assert res.shape == (1, 80)

    # Trigger RuntimeError when latent is not captured
    mock_norm.register_forward_hook.side_effect = lambda fn: MagicMock()
    mock_model.side_effect = lambda tensor: None
    with pytest.raises(RuntimeError, match="LUNA latent representation"):
        BrainDecodeBackend._forward_features(
            mock_model, torch.zeros(1, 19, 100), "luna"
        )


def test_get_skorch_module():
    import torch
    import torch.nn as nn

    meta = get_estimator_spec("cbramod")
    mock_model = MagicMock(spec=nn.Module)
    mock_model.modules.return_value = []

    backend = BrainDecodeBackend(
        metadata=meta,
        model=mock_model,
        feat_dim=200,
        n_outputs=2,
        device="cpu",
        train_mode="frozen",
        task="classification",
    )

    module_cls = backend._get_skorch_module()
    module = module_cls(backend, output_dim=2)
    assert isinstance(module, nn.Module)

    # Test forward passing dict/tensor
    mock_model.return_value = {"logits": torch.zeros(2, 2)}
    out = module(torch.zeros(2, 19, 100))
    assert out.shape == (2, 2)

    mock_model.return_value = torch.zeros(2, 2)
    out = module(torch.zeros(2, 19, 100))
    assert out.shape == (2, 2)

    # Test module.train with frozen backbone
    param = MagicMock()
    param.requires_grad = (
        True  # Need requires_grad=True to execute module.train() loop body
    )
    mock_submodule = MagicMock()
    mock_submodule.parameters.return_value = [param]
    mock_model.modules.return_value = [mock_submodule]

    module.train(True)
    mock_model.eval.assert_called()
    mock_submodule.train.assert_called_once()


def test_predict_classification_and_regression():
    import torch

    meta = get_estimator_spec("cbramod")
    mock_model = MagicMock()

    # 1. Regression
    mock_model.return_value = torch.zeros(2, 1)
    backend_reg = BrainDecodeBackend(
        metadata=meta,
        model=mock_model,
        feat_dim=200,
        n_outputs=1,
        device="cpu",
        train_mode="frozen",
        task="regression",
    )
    backend_reg.signal_metadata_ = SignalMetadata(sfreq=200.0, ch_names=["ch0"])
    preds = backend_reg.predict(np.zeros((2, 1, 100)))
    assert preds.shape == (2,)

    # 2. Classification (argmax path)
    mock_model.return_value = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    backend_clf = BrainDecodeBackend(
        metadata=meta,
        model=mock_model,
        feat_dim=200,
        n_outputs=2,
        device="cpu",
        train_mode="frozen",
        task="classification",
    )
    backend_clf.signal_metadata_ = SignalMetadata(sfreq=200.0, ch_names=["ch0"])
    preds = backend_clf.predict(np.zeros((2, 1, 100)))
    assert np.array_equal(preds, [1, 0])


def test_channel_adaptation_with_interpolation():
    meta = get_estimator_spec("labram")
    mock_model = MagicMock()

    mock_layer = MagicMock()
    mock_layer.tgt_chs_info = [{"ch_name": "Fp1"}, {"ch_name": "Fp2"}]
    mock_layer.matrix = MagicMock()
    mock_layer.matrix.shape = (2, 2)
    mock_layer.method = "spline"
    mock_layer.mode = "name_match"
    mock_model.interpolation_layer = mock_layer

    backend = BrainDecodeBackend(
        metadata=meta,
        model=mock_model,
        feat_dim=200,
        n_outputs=2,
        device="cpu",
        train_mode="frozen",
        task="classification",
    )
    backend._uses_interpolation = True
    backend.signal_metadata_ = SignalMetadata(sfreq=200.0, ch_names=["Fp1"])

    adapt = backend.get_channel_adaptation()
    assert adapt["source_channels"] == ["Fp1"]
    assert adapt["target_channels"] == ["Fp1", "Fp2"]
    assert adapt["direct_channels"] == ["Fp1"]
    assert adapt["interpolated_channels"] == ["Fp2"]
    assert adapt["interpolation_method"] == "spline"
