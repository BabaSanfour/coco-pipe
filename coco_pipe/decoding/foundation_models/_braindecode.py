"""
BrainDecode Backend
===================

Backend that loads pretrained EEG/MEG models from the braindecode library
via ``Model.from_pretrained()``. Supports CBraMod, BIOT, LaBraM, EEGPT,
SignalJEPA, BENDR, CodeBrain, and LUNA.

To add a new braindecode model: add one entry to ``_BD_MODEL_MAP`` and one
``FoundationModelSpec`` entry in ``coco_pipe/decoding/_specs.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import warnings
from typing import TYPE_CHECKING

import numpy as np

from .._specs import SignalMetadata
from ._base import BackendBase

if TYPE_CHECKING:
    from .._specs import FoundationModelSpec

_BD_MODEL_MAP: dict[str, tuple[str, str]] = {
    "cbramod": ("CBraMod", "braindecode.models"),
    "biot": ("BIOT", "braindecode.models"),
    "labram": ("Labram", "braindecode.models"),
    "eegpt": ("EEGPT", "braindecode.models"),
    "signaljepa": ("SignalJEPA_Contextual", "braindecode.models"),
    "bendr": ("BENDR", "braindecode.models"),
    "codebrain": ("CodeBrain", "braindecode.models"),
    "luna": ("LUNA", "braindecode.models"),
}


class BrainDecodeBackend(BackendBase):
    """Backend that delegates to braindecode.models.<Model>.from_pretrained().

    Covers CBraMod, BIOT, LaBraM, EEGPT, SignalJEPA, BENDR, CodeBrain, LUNA.

    Training modes: frozen, full, lora.
    QLoRA is not supported (requires HF quantization in from_pretrained).

    BrainDecode's dict output is unwrapped in ``_probe_feat_dim`` and
    ``transform`` — no dict leaks outside those methods.
    """

    _uses_interpolation: bool = False

    def __init__(
        self,
        metadata: FoundationModelSpec,
        model,
        feat_dim: int,
        n_outputs: int | None,
        device: str,
        train_mode: str,
        task: str,
    ) -> None:
        self._metadata = metadata
        self._model = model
        self._feat_dim = feat_dim
        self._n_outputs = n_outputs
        self._device = device
        self._train_mode = train_mode
        self._task = task
        self.signal_metadata_: SignalMetadata | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Check whether braindecode and torch are installed.

        Returns
        -------
        available : bool
            ``True`` if both ``braindecode`` and ``torch`` are importable.
        """
        return (
            importlib.util.find_spec("braindecode") is not None
            and importlib.util.find_spec("torch") is not None
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_key: str,
        metadata: FoundationModelSpec,
        n_outputs: int | None,
        device: str,
        train_mode: str,
        task: str = "classification",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: str | list[str] = "all-linear",
        lora_dropout: float = 0.05,
        **kw,
    ) -> "BrainDecodeBackend":
        """Load a pretrained braindecode model and return an initialised backend.

        Parameters
        ----------
        model_key : str
            Registry key for the model (e.g. ``"cbramod"``). Must be present
            in ``_BD_MODEL_MAP``.
        metadata : FoundationModelSpec
            Registry spec providing ``hub_repo`` and signal metadata.
        n_outputs : int or None
            Number of output units. ``None`` = headless feature extractor.
        device : str
            Resolved device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        train_mode : str
            One of ``"frozen"``, ``"full"``, ``"lora"``. ``"qlora"`` raises
            ``NotImplementedError``.
        task : str, default ``"classification"``
            ``"classification"`` or ``"regression"``.
        lora_r : int, default 16
            LoRA rank. Ignored unless ``train_mode="lora"``.
        lora_alpha : int, default 32
            LoRA scaling factor.
        lora_target_modules : str or list of str, default ``"all-linear"``
            Which modules to inject LoRA adapters into.
        lora_dropout : float, default 0.05
            Dropout probability inside LoRA adapters.
        **kw
            Extra keyword arguments forwarded to ``Model.from_pretrained``.

        Returns
        -------
        backend : BrainDecodeBackend
            A fully initialised backend instance.

        Raises
        ------
        ImportError
            If braindecode or torch is not installed.
        ValueError
            If ``model_key`` is not in ``_BD_MODEL_MAP``.
        NotImplementedError
            If ``train_mode="qlora"``.
        """
        if not cls.is_available():
            raise ImportError(
                f"Model '{model_key}' requires braindecode and torch.\n"
                'To install: pip install "coco-pipe[braindecode]"'
            )
        if model_key not in _BD_MODEL_MAP:
            raise ValueError(
                f"BrainDecodeBackend does not support '{model_key}'. "
                f"Supported: {sorted(_BD_MODEL_MAP)}"
            )
        if train_mode == "qlora":
            raise NotImplementedError(
                "QLoRA for BrainDecodeBackend requires HuggingFace quantization "
                "support which is not available via braindecode.from_pretrained. "
                "Use train_mode='lora' instead, or switch to hugging_face (REVE)."
            )

        class_name, module_path = _BD_MODEL_MAP[model_key]
        models_module = importlib.import_module(module_path)
        electrode_names = kw.pop("electrode_names", None)
        kw.pop("pooling", None)
        sfreq = float(kw.pop("sfreq", metadata.pretrained_sfreq))
        n_times = int(
            metadata.pretrained_n_times or kw.pop("n_times", round(sfreq * 2))
        )
        kw.pop("n_times", None)
        revision = kw.pop("revision", metadata.checkpoint_revision)
        filename = kw.pop("filename", metadata.checkpoint_filename)
        interpolate_channels = bool(kw.pop("interpolate_channels", False))
        _interpolated_class_map = {
            "labram": ("InterpolatedLaBraM", "InterpolatedLabram"),
            "biot": ("InterpolatedBIOT",),
            "bendr": ("InterpolatedBENDR",),
            "signaljepa": ("InterpolatedSignalJEPA",),
        }
        if interpolate_channels and model_key in _interpolated_class_map:
            for candidate in _interpolated_class_map[model_key]:
                if hasattr(models_module, candidate):
                    class_name = candidate
                    break
            else:
                raise ImportError(
                    f"interpolate_channels=True requires an Interpolated variant "
                    f"for {model_key} (braindecode>=1.5), which is not available."
                )
        model_cls = getattr(models_module, class_name)

        if electrode_names and model_key in {"cbramod", "labram", "luna", "biot", "bendr", "signaljepa"}:
            import mne

            if not interpolate_channels:
                kw.setdefault("n_chans", len(electrode_names))
            kw.setdefault("sfreq", sfreq)
            info = mne.create_info(electrode_names, sfreq=sfreq, ch_types="eeg")
            try:
                info.set_montage("standard_1020", on_missing="warn")
            except Exception as exc:
                warnings.warn(
                    f"Could not assign the standard_1020 montage for "
                    f"{model_key} ({type(exc).__name__}: {exc}); channel "
                    "positions are unset, which can degrade montage-dependent "
                    "embeddings (e.g. LaBraM interpolation).",
                    stacklevel=2,
                )
            kw.setdefault("chs_info", info["chs"])
        if model_key == "luna":
            kw.setdefault("n_times", n_times)
            kw.setdefault("embed_dim", 64)
            kw.setdefault("num_queries", 4)
            kw.setdefault("depth", 8)
        elif model_key == "cbramod":
            kw.setdefault("n_times", n_times)
        elif model_key == "labram":
            # The pretrained temporal embedding has a checkpoint-native width.
            # LaBraM adjusts it dynamically during forward, but overriding
            # n_times while loading makes the state dict shape incompatible.
            kw.pop("n_times", None)
        elif model_key == "signaljepa":
            # braindecode/signaljepa-pretrained was saved with 19-channel
            # scratch mode (pos_encoder_spat shape [19,30]).  Use scratch so
            # the checkpoint loads without shape mismatch.  strict=False lets
            # the head weights be skipped.  NaN was from lr=1e-3 exploding
            # during LoRA training — fixed by lowering lr to 1e-4.
            kw.setdefault("strict", False)
            kw.setdefault("channel_embedding", "scratch")
            kw.setdefault("n_times", n_times)
            kw.setdefault("input_window_seconds", n_times / sfreq)
        elif model_key == "bendr":
            # BENDR config.json stores n_chans_pretrained / chan_proj_max_norm
            # which are not accepted by BENDR.__init__.  Wrap so that the hub
            # mixin injects them into **kwargs but they get stripped before
            # reaching the base class.
            _bendr_extra = {"n_chans_pretrained", "chan_proj_max_norm", "activation"}
            _orig_cls = model_cls

            class _PatchedInterpolatedBENDR(_orig_cls):
                def __init__(self, *args, **kwargs):
                    for _k in _bendr_extra:
                        kwargs.pop(_k, None)
                    super().__init__(*args, **kwargs)

            model_cls = _PatchedInterpolatedBENDR
            kw.setdefault("strict", False)

        load_kwargs = dict(kw)
        if revision is not None:
            load_kwargs["revision"] = revision
        if filename is not None:
            load_kwargs["filename"] = filename
        effective_outputs = (
            1 if model_key in {"cbramod", "luna"} and n_outputs is None else n_outputs
        )
        if effective_outputs is not None:
            load_kwargs["n_outputs"] = effective_outputs
        model = model_cls.from_pretrained(metadata.hub_repo, **load_kwargs)
        model = model.to(device)

        if model_key == "signaljepa":
            # SignalJEPA uses nn.Transformer internally: encoder has 8
            # TransformerEncoderLayer blocks, each containing nn.MultiheadAttention
            # (d_model=64, nhead=8 → d_k=8 per head).
            #
            # ROOT CAUSE: attention logit = (x@W_Q_i)·(x@W_K_j)/sqrt(8).
            # After LoRA training, W_Q/W_K norms can grow so that logits exceed
            # ~89, causing exp() overflow → NaN in softmax → NaN propagates
            # through LayerNorm (LayerNorm(NaN) = NaN) → entire residual stream
            # becomes NaN → all subsequent layers output NaN → NaN predictions.
            # The _NaNGradientFilter prevents weight corruption during training
            # but does NOT stop NaN appearing in forward passes once weights have
            # drifted (including at inference time).
            #
            # PREVIOUS HOOK PROBLEM: the old _qk_maxnorm_hook registered on
            # nn.MultiheadAttention pre-hook received args=(query, key, value)
            # where query/key are the full d_model=64 residual-stream vectors,
            # NOT the per-head Q/K projections. A LayerNorm output has norm
            # exactly sqrt(64)=8.0, so clipping to max-norm=5.0 shrank every
            # single forward pass by factor 5/8=0.625 — degrading valid inputs
            # while not reliably preventing overflow (weight norms were unconstrained).
            #
            # FIX — two defense-in-depth hooks on TransformerEncoderLayer:
            #
            # 1. PRE-HOOK: clamp the residual-stream input to norm ≤ 30.
            #    Normal LayerNorm output has norm = sqrt(64) ≈ 8.  Threshold 30
            #    is 3.75× normal so it never clips healthy activations but stops
            #    actual explosions before they reach attention or FFN.
            #
            # 2. POST-HOOK: replace any surviving NaN/Inf in the output with 0.
            #    This is the final safety net — if something slips through
            #    (e.g., an un-normed path or a bad data sample), the NaN dies
            #    at the layer boundary instead of infecting all downstream layers.
            #
            # Both hooks fire at every forward call (training and inference) and
            # are no-ops for healthy activations, so they don't change model
            # behaviour on clean inputs.
            import torch as _torch
            import torch.nn as _nn

            _RESIDUAL_CLAMP = 30.0  # >> sqrt(64)=8, << actual explosion values

            def _residual_clamp_pre_hook(module, args):
                # args[0] is `src`: the residual-stream token tensor (B, T, d_model)
                x = args[0]
                norms = x.norm(dim=-1, keepdim=True)
                scale = (_RESIDUAL_CLAMP / norms).clamp(max=1.0)
                return (x * scale,) + args[1:]

            def _nanfix_post_hook(module, inp, output):
                if isinstance(output, _torch.Tensor) and (
                    _torch.isnan(output).any() or _torch.isinf(output).any()
                ):
                    return _torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
                return output

            for _m in model.modules():
                if isinstance(_m, _nn.TransformerEncoderLayer):
                    _m.register_forward_pre_hook(_residual_clamp_pre_hook)
                    _m.register_forward_hook(_nanfix_post_hook)

        if train_mode == "frozen":
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        elif train_mode == "lora":
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)
        # full: no changes — all params trainable

        feat_dim = cls._probe_feat_dim(
            model,
            metadata,
            device,
            n_chans=len(electrode_names) if electrode_names else None,
            n_times=n_times,
        )

        if n_outputs is not None:
            try:
                model.reset_head(n_outputs)
            except NotImplementedError:
                if model_key not in {"luna"}:
                    raise
        if train_mode == "frozen" and n_outputs is not None:
            for head_name in ("final_layer", "classifier", "head"):
                head = getattr(model, head_name, None)
                if head is not None:
                    for parameter in head.parameters():
                        parameter.requires_grad = True
                    break

        backend = cls(
            metadata=metadata,
            model=model,
            feat_dim=feat_dim,
            n_outputs=n_outputs,
            device=device,
            train_mode=train_mode,
            task=task,
        )
        if electrode_names:
            backend._expected_n_chans = len(electrode_names)
        backend._uses_interpolation = interpolate_channels and model_key in {"labram", "biot", "bendr", "signaljepa"}
        backend._checkpoint_revision = revision
        backend._checkpoint_filename = filename
        return backend

    @classmethod
    def _probe_feat_dim(
        cls,
        model,
        metadata: FoundationModelSpec,
        device: str,
        *,
        n_chans: int | None = None,
        n_times: int = 400,
    ) -> int:
        """Zero forward pass to get actual feature dim.

        Preferred over hard-coding — registry value is reference, probe is truth.
        """
        import torch

        n_ch = n_chans or metadata.pretrained_n_chans or 19
        probe = torch.zeros(1, n_ch, n_times, device=device)
        with torch.no_grad():
            feats = cls._forward_features(model, probe, metadata.name)
        return int(feats.shape[-1])

    @staticmethod
    def _forward_features(model, tensor, model_key: str):
        if model_key != "luna":
            out = model(tensor, return_features=True)
            return out["features"] if isinstance(out, dict) else out
        captured = {}

        def _capture(_module, _inputs, output):
            captured["latent"] = output

        hook = model.norm.register_forward_hook(_capture)
        try:
            model(tensor)
        finally:
            hook.remove()
        if "latent" not in captured:
            raise RuntimeError("LUNA latent representation was not captured.")
        return captured["latent"]

    def _get_skorch_module(self):
        import torch.nn as nn

        class _BDSklearnModule(nn.Module):
            def __init__(self, backend, output_dim: int) -> None:
                super().__init__()
                self._backend = backend
                self._output_dim = output_dim
                self.model = backend._model

            def forward(self, X):
                out = self.model(X)
                return out["logits"] if isinstance(out, dict) else out

            def train(self, mode: bool = True):
                super().train(mode)
                if mode and self._backend._train_mode == "frozen":
                    self._backend._set_backbone_eval()
                return self

        return _BDSklearnModule

    def _set_backbone_eval(self) -> None:
        self._model.eval()
        for module in self._model.modules():
            if any(
                parameter.requires_grad
                for parameter in module.parameters(recurse=False)
            ):
                module.train()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract backbone embeddings without running the classification head.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_channels, n_times)
            EEG/MEG epochs.

        Returns
        -------
        embeddings : np.ndarray of shape (n_samples, embedding_dim)
            Backbone feature vectors.
        """
        self._validate(X)
        with self._no_grad():
            feats = self._forward_features(
                self._model,
                self._to_tensor(X),
                self._metadata.name,
            )
        return self._from_tensor(feats)

    def checkpoint_components(self) -> dict:
        """BrainDecode keeps one model (LoRA-wrapped when applicable)."""
        return {"model": self._model}

    def get_channel_adaptation(self) -> dict:
        """Describe direct and interpolated channels from the real layer."""
        meta = self.signal_metadata_
        source = list(meta.ch_names) if meta is not None else []
        if not self._uses_interpolation:
            return {
                "source_channels": source,
                "target_channels": source,
                "interpolated_channels": [],
                "zero_filled_channels": [],
                "dropped_channels": [],
            }
        layer = self._model.interpolation_layer
        target = [str(item["ch_name"]) for item in layer.tgt_chs_info]
        source_lookup = {name.casefold() for name in source}
        mode = getattr(layer, "mode", "always")
        direct = (
            [name for name in target if name.casefold() in source_lookup]
            if mode == "name_match"
            else []
        )
        return {
            "source_channels": source,
            "target_channels": target,
            "direct_channels": direct,
            "interpolated_channels": [name for name in target if name not in direct],
            "zero_filled_channels": [],
            "dropped_channels": [],
            "interpolation_method": getattr(layer, "method", None),
            "interpolation_mode": mode,
            "interpolation_matrix_shape": list(layer.matrix.shape),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run a full forward pass and return predictions.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_channels, n_times)
            EEG/MEG epochs.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class indices (classification) or continuous values
            (regression).
        """
        self._validate(X)
        batch_size = 32
        all_logits = []
        with self._no_grad():
            for i in range(0, len(X), batch_size):
                batch = self._to_tensor(X[i : i + batch_size])
                out = self._model(batch)
                logits = out["logits"] if isinstance(out, dict) else out
                all_logits.append(logits)
        logits = __import__("torch").cat(all_logits, dim=0)
        if getattr(self, "_task", "classification") == "regression":
            return self._from_tensor(logits).squeeze(-1)
        return self._from_tensor(logits.argmax(dim=-1))

    def reset_head(self, n_outputs: int) -> "BrainDecodeBackend":
        """Replace the classification head without modifying backbone weights.

        Parameters
        ----------
        n_outputs : int
            Number of output units for the new head.

        Returns
        -------
        self : BrainDecodeBackend
            The backend with the updated head.
        """
        self._model.reset_head(n_outputs)
        self._n_outputs = n_outputs
        return self

    def configure_peft(self, lora_config: dict) -> "BrainDecodeBackend":
        """Apply LoRA adapters to the backbone post-hoc.

        Parameters
        ----------
        lora_config : dict
            Keyword arguments forwarded to ``peft.LoraConfig``.

        Returns
        -------
        self : BrainDecodeBackend
            The backend with LoRA adapters injected and ``train_mode`` set
            to ``"lora"``.
        """
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(**lora_config)
        self._model = get_peft_model(self._model, cfg)
        self._train_mode = "lora"
        return self

    def get_embedding_info(self) -> FoundationModelSpec:
        """Return the registry spec for this model instance.

        Returns
        -------
        spec : FoundationModelSpec
            The spec describing this model's provenance and signal expectations.
        """
        return self._metadata
