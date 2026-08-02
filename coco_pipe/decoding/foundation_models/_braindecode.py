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
import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np

from .._specs import SignalMetadata
from . import _montages
from ._base import BackendBase, resolve_auto_lora_params

if TYPE_CHECKING:
    from .._specs import FoundationModelSpec

_logger = logging.getLogger(__name__)

# Per-model recommended LoRA (r, alpha) based on internal attention d_model.
# alpha = 2*r is the standard ratio (scaling = alpha/r = 2).
# r is chosen so that LoRA rank ≈ 5–16 % of the internal attention dimension.
# resolve_auto_lora_params and LORA_AUTO_PARAMS live in _base.py (shared with HF backend)


def _log_lora_injection(model, target_modules, model_key: str) -> None:
    """Log LoRA injection stats and warn loudly if no adapters were added."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # PEFT marks adapter weights with lora_A / lora_B submodule names
    lora_modules = [
        name for name, m in model.named_modules()
        if hasattr(m, "lora_A") and len(list(m.lora_A.parameters())) > 0
    ]

    pct = 100.0 * trainable / total if total else 0.0
    _logger.info(
        "[LoRA:%s] trainable params: %s / %s (%.2f%%)",
        model_key, f"{trainable:,}", f"{total:,}", pct,
    )

    if lora_modules:
        _logger.info("[LoRA:%s] adapters injected into %d module(s):", model_key, len(lora_modules))
        for name in lora_modules:
            _logger.info("  · %s", name)
    else:
        _logger.warning(
            "[LoRA:%s] *** WARNING: PEFT found NO modules matching target_modules=%r. "
            "The backbone is effectively FROZEN. "
            "Run `print([n for n, _ in model.named_modules()])` to see available names.",
            model_key, target_modules,
        )
        # Also dump the first 30 named modules so it's visible in the SLURM log
        names = [n for n, _ in model.named_modules() if n][:30]
        _logger.warning("[LoRA:%s] Available module names (first 30): %s", model_key, names)

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

_INTERPOLATED_CLASS: dict[str, str] = {
    "labram": "InterpolatedLaBraM",
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
    _channel_adapter = None
    _channel_plan = None

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
    ) -> BrainDecodeBackend:
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
        uses_interpolation = interpolate_channels and model_key in _INTERPOLATED_CLASS
        if (
            interpolate_channels
            and not uses_interpolation
            and not _montages.is_special(model_key)
        ):
            raise ValueError(
                f"interpolate_channels=True is not supported for '{model_key}'. "
                f"Channel interpolation is available for: "
                f"{sorted(_INTERPOLATED_CLASS)}."
            )

        # Faithful fixed-montage construction (BIOT bipolar derivation, BENDR
        # reorder + computed SCALE, EEGPT name-indexed subset). The plan is the
        # shared source of truth with the capability gate (see extraction.py);
        # an unservable montage raises here and the extraction job records it.
        channel_plan = None
        if _montages.is_special(model_key):
            fill_missing = bool(kw.pop("fill_missing_channels", False))
            channel_plan = _montages.plan_channels(
                model_key,
                electrode_names or [],
                fill_missing=fill_missing,
                sfreq=sfreq,
            )
            if channel_plan.status != "available":
                raise ValueError(
                    f"{model_key} cannot be served on this montage: "
                    f"{channel_plan.reason}"
                )
        if uses_interpolation:
            shipped = _INTERPOLATED_CLASS[model_key]
            if not hasattr(models_module, shipped):
                raise ImportError(
                    f"interpolate_channels=True requires {shipped} "
                    "(braindecode>=1.5), which is not available in the "
                    "installed braindecode version."
                )
            model_cls = getattr(models_module, shipped)
        else:
            model_cls = getattr(models_module, class_name)

        # Montage-dependent models need ``chs_info``: LaBraM/SignalJEPA resolve
        # channel identity from it (SignalJEPA indexes its pretrained per-channel
        # embeddings by name), and the LaBraM interpolation wrapper builds its
        # interpolation matrix from it.
        if electrode_names and model_key in {"cbramod", "labram", "luna", "signaljepa"}:
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
            # Load the full 62-row pretrained channel-embedding matrix and index
            # the rows matching the input ``chs_info`` by name (braindecode#991),
            # so any montage that is a subset of the pretraining set works.
            kw.setdefault("channel_embedding", "pretrain_aligned")
        elif model_key == "biot":
            # Plain BIOT on the 16-channel TCP bipolar montage built by exact
            # subtraction (see channel_plan.adapter); never InterpolatedBIOT.
            kw["chs_info"] = channel_plan.target_chs_info
            kw["n_chans"] = channel_plan.model_n_chans
        elif model_key == "bendr":
            # Plain BENDR on 20 channels (19 reordered EEG + computed SCALE);
            # no chs_info → skip BENDR's exact-name check (the adapter already
            # reorders into BENDR_CHANNEL_ORDER).
            kw["n_chans"] = channel_plan.model_n_chans
            kw.pop("chs_info", None)

        load_kwargs = dict(kw)
        if revision is not None:
            load_kwargs["revision"] = revision
        if filename is not None:
            load_kwargs["filename"] = filename
        effective_outputs = (
            1
            if model_key in {"cbramod", "luna", "eegpt"} and n_outputs is None
            else n_outputs
        )
        if effective_outputs is not None:
            load_kwargs["n_outputs"] = effective_outputs
        if model_key == "eegpt":
            # EEGPT's checkpoint stores a 62-channel layout; from_pretrained
            # cannot rebuild it on a subset montage. Build the model on the
            # input channels (chan_proj_type="none") and reconcile the
            # name-indexed state dict (see _load_eegpt).
            model = cls._load_eegpt(
                metadata,
                channel_plan,
                n_times=n_times,
                n_outputs=effective_outputs,
                revision=revision,
                filename=filename,
                device=device,
            )
        else:
            model = model_cls.from_pretrained(metadata.hub_repo, **load_kwargs)
        model = model.to(device)

        if model_key == "signaljepa":
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

            lora_r, lora_alpha = resolve_auto_lora_params(
                model_key, model, lora_target_modules, lora_r, lora_alpha
            )
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)
            _log_lora_injection(model, lora_target_modules, model_key)
        # full: no changes — all params trainable

        probe_n_chans = (
            channel_plan.model_n_chans
            if channel_plan is not None
            else (len(electrode_names) if electrode_names else None)
        )
        feat_dim = cls._probe_feat_dim(
            model,
            metadata,
            device,
            n_chans=probe_n_chans,
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
        backend._channel_adapter = (
            channel_plan.adapter if channel_plan is not None else None
        )
        backend._channel_plan = channel_plan
        return backend

    @classmethod
    def _load_eegpt(
        cls,
        metadata: FoundationModelSpec,
        channel_plan,
        *,
        n_times: int,
        n_outputs: int | None,
        revision: str | None,
        filename: str | None,
        device: str,
    ):
        """Build EEGPT on the input montage and load the name-indexed checkpoint.

        EEGPT identifies channels by name through a learnable channel-embedding
        table indexed by ``chans_id``. The published checkpoint stores a
        62-channel layout, so ``from_pretrained`` cannot rebuild it on a subset
        montage. Instead we build the model with ``chan_proj_type="none"`` on the
        kept channels (``chans_id`` recomputed from their names) and load every
        pretrained weight, dropping only the derived ``chans_id`` buffer and the
        unused classification head.
        """
        import mne
        import torch
        from braindecode.models import EEGPT
        from huggingface_hub import hf_hub_download

        names = channel_plan.model_ch_names
        info = mne.create_info(names, sfreq=metadata.pretrained_sfreq, ch_types="eeg")
        # Positions are unused by EEGPT (channels are identified by name via
        # chans_id); resolve them best-effort and stay quiet on misses.
        info.set_montage("standard_1020", match_case=False, on_missing="ignore")
        model = EEGPT(
            n_chans=len(names),
            n_times=n_times,
            chs_info=info["chs"],
            n_outputs=n_outputs if n_outputs is not None else 1,
            chan_proj_type="none",
        )
        weights = filename or "model.safetensors"
        path = hf_hub_download(metadata.hub_repo, weights, revision=revision)
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(path)
        else:
            state = torch.load(path, map_location="cpu")
        target = model.state_dict()
        reconciled = {}
        for key, value in state.items():
            if key == "chans_id":
                continue  # derived buffer, recomputed from chs_info
            if (
                key in target
                and hasattr(value, "shape")
                and value.shape != target[key].shape
            ):
                continue  # head/probe with a different n_outputs — left fresh
            reconciled[key] = value
        result = model.load_state_dict(reconciled, strict=False)
        backbone_missing = [
            k
            for k in result.missing_keys
            if k != "chans_id" and not k.startswith("final_layer")
        ]
        if backbone_missing or result.unexpected_keys:
            raise RuntimeError(
                "EEGPT checkpoint reconciliation left weights unloaded "
                f"(missing={backbone_missing}, unexpected={result.unexpected_keys}); "
                "embeddings would be partly random."
            )
        return model

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

    def fit(self, X: np.ndarray, y=None, **fit_params):
        """Fit the model, resampling X and constructing the model's montage first."""
        X = self._maybe_resample(X)
        self._validate(X)
        # Map the source montage to the model's native layout (e.g. BIOT bipolar,
        # BENDR reorder+SCALE). Must match transform()/predict() or the model is
        # trained on the wrong channel count. No-op when no adapter is set.
        X = self._construct_channels(X)
        return self._fit_with_skorch(X, y, **fit_params)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, resampling + constructing channels first."""
        X = self._maybe_resample(X)
        self._validate(X)
        X = self._construct_channels(X)
        if self._net_ is None:
            raise RuntimeError("Model must be fitted before predict_proba().")
        return np.asarray(self._net_.predict_proba(X))

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
        X = self._maybe_resample(X)
        self._validate(X)
        X = self._construct_channels(X)
        with self._no_grad():
            feats = self._forward_features(
                self._model,
                self._to_tensor(X),
                self._metadata.name,
            )
        return self._from_tensor(feats)

    def _construct_channels(self, X: np.ndarray) -> np.ndarray:
        """Apply the faithful fixed-montage channel construction, if any.

        Maps the validated source montage ``(n, n_source, t)`` to the model's
        native layout (BIOT bipolar derivation, BENDR reorder + SCALE, EEGPT
        name subset). A no-op when the model consumes channels as-is.
        """
        if self._channel_adapter is None:
            return X
        return self._channel_adapter(X)

    def checkpoint_components(self) -> dict:
        """BrainDecode keeps one model (LoRA-wrapped when applicable)."""
        return {"model": self._model}

    def get_channel_adaptation(self) -> dict:
        """Describe direct and interpolated channels from the real layer."""
        meta = self.signal_metadata_
        source = list(meta.ch_names) if meta is not None else []
        if self._channel_plan is not None:
            plan = self._channel_plan
            provenance = dict(plan.provenance)
            return {
                "source_channels": source,
                "target_channels": list(plan.model_ch_names or []),
                "zero_filled_channels": [],
                **provenance,
                # ``filled_channels`` (provenance) are interpolated unipolar
                # electrodes; surface them under the legacy key too.
                "interpolated_channels": list(provenance.get("filled_channels", [])),
            }
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
        X = self._maybe_resample(X)
        self._validate(X)
        X = self._construct_channels(X)
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

    def reset_head(self, n_outputs: int) -> BrainDecodeBackend:
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

    def configure_peft(self, lora_config: dict) -> BrainDecodeBackend:
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
