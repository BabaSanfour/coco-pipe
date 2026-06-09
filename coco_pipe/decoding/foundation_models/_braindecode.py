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
    "signaljepa": ("SignalJEPA", "braindecode.models"),
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
        model_cls = getattr(importlib.import_module(module_path), class_name)

        model = model_cls.from_pretrained(metadata.hub_repo, n_outputs=n_outputs, **kw)
        model = model.to(device)

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

        feat_dim = cls._probe_feat_dim(model, metadata, device)

        if n_outputs is not None:
            model.reset_head(n_outputs)

        return cls(
            metadata=metadata,
            model=model,
            feat_dim=feat_dim,
            n_outputs=n_outputs,
            device=device,
            train_mode=train_mode,
            task=task,
        )

    @staticmethod
    def _probe_feat_dim(model, metadata: FoundationModelSpec, device: str) -> int:
        """Zero forward pass to get actual feature dim.

        Preferred over hard-coding — registry value is reference, probe is truth.
        """
        import torch

        n_ch = metadata.pretrained_n_chans or 19
        probe = torch.zeros(1, n_ch, 400, device=device)
        with torch.no_grad():
            out = model(probe, return_features=True)
        feats = out["features"] if isinstance(out, dict) else out
        return int(feats.shape[-1])

    def _get_skorch_module(self):
        import torch.nn as nn

        class _BDSklearnModule(nn.Module):
            def __init__(self, backend, output_dim: int) -> None:
                super().__init__()
                self._backend = backend
                self._output_dim = output_dim

            def forward(self, X):
                out = self._backend._model(X)
                return out["logits"] if isinstance(out, dict) else out

            def train(self, mode: bool = True):
                super().train(mode)
                if mode and self._backend._train_mode == "frozen":
                    self._backend._set_backbone_eval()
                return self

        return _BDSklearnModule

    def _set_backbone_eval(self) -> None:
        self._model.eval()

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
            out = self._model(self._to_tensor(X), return_features=True)
            feats = out["features"] if isinstance(out, dict) else out
        return self._from_tensor(feats)

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
        with self._no_grad():
            out = self._model(self._to_tensor(X))
            logits = out["logits"] if isinstance(out, dict) else out
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
