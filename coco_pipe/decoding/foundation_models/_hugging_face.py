"""
HuggingFace Backend
===================

Backend for EEG/MEG foundation models loaded via HuggingFace
``AutoModel.from_pretrained``. Currently supports REVE.

Supports all four training modes: frozen, full, lora, qlora.
qlora requires ``bitsandbytes``; lora/qlora require ``peft``.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np

from .._specs import SignalMetadata
from ._base import BackendBase

if TYPE_CHECKING:
    from .._specs import FoundationModelSpec


class HuggingFaceBackend(BackendBase):
    """Backend for HuggingFace-native models loaded via AutoModel.from_pretrained.

    Training modes: frozen, full, lora, qlora.
    """

    def __init__(
        self,
        metadata: FoundationModelSpec,
        backbone,
        pos_bank,
        head,
        feat_dim: int,
        n_outputs: int | None,
        device: str,
        train_mode: str,
        pooling: str,
        electrode_names: list[str] | None,
        task: str,
    ) -> None:
        self._metadata = metadata
        self._backbone = backbone
        self._pos_bank = pos_bank
        self._head = head
        self._feat_dim = feat_dim
        self._n_outputs = n_outputs
        self._device = device
        self._train_mode = train_mode
        self._pooling = pooling
        self._electrode_names = electrode_names
        self._task = task
        self.signal_metadata_: SignalMetadata | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Check whether transformers and torch are installed.

        Returns
        -------
        available : bool
            ``True`` if both ``transformers`` and ``torch`` are importable.
        """
        return (
            importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("torch") is not None
        )

    @classmethod
    def load(
        cls,
        model_key: str,
        metadata: FoundationModelSpec,
        n_outputs: int | None,
        device: str,
        train_mode: str,
        pooling: str = "mean",
        electrode_names: list[str] | None = None,
        token: str | None = None,
        task: str = "classification",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: tuple[str, ...] = ("query", "value"),
        lora_dropout: float = 0.05,
        **kw,
    ) -> "HuggingFaceBackend":
        """Load a pretrained HuggingFace model and return an initialised backend.

        Parameters
        ----------
        model_key : str
            Registry key. Currently only ``"reve"`` is supported.
        metadata : FoundationModelSpec
            Registry spec providing ``hub_repo`` and signal metadata.
        n_outputs : int or None
            Number of output units. ``None`` = headless feature extractor.
        device : str
            Resolved device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        train_mode : str
            One of ``"frozen"``, ``"full"``, ``"lora"``, ``"qlora"``.
        pooling : str, default ``"mean"``
            How to collapse the token dimension: ``"mean"`` (average pooling)
            or ``"flatten"`` (concatenate all channel embeddings).
        electrode_names : list of str or None
            Channel names used by REVE's positional encoder. Falls back to
            generic ``["e0", "e1", …]`` when ``None``.
        token : str or None
            HuggingFace access token for private repositories.
        task : str, default ``"classification"``
            ``"classification"`` or ``"regression"``.
        lora_r : int, default 16
            LoRA rank. Ignored unless ``train_mode`` is ``"lora"`` or ``"qlora"``.
        lora_alpha : int, default 32
            LoRA scaling factor.
        lora_target_modules : tuple of str, default ``("query", "value")``
            Transformer sub-modules to inject LoRA adapters into.
        lora_dropout : float, default 0.05
            Dropout probability inside LoRA adapters.
        **kw
            Extra keyword arguments forwarded to ``AutoModel.from_pretrained``.

        Returns
        -------
        backend : HuggingFaceBackend
            A fully initialised backend instance.

        Raises
        ------
        ValueError
            If ``model_key`` is not ``"reve"``.
        """
        if model_key != "reve":
            raise ValueError(f"HuggingFaceBackend does not support '{model_key}'.")
        return cls._load_reve(
            metadata,
            n_outputs=n_outputs,
            device=device,
            train_mode=train_mode,
            pooling=pooling,
            electrode_names=electrode_names,
            token=token,
            task=task,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
        )

    @classmethod
    def _load_reve(
        cls,
        metadata: FoundationModelSpec,
        n_outputs: int | None,
        device: str,
        train_mode: str,
        pooling: str,
        electrode_names: list[str] | None,
        token: str | None,
        task: str,
        lora_r: int,
        lora_alpha: int,
        lora_target_modules: tuple[str, ...],
        lora_dropout: float,
    ) -> "HuggingFaceBackend":
        """Load REVE from HuggingFace."""
        import torch
        import torch.nn as nn
        from transformers import AutoModel

        hf_kw: dict = {"trust_remote_code": True, "token": token}

        if train_mode == "qlora":
            from transformers import BitsAndBytesConfig

            hf_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        backbone = AutoModel.from_pretrained(metadata.hub_repo, **hf_kw)
        pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", **hf_kw)
        feat_dim: int = getattr(backbone.config, "hidden_size", 1024)
        if pooling == "flatten":
            n_ch = (
                len(electrode_names)
                if electrode_names
                else metadata.pretrained_n_chans or 19
            )
            feat_dim = feat_dim * n_ch

        if train_mode in ("lora", "qlora"):
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if train_mode == "qlora":
                backbone = prepare_model_for_kbit_training(backbone)
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=list(lora_target_modules),
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            backbone = get_peft_model(backbone, lora_cfg)
        elif train_mode == "frozen":
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()

        head = (
            nn.Linear(feat_dim, n_outputs).to(device)
            if n_outputs is not None
            else nn.Identity().to(device)
        )

        return cls(
            metadata=metadata,
            backbone=backbone.to(device),
            pos_bank=pos_bank.to(device),
            head=head,
            feat_dim=feat_dim,
            n_outputs=n_outputs,
            device=device,
            train_mode=train_mode,
            pooling=pooling,
            electrode_names=electrode_names,
            task=task,
        )

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    def _get_skorch_module(self):
        import torch.nn as nn

        class _REVESklearnModule(nn.Module):
            def __init__(self, backend, output_dim: int) -> None:
                super().__init__()
                self._backend = backend
                self._output_dim = output_dim

            def forward(self, X):
                return self._backend._reve_forward(X, return_embeddings=False)

            def train(self, mode: bool = True):
                super().train(mode)
                if mode and self._backend._train_mode == "frozen":
                    self._backend._backbone.eval()

        return _REVESklearnModule

    def _reve_forward(self, x_tensor, *, return_embeddings: bool):
        """Reorganised from REVEModule.forward()."""
        n_channels = x_tensor.shape[1]
        elec = self._electrode_names or [f"e{i}" for i in range(n_channels)]
        pos = self._pos_bank(elec).unsqueeze(0).expand(len(x_tensor), -1, -1)
        out = self._backbone(x_tensor, pos)
        hidden = out.last_hidden_state
        if self._pooling == "mean":
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden.flatten(start_dim=1)
        return pooled if return_embeddings else self._head(pooled)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract backbone embeddings without running the classification head.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_channels, n_times)
            EEG/MEG epochs.

        Returns
        -------
        embeddings : np.ndarray of shape (n_samples, embedding_dim)
            Pooled backbone representations.
        """
        self._validate(X)
        with self._no_grad():
            out = self._reve_forward(self._to_tensor(X), return_embeddings=True)
        return self._from_tensor(out)

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
            logits = self._reve_forward(self._to_tensor(X), return_embeddings=False)
        if getattr(self, "_task", "classification") == "regression":
            return self._from_tensor(logits).squeeze(-1)
        return self._from_tensor(logits.argmax(dim=-1))

    def reset_head(self, n_outputs: int) -> "HuggingFaceBackend":
        """Replace the classification head without modifying backbone weights.

        Parameters
        ----------
        n_outputs : int
            Number of output units for the new linear head.

        Returns
        -------
        self : HuggingFaceBackend
            The backend with the updated head.
        """
        import torch.nn as nn

        self._head = nn.Linear(self._feat_dim, n_outputs).to(self._device)
        self._n_outputs = n_outputs
        return self

    def configure_peft(self, lora_config: dict) -> "HuggingFaceBackend":
        """Apply LoRA adapters to the backbone post-hoc.

        Parameters
        ----------
        lora_config : dict
            Keyword arguments forwarded to ``peft.LoraConfig``.

        Returns
        -------
        self : HuggingFaceBackend
            The backend with LoRA adapters injected and ``train_mode`` set
            to ``"lora"``.
        """
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(**lora_config)
        self._backbone = get_peft_model(self._backbone, cfg)
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
