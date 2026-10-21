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
import warnings
from typing import TYPE_CHECKING

import numpy as np

from .._specs import SignalMetadata
from ._base import BackendBase, check_weights, resolve_revision, seeded

if TYPE_CHECKING:
    from .._specs import FoundationModelSpec

_POSITION_BANK_REPO = "brain-bzh/reve-positions"


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
        lora_target_modules: str | list[str] = "all-linear",
        lora_dropout: float = 0.05,
        pretrained: bool = True,
        seed: int | None = None,
        **kw,
    ) -> HuggingFaceBackend:
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
            How to collapse the tokens into one vector of ``embed_dim``:
            ``"mean"`` (average over channels and time) or ``"attention"``
            (REVE's pretrained single-query attention read-out over channels
            and time).
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
        lora_target_modules : str or list of str, default ``"all-linear"``
            Modules to inject LoRA adapters into: every linear layer by
            default (REVE names its attention layers ``to_qkv`` and
            ``to_out``).
        lora_dropout : float, default 0.05
            Dropout probability inside LoRA adapters.
        pretrained : bool, default True
            ``False`` keeps the checkpoint's architecture and draws new backbone
            weights from ``seed``; the position bank stays the checkpoint's.
            Not with ``train_mode="qlora"``, which quantizes pretrained weights.
        seed : int or None
            The seed of random weights; only with ``pretrained=False``.
        **kw
            Extra keyword arguments forwarded to ``AutoModel.from_pretrained``.

        Returns
        -------
        backend : HuggingFaceBackend
            A fully initialised backend instance.

        Raises
        ------
        ValueError
            If ``model_key`` is not ``"reve"``, or the weights asked for are not
            pretrained or random with a seed.
        """
        if model_key != "reve":
            raise ValueError(f"HuggingFaceBackend does not support '{model_key}'.")
        check_weights(pretrained, seed)
        if not pretrained and train_mode == "qlora":
            raise ValueError(
                "QLoRA quantizes pretrained weights; random weights take "
                "train_mode 'frozen', 'full' or 'lora'."
            )
        if pooling not in {"mean", "attention"}:
            raise ValueError(
                f"REVE supports pooling='mean' or 'attention', got {pooling!r}."
            )
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
            seed=None if pretrained else seed,
            **kw,
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
        lora_target_modules: str | list[str],
        lora_dropout: float,
        seed: int | None = None,
        **kw,
    ) -> HuggingFaceBackend:
        """Load REVE from HuggingFace, with random backbone weights from ``seed``."""
        import torch
        import torch.nn as nn
        from transformers import AutoModel

        if token is None:
            try:
                from huggingface_hub import get_token

                token = get_token()
            except Exception:
                token = None

        if not electrode_names:
            warnings.warn(
                "REVE uses channel names for positional encoding, but "
                "none were provided; generic names yield incorrect "
                "positions and invalid embeddings. Pass "
                "signal_metadata.ch_names.",
                stacklevel=2,
            )

        hf_kw: dict = {
            "trust_remote_code": True,
            "token": token,
            **{
                key: value
                for key, value in kw.items()
                if key
                in {
                    "revision",
                    "cache_dir",
                    "local_files_only",
                    "force_download",
                }
                and value is not None
            },
        }

        # The position bank lives in its own repo that is versioned
        # independently of the backbone: each loads the commit resolved for it.
        # Quantization likewise only applies to the backbone.
        hub_kw = {key: hf_kw[key] for key in ("token", "cache_dir") if key in hf_kw}
        revisions = {
            metadata.hub_repo: resolve_revision(
                metadata.hub_repo, hf_kw.get("revision"), **hub_kw
            ),
            _POSITION_BANK_REPO: resolve_revision(_POSITION_BANK_REPO, **hub_kw),
        }
        pos_kw = {**hf_kw, "revision": revisions[_POSITION_BANK_REPO]}
        hf_kw["revision"] = revisions[metadata.hub_repo]

        if train_mode == "qlora":
            from transformers import BitsAndBytesConfig

            hf_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        backbone = AutoModel.from_pretrained(metadata.hub_repo, **hf_kw)
        if seed is not None:
            # The checkpoint's architecture, from its own class and config, with
            # new weights drawn from the seed.
            with seeded(seed):
                backbone = type(backbone)._from_config(
                    backbone.config, dtype=backbone.dtype
                )
        pos_bank = AutoModel.from_pretrained(_POSITION_BANK_REPO, **pos_kw)
        feat_dim: int = getattr(
            backbone.config,
            "hidden_size",
            metadata.embedding_dim,
        )

        if train_mode in ("lora", "qlora"):
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if train_mode == "qlora":
                # REVE does not support gradient checkpointing, which peft
                # turns on by default and which then fails the load
                backbone = prepare_model_for_kbit_training(
                    backbone, use_gradient_checkpointing=False
                )
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=(
                    lora_target_modules
                    if isinstance(lora_target_modules, str)
                    else list(lora_target_modules)
                ),
                lora_dropout=lora_dropout,
                bias="none",
            )
            backbone = get_peft_model(backbone, lora_cfg)
        elif train_mode == "frozen":
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()
        if train_mode != "full":
            for param in pos_bank.parameters():
                param.requires_grad = False
            pos_bank.eval()

        head = (
            nn.Linear(feat_dim, n_outputs).to(device)
            if n_outputs is not None
            else nn.Identity().to(device)
        )

        backend = cls(
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
        backend.checkpoint_revisions = revisions
        backend.weights_seed = None if seed is None else int(seed)
        return backend

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
                self.backbone = backend._backbone
                self.position_bank = backend._pos_bank
                self.head = backend._head

            def forward(self, X):
                return self._backend._reve_forward(X, return_embeddings=False)

            def train(self, mode: bool = True):
                super().train(mode)
                if mode and self._backend._train_mode == "frozen":
                    self._backend._backbone.eval()
                if mode and self._backend._train_mode != "full":
                    self._backend._pos_bank.eval()
                return self

        return _REVESklearnModule

    def _reve_forward(
        self, x_tensor, *, return_embeddings: bool, return_tokens: bool = False
    ):
        """Reorganised from REVEModule.forward()."""
        n_channels = x_tensor.shape[1]
        elec = self._electrode_names or [f"e{i}" for i in range(n_channels)]
        pos = self._pos_bank(elec).unsqueeze(0).expand(len(x_tensor), -1, -1)
        # REVE's forward returns a raw tensor of per-patch embeddings with shape
        # (batch, channels, time_patches, embed_dim) -- not a HF ModelOutput.
        out = self._backbone(x_tensor, pos)
        pooled = self._pool(out, self._pooling)
        if return_tokens:
            return pooled, out
        return pooled if return_embeddings else self._head(pooled)

    def _pool(self, out, pooling: str):
        """Collapse ``(batch, channel, time_patch, embed_dim)`` tokens."""
        if pooling == "attention":
            # REVE's pretrained single-query attention read-out over the
            # flattened channel+time tokens -> (batch, embed_dim). Under
            # LoRA/QLoRA the backbone is a PeftModel wrapping the Reve module,
            # so reach the base model that owns the method.
            base = getattr(self._backbone, "get_base_model", None)
            base = base() if base is not None else self._backbone
            return base.attention_pooling(out)
        # average over time patches, then over channels -> (batch, embed_dim)
        return out.mean(dim=2).mean(dim=1)

    def pool_tokens(self, tokens: np.ndarray, pooling: str) -> np.ndarray:
        """Pool the tokens of ``transform`` by ``"mean"`` or ``"attention"``."""
        import torch

        if pooling not in {"mean", "attention"}:
            raise ValueError(
                f"REVE supports pooling='mean' or 'attention', got {pooling!r}."
            )
        batch_size = self._eval_batch_size()
        chunks = []
        with self._no_grad():
            for start in range(0, len(tokens), batch_size):
                # the tokens' own dtype: the values the forward pass pooled
                batch = torch.from_numpy(tokens[start : start + batch_size])
                pooled = self._pool(batch.to(self._device), pooling)
                chunks.append(self._from_tensor(pooled))
        return np.concatenate(chunks)

    def transform(
        self, X: np.ndarray, *, return_tokens: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
        X = self._native_input(X)
        batch_size = self._eval_batch_size()
        pooled_chunks: list[np.ndarray] = []
        token_chunks: list[np.ndarray] = []
        with self._no_grad():
            for start in range(0, len(X), batch_size):
                out = self._reve_forward(
                    self._to_tensor(X[start : start + batch_size]),
                    return_embeddings=True,
                    return_tokens=return_tokens,
                )
                if return_tokens:
                    pooled, tokens = out
                    pooled_chunks.append(self._from_tensor(pooled))
                    token_chunks.append(self._from_tensor(tokens))
                else:
                    pooled_chunks.append(self._from_tensor(out))
        pooled = np.concatenate(pooled_chunks, axis=0)
        if return_tokens:
            return pooled, np.concatenate(token_chunks, axis=0)
        return pooled

    def get_token_output_metadata(self) -> dict[str, object]:
        """Describe REVE's unmodified channel-by-time-patch backbone output."""
        return {
            "token_source": "reve_backbone_output",
            "token_axes": ["window", "channel", "time_patch", "feature"],
            "token_observation_axes": ["channel", "time_patch"],
            "token_feature_axis": "feature",
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
        X = self._native_input(X)
        batch_size = self._eval_batch_size()
        is_regression = self._task == "regression"
        chunks = []
        with self._no_grad():
            for start in range(0, len(X), batch_size):
                logits = self._reve_forward(
                    self._to_tensor(X[start : start + batch_size]),
                    return_embeddings=False,
                )
                if is_regression:
                    chunks.append(self._from_tensor(logits).squeeze(-1))
                else:
                    chunks.append(self._from_tensor(logits.argmax(dim=-1)))
        return np.concatenate(chunks, axis=0)

    def checkpoint_components(self) -> dict:
        """REVE keeps backbone, position bank, and head as separate modules."""
        return {
            name: component
            for name, component in {
                "backbone": self._backbone,
                "position_bank": self._pos_bank,
                "head": self._head,
            }.items()
            if component is not None
        }

    def reset_head(self, n_outputs: int) -> HuggingFaceBackend:
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

    def configure_peft(self, lora_config: dict) -> HuggingFaceBackend:
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
