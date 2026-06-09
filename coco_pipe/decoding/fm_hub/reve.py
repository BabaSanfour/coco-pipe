"""
REVE Foundation Model Provider
==============================
Implementation for the Representation for EEG with Versatile Embeddings (REVE).
"""

from typing import List, Optional, Type

import torch
import torch.nn as nn

from .base import BaseFoundationModel, BaseTransformerModule, EmbeddingInfo


class REVEModule(BaseTransformerModule):
    """
    Pure PyTorch implementation of the REVE architecture.

    This module handles EEG-specific positional encoding using a position bank
    and performs feature extraction through a Transformer backbone.
    """

    def __init__(
        self,
        model_name: str = "brain-bzh/reve-large",
        output_dim: int = 2,
        electrode_names: Optional[List[str]] = None,
        train_mode: str = "frozen",
        pooling: str = "mean",
        token: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        from transformers import AutoModel

        self.pooling = pooling
        self.electrode_names = electrode_names

        # 1. Initialize Backbone (Generic Transformer logic)
        self.backbone = self.load_backbone(model_name, train_mode, token, **kwargs)

        # 2. REVE-Specific: Position Bank
        # The position bank maps electrode names to spatial embeddings
        hf_kwargs = {"trust_remote_code": True, "token": token}
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions", **hf_kwargs
        )

        # 3. Task-Specific Head
        feat_dim = (
            self.backbone.config.hidden_size
            if hasattr(self.backbone, "config")
            else 1024
        )
        self.head = nn.Linear(feat_dim, output_dim)

    def forward(
        self, X: torch.Tensor, return_embeddings: bool = False, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for REVE.

        Parameters
        ----------
        X : torch.Tensor
            Input EEG data of shape (batch, channels, time).
        return_embeddings : bool
            If True, returns the 1024D pooled representation instead of logits.
        """
        # Resolve electrode names for positional encoding
        n_channels = X.shape[1]
        elec = self.electrode_names or [f"e{i}" for i in range(n_channels)]

        # Lookup spatial positions and expand to batch size
        pos = self.pos_bank(elec).unsqueeze(0).expand(len(X), -1, -1)

        # Transform backbone pass
        out = self.backbone(X, pos)
        hidden = out.last_hidden_state

        # Temporal Pooling
        if self.pooling == "mean":
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden[:, 0, :]  # CLS-style pooling

        if return_embeddings:
            return pooled

        return self.head(pooled)


class REVEModel(BaseFoundationModel):
    """
    Unified REVE Model wrapper for the coco-pipe decoding engine.

    Provides a Scikit-Learn compatible interface for REVE-large, supporting
    frozen extraction, linear probing, and parameter-efficient fine-tuning (LoRA).
    """

    def __init__(self, **kwargs):
        model_name = kwargs.pop("model_name", "brain-bzh/reve-large")
        super().__init__(model_name=model_name, **kwargs)
        self.provider = "reve"

    def get_module_cls(self) -> Type[nn.Module]:
        """Return the REVEModule class."""
        return REVEModule

    def _get_net_params(self) -> dict:
        """Add REVE-specific parameters for skorch initialization."""
        params = super()._get_net_params()
        params.update(
            {
                "module__electrode_names": self.kwargs.get("electrode_names"),
                "module__pooling": self.kwargs.get("pooling", "mean"),
                "module__token": self.kwargs.get("token"),
            }
        )
        return params

    def get_embedding_info(self) -> EmbeddingInfo:
        """Return metadata about REVE embeddings."""
        return EmbeddingInfo(
            n_embeddings=1024,
            embedding_name=f"REVE ({self.kwargs.get('pooling', 'mean')})",
            provider="reve",
            model_name=self.model_name,
            sfreq=self.kwargs.get("sfreq", 200.0),
        )
