"""
Base classes for the Foundation Model Hub (fm_hub).
=================================================
Unified API for foundation models using skorch for robust Scikit-Learn
compatibility and training orchestration.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class EmbeddingInfo:
    """
    Metadata about the embeddings produced by a foundation model.

    Parameters
    ----------
    n_embeddings : int
        The dimensionality of the embedding vector.
    embedding_name : str
        A descriptive name for the embedding (e.g., "CLS token", "Mean Pool").
    provider : str
        The model provider (e.g., "reve", "neuroai").
    model_name : str
        The specific model identifier (e.g., "brain-bzh/reve-large").
    sfreq : float
        The sampling frequency the model expects.
    """

    n_embeddings: int
    embedding_name: str
    provider: str
    model_name: str
    sfreq: float


class BaseTransformerModule(nn.Module):
    """
    Shared PyTorch base for Transformer models (REVE, CBRAMOD, etc.).

    This class centralizes the complex logic of Parameter-Efficient Fine-Tuning
    (PEFT), Quantization, and parameter freezing, ensuring consistent behavior
    across different foundation model architectures.
    """

    def load_backbone(
        self, model_name: str, train_mode: str, token: Optional[str] = None, **kwargs
    ) -> nn.Module:
        """
        Initialize and configure a HuggingFace Transformer backbone.

        Parameters
        ----------
        model_name : str
            HuggingFace model ID.
        train_mode : str
            One of {"frozen", "full", "lora", "qlora"}.
        token : str, optional
            HuggingFace API token for private models.
        **kwargs : dict
            Additional parameters passed to LoRA/Quantization configs.

        Returns
        -------
        nn.Module
            The configured backbone (potentially wrapped with LoRA/Quantization).
        """
        from transformers import AutoModel, BitsAndBytesConfig

        hf_kwargs = {"trust_remote_code": True, "token": token}

        # 1. Handle 4-bit Quantization (QLoRA)
        if train_mode == "qlora":
            hf_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        backbone = AutoModel.from_pretrained(model_name, **hf_kwargs)

        # 2. Handle PEFT (LoRA)
        if train_mode in {"lora", "qlora"}:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if train_mode == "qlora":
                backbone = prepare_model_for_kbit_training(backbone)

            lora_config = LoraConfig(
                r=kwargs.get("lora_r", 16),
                lora_alpha=kwargs.get("lora_alpha", 32),
                target_modules=kwargs.get("lora_target_modules", ["query", "value"]),
                lora_dropout=kwargs.get("lora_dropout", 0.05),
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            backbone = get_peft_model(backbone, lora_config)

        # 3. Handle Parameter Freezing (Linear Probing)
        if train_mode == "frozen":
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()

        return backbone


class BaseFoundationModel(BaseEstimator, TransformerMixin):
    """
    Abstract base class for all foundation models in fm_hub.

    Provides a standardized Scikit-Learn wrapper around skorch, enabling
    automated training history collection, hardware-agnostic device management,
    and diagnostic reporting.
    """

    def __init__(
        self,
        model_name: str,
        train_mode: str = "frozen",
        task: str = "classification",
        device: Optional[str] = None,
        **kwargs,
    ):
        valid_modes = {"frozen", "full", "lora", "qlora"}
        if train_mode not in valid_modes:
            raise ValueError(
                f"train_mode must be one of {valid_modes}, got {train_mode}"
            )

        self.model_name = model_name
        self.train_mode = train_mode
        self.task = task
        self.device = device
        self.kwargs = kwargs
        self.net_ = None

    @abstractmethod
    def get_module_cls(self) -> Type[nn.Module]:
        """Return the PyTorch Module class to be instantiated by skorch."""
        pass

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "BaseFoundationModel":
        """
        Fit the foundation model or its head using skorch.
        """
        from skorch import NeuralNetClassifier, NeuralNetRegressor

        self.device_ = self._resolve_device()

        # Infer output dimensionality
        if self.task == "classification" and y is not None:
            self.out_dim_ = len(np.unique(y))
        else:
            self.out_dim_ = 1

        net_cls = (
            NeuralNetClassifier if self.task == "classification" else NeuralNetRegressor
        )

        # Configure skorch Net
        self.net_ = net_cls(
            module=self.get_module_cls(),
            module__model_name=self.model_name,
            module__train_mode=self.train_mode,
            module__output_dim=self.out_dim_,
            device=self.device_,
            **self._get_net_params(),
        )

        self.net_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract high-dimensional embeddings from the model.
        """
        if self.net_ is None:
            raise RuntimeError("Model must be fitted before transform.")

        return self.net_.forward(X, training=False, return_embeddings=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform task-specific inference."""
        if self.net_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        return self.net_.predict(X)

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Return the skorch training history."""
        return self.net_.history if self.net_ else []

    def get_artifact_metadata(self) -> Dict[str, Any]:
        """Diagnostic reporting for the training engine."""
        return {
            "model_type": "foundation",
            "model_card": {
                "model_name": self.model_name,
                "train_mode": self.train_mode,
                "task": self.task,
            },
            "history": self.get_training_history(),
        }

    def _get_net_params(self) -> Dict[str, Any]:
        """Package standard skorch parameters."""
        return {
            "max_epochs": self.kwargs.get("max_epochs", 10),
            "lr": self.kwargs.get("lr", 0.001),
            "batch_size": self.kwargs.get("batch_size", 32),
        }

    def _resolve_device(self) -> str:
        """Determine the best available compute device."""
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
