"""
Foundation Model Hub (fm_hub)
============================
Unified access to foundation models for both extraction and fine-tuning.
"""

from ._factory import build_foundation_model
from .base import BaseFoundationModel, EmbeddingInfo
from .reve import REVEModel

__all__ = [
    "BaseFoundationModel",
    "EmbeddingInfo",
    "build_foundation_model",
    "REVEModel",
]
