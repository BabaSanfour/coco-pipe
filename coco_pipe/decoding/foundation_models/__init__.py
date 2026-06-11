"""
Foundation Models
=================
Pretrained EEG/MEG foundation model backends and loading utilities.
"""

from ._loader import load, register_backend, unregister_backend
from ._prepare import normalize_channel_names
from .estimators import FoundationClassifier, FrozenBackboneTransformer
from .extraction import (
    CapabilityResult,
    FoundationEmbeddingExtractor,
    FoundationEmbeddingResult,
    check_capability,
)

__all__ = [
    "load",
    "CapabilityResult",
    "FoundationEmbeddingExtractor",
    "FoundationEmbeddingResult",
    "check_capability",
    "normalize_channel_names",
    "register_backend",
    "unregister_backend",
    "FoundationClassifier",
    "FrozenBackboneTransformer",
]
