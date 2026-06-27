"""
Foundation Models
=================
Pretrained EEG/MEG foundation model backends and loading utilities.
"""

from ._channels import normalize_channel_names
from ._loader import load, register_backend, unregister_backend
from .estimators import (
    FoundationClassifier,
    FrozenBackboneTransformer,
    clear_frozen_embedding_cache,
)
from .extraction import (
    CapabilityResult,
    FoundationEmbeddingExtractor,
    FoundationEmbeddingResult,
    check_capability,
    normalize_inclusive_endpoint,
)

__all__ = [
    "CapabilityResult",
    "FoundationClassifier",
    "FoundationEmbeddingExtractor",
    "FoundationEmbeddingResult",
    "FrozenBackboneTransformer",
    "check_capability",
    "clear_frozen_embedding_cache",
    "load",
    "normalize_channel_names",
    "normalize_inclusive_endpoint",
    "register_backend",
    "unregister_backend",
]
