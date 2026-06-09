"""
Cache-key helpers for decoding feature extraction.
==================================================

The decoding module uses these helpers to generate stable, split-safe keys for
caching intermediate artifacts like embeddings or fitted preprocessing steps.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence


def make_feature_cache_key(
    train_sample_ids: Sequence[Any],
    test_sample_ids: Sequence[Any],
    preprocessing_fingerprint: str,
    backbone_fingerprint: str,
    extra_metadata: dict[str, Any] | None = None,
    sort_ids: bool = True,
) -> str:
    """
    Build a stable cache key for split-specific feature extraction artifacts.

    This generates a SHA256 hex digest of a JSON-serialized payload containing
    the identities of the train/test samples and the configuration of the
    preprocessing and backbone modules. This ensures that fitted transforms
    or extracted embeddings cannot be reused for incompatible splits or
    different model configurations, preventing data leakage and silent errors.

    The sample IDs are converted to strings to ensure stability across
    different ID types. By default, IDs are sorted to ensure the cache key
    is order-insensitive. If order-dependent preprocessing is used,
    set `sort_ids=False`.

    Parameters
    ----------
    train_sample_ids : Sequence[Any]
        Sample IDs identifying the training fold.
    test_sample_ids : Sequence[Any]
        Sample IDs identifying the test/validation fold.
    preprocessing_fingerprint : str
        A unique hash or string representing the preprocessing configuration.
    backbone_fingerprint : str
        A unique hash or string representing the model/extractor configuration.
    extra_metadata : dict[str, Any], optional
        Additional dimensions that affect the output (e.g., time indices,
        target labels, or stage names). Default is None.
    sort_ids : bool, default=True
        Whether to sort the sample IDs before hashing. Sorting makes the
        key order-insensitive, which is usually desired for reproducibility.

    Returns
    -------
    key : str
        The SHA256 hex digest of the normalized JSON payload.
    """
    # 1. Normalize identifiers
    train_ids = [str(value) for value in train_sample_ids]
    test_ids = [str(value) for value in test_sample_ids]

    if sort_ids:
        train_ids.sort()
        test_ids.sort()

    payload = {
        "train_sample_ids": train_ids,
        "test_sample_ids": test_ids,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "backbone_fingerprint": backbone_fingerprint,
    }

    # 2. Handle metadata path (explicit for coverage)
    if extra_metadata is not None:
        payload["extra_metadata"] = extra_metadata
    else:
        payload["extra_metadata"] = {}

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
