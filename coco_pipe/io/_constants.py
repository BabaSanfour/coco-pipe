"""Type aliases and runtime constants shared across the io layer.

Centralises the small set of domain literals (``QualityStatus``,
``QCFlagLevel``) and the status-ordering dict so that :mod:`coco_pipe.io.quality`
stays focused on logic and tests can import the aliases without pulling in the
full module dependency tree.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

QualityStatus = Literal["OK", "WARN", "FAIL"]
"""Status value produced by :class:`~coco_pipe.io.quality.CheckResult`."""

QCFlagLevel = Literal["pass", "warn", "fail"]
"""Severity level for QC flag dicts produced by ``make_qc_flag``."""

QualityInput = pd.DataFrame | np.ndarray
"""Accepted input type for column-level quality checks."""

STATUS_ORDER: dict[str, int] = {"pass": 0, "warn": 1, "fail": 2}
"""Ordinal ranking used by ``resolve_qc_status`` to pick the worst level."""

AGGREGATION_LEVELS: tuple[str, ...] = ("epoch", "recording", "subject")
"""Canonical observation-granularity ladder, coarsening left → right.

- ``epoch``     — one row per analysis window/epoch (finest).
- ``recording`` — one row per recording = ``(subject, session, run)``.
- ``subject``   — one row per subject, pooled across that subject's recordings.

The single shared vocabulary for descriptor tables, foundation-embedding
representations, and raw-EEG aggregation, so every pipeline names the same
granularity the same way.
"""


EMBEDDING_COMBINED_TABLE_LABELS: dict[str, str] = {
    "epoch": "epoch_embeddings",
    "recording": "recording_embeddings",
    "subject": "subject_embeddings",
}
"""Maps a ``representation`` granularity to the combined embedding table filename.
"""


GROUP_BY_COLUMN: dict[str, str] = {
    "family": "family",
    "subfamily": "subfamily",
    "measure": "measure",
    "feature": "column",
}
"""Maps a ``group_by`` granularity to the ``classify_descriptor_columns`` column
that carries its label. Shared by the QC drop path and downstream callers so the
set of valid ``group_by`` values lives in one place."""

REQUIRED_ARRAYS = {
    "window_embeddings",
    "recording_embedding",
    "window_start",
    "window_stop",
    "window_index",
}

TOKEN_REQUIRED_ARRAYS = {
    "token_embeddings",
    "window_start",
    "window_stop",
    "window_index",
}

ARTIFACT_SUFFIX = {"embedding": "_embedding.npz", "token": "_tokens.npz"}
"""Filename suffix per embedding-derivative kind. Enforced at save time so the
suffix is the authoritative marker of an artifact's kind for discovery/loading."""

ANALYSIS_MODES = (
    "flat",
    "sensor",
    "family",
    "subfamily",
    "sensor_within_family",
    "sensor_within_subfamily",
    "feature",
    "feature_within_family",
    "descriptor",
    "descriptor_sensor",
)

DESCRIPTOR_ONLY_ANALYSIS_MODES = frozenset(ANALYSIS_MODES) - {"flat", "sensor"}
