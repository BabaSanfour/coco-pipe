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

EMBEDDING_VIEWS: tuple[str, ...] = ("token", "window", "channel", "patch", "cov")
"""Representations of a foundation model's output, one BIDS suffix each.

A view is stored under the suffix ``<view>embeddings``:

- ``token``   — the model's token tensor, nothing pooled.
- ``window``  — one vector per window, the tokens averaged over every axis.
- ``channel`` — one vector per channel of a window, averaged over time patches.
- ``patch``   — one vector per time patch of a window, averaged over channels.

Window vectors are normalised when ``NormalizeEmbeddings`` says so; channel and
patch views never are.
- ``cov``     — the tangent-space token covariance of each window (RA).
"""

EMBEDDING_CONFIG_KEYS: dict[str, str] = {
    "model_key": "ModelKey",
    "model_name": "ModelName",
    "model_checkpoint": "Checkpoint",
    "checkpoint_filename": "CheckpointFile",
    "checkpoint_revisions": "CheckpointRevisions",
    "weights": "Weights",
    "seed": "Seed",
    "backend": "Backend",
    "pretrained_sfreq": "PretrainedSamplingFrequency",
    "model_sfreq": "ModelSamplingFrequency",
    "window_duration_seconds": "WindowDuration",
    "window_n_times_model": "WindowSamples",
    "within_window_pooling": "WindowPooling",
    "normalize_embeddings": "NormalizeEmbeddings",
    "expects_microvolts": "ExpectsMicrovolts",
    "input_scale": "InputScale",
    "input_normalization": "InputNormalization",
    "input_reference": "InputReference",
    "input_clip": "InputClip",
    "finetune_input_scale": "FinetuneInputScale",
    "expected_passband": "ExpectedPassband",
    "expected_reference": "ExpectedReference",
    "transform": "Transform",
}
"""The extraction settings an embedding file records: metadata key -> JSON key.

They are the same for every recording a model embeds with one configuration:
the model and its weights (``Weights`` is ``"pretrained"``, ``"random"`` with
its ``Seed``, or ``"finetuned"``), the sampling rates the model was trained at
and is fed (Hz), the window length (``WindowDuration`` in seconds,
``WindowSamples`` fed to the model), the pooling and normalisation of window
vectors, how the input is scaled, and, in a file a subject transform wrote,
the transform (``Transform``, see
:func:`~coco_pipe.io.write_transformed_embeddings`). Their keys are CamelCase, as BIDS
recommends for JSON keys. Each embedding file stores them as the JSON text
attribute ``config``, which is also the content of the dataset's sidecar for
the file's label and suffix, and ``config_hash`` is their hash.
"""

LOADED_LEVELS: tuple[str, ...] = ("epoch", "recording")
"""The levels the BIDS and embedding loaders return (their ``representation``)."""

AGGREGATION_LEVELS: tuple[str, ...] = ("epoch", "recording", "subject")
"""Canonical observation-granularity ladder, coarsening left → right.

- ``epoch``     — one row per analysis window/epoch (finest).
- ``recording`` — one row per recording and condition: the rows of one
  ``recording_id`` (its subject, session, task, acquisition and run) and one
  ``condition``, averaged, named ``<recording_id>_cond-<condition>``
  (:func:`~coco_pipe.io.aggregate_recordings`).
- ``subject``   — one row per subject, pooled across that subject's recordings.

The single shared vocabulary for descriptor tables, foundation-embedding
representations, and raw-EEG aggregation, so every pipeline names the same
granularity the same way.
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
