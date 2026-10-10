"""
Dimensionality-reduction constants.
====================================

Centralises the literal types, metric taxonomies, artifact-layout names, and
run-inventory key fields shared across :mod:`coco_pipe.dim_reduction.artifacts`
and :mod:`coco_pipe.dim_reduction.pipeline`, so the logic modules stay focused
and the single source of truth for these values lives in one place.

This module imports nothing from the rest of the package, so it is safe to
import from anywhere without risking a cycle.
"""

from __future__ import annotations

from typing import Literal

# --- Run-mode literals -----------------------------------------------------

ErrorMode = Literal["raise", "record"]
"""Error-handling mode for :func:`~coco_pipe.dim_reduction.pipeline.run_fit`
and :func:`~coco_pipe.dim_reduction.pipeline.run_eval`."""

POOLED_CONDITION: str = "pooled_all"
"""Canonical condition name used for the pooled (multi-condition) container."""

# --- Metric taxonomies -----------------------------------------------------

SEPARATION_RF_METRIC_KEY: str = "separation_rf_balanced_accuracy"
"""Canonical key for the random-forest separation metric."""

SEPARATION_METRIC_KEY: str = "separation_logreg_balanced_accuracy"
"""Canonical key for the logistic-regression separation metric."""

FIT_METRIC_COLUMNS: list[str] = [
    "trustworthiness",
    "continuity",
    "lcmc",
    "shepard_correlation",
    "mrre_intrusion",
    "mrre_extrusion",
    "mrre_total",
]
"""Geometry quality metrics recorded in fit run inventory rows."""

EVAL_METRIC_COLUMNS: list[str] = [SEPARATION_RF_METRIC_KEY, SEPARATION_METRIC_KEY]
"""Eval metrics recorded in eval run inventory rows."""

DEFAULT_MAX_CORANKING_SAMPLES: int = 3000
"""Row cap for the co-ranking geometry metrics (trustworthiness/continuity/lcmc/
mrre). The co-ranking matrix is dense ``(n, n)`` in the sample count, so at epoch
granularity (tens of thousands of rows) it is the dominant OOM driver. These
metrics are neighborhood-preservation *estimates*, so above this cap they are
computed on one random subsample of rows (shared between the original and
embedded spaces) instead of the full set. ``None`` disables the cap."""

# --- Run-inventory identity -------------------------------------------------

FIT_RUN_KEY_FIELDS: tuple[str, ...] = ("fit_id",)
"""Fields that uniquely identify a fit run entry."""

EVAL_RUN_KEY_FIELDS: tuple[str, ...] = (
    "fit_id",
    "eval_name",
    "target_col",
    "group_col",
)
"""Fields that uniquely identify an eval run entry."""

FIT_PROVENANCE_FIELDS: tuple[str, ...] = (
    "fit_id",
    "scope",
    "condition",
    "analysis_mode",
    "unit_type",
    "unit_name",
    "unit_key",
    "family",
    "subfamily",
    "container_signature",
    "input_mode",
    "representation",
    "aggregation_unit",
    "run_label",
    "reducer",
    "descriptor_max_abs_value",
    "embedding_model_key",
    "embedding_representation",
    "embedding_aggregate_by",
)
"""Fit-payload fields inherited verbatim by an eval payload."""

# --- Artifact layout --------------------------------------------------------

FIT_ARRAYS_NAME = "fit.npz"
"""Compact fit artifact: arrays bundle (embedding + ids + diagnostics)."""
FIT_META_NAME = "fit.json"
"""Compact fit artifact: metadata bundle (fit payload + metrics)."""
EVAL_NAME = "eval.json"
"""Compact eval artifact: the single eval payload file."""

ARTIFACTS_DIRNAME = "artifacts"
"""Top-level directory under ``output_root`` that holds all artifacts."""
FITS_SUBDIR = "fits"
"""Subdirectory of ``artifacts`` for fit artifact directories."""
EVALS_SUBDIR = "evals"
"""Subdirectory of ``artifacts`` for eval artifact directories."""

STEM_SLUG_MAX_LEN = 32
"""Maximum slug length for each component of an artifact stem."""
