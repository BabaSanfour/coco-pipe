"""Descriptor module constants — column contracts and QC schemas.

Two concerns live here:

* **Column-naming contract** — ``KNOWN_FAMILY_TOKENS`` and
  ``DESCRIPTOR_SCOPE_RE`` define the token vocabulary and regex used by both
  the IO parser (:func:`~coco_pipe.io.descriptors.parse_descriptor_feature_column`)
  and the family-level QC (:mod:`coco_pipe.descriptors.qc`).

* **QC output schemas** — the ``_*_COLUMNS`` lists and
  ``_FAILURE_FAMILY_ALIASES`` dict are the canonical column orderings for
  every DataFrame produced by :mod:`coco_pipe.descriptors.qc`.  Centralising
  them here makes schema changes a single-line edit.
"""

import re

KNOWN_FAMILY_TOKENS: tuple[str, ...] = ("band", "param", "complexity")
"""Column-name prefix tokens produced by DescriptorPipeline extractors."""

# The greedy first group selects the last scope marker, which preserves earlier
# `_ch-` fragments in cross-channel measure names.
DESCRIPTOR_SCOPE_RE: re.Pattern[str] = re.compile(r"^(.+)_(chgrp|ch)-(.+)$")


_FAILURE_FAMILY_ALIASES: dict[str, str] = {
    "bands": "band",
    "parametric": "param",
}
"""Normalise loose failure-log family labels to the canonical token form."""

_CLASSIFICATION_COLUMNS: list[str] = [
    "column",
    "family",
    "scope",
    "channel",
    "measure",
    "subfamily",
    "descriptor",
]
"""Output schema of :func:`~coco_pipe.descriptors.qc.classify_descriptor_columns`."""

_MISSINGNESS_COLUMNS: list[str] = [
    "column",
    "missing_count",
    "missing_rate",
    "nonfinite_count",
    "nonfinite_rate",
    "family",
    "scope",
    "channel",
    "measure",
]
"""Output schema of :func:`~coco_pipe.descriptors.qc.compute_family_missingness`."""

_CONSTANT_COLUMNS: list[str] = [
    "column",
    "std",
    "is_all_nan",
    "is_constant",
    "family",
    "scope",
    "channel",
    "measure",
]
"""Output schema of
:func:`~coco_pipe.descriptors.qc.compute_family_constant_summary`."""

_FAMILY_QC_COLUMNS: list[str] = [
    "family",
    "n_features",
    "missing_rate_mean",
    "missing_rate_max",
    "nonfinite_rate_mean",
    "n_all_nan_features",
    "n_constant_features",
    "failure_count",
    "failure_rate",
]
"""Output schema of :func:`~coco_pipe.descriptors.qc.aggregate_family_qc`."""
