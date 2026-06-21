"""Descriptor module constants — column contract, sub-family vocabulary, and
QC schemas.

Three concerns live here, each a single-place edit point:

* **Column-naming contract** — ``KNOWN_FAMILY_TOKENS`` and
  ``DESCRIPTOR_SCOPE_RE`` define the token vocabulary and scope regex used by
  the column parser (:mod:`coco_pipe.descriptors.naming`) and the family-level
  QC (:mod:`coco_pipe.descriptors.qc`).
* **Sub-family vocabulary** — ``AGG_STAT_PREFIXES``,
  ``BAND_SUBFAMILY_PATTERNS``, ``PARAM_SUBFAMILY``, and
  ``COMPLEXITY_SUBFAMILY`` drive
  :func:`coco_pipe.descriptors.qc.descriptor_subfamily`.
* **QC output schemas** — the ``_*_COLUMNS`` lists and
  ``FAILURE_FAMILY_ALIASES`` dict are the canonical column orderings for every
  DataFrame produced by :mod:`coco_pipe.descriptors.qc`.
"""

import re

__all__ = [
    "AGG_STAT_PREFIXES",
    "BAND_SUBFAMILY_PATTERNS",
    "CLASSIFICATION_COLUMNS",
    "COMPLEXITY_SUBFAMILY",
    "CONSTANT_COLUMNS",
    "DEFAULT_RATIO_PREFIXES",
    "DESCRIPTOR_SCOPE_RE",
    "FAILURE_FAMILY_ALIASES",
    "FAMILY_QC_COLUMNS",
    "KNOWN_FAMILY_TOKENS",
    "MISSINGNESS_COLUMNS",
    "PARAM_SUBFAMILY",
]

# --------------------------------------------------------------------------- #
# Column-naming contract
# --------------------------------------------------------------------------- #
KNOWN_FAMILY_TOKENS: tuple[str, ...] = ("band", "param", "complexity")
"""Column-name prefix tokens produced by DescriptorPipeline extractors."""

DESCRIPTOR_SCOPE_RE: re.Pattern[str] = re.compile(r"^(.+)_(chgrp|ch)-(.+)$")
"""Scope regex splitting ``{body}_{ch|chgrp}-{sensor}``. The greedy first group
selects the *last* scope marker, preserving earlier ``_ch-`` fragments in
cross-channel measure names."""


# --------------------------------------------------------------------------- #
# Sub-family vocabulary (drives descriptor_subfamily)
# --------------------------------------------------------------------------- #
AGG_STAT_PREFIXES: frozenset[str] = frozenset(
    {"mean", "median", "iqr", "mad", "std", "var", "min", "max"}
)
"""Aggregation-stat tokens that may prefix a subject-level measure (e.g.
``median_log_abs_alpha``); stripped before sub-family derivation."""

BAND_SUBFAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("corr_log_abs", "corr_log_abs"),
    ("corr_rel", "corr_rel"),
    ("corr_ratio", "corr_ratio"),
    ("corr_abs", "corr_abs"),
    ("log_abs", "log_abs"),
    ("ratio", "ratio"),
    ("rel", "rel"),
    ("abs", "abs"),
)
"""Band output-type ``(pattern, label)`` pairs, longest/corrected first so e.g.
``corr_log_abs`` matches before ``log_abs`` and ``abs``."""

PARAM_SUBFAMILY: dict[str, str] = {
    "offset": "aperiodic",
    "exponent": "aperiodic",
    "knee": "aperiodic",
    "r_squared": "fit_quality",
    "fit_error": "fit_quality",
    "peak_count": "peaks",
    "peak_freq_dom": "peaks",
    "peak_power_dom": "peaks",
    "peak_bandwidth_dom": "peaks",
    "alpha_peak_freq": "peaks",
    "alpha_peak_power": "peaks",
}
"""Parametric measure → sub-family (``aperiodic`` / ``peaks`` / ``fit_quality``)."""

COMPLEXITY_SUBFAMILY: dict[str, str] = {
    "sample_entropy": "entropy",
    "perm_entropy": "entropy",
    "spectral_entropy": "entropy",
    "svd_entropy": "entropy",
    "fuzzy_entropy": "entropy",
    "dispersion_entropy": "entropy",
    "higuchi_fd": "fractal_complexity",
    "petrosian_fd": "fractal_complexity",
    "hurst_exponent": "fractal_complexity",
    "lziv_complexity": "fractal_complexity",
    "hjorth_mobility": "signal_dynamics",
    "hjorth_complexity": "signal_dynamics",
    "kurtosis": "signal_dynamics",
    "zero_crossings": "signal_dynamics",
}
"""Complexity measure → sub-family (``entropy`` / ``fractal_complexity`` /
``signal_dynamics``)."""


# --------------------------------------------------------------------------- #
# QC output schemas
# --------------------------------------------------------------------------- #
FAILURE_FAMILY_ALIASES: dict[str, str] = {
    "bands": "band",
    "parametric": "param",
}
"""Normalise loose failure-log family labels to the canonical token form."""

CLASSIFICATION_COLUMNS: list[str] = [
    "column",
    "family",
    "scope",
    "channel",
    "measure",
    "subfamily",
    "descriptor",
]
"""Output schema of :func:`~coco_pipe.descriptors.qc.classify_descriptor_columns`."""

MISSINGNESS_COLUMNS: list[str] = [
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

CONSTANT_COLUMNS: list[str] = [
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

FAMILY_QC_COLUMNS: list[str] = [
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

# -------------------------------------------------------------------------- #
# Tables
# -------------------------------------------------------------------------- #
DEFAULT_RATIO_PREFIXES: tuple[tuple[str, str], ...] = (
    ("band_abs_", "agg_band_ratio_"),
    ("band_corr_abs_", "agg_band_corr_ratio_"),
)
"""Default ``(input_prefix, output_prefix)`` pairs for band-ratio columns."""
