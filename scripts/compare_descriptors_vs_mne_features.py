#!/usr/bin/env python3
"""
Standalone comparison harness for `coco-pipe.descriptors` vs `mne-features`.

This script is intentionally comparison-only. It does not modify the
descriptor extraction runtime. It produces four artifacts:

- coverage and provenance matrix
- validity/stability review table
- benchmark results table
- recommendation memo
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "outputs" / "descriptors_vs_mne_features"
DEFAULT_COCO_PYTHON = REPO_ROOT / ".venv_coco_pipe" / "bin" / "python"
DEFAULT_MNE_FEATURES_PYTHON = REPO_ROOT / ".venv_mne_features" / "bin" / "python"
DEFAULT_REAL_DATA_BIDS_ROOT = REPO_ROOT / "PhysioNet_EEGBCI" / "BIDS"

RATING_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class ReviewRecord:
    feature_name: str
    library: str
    scientific_maturity: str
    implementation_maturity: str
    parameter_sensitivity: str
    backend_risk: str
    numerical_risk: str
    interpretability: str
    review_notes: str


@dataclass(frozen=True)
class CoverageRecord:
    domain: str
    feature_name_coco_pipe: str
    feature_name_mne_features: str
    relationship: str
    implemented_in_coco: bool
    implemented_in_mne_features: bool
    coco_source_type: str
    mne_features_source_type: str
    coco_source_backend: str
    mne_features_source_backend: str
    semantic_notes: str
    benchmarkable_now: bool
    future_usefulness: str
    recommended_future_action: str
    coco_review: ReviewRecord | None = field(default=None, repr=False)
    mne_review: ReviewRecord | None = field(default=None, repr=False)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("coco_review", None)
        row.pop("mne_review", None)
        return row


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    dataset_kind: str
    feature_subset: str
    library: str
    n_epochs: int
    n_channels: int
    n_times: int
    n_jobs: int
    parallel_mode: str
    real_data_npz: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset_kind": self.dataset_kind,
            "feature_subset": self.feature_subset,
            "library": self.library,
            "n_epochs": self.n_epochs,
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "n_jobs": self.n_jobs,
            "parallel_mode": self.parallel_mode,
            "real_data_npz": self.real_data_npz,
        }


def _make_review(
    *,
    feature_name: str,
    library: str,
    scientific_maturity: str,
    implementation_maturity: str,
    parameter_sensitivity: str,
    backend_risk: str,
    numerical_risk: str,
    interpretability: str,
    review_notes: str,
) -> ReviewRecord:
    return ReviewRecord(
        feature_name=feature_name,
        library=library,
        scientific_maturity=scientific_maturity,
        implementation_maturity=implementation_maturity,
        parameter_sensitivity=parameter_sensitivity,
        backend_risk=backend_risk,
        numerical_risk=numerical_risk,
        interpretability=interpretability,
        review_notes=review_notes,
    )


def _implementation_maturity(source_type: str) -> str:
    if source_type in {"backend_wrapper", "numpy_scipy_custom"}:
        return "high"
    if source_type in {"specparam_wrapper", "mne_wrapper", "pywt_wrapper"}:
        return "medium"
    if source_type == "project_specific_logic":
        return "medium"
    if source_type in {"handcoded_numpy", "handcoded_scipy", "handcoded_numba"}:
        return "medium"
    return "medium"


def _backend_risk(source_backend: str, source_type: str) -> str:
    backend = source_backend.lower()
    if source_type in {"numpy_scipy_custom", "handcoded_numpy", "handcoded_scipy"}:
        return "low"
    if "antropy" in backend or "scipy" in backend or "numpy" in backend:
        return "low"
    if "neurokit2" in backend or "specparam" in backend or "mne" in backend:
        return "medium"
    if "pywt" in backend or "sklearn" in backend or "numba" in backend:
        return "medium"
    return "medium"


def _scientific_maturity(feature_name: str, domain: str, future_usefulness: str) -> str:
    if feature_name in {"pool_channels", "shared_psd_planner"}:
        return "high"
    if feature_name in {
        "absolute_power",
        "log_absolute_power",
        "relative_power",
        "sample_entropy",
        "spectral_entropy",
        "hjorth_mobility",
        "hjorth_complexity",
        "offset",
        "exponent",
        "rms",
        "phase_lock_val",
    }:
        return "high"
    if feature_name in {
        "corrected_ratios",
        "nonlin_interdep",
        "svd_fisher_info",
    }:
        return "low"
    if domain == "parametric" or future_usefulness == "high":
        return "medium"
    return "medium"


def _parameter_sensitivity(feature_name: str, domain: str) -> str:
    if feature_name in {"pool_channels", "shared_psd_planner"}:
        return "low"
    if "entropy" in feature_name or feature_name in {
        "higuchi_fd",
        "katz_fd",
        "petrosian_fd",
        "hurst_exponent",
        "hurst_exp",
        "nonlin_interdep",
    }:
        return "high"
    if feature_name.startswith("corrected_") or domain == "parametric":
        return "high"
    if domain in {"spectral", "bivariate", "wavelet"}:
        return "medium"
    return "low"


def _numerical_risk(feature_name: str, domain: str) -> str:
    if feature_name in {"pool_channels", "shared_psd_planner"}:
        return "low"
    if feature_name.startswith("corrected_"):
        return "high"
    if feature_name in {
        "knee",
        "fit_error",
        "r_squared",
        "peak_count",
        "peak_freq_dom",
        "peak_power_dom",
        "peak_bandwidth_dom",
        "alpha_peak_freq",
        "alpha_peak_power",
        "sample_entropy",
        "approx_entropy",
        "svd_entropy",
        "hurst_exponent",
        "hurst_exp",
        "dispersion_entropy",
        "fuzzy_entropy",
        "shannon_entropy",
        "nonlin_interdep",
    }:
        return "medium"
    if domain == "bivariate":
        return "medium"
    return "low"


def _interpretability(feature_name: str, domain: str) -> str:
    if feature_name in {"pool_channels", "shared_psd_planner"}:
        return "high"
    if domain == "utility/amplitude":
        return "high"
    if domain == "spectral" and feature_name != "corrected_ratios":
        return "high"
    if domain == "parametric" and feature_name not in {
        "fit_error",
        "r_squared",
        "knee",
    }:
        return "high"
    if feature_name in {"corrected_ratios", "svd_fisher_info", "nonlin_interdep"}:
        return "low"
    return "medium"


def _default_review_notes(
    *,
    feature_name: str,
    library: str,
    domain: str,
    source_backend: str,
    source_type: str,
) -> str:
    if feature_name == "pool_channels":
        return (
            "Workflow capability rather than a raw EEG descriptor; its value is in "
            "stable region-level derivation from sensor outputs without recomputation."
        )
    if feature_name == "shared_psd_planner":
        return (
            "Runtime architecture capability rather than a single descriptor; it "
            "improves consistency and reuse across spectral families."
        )
    if domain == "parametric":
        return (
            f"Implemented through {source_backend}. Scientifically useful for spectral-model "
            "interpretation, but reliability depends on fit assumptions and peak quality."
        )
    if domain == "spectral":
        return (
            f"Implemented through {source_backend}. The main caveat is alignment of PSD "
            "choices, normalization, and band definitions."
        )
    if domain == "bivariate":
        return (
            f"Implemented through {source_backend}. Potentially useful later, but the "
            "current project would need a separate pairwise-output design before adoption."
        )
    if domain == "wavelet":
        return (
            f"Implemented through {source_backend}. Interesting as a future extension, "
            "though the interpretation depends on transform settings."
        )
    return (
        f"Implemented through {source_backend or source_type}. This looks technically "
        "stable enough for benchmarking, with the main caveat being feature-specific "
        "parameter sensitivity and edge-case handling."
    )


def _default_review_for_record(
    record: CoverageRecord, *, library: str
) -> ReviewRecord | None:
    if library == "coco":
        if not record.implemented_in_coco:
            return None
        feature_name = record.feature_name_coco_pipe
        source_type = record.coco_source_type
        source_backend = record.coco_source_backend
        label = "coco-pipe"
    else:
        if not record.implemented_in_mne_features:
            return None
        feature_name = record.feature_name_mne_features
        source_type = record.mne_features_source_type
        source_backend = record.mne_features_source_backend
        label = "mne-features"

    return _make_review(
        feature_name=feature_name,
        library=label,
        scientific_maturity=_scientific_maturity(
            feature_name, record.domain, record.future_usefulness
        ),
        implementation_maturity=_implementation_maturity(source_type),
        parameter_sensitivity=_parameter_sensitivity(feature_name, record.domain),
        backend_risk=_backend_risk(source_backend, source_type),
        numerical_risk=_numerical_risk(feature_name, record.domain),
        interpretability=_interpretability(feature_name, record.domain),
        review_notes=_default_review_notes(
            feature_name=feature_name,
            library=label,
            domain=record.domain,
            source_backend=source_backend,
            source_type=source_type,
        ),
    )


def _populate_missing_reviews(records: list[CoverageRecord]) -> list[CoverageRecord]:
    populated: list[CoverageRecord] = []
    for record in records:
        coco_review = record.coco_review or _default_review_for_record(
            record, library="coco"
        )
        mne_review = record.mne_review or _default_review_for_record(
            record, library="mne"
        )
        populated.append(
            replace(record, coco_review=coco_review, mne_review=mne_review)
        )
    return populated


def build_coverage_records() -> list[CoverageRecord]:
    records = [
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="sample_entropy",
            feature_name_mne_features="samp_entropy",
            relationship="exact_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="backend_wrapper",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="antropy.sample_entropy",
            mne_features_source_backend="numpy + sklearn.neighbors.KDTree",
            semantic_notes="Same metric family after "
            "aligning embedding and distance metric.",
            benchmarkable_now=True,
            future_usefulness="high",
            recommended_future_action="use_as_reference",
            coco_review=_make_review(
                feature_name="sample_entropy",
                library="coco-pipe",
                scientific_maturity="high",
                implementation_maturity="high",
                parameter_sensitivity="high",
                backend_risk="low",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Antropy backend is mature; output is still sensitive to embedding and signal length.",
            ),
            mne_review=_make_review(
                feature_name="samp_entropy",
                library="mne-features",
                scientific_maturity="high",
                implementation_maturity="medium",
                parameter_sensitivity="high",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Hand-coded implementation is transparent but depends on sklearn KDTree and older package maintenance.",
            ),
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="approx_entropy",
            feature_name_mne_features="app_entropy",
            relationship="exact_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="backend_wrapper",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="antropy.app_entropy",
            mne_features_source_backend="numpy + sklearn.neighbors.KDTree",
            semantic_notes="Same entropy family; compare only "
            "with aligned embedding and metric settings.",
            benchmarkable_now=True,
            future_usefulness="high",
            recommended_future_action="use_as_reference",
            coco_review=_make_review(
                feature_name="approx_entropy",
                library="coco-pipe",
                scientific_maturity="medium",
                implementation_maturity="high",
                parameter_sensitivity="high",
                backend_risk="low",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Common classical metric but more redundant than sample entropy in many EEG settings.",
            ),
            mne_review=_make_review(
                feature_name="app_entropy",
                library="mne-features",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="high",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Reasonable reference implementation, though the library itself is less actively developed.",
            ),
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="spectral_entropy",
            feature_name_mne_features="spect_entropy",
            relationship="approx_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="backend_wrapper",
            mne_features_source_type="mne_wrapper",
            coco_source_backend="antropy.spectral_entropy",
            mne_features_source_backend="mne_features.utils.power_spectrum -> mne PSD utilities",
            semantic_notes="Conceptually aligned, but PSD defaults "
            "and normalization choices differ by implementation.",
            benchmarkable_now=True,
            future_usefulness="high",
            recommended_future_action="use_as_reference",
            coco_review=_make_review(
                feature_name="spectral_entropy",
                library="coco-pipe",
                scientific_maturity="high",
                implementation_maturity="high",
                parameter_sensitivity="medium",
                backend_risk="low",
                numerical_risk="low",
                interpretability="medium",
                review_notes="Good modern wrapper path, but validity depends on matching PSD method and frequency window.",
            ),
            mne_review=_make_review(
                feature_name="spect_entropy",
                library="mne-features",
                scientific_maturity="high",
                implementation_maturity="medium",
                parameter_sensitivity="medium",
                backend_risk="medium",
                numerical_risk="low",
                interpretability="medium",
                review_notes="Reasonable PSD-based formulation built on MNE helpers; import/runtime burden is heavier.",
            ),
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="svd_entropy",
            feature_name_mne_features="svd_entropy",
            relationship="exact_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="backend_wrapper",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="antropy.svd_entropy",
            mne_features_source_backend="numpy SVD on delay embedding",
            semantic_notes="Same family when tau and emb are aligned.",
            benchmarkable_now=True,
            future_usefulness="medium",
            recommended_future_action="use_as_reference",
            coco_review=_make_review(
                feature_name="svd_entropy",
                library="coco-pipe",
                scientific_maturity="medium",
                implementation_maturity="high",
                parameter_sensitivity="medium",
                backend_risk="low",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Stable wrapper around a mature backend with explicit kwargs support.",
            ),
            mne_review=_make_review(
                feature_name="svd_entropy",
                library="mne-features",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="medium",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Transparent hand-coded implementation; useful as an independent reference.",
            ),
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="higuchi_fd",
            feature_name_mne_features="higuchi_fd",
            relationship="exact_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="backend_wrapper",
            mne_features_source_type="handcoded_numba",
            coco_source_backend="antropy.higuchi_fd",
            mne_features_source_backend="numba-accelerated custom implementation",
            semantic_notes="Same fractal-dimension family with aligned kmax.",
            benchmarkable_now=True,
            future_usefulness="high",
            recommended_future_action="use_as_reference",
            coco_review=_make_review(
                feature_name="higuchi_fd",
                library="coco-pipe",
                scientific_maturity="high",
                implementation_maturity="high",
                parameter_sensitivity="medium",
                backend_risk="low",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Well-established EEG complexity feature via a maintained backend.",
            ),
            mne_review=_make_review(
                feature_name="higuchi_fd",
                library="mne-features",
                scientific_maturity="high",
                implementation_maturity="medium",
                parameter_sensitivity="medium",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Custom numba path is performant and independent, but tied to an older package stack.",
            ),
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="katz_fd",
            feature_name_mne_features="katz_fd",
            relationship="exact_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="backend_wrapper",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="antropy.katz_fd",
            mne_features_source_backend="custom numpy formula",
            semantic_notes="Same Katz fractal-dimension family.",
            benchmarkable_now=True,
            future_usefulness="medium",
            recommended_future_action="use_as_reference",
            coco_review=_make_review(
                feature_name="katz_fd",
                library="coco-pipe",
                scientific_maturity="medium",
                implementation_maturity="high",
                parameter_sensitivity="low",
                backend_risk="low",
                numerical_risk="low",
                interpretability="medium",
                review_notes="Straightforward wrapper over a stable classical formula.",
            ),
            mne_review=_make_review(
                feature_name="katz_fd",
                library="mne-features",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="low",
                backend_risk="medium",
                numerical_risk="low",
                interpretability="medium",
                review_notes="Simple direct implementation and useful as a reference check.",
            ),
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="hjorth_mobility",
            feature_name_mne_features="hjorth_mobility",
            relationship="exact_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="backend_wrapper",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="antropy.hjorth_params",
            mne_features_source_backend="custom numpy Hjorth formula",
            semantic_notes="Same Hjorth mobility metric.",
            benchmarkable_now=True,
            future_usefulness="high",
            recommended_future_action="use_as_reference",
            coco_review=_make_review(
                feature_name="hjorth_mobility",
                library="coco-pipe",
                scientific_maturity="high",
                implementation_maturity="high",
                parameter_sensitivity="low",
                backend_risk="low",
                numerical_risk="low",
                interpretability="high",
                review_notes="Simple and interpretable classical EEG summary through a mature backend.",
            ),
            mne_review=_make_review(
                feature_name="hjorth_mobility",
                library="mne-features",
                scientific_maturity="high",
                implementation_maturity="medium",
                parameter_sensitivity="low",
                backend_risk="medium",
                numerical_risk="low",
                interpretability="high",
                review_notes="Straightforward direct implementation; good validation reference.",
            ),
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="hjorth_complexity",
            feature_name_mne_features="hjorth_complexity",
            relationship="exact_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="backend_wrapper",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="antropy.hjorth_params",
            mne_features_source_backend="custom numpy Hjorth formula",
            semantic_notes="Same Hjorth complexity metric.",
            benchmarkable_now=True,
            future_usefulness="high",
            recommended_future_action="use_as_reference",
            coco_review=_make_review(
                feature_name="hjorth_complexity",
                library="coco-pipe",
                scientific_maturity="high",
                implementation_maturity="high",
                parameter_sensitivity="low",
                backend_risk="low",
                numerical_risk="low",
                interpretability="high",
                review_notes="Strong, interpretable classical descriptor already well integrated.",
            ),
            mne_review=_make_review(
                feature_name="hjorth_complexity",
                library="mne-features",
                scientific_maturity="high",
                implementation_maturity="medium",
                parameter_sensitivity="low",
                backend_risk="medium",
                numerical_risk="low",
                interpretability="high",
                review_notes="Useful reference implementation for cross-checking values.",
            ),
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="hurst_exponent",
            feature_name_mne_features="hurst_exp",
            relationship="approx_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="backend_wrapper",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="neurokit2.fractal_hurst",
            mne_features_source_backend="custom rescaled-range style implementation",
            semantic_notes="Same broad concept, but method details differ "
            "enough that only trend comparison is appropriate.",
            benchmarkable_now=True,
            future_usefulness="medium",
            recommended_future_action="use_as_reference",
            coco_review=_make_review(
                feature_name="hurst_exponent",
                library="coco-pipe",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="medium",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Useful but method-specific; NeuroKit adds dependency breadth and some output-shape normalization work.",
            ),
            mne_review=_make_review(
                feature_name="hurst_exp",
                library="mne-features",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="medium",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Classical hand-coded version that is reasonable as a reference but not obviously more valid.",
            ),
        ),
        CoverageRecord(
            domain="utility/amplitude",
            feature_name_coco_pipe="zero_crossings",
            feature_name_mne_features="zero_crossings",
            relationship="exact_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="numpy_scipy_custom",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="numpy signbit transitions",
            mne_features_source_backend="custom numpy thresholded crossings",
            semantic_notes="Comparable when the threshold convention is aligned.",
            benchmarkable_now=True,
            future_usefulness="medium",
            recommended_future_action="use_as_reference",
            coco_review=_make_review(
                feature_name="zero_crossings",
                library="coco-pipe",
                scientific_maturity="medium",
                implementation_maturity="high",
                parameter_sensitivity="low",
                backend_risk="low",
                numerical_risk="low",
                interpretability="high",
                review_notes="Very stable implementation; scientific value is modest but robust.",
            ),
            mne_review=_make_review(
                feature_name="zero_crossings",
                library="mne-features",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="low",
                backend_risk="medium",
                numerical_risk="low",
                interpretability="high",
                review_notes="Simple reference implementation with explicit threshold parameter.",
            ),
        ),
        CoverageRecord(
            domain="utility/amplitude",
            feature_name_coco_pipe="kurtosis",
            feature_name_mne_features="kurtosis",
            relationship="exact_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="numpy_scipy_custom",
            mne_features_source_type="handcoded_scipy",
            coco_source_backend="scipy.stats.kurtosis",
            mne_features_source_backend="scipy.stats.kurtosis",
            semantic_notes="Direct overlap once fisher/bias conventions are aligned.",
            benchmarkable_now=True,
            future_usefulness="medium",
            recommended_future_action="use_as_reference",
            coco_review=_make_review(
                feature_name="kurtosis",
                library="coco-pipe",
                scientific_maturity="medium",
                implementation_maturity="high",
                parameter_sensitivity="low",
                backend_risk="low",
                numerical_risk="medium",
                interpretability="medium",
                match=(
                    "Outer CV strategy is group-based, but tuning.cv strategy "
                    "'stratified' is not"
                ),
            ),
            mne_review=_make_review(
                feature_name="kurtosis",
                library="mne-features",
                scientific_maturity="medium",
                implementation_maturity="high",
                parameter_sensitivity="low",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Same core backend; mostly useful as a validation reference.",
            ),
        ),
        CoverageRecord(
            domain="utility/amplitude",
            feature_name_coco_pipe="rms",
            feature_name_mne_features="rms",
            relationship="exact_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="numpy_scipy_custom",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="numpy sqrt(mean(x**2))",
            mne_features_source_backend="custom numpy formula",
            semantic_notes="Direct overlap.",
            benchmarkable_now=True,
            future_usefulness="low",
            recommended_future_action="use_as_reference",
            coco_review=_make_review(
                feature_name="rms",
                library="coco-pipe",
                scientific_maturity="high",
                implementation_maturity="high",
                parameter_sensitivity="low",
                backend_risk="low",
                numerical_risk="low",
                interpretability="high",
                review_notes="Stable amplitude descriptor, though largely redundant with existing spectral magnitude summaries.",
            ),
            mne_review=_make_review(
                feature_name="rms",
                library="mne-features",
                scientific_maturity="high",
                implementation_maturity="high",
                parameter_sensitivity="low",
                backend_risk="medium",
                numerical_risk="low",
                interpretability="high",
                review_notes="Simple reference implementation; not a strong reason to switch runtime libraries.",
            ),
        ),
        CoverageRecord(
            domain="spectral",
            feature_name_coco_pipe="absolute_power",
            feature_name_mne_features="pow_freq_bands",
            relationship="approx_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="project_specific_logic",
            mne_features_source_type="mne_wrapper",
            coco_source_backend="mne PSD utilities + numpy band integration",
            mne_features_source_backend="mne_features.utils.power_spectrum + numpy band sums",
            semantic_notes="Comparable when bands, PSD method, "
            "normalization, and log settings are aligned.",
            benchmarkable_now=True,
            future_usefulness="high",
            recommended_future_action="keep_native",
            coco_review=_make_review(
                feature_name="absolute_power",
                library="coco-pipe",
                scientific_maturity="high",
                implementation_maturity="high",
                parameter_sensitivity="medium",
                backend_risk="medium",
                numerical_risk="low",
                interpretability="high",
                review_notes="Strong native implementation with explicit band config and reuse across outputs.",
            ),
            mne_review=_make_review(
                feature_name="pow_freq_bands",
                library="mne-features",
                scientific_maturity="high",
                implementation_maturity="medium",
                parameter_sensitivity="medium",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="high",
                review_notes="Broad spectral utility, but less tailored to corrected or planner-shared workflows.",
            ),
        ),
        CoverageRecord(
            domain="spectral",
            feature_name_coco_pipe="log_absolute_power",
            feature_name_mne_features="pow_freq_bands",
            relationship="approx_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="project_specific_logic",
            mne_features_source_type="mne_wrapper",
            coco_source_backend="absolute band power + numpy log10 transform",
            mne_features_source_backend="pow_freq_bands(log=True)",
            semantic_notes="Comparable after matching PSD and band settings; "
            "mne-features uses log-ratio semantics inside the same utility.",
            benchmarkable_now=True,
            future_usefulness="high",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="spectral",
            feature_name_coco_pipe="relative_power",
            feature_name_mne_features="pow_freq_bands",
            relationship="approx_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="project_specific_logic",
            mne_features_source_type="mne_wrapper",
            coco_source_backend="absolute band power / total PSD power",
            mne_features_source_backend="pow_freq_bands(normalize=True)",
            semantic_notes="Close overlap if the same "
            "PSD range is used for normalization.",
            benchmarkable_now=True,
            future_usefulness="high",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="spectral",
            feature_name_coco_pipe="ratios",
            feature_name_mne_features="pow_freq_bands",
            relationship="approx_overlap",
            implemented_in_coco=True,
            implemented_in_mne_features=True,
            coco_source_type="project_specific_logic",
            mne_features_source_type="mne_wrapper",
            coco_source_backend="explicit ratio_pairs on absolute band powers",
            mne_features_source_backend="pow_freq_bands(ratios='all'|'only')",
            semantic_notes="Conceptually similar, but coco-pipe exposes curated "
            "explicit pairs while mne-features generates combinatorial ratios.",
            benchmarkable_now=True,
            future_usefulness="medium",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="spectral",
            feature_name_coco_pipe="corrected_absolute_power",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="project_specific_logic",
            mne_features_source_type="",
            coco_source_backend="specparam periodic PSD + numpy band integration",
            mne_features_source_backend="",
            semantic_notes="Periodic-only spectral power has "
            "no direct counterpart in mne-features.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="keep_native",
            coco_review=_make_review(
                feature_name="corrected_absolute_power",
                library="coco-pipe",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="high",
                backend_risk="medium",
                numerical_risk="high",
                interpretability="medium",
                review_notes="Scientifically useful but dependent on specparam fit quality and periodic residual stability.",
            ),
        ),
        CoverageRecord(
            domain="spectral",
            feature_name_coco_pipe="corrected_log_absolute_power",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="project_specific_logic",
            mne_features_source_type="",
            coco_source_backend="corrected absolute power + numpy log10",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="keep_native",
            coco_review=_make_review(
                feature_name="corrected_log_absolute_power",
                library="coco-pipe",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="high",
                backend_risk="medium",
                numerical_risk="high",
                interpretability="medium",
                review_notes="Often the most analysis-friendly corrected spectral summary, but still inherits periodic-fit fragility.",
            ),
        ),
        CoverageRecord(
            domain="spectral",
            feature_name_coco_pipe="corrected_relative_power",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="project_specific_logic",
            mne_features_source_type="",
            coco_source_backend="corrected band power / corrected total power",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="keep_native",
            coco_review=_make_review(
                feature_name="corrected_relative_power",
                library="coco-pipe",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="high",
                backend_risk="medium",
                numerical_risk="high",
                interpretability="medium",
                review_notes="Potentially informative but especially sensitive to tiny corrected denominators.",
            ),
        ),
        CoverageRecord(
            domain="spectral",
            feature_name_coco_pipe="corrected_ratios",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="project_specific_logic",
            mne_features_source_type="",
            coco_source_backend="explicit ratios on corrected band powers",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="low",
            recommended_future_action="keep_native",
            coco_review=_make_review(
                feature_name="corrected_ratios",
                library="coco-pipe",
                scientific_maturity="low",
                implementation_maturity="medium",
                parameter_sensitivity="high",
                backend_risk="medium",
                numerical_risk="high",
                interpretability="low",
                review_notes="Scientifically fragile because corrected denominators can be tiny or absent.",
            ),
        ),
        CoverageRecord(
            domain="parametric",
            feature_name_coco_pipe="offset",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="specparam_wrapper",
            mne_features_source_type="",
            coco_source_backend="specparam aperiodic fit",
            mne_features_source_backend="",
            semantic_notes="No direct parametric aperiodic modeling in mne-features.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="keep_native",
            coco_review=_make_review(
                feature_name="offset",
                library="coco-pipe",
                scientific_maturity="high",
                implementation_maturity="high",
                parameter_sensitivity="medium",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="high",
                review_notes="Well-established aperiodic summary that is central to the package's domain-specific value.",
            ),
        ),
        CoverageRecord(
            domain="parametric",
            feature_name_coco_pipe="exponent",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="specparam_wrapper",
            mne_features_source_type="",
            coco_source_backend="specparam aperiodic fit",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="keep_native",
            coco_review=_make_review(
                feature_name="exponent",
                library="coco-pipe",
                scientific_maturity="high",
                implementation_maturity="high",
                parameter_sensitivity="medium",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="high",
                review_notes="Core EEG aperiodic descriptor and a major reason not to replace the runtime with mne-features.",
            ),
        ),
        CoverageRecord(
            domain="parametric",
            feature_name_coco_pipe="knee",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="specparam_wrapper",
            mne_features_source_type="",
            coco_source_backend="specparam knee mode",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="parametric",
            feature_name_coco_pipe="fit_error",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="specparam_wrapper",
            mne_features_source_type="",
            coco_source_backend="specparam model diagnostics",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="parametric",
            feature_name_coco_pipe="r_squared",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="specparam_wrapper",
            mne_features_source_type="",
            coco_source_backend="specparam model diagnostics",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="parametric",
            feature_name_coco_pipe="peak_count",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="specparam_wrapper",
            mne_features_source_type="",
            coco_source_backend="specparam periodic peaks",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="parametric",
            feature_name_coco_pipe="peak_freq_dom",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="specparam_wrapper",
            mne_features_source_type="",
            coco_source_backend="specparam periodic peaks",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="parametric",
            feature_name_coco_pipe="peak_power_dom",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="specparam_wrapper",
            mne_features_source_type="",
            coco_source_backend="specparam periodic peaks",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="parametric",
            feature_name_coco_pipe="peak_bandwidth_dom",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="specparam_wrapper",
            mne_features_source_type="",
            coco_source_backend="specparam periodic peaks",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="parametric",
            feature_name_coco_pipe="alpha_peak_freq",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="project_specific_logic",
            mne_features_source_type="",
            coco_source_backend="specparam periodic peaks + alpha-band selection",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="parametric",
            feature_name_coco_pipe="alpha_peak_power",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="project_specific_logic",
            mne_features_source_type="",
            coco_source_backend="specparam periodic peaks + alpha-band selection",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="perm_entropy",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="backend_wrapper",
            mne_features_source_type="",
            coco_source_backend="antropy.perm_entropy",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="keep_native",
            coco_review=_make_review(
                feature_name="perm_entropy",
                library="coco-pipe",
                scientific_maturity="high",
                implementation_maturity="high",
                parameter_sensitivity="medium",
                backend_risk="low",
                numerical_risk="low",
                interpretability="medium",
                review_notes="Widely used entropy measure already well supported natively.",
            ),
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="petrosian_fd",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="backend_wrapper",
            mne_features_source_type="",
            coco_source_backend="antropy.petrosian_fd",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="shannon_entropy",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="backend_wrapper",
            mne_features_source_type="",
            coco_source_backend="neurokit2.entropy_shannon",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="fuzzy_entropy",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="backend_wrapper",
            mne_features_source_type="",
            coco_source_backend="neurokit2.entropy_fuzzy",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="dispersion_entropy",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="backend_wrapper",
            mne_features_source_type="",
            coco_source_backend="neurokit2.entropy_dispersion",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="lziv_complexity",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="backend_wrapper",
            mne_features_source_type="",
            coco_source_backend="antropy.lziv_complexity",
            mne_features_source_backend="",
            semantic_notes="No direct mne-features equivalent.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="utility/amplitude",
            feature_name_coco_pipe="pool_channels",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="project_specific_logic",
            mne_features_source_type="",
            coco_source_backend="DescriptorPipeline.pool_channels",
            mne_features_source_backend="",
            semantic_notes="Workflow utility for grouped-channel derivation "
            "rather than a feature formula.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="utility/amplitude",
            feature_name_coco_pipe="shared_psd_planner",
            feature_name_mne_features="",
            relationship="coco_only",
            implemented_in_coco=True,
            implemented_in_mne_features=False,
            coco_source_type="project_specific_logic",
            mne_features_source_type="",
            coco_source_backend="DescriptorPipeline planner and PSD reuse groups",
            mne_features_source_backend="",
            semantic_notes="Execution architecture, not a single descriptor formula.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="keep_native",
        ),
        CoverageRecord(
            domain="utility/amplitude",
            feature_name_coco_pipe="",
            feature_name_mne_features="mean",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="",
            mne_features_source_backend="numpy mean",
            semantic_notes="Basic summary statistic not exposed in coco descriptors.",
            benchmarkable_now=False,
            future_usefulness="low",
            recommended_future_action="ignore_for_now",
            mne_review=_make_review(
                feature_name="mean",
                library="mne-features",
                scientific_maturity="high",
                implementation_maturity="high",
                parameter_sensitivity="low",
                backend_risk="low",
                numerical_risk="low",
                interpretability="high",
                review_notes="Technically stable but not compelling as a dedicated EEG descriptor for this project.",
            ),
        ),
        CoverageRecord(
            domain="utility/amplitude",
            feature_name_coco_pipe="",
            feature_name_mne_features="variance",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="",
            mne_features_source_backend="numpy variance",
            semantic_notes="Basic dispersion statistic only.",
            benchmarkable_now=False,
            future_usefulness="low",
            recommended_future_action="ignore_for_now",
        ),
        CoverageRecord(
            domain="utility/amplitude",
            feature_name_coco_pipe="",
            feature_name_mne_features="std",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="",
            mne_features_source_backend="numpy standard deviation",
            semantic_notes="Basic amplitude variability only.",
            benchmarkable_now=False,
            future_usefulness="low",
            recommended_future_action="ignore_for_now",
        ),
        CoverageRecord(
            domain="utility/amplitude",
            feature_name_coco_pipe="",
            feature_name_mne_features="ptp_amp",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="",
            mne_features_source_backend="numpy peak-to-peak amplitude",
            semantic_notes="Classical amplitude spread metric.",
            benchmarkable_now=False,
            future_usefulness="low",
            recommended_future_action="ignore_for_now",
        ),
        CoverageRecord(
            domain="utility/amplitude",
            feature_name_coco_pipe="",
            feature_name_mne_features="skewness",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_scipy",
            coco_source_backend="",
            mne_features_source_backend="scipy.stats.skew",
            semantic_notes="Useful amplitude-shape descriptor that could "
            "be added later with little ambiguity.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="borrow_formula",
            mne_review=_make_review(
                feature_name="skewness",
                library="mne-features",
                scientific_maturity="medium",
                implementation_maturity="high",
                parameter_sensitivity="low",
                backend_risk="low",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Low-complexity candidate for future addition; artifact-sensitive but simple and stable.",
            ),
        ),
        CoverageRecord(
            domain="utility/amplitude",
            feature_name_coco_pipe="",
            feature_name_mne_features="quantile",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="",
            mne_features_source_backend="numpy quantile",
            semantic_notes="General amplitude summary with tunable q parameter.",
            benchmarkable_now=False,
            future_usefulness="low",
            recommended_future_action="ignore_for_now",
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="",
            feature_name_mne_features="decorr_time",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="",
            mne_features_source_backend="FFT-based unbiased autocorrelation",
            semantic_notes="Potentially useful classical temporal dependence measure.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="borrow_formula",
            mne_review=_make_review(
                feature_name="decorr_time",
                library="mne-features",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="medium",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Interesting classical feature, but not core to the current study needs.",
            ),
        ),
        CoverageRecord(
            domain="spectral",
            feature_name_coco_pipe="",
            feature_name_mne_features="hjorth_mobility_spect",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="mne_wrapper",
            coco_source_backend="",
            mne_features_source_backend="PSD-based spectral Hjorth mobility",
            semantic_notes="Spectral variant, not the "
            "same as time-domain Hjorth mobility.",
            benchmarkable_now=False,
            future_usefulness="low",
            recommended_future_action="ignore_for_now",
        ),
        CoverageRecord(
            domain="spectral",
            feature_name_coco_pipe="",
            feature_name_mne_features="hjorth_complexity_spect",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="mne_wrapper",
            coco_source_backend="",
            mne_features_source_backend="PSD-based spectral Hjorth complexity",
            semantic_notes="Spectral variant, not present in coco descriptors.",
            benchmarkable_now=False,
            future_usefulness="low",
            recommended_future_action="ignore_for_now",
        ),
        CoverageRecord(
            domain="utility/amplitude",
            feature_name_coco_pipe="",
            feature_name_mne_features="line_length",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="",
            mne_features_source_backend="custom numpy line-length formula",
            semantic_notes="Simple and often useful EEG seizure/roughness feature.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="borrow_formula",
            mne_review=_make_review(
                feature_name="line_length",
                library="mne-features",
                scientific_maturity="high",
                implementation_maturity="high",
                parameter_sensitivity="low",
                backend_risk="low",
                numerical_risk="low",
                interpretability="high",
                review_notes="Strong candidate for future native addition because it is simple, stable, and widely understood.",
            ),
        ),
        CoverageRecord(
            domain="spectral",
            feature_name_coco_pipe="",
            feature_name_mne_features="spect_slope",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_scipy",
            coco_source_backend="",
            mne_features_source_backend="PSD + linear regression",
            semantic_notes="Different from specparam exponent; "
            "simpler classical spectral slope estimate.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="borrow_formula",
            mne_review=_make_review(
                feature_name="spect_slope",
                library="mne-features",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="high",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Could be useful as a simpler spectral utility, but should not replace specparam exponent.",
            ),
        ),
        CoverageRecord(
            domain="complexity",
            feature_name_coco_pipe="",
            feature_name_mne_features="svd_fisher_info",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_numpy",
            coco_source_backend="",
            mne_features_source_backend="custom embedding/SVD formulation",
            semantic_notes="Advanced classical complexity utility "
            "without current coco counterpart.",
            benchmarkable_now=False,
            future_usefulness="low",
            recommended_future_action="ignore_for_now",
        ),
        CoverageRecord(
            domain="spectral",
            feature_name_coco_pipe="",
            feature_name_mne_features="energy_freq_bands",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="mne_wrapper",
            coco_source_backend="",
            mne_features_source_backend="MNE filtering + band energy",
            semantic_notes="Energy-by-band metric distinct from PSD-integrated power.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="borrow_formula",
            mne_review=_make_review(
                feature_name="energy_freq_bands",
                library="mne-features",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="medium",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Potentially useful but overlaps conceptually with spectral power summaries.",
            ),
        ),
        CoverageRecord(
            domain="spectral",
            feature_name_coco_pipe="",
            feature_name_mne_features="spect_edge_freq",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="mne_wrapper",
            coco_source_backend="",
            mne_features_source_backend="PSD cumulative energy edge calculation",
            semantic_notes="Classical EEG summary that could "
            "be added later with clear semantics.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="borrow_formula",
            mne_review=_make_review(
                feature_name="spect_edge_freq",
                library="mne-features",
                scientific_maturity="high",
                implementation_maturity="medium",
                parameter_sensitivity="medium",
                backend_risk="medium",
                numerical_risk="low",
                interpretability="high",
                review_notes="Good future candidate because it is interpretable and common in EEG summaries.",
            ),
        ),
        CoverageRecord(
            domain="wavelet",
            feature_name_coco_pipe="",
            feature_name_mne_features="wavelet_coef_energy",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="pywt_wrapper",
            coco_source_backend="",
            mne_features_source_backend="pywt wavelet coefficients",
            semantic_notes="Would belong to the planned wavelet "
            "track, not the current complexity family.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="sidecar_only",
            mne_review=_make_review(
                feature_name="wavelet_coef_energy",
                library="mne-features",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="high",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Useful only within a deliberate wavelet-oriented design, not as an ad hoc scalar add-on.",
            ),
        ),
        CoverageRecord(
            domain="wavelet",
            feature_name_coco_pipe="",
            feature_name_mne_features="teager_kaiser_energy",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="pywt_wrapper",
            coco_source_backend="",
            mne_features_source_backend="wavelet-assisted Teager-Kaiser energy",
            semantic_notes="Wavelet-adjacent energy feature better treated "
            "as a separate future track.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="sidecar_only",
            mne_review=_make_review(
                feature_name="teager_kaiser_energy",
                library="mne-features",
                scientific_maturity="medium",
                implementation_maturity="medium",
                parameter_sensitivity="high",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Potentially useful, but not something to fold into the current descriptor family without a wavelet design.",
            ),
        ),
        CoverageRecord(
            domain="bivariate",
            feature_name_coco_pipe="",
            feature_name_mne_features="phase_lock_val",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_scipy",
            coco_source_backend="",
            mne_features_source_backend="scipy.signal.hilbert + custom PLV formula",
            semantic_notes="Potential future sidecar or bivariate family, "
            "but not part of current descriptor runtime.",
            benchmarkable_now=False,
            future_usefulness="high",
            recommended_future_action="sidecar_only",
            mne_review=_make_review(
                feature_name="phase_lock_val",
                library="mne-features",
                scientific_maturity="high",
                implementation_maturity="medium",
                parameter_sensitivity="medium",
                backend_risk="medium",
                numerical_risk="medium",
                interpretability="medium",
                review_notes="Scientifically useful, but for long-term connectivity work `mne-connectivity` is likely the better backend.",
            ),
        ),
        CoverageRecord(
            domain="bivariate",
            feature_name_coco_pipe="",
            feature_name_mne_features="max_cross_corr",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_numba",
            coco_source_backend="",
            mne_features_source_backend="custom numba cross-correlation scan",
            semantic_notes="Classical pairwise feature candidate "
            "for a future bivariate sidecar.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="sidecar_only",
        ),
        CoverageRecord(
            domain="bivariate",
            feature_name_coco_pipe="",
            feature_name_mne_features="time_corr",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_scipy",
            coco_source_backend="",
            mne_features_source_backend="correlation matrix + eigenvalue summaries",
            semantic_notes="Interesting classical pairwise summary, "
            "but not in current scope.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="sidecar_only",
        ),
        CoverageRecord(
            domain="bivariate",
            feature_name_coco_pipe="",
            feature_name_mne_features="spect_corr",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="mne_wrapper",
            coco_source_backend="",
            mne_features_source_backend="PSD-based pairwise spectral correlation",
            semantic_notes="Potential future bivariate utility, "
            "distinct from corrected spectral modeling.",
            benchmarkable_now=False,
            future_usefulness="medium",
            recommended_future_action="sidecar_only",
        ),
        CoverageRecord(
            domain="bivariate",
            feature_name_coco_pipe="",
            feature_name_mne_features="nonlin_interdep",
            relationship="mne_only",
            implemented_in_coco=False,
            implemented_in_mne_features=True,
            coco_source_type="",
            mne_features_source_type="handcoded_scipy",
            coco_source_backend="",
            mne_features_source_backend="embedding + nearest-neighbor distances",
            semantic_notes="More specialized bivariate measure "
            "with limited immediate use.",
            benchmarkable_now=False,
            future_usefulness="low",
            recommended_future_action="ignore_for_now",
        ),
    ]
    return _populate_missing_reviews(records)


def build_validity_review_rows(records: list[CoverageRecord]) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        for review in (record.coco_review, record.mne_review):
            if review is not None:
                key = (review.feature_name, review.library)
                rows_by_key.setdefault(key, asdict(review))
    rows = list(rows_by_key.values())
    rows.sort(key=lambda row: (row["feature_name"], row["library"]))
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def build_benchmark_cases(
    *,
    profile: str,
    real_data_npz: str | None,
) -> list[BenchmarkCase]:
    sizes = {
        "small": (32, 19, 256),
        "medium": (64, 19, 512),
        "large": (128, 19, 1024),
        "high_channel": (64, 64, 512),
    }
    if profile == "quick":
        size_keys = ["medium"]
        overlap_subsets = ["overlap_combined"]
        native_subsets = [
            "coco_pipe_current_study_like",
            "mne_features_univariate_broad",
        ]
        n_jobs_values = [1, 8]
    else:
        size_keys = ["small", "medium", "large", "high_channel"]
        overlap_subsets = [
            "overlap_complexity_exact",
            "overlap_spectral_exact_or_nearest",
            "overlap_combined",
        ]
        native_subsets = [
            "coco_pipe_current_study_like",
            "mne_features_univariate_broad",
            "mne_features_bivariate_small",
        ]
        n_jobs_values = [1, 2, 4, 8]

    cases: list[BenchmarkCase] = []
    for subset in overlap_subsets + native_subsets:
        libraries = ["coco-pipe", "mne-features"]
        if subset == "coco_pipe_current_study_like":
            libraries = ["coco-pipe"]
        elif subset.startswith("mne_features_"):
            libraries = ["mne-features"]

        for size_key in size_keys:
            n_epochs, n_channels, n_times = sizes[size_key]
            for library in libraries:
                for n_jobs in n_jobs_values:
                    parallel_mode = "sequential" if n_jobs == 1 else "default_parallel"
                    cases.append(
                        BenchmarkCase(
                            case_id=f"{subset}__{size_key}__{library}__nj{n_jobs}",
                            dataset_kind=size_key,
                            feature_subset=subset,
                            library=library,
                            n_epochs=n_epochs,
                            n_channels=n_channels,
                            n_times=n_times,
                            n_jobs=n_jobs,
                            parallel_mode=parallel_mode,
                        )
                    )

    if real_data_npz:
        real_subsets = [
            "overlap_complexity_exact",
            "overlap_spectral_exact_or_nearest",
            "overlap_combined",
            "coco_pipe_current_study_like",
            "mne_features_univariate_broad",
            "mne_features_bivariate_small",
        ]
        for subset in real_subsets:
            libraries = ["coco-pipe", "mne-features"]
            if subset == "coco_pipe_current_study_like":
                libraries = ["coco-pipe"]
            elif subset.startswith("mne_features_"):
                libraries = ["mne-features"]

            for library in libraries:
                for n_jobs in (1, 8) if profile == "quick" else (1, 2, 4, 8):
                    parallel_mode = "sequential" if n_jobs == 1 else "default_parallel"
                    cases.append(
                        BenchmarkCase(
                            case_id=f"{subset}__real_micro__{library}__nj{n_jobs}",
                            dataset_kind="real_micro",
                            feature_subset=subset,
                            library=library,
                            n_epochs=0,
                            n_channels=0,
                            n_times=0,
                            n_jobs=n_jobs,
                            parallel_mode=parallel_mode,
                            real_data_npz=real_data_npz,
                        )
                    )
    return cases


def _default_environment(*, temp_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    home = temp_root / "home"
    mpl = temp_root / "mpl"
    home.mkdir(parents=True, exist_ok=True)
    mpl.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(home)
    env["MPLCONFIGDIR"] = str(mpl)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _run_subprocess_json(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    stdout = completed.stdout.strip()
    return json.loads(stdout)


def _build_real_case_npz(
    *,
    coco_python: Path,
    real_data_bids_root: Path,
    temp_root: Path,
) -> Path | None:
    if not real_data_bids_root.exists():
        return None

    output_npz = temp_root / "real_micro_case.npz"
    payload = {
        "bids_root": str(real_data_bids_root),
        "subject": "001",
        "session": "01",
        "task": "motorimagery",
        "run": "03",
        "duration": 2.0,
        "max_epochs": 12,
        "max_channels": 19,
        "output_npz": str(output_npz),
    }
    payload_path = temp_root / "real_case_payload.json"
    _write_json(payload, payload_path)

    command = [
        str(coco_python),
        str(Path(__file__).resolve()),
        "--internal-build-real-case",
        str(payload_path),
    ]
    env = _default_environment(temp_root=temp_root / "real_case_env")
    try:
        _run_subprocess_json(command, env=env, cwd=REPO_ROOT)
    except RuntimeError:
        return None
    return output_npz if output_npz.exists() else None


def run_benchmarks(
    *,
    output_dir: Path,
    profile: str,
    coco_python: Path,
    mne_features_python: Path,
    real_data_bids_root: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="descriptor_compare_") as tmp_dir:
        temp_root = Path(tmp_dir)
        real_npz = _build_real_case_npz(
            coco_python=coco_python,
            real_data_bids_root=real_data_bids_root,
            temp_root=temp_root,
        )
        cases = build_benchmark_cases(
            profile=profile,
            real_data_npz=str(real_npz) if real_npz else None,
        )
        rows: list[dict[str, Any]] = []
        for case in cases:
            python_bin = (
                coco_python if case.library == "coco-pipe" else mne_features_python
            )
            payload_path = temp_root / f"{case.case_id}.json"
            _write_json(case.to_payload(), payload_path)
            env = _default_environment(temp_root=temp_root / case.case_id)
            command = [
                str(python_bin),
                str(Path(__file__).resolve()),
                "--internal-benchmark",
                str(payload_path),
            ]
            try:
                row = _run_subprocess_json(command, env=env, cwd=REPO_ROOT)
            except RuntimeError as error:
                row = {
                    "case_id": case.case_id,
                    "library": case.library,
                    "dataset_kind": case.dataset_kind,
                    "n_epochs": case.n_epochs,
                    "n_channels": case.n_channels,
                    "n_times": case.n_times,
                    "feature_subset": case.feature_subset,
                    "n_features_out": None,
                    "python_version": None,
                    "library_versions": None,
                    "cold_import_s": None,
                    "cold_extract_s": None,
                    "warm_extract_s": None,
                    "warm_extract_median_s": None,
                    "peak_memory_mb": None,
                    "parallel_mode": case.parallel_mode,
                    "n_jobs": case.n_jobs,
                    "environment_notes": str(error),
                }
            rows.append(row)
        return rows


def render_recommendation(
    *,
    coverage_rows: list[CoverageRecord],
    benchmark_rows: list[dict[str, Any]],
) -> str:
    exact_overlap = [
        row for row in coverage_rows if row.relationship == "exact_overlap"
    ]
    borrow_formula = [
        row
        for row in coverage_rows
        if row.recommended_future_action == "borrow_formula"
    ]
    keep_native = [
        row for row in coverage_rows if row.recommended_future_action == "keep_native"
    ]
    sidecar_only = [
        row for row in coverage_rows if row.recommended_future_action == "sidecar_only"
    ]
    oracle = [
        row
        for row in coverage_rows
        if row.recommended_future_action == "use_as_reference"
    ]

    warm_rows = [
        row
        for row in benchmark_rows
        if row.get("dataset_kind") == "medium"
        and row.get("feature_subset") == "overlap_combined"
        and row.get("n_jobs") == 1
        and row.get("warm_extract_s") is not None
    ]
    warm_rows.sort(key=lambda row: row["library"])
    timing_lines = []
    for row in warm_rows:
        timing_lines.append(
            f"- {row['library']}: warm extract {row['warm_extract_s']} s for "
            f"{row['n_features_out']} features ({row['dataset_kind']}, {row['feature_subset']})"
        )

    return (
        textwrap.dedent(
            f"""\
        # Recommendation Memo: `mne-features` vs `coco-pipe.descriptors`

        ## High-level conclusion
        Keep `coco-pipe.descriptors` as the primary runtime. Use `mne-features`
        as a reference/oracle for overlapping classical descriptors and as a
        source of future feature ideas, not as a wholesale replacement.

        ## What `coco-pipe` does better
        - Corrected spectral outputs and corrected ratios/log power.
        - Parametric/aperiodic spectral modeling through `specparam`.
        - Explicit descriptor family config and runtime planner.
        - Shared PSD reuse and pooled-channel derivation.
        - Structured failure collection that fits the current study workflow.

        ## What `mne-features` does better
        - Broader catalog of classical univariate and bivariate EEG features.
        - Independent implementations for many overlap features, useful as
          validation or regression oracles.
        - Several straightforward future candidates:
        {chr(10).join(
                f"- {row.feature_name_mne_features}" for row in borrow_formula[:8]
            )}

        ## Which overlap implementations look stronger
        - Prefer `coco-pipe` as the runtime path for overlap features that are
          already present and integrated.
        - Use `mne-features` as a comparison/reference path for classical
          overlaps such as:
        {chr(10).join(
                f"- {row.feature_name_coco_pipe} <-> {row.feature_name_mne_features}"
                for row in exact_overlap[:8]
            )}

        ## Runtime notes from the current benchmark harness
        {
                chr(10).join(timing_lines)
                if timing_lines
                else "- No medium overlap timing rows were produced in this run."
            }

        ## Features/behaviors to keep native in `coco-pipe`
        {chr(10).join(f"- {row.feature_name_coco_pipe}" for row in keep_native[:12])}

        ## Good future candidates inspired by `mne-features`
        {chr(10).join(
                f"- {row.feature_name_mne_features}" for row in borrow_formula[:12]
            )}

        ## Best use of `mne-features` going forward
        - Validation oracle for overlap features:
        {chr(10).join(
                f"- {row.feature_name_coco_pipe} <-> {row.feature_name_mne_features}"
                for row in oracle[:10]
            )}
        - Sidecar-only candidates for future separate design:
        {chr(10).join(
                f"- {row.feature_name_mne_features or row.feature_name_coco_pipe}"
                for row in sidecar_only[:10]
            )}
        """
        ).strip()
        + "\n"
    )


def _library_versions(names: list[str]) -> str:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return json.dumps(versions, sort_keys=True)


def _build_signal_array(n_epochs: int, n_channels: int, n_times: int) -> Any:
    import numpy as np

    rng = np.random.default_rng(0)
    time = np.linspace(0.0, 1.0, n_times, endpoint=False)
    X = rng.normal(scale=0.4, size=(n_epochs, n_channels, n_times))
    for channel in range(n_channels):
        alpha = np.sin(2 * np.pi * (8.0 + 0.1 * channel) * time)
        beta = 0.5 * np.sin(2 * np.pi * (18.0 + 0.05 * channel) * time + 0.2)
        theta = 0.3 * np.sin(2 * np.pi * 5.0 * time + 0.1 * channel)
        X[:, channel, :] += alpha + beta + theta
    return X.astype(float)


def _load_benchmark_data(payload: dict[str, Any]) -> tuple[Any, float, list[str], str]:
    import numpy as np

    if payload["dataset_kind"] == "real_micro":
        archive = np.load(payload["real_data_npz"], allow_pickle=True)
        X = archive["X"]
        sfreq = float(archive["sfreq"])
        channel_names = [str(name) for name in archive["channel_names"].tolist()]
        return X, sfreq, channel_names, archive["source_description"].item()

    X = _build_signal_array(
        payload["n_epochs"], payload["n_channels"], payload["n_times"]
    )
    channel_names = [f"C{idx:02d}" for idx in range(X.shape[1])]
    return X, 128.0, channel_names, "synthetic deterministic EEG-like signal"


def _peak_memory_mb() -> float:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(float(rss) / (1024 * 1024), 3)
    return round(float(rss) / 1024, 3)


def _coco_config_for_subset(subset: str) -> dict[str, Any]:
    exact_overlap_measures = [
        "sample_entropy",
        "approx_entropy",
        "svd_entropy",
        "higuchi_fd",
        "katz_fd",
        "hjorth_mobility",
        "hjorth_complexity",
        "zero_crossings",
        "kurtosis",
        "rms",
    ]
    current_study_measures = [
        "sample_entropy",
        "perm_entropy",
        "spectral_entropy",
        "hjorth_mobility",
        "hjorth_complexity",
        "lziv_complexity",
        "higuchi_fd",
        "petrosian_fd",
        "svd_entropy",
        "fuzzy_entropy",
        "hurst_exponent",
        "dispersion_entropy",
        "kurtosis",
        "zero_crossings",
    ]

    base = {
        "precision": "float32",
        "families": {
            "bands": {"enabled": False},
            "parametric": {"enabled": False},
            "complexity": {"enabled": False},
        },
        "runtime": {
            "execution_backend": "sequential",
            "n_jobs": 1,
            "obs_chunk": 128,
            "on_error": "raise",
        },
    }

    if subset == "overlap_complexity_exact":
        base["families"]["complexity"] = {
            "enabled": True,
            "backend": "auto",
            "measures": exact_overlap_measures,
        }
    elif subset == "overlap_spectral_exact_or_nearest":
        base["families"]["bands"] = {
            "enabled": True,
            "psd_method": "welch",
            "fmin": 1.0,
            "fmax": 45.0,
            "bands": {
                "delta": (1.0, 4.0),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 13.0),
                "beta": (13.0, 30.0),
                "gamma": (30.0, 45.0),
            },
            "outputs": ["absolute_power"],
        }
    elif subset == "overlap_combined":
        base["families"]["bands"] = {
            "enabled": True,
            "psd_method": "welch",
            "fmin": 1.0,
            "fmax": 45.0,
            "bands": {
                "delta": (1.0, 4.0),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 13.0),
                "beta": (13.0, 30.0),
                "gamma": (30.0, 45.0),
            },
            "outputs": ["absolute_power"],
        }
        base["families"]["complexity"] = {
            "enabled": True,
            "backend": "auto",
            "measures": exact_overlap_measures + ["spectral_entropy", "hurst_exponent"],
        }
    elif subset == "coco_pipe_current_study_like":
        base["families"]["bands"] = {
            "enabled": True,
            "psd_method": "welch",
            "fmin": 1.0,
            "fmax": 45.0,
            "outputs": [
                "absolute_power",
                "log_absolute_power",
                "relative_power",
                "corrected_absolute_power",
                "corrected_log_absolute_power",
                "corrected_relative_power",
            ],
            "bands": {
                "delta": (1.0, 4.0),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 13.0),
                "beta": (13.0, 30.0),
                "gamma": (30.0, 45.0),
            },
        }
        base["families"]["parametric"] = {
            "enabled": True,
            "psd_method": "welch",
            "freq_range": (1.0, 45.0),
            "outputs": ["aperiodic", "fit_quality", "peak_summary"],
        }
        base["families"]["complexity"] = {
            "enabled": True,
            "backend": "auto",
            "measures": current_study_measures,
        }
    else:
        raise ValueError(f"Unsupported coco subset: {subset}")
    return base


def _mne_feature_subset(subset: str) -> tuple[list[str], dict[str, Any]]:
    import numpy as np

    exact_overlap = [
        "samp_entropy",
        "app_entropy",
        "svd_entropy",
        "higuchi_fd",
        "katz_fd",
        "hjorth_mobility",
        "hjorth_complexity",
        "zero_crossings",
        "kurtosis",
        "rms",
    ]
    base_params = {
        "pow_freq_bands__freq_bands": np.array([1.0, 4.0, 8.0, 13.0, 30.0, 45.0]),
        "pow_freq_bands__normalize": False,
        "pow_freq_bands__ratios": None,
        "pow_freq_bands__ratios_triu": False,
        "pow_freq_bands__log": False,
        "spect_entropy__psd_method": "welch",
        "spect_slope__psd_method": "welch",
        "spect_edge_freq__psd_method": "welch",
    }
    selected_funcs: list[str]
    if subset == "overlap_complexity_exact":
        selected_funcs = exact_overlap
    elif subset == "overlap_spectral_exact_or_nearest":
        selected_funcs = ["pow_freq_bands"]
    elif subset == "overlap_combined":
        selected_funcs = exact_overlap + [
            "pow_freq_bands",
            "spect_entropy",
            "hurst_exp",
        ]
    elif subset == "mne_features_univariate_broad":
        selected_funcs = [
            "mean",
            "variance",
            "std",
            "ptp_amp",
            "skewness",
            "kurtosis",
            "rms",
            "quantile",
            "hurst_exp",
            "app_entropy",
            "samp_entropy",
            "decorr_time",
            "pow_freq_bands",
            "hjorth_mobility_spect",
            "hjorth_complexity_spect",
            "hjorth_mobility",
            "hjorth_complexity",
            "higuchi_fd",
            "katz_fd",
            "zero_crossings",
            "line_length",
            "spect_entropy",
            "svd_entropy",
            "spect_slope",
            "svd_fisher_info",
            "energy_freq_bands",
            "spect_edge_freq",
            "wavelet_coef_energy",
            "teager_kaiser_energy",
        ]
    elif subset == "mne_features_bivariate_small":
        selected_funcs = [
            "phase_lock_val",
            "max_cross_corr",
            "time_corr",
            "spect_corr",
            "nonlin_interdep",
        ]
    else:
        raise ValueError(f"Unsupported mne-features subset: {subset}")

    funcs_params = {
        key: value
        for key, value in base_params.items()
        if key.split("__", 1)[0] in selected_funcs
    }
    return selected_funcs, funcs_params


def _benchmark_coco(payload: dict[str, Any]) -> dict[str, Any]:
    import time

    from coco_pipe.descriptors import DescriptorPipeline

    X, sfreq, channel_names, source_description = _load_benchmark_data(payload)
    config = _coco_config_for_subset(payload["feature_subset"])
    if payload["n_jobs"] > 1:
        config["runtime"]["execution_backend"] = "joblib"
        config["runtime"]["n_jobs"] = int(payload["n_jobs"])
        config["runtime"]["obs_chunk"] = 16

    import_start = time.perf_counter()
    pipe = DescriptorPipeline(config)
    cold_import_s = time.perf_counter() - import_start

    extract_start = time.perf_counter()
    first_result = pipe.extract(X=X, sfreq=sfreq, channel_names=channel_names)
    cold_extract_s = time.perf_counter() - extract_start

    warm_start = time.perf_counter()
    pipe.extract(X=X, sfreq=sfreq, channel_names=channel_names)
    warm_extract_s = time.perf_counter() - warm_start

    warm_runs: list[float] = []
    for _ in range(5):
        run_start = time.perf_counter()
        pipe.extract(X=X, sfreq=sfreq, channel_names=channel_names)
        warm_runs.append(time.perf_counter() - run_start)

    return {
        "case_id": payload["case_id"],
        "library": "coco-pipe",
        "dataset_kind": payload["dataset_kind"],
        "n_epochs": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "feature_subset": payload["feature_subset"],
        "n_features_out": len(first_result["descriptor_names"]),
        "python_version": sys.version.split()[0],
        "library_versions": _library_versions(
            ["coco-pipe", "antropy", "neurokit2", "specparam", "mne"]
        ),
        "cold_import_s": round(cold_import_s, 6),
        "cold_extract_s": round(cold_extract_s, 6),
        "warm_extract_s": round(warm_extract_s, 6),
        "warm_extract_median_s": round(statistics.median(warm_runs), 6),
        "peak_memory_mb": _peak_memory_mb(),
        "parallel_mode": payload["parallel_mode"],
        "n_jobs": payload["n_jobs"],
        "environment_notes": source_description,
    }


def _benchmark_mne_features(payload: dict[str, Any]) -> dict[str, Any]:
    import time

    X, sfreq, channel_names, source_description = _load_benchmark_data(payload)

    import_start = time.perf_counter()
    from mne_features.feature_extraction import extract_features

    cold_import_s = time.perf_counter() - import_start
    selected_funcs, funcs_params = _mne_feature_subset(payload["feature_subset"])
    n_jobs = int(payload["n_jobs"])

    cold_start = time.perf_counter()
    first_result = extract_features(
        X,
        sfreq=sfreq,
        selected_funcs=selected_funcs,
        funcs_params=funcs_params,
        n_jobs=n_jobs,
        ch_names=channel_names,
    )
    cold_extract_s = time.perf_counter() - cold_start

    warm_start = time.perf_counter()
    extract_features(
        X,
        sfreq=sfreq,
        selected_funcs=selected_funcs,
        funcs_params=funcs_params,
        n_jobs=n_jobs,
        ch_names=channel_names,
    )
    warm_extract_s = time.perf_counter() - warm_start

    warm_runs: list[float] = []
    for _ in range(5):
        run_start = time.perf_counter()
        extract_features(
            X,
            sfreq=sfreq,
            selected_funcs=selected_funcs,
            funcs_params=funcs_params,
            n_jobs=n_jobs,
            ch_names=channel_names,
        )
        warm_runs.append(time.perf_counter() - run_start)

    return {
        "case_id": payload["case_id"],
        "library": "mne-features",
        "dataset_kind": payload["dataset_kind"],
        "n_epochs": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "feature_subset": payload["feature_subset"],
        "n_features_out": int(first_result.shape[1]),
        "python_version": sys.version.split()[0],
        "library_versions": _library_versions(
            ["mne-features", "mne", "numpy", "scipy", "PyWavelets", "numba"]
        ),
        "cold_import_s": round(cold_import_s, 6),
        "cold_extract_s": round(cold_extract_s, 6),
        "warm_extract_s": round(warm_extract_s, 6),
        "warm_extract_median_s": round(statistics.median(warm_runs), 6),
        "peak_memory_mb": _peak_memory_mb(),
        "parallel_mode": payload["parallel_mode"],
        "n_jobs": n_jobs,
        "environment_notes": source_description,
    }


def _handle_internal_build_real_case(payload_path: Path) -> int:
    import mne
    import numpy as np

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    bids_root = Path(payload["bids_root"])
    eeg_dir = (
        bids_root / f"sub-{payload['subject']}" / f"ses-{payload['session']}" / "eeg"
    )
    vhdr_path = eeg_dir / (
        f"sub-{payload['subject']}_ses-{payload['session']}_task-"
        f"{payload['task']}_run-{payload['run']}_eeg.vhdr"
    )
    if not vhdr_path.exists():
        raise FileNotFoundError(f"Could not find BrainVision file: {vhdr_path}")

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
    raw.pick(["eeg"])
    if payload["max_channels"]:
        raw.pick(raw.ch_names[: int(payload["max_channels"])])
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=float(payload["duration"]),
        overlap=0.0,
        preload=True,
        verbose="ERROR",
    )
    X = epochs.get_data(copy=True)
    if payload["max_epochs"]:
        X = X[: int(payload["max_epochs"])]
    if X.size == 0:
        raise RuntimeError("Real-data micro-case produced no epochs.")

    output_npz = Path(payload["output_npz"])
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        X=X,
        sfreq=float(raw.info["sfreq"]),
        channel_names=np.asarray(raw.ch_names, dtype=object),
        source_description=np.asarray(str(vhdr_path), dtype=object),
    )
    print(json.dumps({"status": "ok", "output_npz": str(output_npz)}))
    return 0


def _handle_internal_benchmark(payload_path: Path) -> int:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    library = payload["library"]
    if library == "coco-pipe":
        row = _benchmark_coco(payload)
    elif library == "mne-features":
        row = _benchmark_mne_features(payload)
    else:
        raise ValueError(f"Unknown benchmark library: {library}")
    print(json.dumps(row))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare coco-pipe.descriptors against mne-features."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where comparison artifacts will be written.",
    )
    parser.add_argument(
        "--coco-python",
        type=Path,
        default=DEFAULT_COCO_PYTHON,
        help="Python executable for the coco-pipe benchmark environment.",
    )
    parser.add_argument(
        "--mne-features-python",
        type=Path,
        default=DEFAULT_MNE_FEATURES_PYTHON,
        help="Python executable for the mne-features benchmark environment.",
    )
    parser.add_argument(
        "--real-data-bids-root",
        type=Path,
        default=DEFAULT_REAL_DATA_BIDS_ROOT,
        help="Optional BIDS root used to build the real-data micro benchmark case.",
    )
    parser.add_argument(
        "--benchmark-profile",
        choices=("quick", "full"),
        default="full",
        help="Benchmark scope to execute.",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Only emit the coverage, validity, and recommendation artifacts.",
    )
    parser.add_argument(
        "--internal-build-real-case",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-benchmark",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.internal_build_real_case:
        return _handle_internal_build_real_case(args.internal_build_real_case)
    if args.internal_benchmark:
        return _handle_internal_benchmark(args.internal_benchmark)

    coverage_records = build_coverage_records()
    review_rows = build_validity_review_rows(coverage_records)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_rows = [record.to_row() for record in coverage_records]
    _write_csv(coverage_rows, output_dir / "coverage_provenance_matrix.csv")
    _write_json(coverage_rows, output_dir / "coverage_provenance_matrix.json")

    _write_csv(review_rows, output_dir / "validity_stability_review.csv")
    _write_json(review_rows, output_dir / "validity_stability_review.json")

    benchmark_rows: list[dict[str, Any]] = []
    if not args.skip_benchmarks:
        benchmark_rows = run_benchmarks(
            output_dir=output_dir,
            profile=args.benchmark_profile,
            coco_python=args.coco_python,
            mne_features_python=args.mne_features_python,
            real_data_bids_root=args.real_data_bids_root,
        )
    _write_csv(benchmark_rows, output_dir / "benchmark_results.csv")
    _write_json(benchmark_rows, output_dir / "benchmark_results.json")

    memo = render_recommendation(
        coverage_rows=coverage_records,
        benchmark_rows=benchmark_rows,
    )
    (output_dir / "recommendation_memo.md").write_text(memo, encoding="utf-8")

    print(
        json.dumps(
            {
                "coverage_rows": len(coverage_rows),
                "review_rows": len(review_rows),
                "benchmark_rows": len(benchmark_rows),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
