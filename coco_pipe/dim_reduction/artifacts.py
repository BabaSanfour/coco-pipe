"""
Artifact persistence helpers for dimensionality-reduction runs.

This module owns the save/load/update layer that the dim-reduction pipeline
uses to checkpoint fit results and post-hoc eval results to disk.  Keeping
these functions in coco-pipe (rather than in a consumer project) means any
project that runs the dim-reduction pipeline can share the same artifact
format without duplicating the serialization logic.

Public API
----------
save_fit_artifact
    Write embedding, ids, fit payload, metrics, and diagnostics to a
    directory and stamp it with ``_SUCCESS``.
save_eval_artifact
    Write an eval payload to a directory and stamp it with ``_SUCCESS``.
load_fit_artifact
    Read a fit artifact directory back into a dict.
load_fit_runs
    Read a JSON runs inventory file back into a list of dicts.
update_runs
    Upsert a record into a JSON runs inventory, keeping it sorted.

Constants
---------
SEPARATION_METRIC_KEY
    Canonical key for the logistic-regression separation metric returned by
    :func:`coco_pipe.dim_reduction.evaluation.core.evaluate_embedding`.
FIT_METRIC_COLUMNS
    Ordered list of geometry quality metric names written by fit artifacts.
EVAL_METRIC_COLUMNS
    Ordered list of eval metric names written by eval artifacts.
FIT_RUN_KEY_FIELDS
    Fields that uniquely identify a fit run in the runs inventory.
EVAL_RUN_KEY_FIELDS
    Fields that uniquely identify an eval run in the runs inventory.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from coco_pipe.io import read_json, save_npz, write_json

if TYPE_CHECKING:
    from coco_pipe.io.structures import DataContainer

__all__ = [
    # Persistence
    "save_fit_artifact",
    "save_eval_artifact",
    "load_fit_artifact",
    "load_fit_runs",
    "update_runs",
    # Constants
    "SEPARATION_METRIC_KEY",
    "FIT_METRIC_COLUMNS",
    "EVAL_METRIC_COLUMNS",
    "FIT_RUN_KEY_FIELDS",
    "EVAL_RUN_KEY_FIELDS",
    # Record builders
    "_build_result_record",
    "_build_fit_record",
    "_build_eval_record",
    # Run-level helpers
    "_write_run_status",
    "_availability_record",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

EVAL_METRIC_COLUMNS: list[str] = [SEPARATION_METRIC_KEY]
"""Eval metrics recorded in eval run inventory rows."""

FIT_RUN_KEY_FIELDS: tuple[str, ...] = ("fit_id",)
"""Fields that uniquely identify a fit run entry."""

EVAL_RUN_KEY_FIELDS: tuple[str, ...] = (
    "fit_id",
    "eval_name",
    "target_col",
    "group_col",
)
"""Fields that uniquely identify an eval run entry."""


FIT_ARRAYS_NAME = "fit.npz"
"""Compact fit artifact: arrays bundle (embedding + ids + diagnostics)."""
FIT_META_NAME = "fit.json"
"""Compact fit artifact: metadata bundle (fit payload + metrics)."""


def save_fit_artifact(
    path: Path,
    embedding: np.ndarray,
    ids: np.ndarray,
    fit_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    """Write a fit artifact to *path* and stamp it with ``_SUCCESS``.

    The artifact directory receives three files (a compact layout that keeps the
    inode count low for high-cardinality analysis modes):

    - ``fit.npz`` — embedding, ids, and reducer diagnostics in one archive
    - ``fit.json`` — the fit payload and geometry quality metrics
    - ``_SUCCESS`` — sentinel that marks the artifact as complete

    :func:`load_fit_artifact` also reads the older seven-file layout, so existing
    artifacts remain loadable.

    Parameters
    ----------
    path:
        Target directory.  Created (including parents) if it does not exist.
    embedding:
        Array of shape ``(n_obs, n_components)``.
    ids:
        1-D array of observation identifiers aligned with *embedding*.
    fit_payload:
        Serialisable dict describing the fit (reducer, config, provenance, …).
        The key ``"artifact_stem"`` is recorded for provenance.
    metrics_payload:
        Serialisable dict of geometry quality metrics (trustworthiness, …).
    diagnostics:
        Serialisable dict of reducer-internal diagnostics.  May be empty.
    """
    path.mkdir(parents=True, exist_ok=True)
    stem = str(fit_payload.get("artifact_stem") or "dim_reduction_fit")
    save_npz(
        path / FIT_ARRAYS_NAME,
        embedding=np.asarray(embedding),
        ids=np.asarray(ids, dtype=object),
        diagnostics=np.asarray([diagnostics], dtype=object),
    )
    write_json(
        path / FIT_META_NAME,
        {"artifact_stem": stem, "fit": fit_payload, "metrics": metrics_payload},
        indent=2,
    )
    (path / "_SUCCESS").write_text("ok\n", encoding="utf-8")


EVAL_NAME = "eval.json"
"""Compact eval artifact: the single eval payload file."""


def save_eval_artifact(path: Path, eval_payload: dict[str, Any]) -> None:
    """Write an eval artifact to *path* and stamp it with ``_SUCCESS``.

    The artifact directory receives two files:

    - ``eval.json`` — the full eval payload
    - ``_SUCCESS`` — sentinel that marks the artifact as complete

    :func:`_load_eval_payload` also reads the older ``<stem>_eval.json`` layout.

    Parameters
    ----------
    path:
        Target directory.  Created (including parents) if it does not exist.
    eval_payload:
        Serialisable dict describing the post-hoc evaluation results.  The key
        ``"artifact_stem"`` is recorded for provenance when present.
    """
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / EVAL_NAME, eval_payload, indent=2)
    (path / "_SUCCESS").write_text("ok\n", encoding="utf-8")


def _load_eval_payload(path: Path) -> dict[str, Any]:
    """Load an eval payload from an artifact directory.

    Resolution order:

    1. Read ``artifact_manifest.json`` and follow the ``"eval"`` key.
    2. Fall back to ``eval.json`` directly.
    3. Glob for ``*_eval.json`` and return the first match.

    Raises :class:`FileNotFoundError` when no eval file can be located.
    """
    manifest_path = path / "artifact_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        eval_path = path / manifest.get("eval", "eval.json")
        if eval_path.exists():
            return read_json(eval_path)
    eval_path = path / "eval.json"
    if eval_path.exists():
        return read_json(eval_path)
    matches = sorted(path.glob("*_eval.json"))
    if matches:
        return read_json(matches[0])
    raise FileNotFoundError(f"No eval payload found in {path}")


def load_fit_artifact(path: Path) -> dict[str, Any]:
    """Load a fit artifact directory into a dict.

    The compact ``fit.npz`` + ``fit.json`` layout written by
    :func:`save_fit_artifact` is read first; if absent, the legacy seven-file
    layout is resolved via ``artifact_manifest.json`` then a glob fallback so
    artifacts written by earlier versions remain loadable.

    Parameters
    ----------
    path:
        Artifact directory written by :func:`save_fit_artifact`.

    Returns
    -------
    dict with keys:
        ``embedding``, ``ids``, ``fit``, ``metrics``, ``diagnostics``,
        ``manifest``, ``path``.
    """
    arrays_path = path / FIT_ARRAYS_NAME
    meta_path = path / FIT_META_NAME
    if arrays_path.exists() and meta_path.exists():
        meta = read_json(meta_path)
        with np.load(arrays_path, allow_pickle=True) as npz:
            embedding = npz["embedding"]
            ids = npz["ids"]
            diagnostics = (
                dict(npz["diagnostics"][0]) if "diagnostics" in npz.files else {}
            )
        return {
            "embedding": embedding,
            "ids": ids,
            "fit": meta.get("fit", {}),
            "metrics": meta.get("metrics", {}),
            "diagnostics": diagnostics,
            "manifest": {"artifact_stem": meta.get("artifact_stem")},
            "path": path,
        }

    # --- Legacy seven-file layout fallback ---
    manifest_path = path / "artifact_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    fit_path = path / manifest.get("fit", "fit.json")
    metrics_path = path / manifest.get("metrics", "metrics.json")
    embedding_path = path / manifest.get("embedding", "embedding.npy")
    ids_path = path / manifest.get("ids", "ids.npy")
    diagnostics_path = path / manifest.get("diagnostics", "diagnostics.npz")

    if not fit_path.exists():
        fit_matches = sorted(path.glob("*_fit.json"))
        if fit_matches:
            fit_path = fit_matches[0]
    if not metrics_path.exists():
        metrics_matches = sorted(path.glob("*_metrics.json"))
        if metrics_matches:
            metrics_path = metrics_matches[0]
    if not embedding_path.exists():
        embedding_matches = sorted(path.glob("*_embedding.npy"))
        if embedding_matches:
            embedding_path = embedding_matches[0]
    if not ids_path.exists():
        ids_matches = sorted(path.glob("*_ids.npy"))
        if ids_matches:
            ids_path = ids_matches[0]
    if not diagnostics_path.exists():
        diagnostics_matches = sorted(path.glob("*_diagnostics.npz"))
        if diagnostics_matches:
            diagnostics_path = diagnostics_matches[0]

    diagnostics: dict[str, Any] = {}
    if diagnostics_path.exists():
        with np.load(diagnostics_path, allow_pickle=True) as npz:
            diagnostics = dict(npz["payload"][0])

    fit = read_json(fit_path)
    metrics = read_json(metrics_path)
    return {
        "embedding": np.load(embedding_path, allow_pickle=True),
        "ids": np.load(ids_path, allow_pickle=True),
        "fit": fit,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "manifest": manifest,
        "path": path,
    }


def load_fit_runs(path: Path) -> list[dict[str, Any]]:
    """Load a fit-runs inventory JSON file into a list of dicts.

    Parameters
    ----------
    path:
        Path to the JSON file written by :func:`update_runs`.

    Returns
    -------
    list of dict

    Raises
    ------
    RuntimeError
        If *path* does not exist.
    ValueError
        If the file does not contain a JSON list.
    """
    if not path.exists():
        raise RuntimeError(f"No fit runs found in {path}.")
    runs = read_json(path)
    if not isinstance(runs, list):
        raise ValueError(f"Expected list payload in {path}.")
    return runs


def update_runs(path: Path, record: dict[str, Any], key_fields: Sequence[str]) -> None:
    """Upsert *record* into a JSON runs inventory at *path*.

    If *path* already exists, the list is loaded and the record whose
    ``key_fields`` values match *record*'s is replaced (upsert semantics).
    If no match is found, *record* is appended.  The list is then sorted by
    ``key_fields`` followed by common run-taxonomy fields and written back.

    Parameters
    ----------
    path:
        Target JSON file.  The parent directory is created if necessary.
    record:
        Dict to upsert.  Must contain all fields listed in *key_fields*.
    key_fields:
        Ordered sequence of field names that uniquely identify a run.
    """
    if path.exists():
        runs = read_json(path)
        if not isinstance(runs, list):
            raise ValueError(f"Expected list payload in {path}.")
    else:
        runs = []

    key = tuple(record.get(field) for field in key_fields)
    for idx, existing in enumerate(runs):
        if tuple(existing.get(field) for field in key_fields) == key:
            runs[idx] = dict(record)
            break
    else:
        runs.append(dict(record))

    sort_fields = list(key_fields) + [
        "scope",
        "condition",
        "analysis_mode",
        "family",
        "unit_name",
        "reducer",
        "n_components",
    ]
    runs.sort(key=lambda item: tuple(str(item.get(field, "")) for field in sort_fields))
    write_json(path, runs, indent=2)


def _build_result_record(
    payload: dict[str, Any],
    artifact_path: Path,
    output_root: Path,
    metric_columns: Sequence[str],
    metrics_payload: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """Build a flat run-inventory record from an artifact payload dict.

    Strips bulky sub-dicts (``metrics``, ``records``, ``metadata``,
    ``artifacts``) from *payload*, adds a relative ``artifact_path``,
    promotes each metric in *metric_columns* to a top-level float (or
    ``nan``), and stamps the record with a ``status`` of ``"success"`` or
    ``"failed"``.
    """
    record = {
        key: value
        for key, value in dict(payload).items()
        if key not in {"metrics", "records", "metadata", "artifacts"}
    }
    record["artifact_path"] = str(artifact_path.relative_to(output_root))
    for metric_name in metric_columns:
        value = None if metrics_payload is None else metrics_payload.get(metric_name)
        record[metric_name] = np.nan if value is None else float(value)
    if error is not None:
        record["status"] = "failed"
        record["error"] = error
    else:
        record["status"] = "success"
    return record


def _build_fit_record(
    fit_payload: dict[str, Any],
    artifact_path: Path,
    output_root: Path,
    metrics_payload: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """Build a fit run-inventory record from *fit_payload*.

    Thin wrapper around :func:`_build_result_record` that pre-fills
    *metric_columns* with :data:`FIT_METRIC_COLUMNS`.
    """
    return _build_result_record(
        fit_payload,
        artifact_path,
        output_root,
        FIT_METRIC_COLUMNS,
        metrics_payload,
        error,
    )


def _build_eval_record(
    eval_payload: dict[str, Any],
    artifact_path: Path,
    output_root: Path,
    metrics_payload: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """Build an eval run-inventory record from *eval_payload*.

    Thin wrapper around :func:`_build_result_record` that pre-fills
    *metric_columns* with :data:`EVAL_METRIC_COLUMNS`.
    """
    return _build_result_record(
        eval_payload,
        artifact_path,
        output_root,
        EVAL_METRIC_COLUMNS,
        metrics_payload,
        error,
    )


def _write_run_status(
    output_root: Path,
    fit_runs_path: Path,
    eval_runs_path: Path,
    *,
    fatal_error: Optional[str] = None,
    report_path: Optional[Path] = None,
    run_metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Write ``run_summary.json`` and a run-marker sentinel to *output_root*.

    The marker file is one of ``_RUN_SUCCESS``, ``_RUN_PARTIAL``, or
    ``_RUN_FAILED``.  Any pre-existing marker files are removed first so only
    one is present at a time.

    Run status logic:

    - *success*  — at least one successful fit or eval, no failures
    - *partial*  — both successes and failures present
    - *failed*   — only failures, or *fatal_error* is set with no successes

    Parameters
    ----------
    output_root:
        Directory that receives ``run_summary.json`` and the marker file.
    fit_runs_path:
        Path to the fit runs JSON inventory (may not exist yet).
    eval_runs_path:
        Path to the eval runs JSON inventory (may not exist yet).
    fatal_error:
        If set, the run is marked as at least partially failed.
    report_path:
        Path to the generated HTML report, if any.
    run_metadata:
        Extra key/value pairs merged into the summary payload.
    """
    fit_runs = read_json(fit_runs_path) if fit_runs_path.exists() else []
    eval_runs = read_json(eval_runs_path) if eval_runs_path.exists() else []

    fit_success = sum(record.get("status") == "success" for record in fit_runs)
    fit_failed = sum(record.get("status") == "failed" for record in fit_runs)
    eval_success = sum(record.get("status") == "success" for record in eval_runs)
    eval_failed = sum(record.get("status") == "failed" for record in eval_runs)
    any_success = (fit_success + eval_success) > 0
    any_failed = (fit_failed + eval_failed) > 0 or fatal_error is not None

    if any_failed and any_success:
        run_status = "partial"
    elif any_failed:
        run_status = "failed"
    elif any_success:
        run_status = "success"
    else:
        run_status = "failed" if fatal_error is not None else "partial"

    summary_payload: dict[str, Any] = {
        "status": run_status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fit_total": len(fit_runs),
        "fit_success": fit_success,
        "fit_failed": fit_failed,
        "eval_total": len(eval_runs),
        "eval_success": eval_success,
        "eval_failed": eval_failed,
        "report_path": str(report_path) if report_path is not None else None,
        "report_exists": bool(report_path is not None and report_path.exists()),
        "fatal_error": fatal_error,
    }
    if run_metadata:
        summary_payload.update(run_metadata)

    write_json(output_root / "run_summary.json", summary_payload, indent=2)

    for marker_name in ("_RUN_SUCCESS", "_RUN_PARTIAL", "_RUN_FAILED"):
        marker = output_root / marker_name
        if marker.exists():
            marker.unlink()

    marker_name = {
        "success": "_RUN_SUCCESS",
        "partial": "_RUN_PARTIAL",
        "failed": "_RUN_FAILED",
    }[run_status]
    (output_root / marker_name).write_text("ok\n", encoding="utf-8")


def _availability_record(
    *,
    scope: str,
    condition: str,
    unit_spec: Optional[dict[str, Any]],
    container: "DataContainer",
    requested_components: Sequence[int],
    valid_components: Sequence[int],
) -> dict[str, Any]:
    """Build a data-availability record from container shape information.

    Captures matrix dimensions and which n_components values are feasible vs
    skipped, for inclusion in run-level provenance metadata.
    """
    X = np.asarray(container.X)
    n_features = int(X.shape[1]) if X.ndim == 2 else None
    return {
        "scope": scope,
        "condition": condition,
        "unit_type": None if unit_spec is None else unit_spec["unit_type"],
        "unit_name": None if unit_spec is None else unit_spec["unit_name"],
        "unit_key": None if unit_spec is None else unit_spec["unit_key"],
        "matrix_shape": list(X.shape),
        "n_samples": int(X.shape[0]) if X.ndim >= 1 else 0,
        "n_features": n_features,
        "requested_n_components": [int(v) for v in requested_components],
        "valid_n_components": [int(v) for v in valid_components],
        "skipped_n_components": [
            int(v)
            for v in requested_components
            if int(v) not in set(map(int, valid_components))
        ],
    }
