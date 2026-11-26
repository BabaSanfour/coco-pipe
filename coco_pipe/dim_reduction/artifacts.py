"""
Artifact persistence helpers for dimensionality-reduction runs.

This module owns the save/load/update layer that the dim-reduction pipeline
uses to checkpoint fit results to disk; eval results live only in the runs
inventory.  Keeping
these functions in coco-pipe (rather than in a consumer project) means any
project that runs the dim-reduction pipeline can share the same artifact
format without duplicating the serialization logic.

Public API
----------
save_fit_artifact
    Write embedding, ids, fit payload, metrics, and diagnostics to one
    ``.npz`` file.
load_fit_artifact
    Read a fit artifact back into a dict.
slice_fit_artifact
    One dimension of a fit artifact that holds a nested reducer's sweep.
load_fit_runs
    Read a JSON runs inventory file back into a list of dicts.
update_runs
    Upsert a record into a JSON runs inventory, keeping it sorted.

Constants
---------
SEPARATION_METRIC_KEY
    Canonical key for the logistic-regression separation metric returned by
    :func:`coco_pipe.dim_reduction.evaluation.core.evaluate_embedding`.
SEPARATION_RF_METRIC_KEY
    Canonical key for the random-forest separation metric.
FIT_METRIC_COLUMNS
    Ordered list of geometry quality metric names written by fit artifacts.
EVAL_METRIC_COLUMNS
    Ordered list of eval metric names in eval inventory records.
FIT_RUN_KEY_FIELDS
    Fields that uniquely identify a fit run in the runs inventory.
EVAL_RUN_KEY_FIELDS
    Fields that uniquely identify an eval run in the runs inventory.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from coco_pipe.dim_reduction._constants import (
    EVAL_METRIC_COLUMNS,
    EVAL_RUN_KEY_FIELDS,
    FIT_METRIC_COLUMNS,
    FIT_RUN_KEY_FIELDS,
    SEPARATION_METRIC_KEY,
    SEPARATION_RF_METRIC_KEY,
)
from coco_pipe.io import read_json, save_npz, write_json
from coco_pipe.io.structures import DataContainer

__all__ = [
    "EVAL_METRIC_COLUMNS",
    "EVAL_RUN_KEY_FIELDS",
    "FIT_METRIC_COLUMNS",
    "FIT_RUN_KEY_FIELDS",
    # Constants
    "SEPARATION_METRIC_KEY",
    "SEPARATION_RF_METRIC_KEY",
    "build_availability_record",
    # Record builder
    "build_record",
    "load_fit_artifact",
    "load_fit_runs",
    # Persistence
    "save_fit_artifact",
    "slice_fit_artifact",
    "update_runs",
    # Run-level helpers
    "write_run_status",
]


def save_fit_artifact(
    path: Path,
    embedding: np.ndarray,
    ids: np.ndarray,
    fit_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
    diagnostics: dict[str, Any],
    sweep: dict[int, dict[str, Any]] | None = None,
) -> None:
    """Write a fit artifact: one ``.npz`` file at *path*.

    The file holds the embedding, the ids, the reducer diagnostics and, as JSON,
    the fit payload and its metrics. It is written atomically, so a file at
    *path* is a complete artifact: one file per fit keeps the file count low for
    analysis modes with many units.

    Parameters
    ----------
    path:
        Target ``.npz`` file. Parent directories are created.
    embedding:
        Array of shape ``(n_obs, n_dims)``.
    ids:
        1-D array of observation identifiers aligned with *embedding*.
    fit_payload:
        Serialisable dict describing the fit (reducer, config, provenance, …).
        The key ``"artifact_stem"`` is recorded for provenance.
    metrics_payload:
        Serialisable dict of geometry quality metrics (trustworthiness, …).
    diagnostics:
        Reducer-internal diagnostics.  May be empty.
    sweep:
        For a nested reducer fitted once at its largest dimension, each swept
        ``n_components`` -> ``{"fit": payload, "metrics": metrics}``; the file
        then holds the whole sweep, read one dimension at a time with
        :func:`slice_fit_artifact`.
    """
    stem = str(fit_payload.get("artifact_stem") or "dim_reduction_fit")
    meta = {"artifact_stem": stem, "fit": fit_payload, "metrics": metrics_payload}
    if sweep:
        meta["sweep"] = {str(n): entry for n, entry in sweep.items()}
    save_npz(
        path,
        embedding=np.asarray(embedding),
        ids=np.asarray(ids, dtype=object),
        diagnostics=np.asarray([diagnostics], dtype=object),
        meta=np.asarray(json.dumps(meta)),
    )


def _embedding_container(
    embedding: np.ndarray, ids: np.ndarray, fit: dict[str, Any]
) -> DataContainer | None:
    """Reconstruct the embedding as a ``~coco_pipe.io.DataContainer`` from artifact
    arrays.

    Returns a ``('obs', 'component')`` container carrying the observation ids,
    a ``component`` coordinate, and the fit payload under ``meta['fit']``. The
    obs-level coordinates from the original fit container are not persisted in
    the compact artifact layout, so they are intentionally absent here; eval
    sources its labels/groups from a fresh container instead.

    Returns ``None`` for non-2D embeddings (e.g. native trajectory tensors),
    which do not map onto a ``component`` axis.
    """
    embedding = np.asarray(embedding)
    if embedding.ndim != 2:
        return None
    names = [f"component_{i + 1}" for i in range(embedding.shape[1])]
    return DataContainer(
        X=embedding,
        dims=("obs", "component"),
        coords={"component": np.asarray(names)},
        ids=np.asarray(ids),
        meta={"fit": dict(fit)},
    )


def load_fit_artifact(path: Path, n_components: int | None = None) -> dict[str, Any]:
    """Load a fit artifact written by :func:`save_fit_artifact`.

    Parameters
    ----------
    path:
        The artifact's ``.npz`` file.
    n_components:
        For a file holding a sweep, the dimension to return (see
        :func:`slice_fit_artifact`); ``None`` returns the whole file.

    Returns
    -------
    dict with keys:
        ``embedding``, ``ids``, ``fit``, ``metrics``, ``diagnostics``,
        ``manifest``, ``path``, and ``embedding_container``. The
        ``embedding_container`` is a :class:`~coco_pipe.io.structures.DataContainer`
        view of the embedding (or ``None`` for non-2D embeddings); the
        ``embedding`` and ``ids`` arrays remain for direct array access.
    """
    with np.load(path, allow_pickle=True) as npz:
        embedding = npz["embedding"]
        ids = npz["ids"]
        diagnostics = dict(npz["diagnostics"][0])
        meta = json.loads(str(npz["meta"]))
    fit = meta["fit"]
    artifact = {
        "embedding": embedding,
        "ids": ids,
        "fit": fit,
        "metrics": meta["metrics"],
        "diagnostics": diagnostics,
        "manifest": {"artifact_stem": meta["artifact_stem"]},
        "path": path,
        "embedding_container": _embedding_container(embedding, ids, fit),
        "sweep": meta.get("sweep"),
    }
    return (
        artifact if n_components is None else slice_fit_artifact(artifact, n_components)
    )


def slice_fit_artifact(artifact: dict[str, Any], n_components: int) -> dict[str, Any]:
    """One dimension of a loaded fit artifact that holds a sweep.

    The embedding, components and explained-variance ratios are cut to their
    first *n_components*, and the fit payload and metrics are that dimension's.
    An artifact without a sweep is returned unchanged.
    """
    sweep = artifact.get("sweep")
    if not sweep:
        return artifact
    if str(n_components) not in sweep:
        raise ValueError(
            f"{artifact['path']} holds n_components {sorted(map(int, sweep))}, "
            f"not {n_components}."
        )
    entry = sweep[str(n_components)]
    embedding = np.asarray(artifact["embedding"])[:, :n_components]
    diagnostics = dict(artifact["diagnostics"])
    for key in ("components", "explained_variance_ratio"):
        if diagnostics.get(key) is not None:
            diagnostics[key] = np.asarray(diagnostics[key])[:n_components]
    return {
        **artifact,
        "embedding": embedding,
        "fit": entry["fit"],
        "metrics": entry["metrics"],
        "diagnostics": diagnostics,
        "embedding_container": _embedding_container(
            embedding, artifact["ids"], entry["fit"]
        ),
        "sweep": None,
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

    sort_fields = [
        *list(key_fields),
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


def build_record(
    payload: dict[str, Any],
    artifact_path: Path,
    output_root: Path,
    metric_columns: Sequence[str],
    metrics_payload: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a flat run-inventory record from an artifact payload dict.

    Used for both fit and eval records — pass :data:`FIT_METRIC_COLUMNS` or
    :data:`EVAL_METRIC_COLUMNS` as *metric_columns*.

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


def write_run_status(
    output_root: Path,
    fit_runs_path: Path,
    eval_runs_path: Path,
    *,
    run_summary_path: Path | None = None,
    fatal_error: str | None = None,
    report_path: Path | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    run_summary_path:
        Optional explicit path for the summary JSON. Defaults to
        ``output_root / "run_summary.json"``.
    fatal_error:
        If set, the run is marked as at least partially failed.
    report_path:
        Path to the generated HTML report, if any.
    run_metadata:
        Extra key/value pairs merged into the summary payload.

    Returns
    -------
    dict
        The summary payload written to disk.
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
        "generated_at": datetime.now(UTC).isoformat(),
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

    summary_path = run_summary_path or (output_root / "run_summary.json")
    write_json(summary_path, summary_payload, indent=2)

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
    return summary_payload


def build_availability_record(
    *,
    scope: str,
    condition: str,
    unit_spec: dict[str, Any] | None,
    container: DataContainer,
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
