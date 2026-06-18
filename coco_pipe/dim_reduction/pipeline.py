"""
Checkpointed fit/eval pipeline for dimensionality-reduction runs.

This module contains the core execution primitives that any project using
coco-pipe's dim-reduction stack can share.  Each function is intentionally
side-effect-free aside from writing artifact files — the caller controls
all paths and inventory updates.

Public API
----------
run_fit
    Fit one reducer variant on one analysis unit, writing a checkpointed
    artifact directory.
run_eval
    Run one post-hoc evaluation of a saved embedding, writing a checkpointed
    eval artifact directory.
build_auto_pooled_eval_spec
    Build the automatic ``condition_separation`` eval spec used when pooling
    is active.

Private helpers (used by the task-builder/executor layer)
---------------------------------------------------------
_prepare_eval_inputs
    Align a DataContainer to saved fit ids and apply label/filter logic.
_build_fit_task / _execute_fit_task
    Construct and execute a serialisable fit task dict.
_build_eval_task / _execute_eval_task
    Construct and execute a serialisable eval task dict.
_valid_n_components_for_container
    Check whether *n_components* is feasible for a container's matrix shape.
_valid_component_sweep
    Filter a list of component counts to feasible values.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from coco_pipe.dim_reduction.artifacts import (
    EVAL_METRIC_COLUMNS,
    FIT_METRIC_COLUMNS,
    _build_eval_record,
    _build_fit_record,
    load_fit_artifact,
    save_eval_artifact,
    save_fit_artifact,
)
from coco_pipe.dim_reduction.config import _MISSING_EVAL_VALUES, DEFAULT_EVAL_GROUP_COL
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding
from coco_pipe.io.structures import DataContainer
from coco_pipe.utils import _slug

__all__ = [
    "POOLED_CONDITION",
    "run_fit",
    "run_eval",
    "build_auto_pooled_eval_spec",
    "_prepare_eval_inputs",
    "_build_fit_task",
    "_execute_fit_task",
    "_build_eval_task",
    "_execute_eval_task",
    "_valid_n_components_for_container",
    "_valid_component_sweep",
]

logger = logging.getLogger(__name__)

POOLED_CONDITION: str = "pooled_all"
"""Canonical condition name used for the pooled (multi-condition) container."""


def _as_array(value: Any) -> np.ndarray:
    """Return the embedding array from an ndarray or a ``DataContainer``."""
    if isinstance(value, DataContainer):
        return np.asarray(value.X)
    return np.asarray(value)


# ---------------------------------------------------------------------------
# Core fit / eval
# ---------------------------------------------------------------------------


def run_fit(
    fit_payload: dict[str, Any],
    container: DataContainer,
    out_path: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Fit one reducer variant and checkpoint the result to *out_path*.

    If ``_SUCCESS`` already exists in *out_path* and *overwrite* is ``False``
    the existing artifact is loaded and its inventory record is returned
    immediately (checkpoint resume).

    Parameters
    ----------
    fit_payload:
        Provenance/config dict describing this fit (reducer, n_components,
        scope, condition, unit info, input signature, …).
    container:
        Data container for this analysis unit.  Must have ``ids``.
    out_path:
        Artifact directory to write (or resume from).
    output_root:
        Root of the entire run output.  Used for relative-path computation in
        the returned inventory record.
    overwrite:
        When ``True``, an existing *out_path* directory is deleted before
        fitting.

    Returns
    -------
    dict
        A flat inventory record suitable for passing to :func:`update_runs`.
    """
    success_marker = out_path / "_SUCCESS"
    if success_marker.exists() and not overwrite:
        artifact = load_fit_artifact(out_path)
        return _build_fit_record(
            fit_payload=artifact["fit"],
            artifact_path=out_path,
            output_root=output_root,
            metrics_payload=artifact["metrics"],
        )

    if overwrite and out_path.exists():
        shutil.rmtree(out_path)

    X = np.asarray(container.X)
    if X.ndim != 2:
        raise ValueError("run_fit expects a 2D matrix.")
    if container.ids is None:
        raise ValueError("Dim-reduction fits expect container.ids to be present.")
    ids = np.asarray(container.ids, dtype=object).astype(str)

    reducer = DimReduction(
        method=fit_payload["reducer"], n_components=fit_payload["n_components"]
    )
    embedding_container = reducer.fit_transform(container)
    embedding = np.asarray(embedding_container.X)
    score_payload = reducer.score(embedding_container, X=container)
    score_metrics = dict(reducer.get_metrics())
    metrics_payload = {
        metric_name: (
            None
            if np.isnan(score_metrics.get(metric_name, np.nan))
            else float(score_metrics.get(metric_name))
        )
        for metric_name in FIT_METRIC_COLUMNS
    }

    summary = reducer.get_summary()
    diagnostics = dict(summary.get("diagnostics") or {})
    diagnostics["score_payload"] = score_payload
    diagnostics["summary"] = summary
    try:
        components = reducer.get_components()
    except Exception:
        components = None
    if components is not None:
        diagnostics["components"] = components
    explained_variance = getattr(reducer.reducer, "explained_variance_ratio_", None)
    if explained_variance is not None:
        diagnostics["explained_variance_ratio"] = np.asarray(explained_variance)

    save_fit_artifact(
        out_path, embedding, ids, fit_payload, metrics_payload, diagnostics
    )
    return _build_fit_record(
        fit_payload=fit_payload,
        artifact_path=out_path,
        output_root=output_root,
        metrics_payload=metrics_payload,
    )


def run_eval(
    fit_payload: dict[str, Any],
    fit_artifact: dict[str, Any],
    container: DataContainer,
    eval_spec: dict[str, Any],
    out_path: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Run one post-hoc evaluation and checkpoint the result to *out_path*.

    If ``_SUCCESS`` already exists in *out_path* and *overwrite* is ``False``
    the existing eval artifact is loaded and its inventory record is returned
    immediately.

    Parameters
    ----------
    fit_payload:
        Provenance dict from the fit artifact (``fit_artifact["fit"]``).
    fit_artifact:
        Full fit artifact dict as returned by :func:`load_fit_artifact`.
    container:
        Data container for the analysis unit (must contain the columns
        referenced by *eval_spec*).
    eval_spec:
        Eval specification dict with keys ``name``, ``target_col``,
        ``group_col``, ``filters``, ``label_map``.
    out_path:
        Artifact directory to write (or resume from).
    output_root:
        Root of the entire run output.
    overwrite:
        When ``True``, an existing *out_path* directory is deleted before
        evaluating.

    Returns
    -------
    dict
        A flat eval inventory record suitable for passing to
        :func:`update_runs`.
    """
    from coco_pipe.dim_reduction.artifacts import (
        _load_eval_payload,  # local to avoid cycle
    )

    success_marker = out_path / "_SUCCESS"
    if success_marker.exists() and not overwrite:
        eval_payload = _load_eval_payload(out_path)
        return _build_eval_record(
            eval_payload=eval_payload,
            artifact_path=out_path,
            output_root=output_root,
            metrics_payload=eval_payload.get("metrics"),
        )

    if overwrite and out_path.exists():
        shutil.rmtree(out_path)

    selected_index, selected_ids, labels, groups = _prepare_eval_inputs(
        container=container,
        fit_ids=np.asarray(fit_artifact["ids"], dtype=object).astype(str),
        eval_spec=eval_spec,
    )
    eval_id = hashlib.sha256(
        json.dumps(
            {
                "fit_id": fit_payload["fit_id"],
                "eval_name": eval_spec["name"],
                "target_col": eval_spec["target_col"],
                "group_col": eval_spec["group_col"],
                "filters": eval_spec["filters"],
                "label_map": eval_spec["label_map"],
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]

    embedding = _as_array(fit_artifact["embedding"])[selected_index.to_numpy()]
    if embedding.ndim != 2:
        raise ValueError("run_eval expects a 2D embedding artifact.")

    score_payload = evaluate_embedding(
        embedding,
        method_name=fit_payload["reducer"],
        metrics=EVAL_METRIC_COLUMNS,
        labels=labels,
        groups=groups,
    )
    metrics_payload = dict(score_payload["metrics"])
    artifact_stem = "_".join(
        [
            "sub-all",
            "ses-all",
            f"scope-{_slug(fit_payload['scope'], max_len=32)}",
            f"cond-{_slug(fit_payload['condition'], max_len=32)}",
            f"unit-{_slug(fit_payload['unit_key'], max_len=32)}",
            f"reducer-{_slug(fit_payload['reducer'], max_len=32)}",
            f"components-{int(fit_payload['n_components'])}",
            f"eval-{_slug(eval_spec['name'], max_len=32)}",
        ]
    )
    eval_payload = {
        "eval_id": eval_id,
        "fit_id": fit_payload["fit_id"],
        "scope": fit_payload["scope"],
        "condition": fit_payload["condition"],
        "analysis_mode": fit_payload["analysis_mode"],
        "unit_type": fit_payload["unit_type"],
        "unit_name": fit_payload["unit_name"],
        "unit_key": fit_payload["unit_key"],
        "family": fit_payload.get("family"),
        "eval_name": eval_spec["name"],
        "input_mode": fit_payload["input_mode"],
        "representation": fit_payload["representation"],
        "aggregation_unit": fit_payload.get("aggregation_unit"),
        "run_label": fit_payload.get("run_label"),
        "reducer": fit_payload["reducer"],
        "n_components": int(fit_payload["n_components"]),
        "target_col": eval_spec["target_col"],
        "group_col": eval_spec["group_col"],
        "filters": list(eval_spec["filters"]),
        "label_map": dict(eval_spec["label_map"]),
        "descriptor_families": list(fit_payload.get("descriptor_families", [])),
        "descriptor_max_abs_value": fit_payload.get("descriptor_max_abs_value"),
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(selected_ids)),
        "n_groups": int(pd.Index(groups).nunique()),
        "n_labels": int(pd.Index(labels).nunique()),
        "metrics": metrics_payload,
        "records": score_payload.get("records", []),
        "metadata": score_payload.get("metadata", {}),
        "artifacts": score_payload.get("artifacts", {}),
        "artifact_stem": f"{artifact_stem}_eval-{eval_id}",
    }
    save_eval_artifact(out_path, eval_payload)
    return _build_eval_record(
        eval_payload=eval_payload,
        artifact_path=out_path,
        output_root=output_root,
        metrics_payload=metrics_payload,
    )


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def _prepare_eval_inputs(
    container: DataContainer,
    fit_ids: np.ndarray,
    eval_spec: dict[str, Any],
) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    """Align *container* observations to *fit_ids* and apply eval filters.

    The saved fit may cover a different (or differently ordered) subset of
    observations than the current container, so this function aligns by
    observation id (with occurrence-count disambiguation for duplicate ids),
    applies column filters from *eval_spec*, resolves labels and groups, and
    masks out missing values.

    Parameters
    ----------
    container:
        DataContainer that holds the metadata columns referenced by
        *eval_spec*.
    fit_ids:
        Observation ids in the order they appear in the saved embedding.
    eval_spec:
        Eval specification dict (``name``, ``target_col``, ``group_col``,
        ``filters``, ``label_map``).

    Returns
    -------
    tuple of (selected_index, selected_ids, labels, groups)
        *selected_index* is the pandas :class:`~pandas.Index` into the
        aligned frame (suitable for slicing the embedding array).
        The remaining three are numpy arrays of strings.

    Raises
    ------
    ValueError
        On missing columns or structural issues.
    RuntimeError
        When no valid samples remain after alignment and filtering.
    """
    if container.ids is None:
        raise ValueError("Dim-reduction fit/eval expects container.ids to be present.")

    container_ids = np.asarray(container.ids, dtype=object).astype(str)
    frame = pd.DataFrame({"obs_id": container_ids})
    n_obs = len(container_ids)
    for key, values in container.coords.items():
        arr = np.asarray(values)
        if arr.ndim == 1 and len(arr) == n_obs and key != "feature":
            frame[key] = arr
    if container.y is not None and "y" not in frame.columns:
        frame["y"] = np.asarray(container.y)

    # Build occurrence-disambiguated keys for the fit ids
    aligned_keys: list[str] = []
    counts: dict[str, int] = {}
    for obs_id in np.asarray(fit_ids, dtype=object).astype(str):
        occurrence = counts.get(obs_id, 0)
        aligned_keys.append(f"{obs_id}__{occurrence}")
        counts[obs_id] = occurrence + 1

    # Build the same keys for the container frame
    counts = {}
    frame_keys: list[str] = []
    for obs_id in frame["obs_id"].astype(str):
        occurrence = counts.get(obs_id, 0)
        frame_keys.append(f"{obs_id}__{occurrence}")
        counts[obs_id] = occurrence + 1
    frame["_obs_key"] = frame_keys

    aligned_frame = frame.drop_duplicates("_obs_key", keep="first").set_index(
        "_obs_key"
    )
    missing = [key for key in aligned_keys if key not in aligned_frame.index]
    if missing:
        raise RuntimeError(
            "Saved fit ids could not be aligned to the current container."
        )
    aligned_frame = aligned_frame.loc[aligned_keys].reset_index(drop=True)

    for filter_spec in eval_spec["filters"]:
        column = filter_spec["column"]
        if column not in aligned_frame.columns:
            raise ValueError(f"Eval filter column '{column}' is not available.")
        values = {str(v) for v in filter_spec["values"]}
        aligned_frame = aligned_frame[
            aligned_frame[column].astype(str).isin(values)
        ].copy()

    if eval_spec["target_col"] not in aligned_frame.columns:
        raise ValueError(
            f"Eval target column '{eval_spec['target_col']}' is not available."
        )
    if eval_spec["group_col"] not in aligned_frame.columns:
        raise ValueError(
            f"Eval group column '{eval_spec['group_col']}' is not available."
        )

    labels = aligned_frame[eval_spec["target_col"]]
    if eval_spec["label_map"]:
        labels = labels.map(lambda value: eval_spec["label_map"].get(str(value), value))
    labels = labels.astype("string").str.strip()
    labels = labels.mask(labels.str.lower().isin(_MISSING_EVAL_VALUES))
    groups = aligned_frame[eval_spec["group_col"]].astype("string").str.strip()
    groups = groups.mask(groups.str.lower().isin(_MISSING_EVAL_VALUES))
    valid_mask = labels.notna() & groups.notna()
    if not valid_mask.any():
        raise RuntimeError(f"Eval '{eval_spec['name']}' produced no valid samples.")

    selected_frame = aligned_frame.loc[valid_mask].copy()
    return (
        selected_frame.index,
        selected_frame["obs_id"].astype(str).to_numpy(),
        labels.loc[valid_mask].astype(str).to_numpy(),
        groups.loc[valid_mask].astype(str).to_numpy(),
    )


# ---------------------------------------------------------------------------
# Component sweep helpers
# ---------------------------------------------------------------------------


def _valid_n_components_for_container(
    container: DataContainer, n_components: int
) -> bool:
    """Return ``True`` if *n_components* is feasible for *container*'s matrix."""
    X = np.asarray(container.X)
    if X.ndim != 2:
        return False
    if n_components < 1:
        return False
    max_components = min(int(X.shape[0]), int(X.shape[1]))
    return int(n_components) <= max_components


def _valid_component_sweep(
    container: DataContainer, requested: Sequence[int]
) -> list[int]:
    """Filter *requested* to the component counts feasible for *container*.

    Logs a message if any values are skipped.
    """
    valid = [
        int(v)
        for v in requested
        if _valid_n_components_for_container(container, int(v))
    ]
    skipped = [int(v) for v in requested if int(v) not in valid]
    if skipped:
        logger.info(
            "Skipping n_components values %s for matrix shape %s.",
            skipped,
            tuple(np.asarray(container.X).shape),
        )
    return valid


# ---------------------------------------------------------------------------
# Eval-spec helpers
# ---------------------------------------------------------------------------


def build_auto_pooled_eval_spec(
    conditions: list[str],
    run_pooled: bool,
) -> Optional[dict[str, Any]]:
    """Return a ``condition_separation`` eval spec, or ``None``.

    The spec is only produced when *run_pooled* is ``True`` and at least two
    conditions are present — otherwise condition separation is not meaningful.

    Parameters
    ----------
    conditions:
        List of condition names that will be included in the pooled container.
    run_pooled:
        Whether the caller intends to run a pooled analysis.
    """
    if not run_pooled or len(conditions) < 2:
        return None
    return {
        "name": "condition_separation",
        "target_col": "condition",
        "group_col": DEFAULT_EVAL_GROUP_COL,
        "filters": [],
        "label_map": {},
    }


# ---------------------------------------------------------------------------
# Task builders and executors (joblib-compatible)
# ---------------------------------------------------------------------------


def _build_fit_task(
    args: Any,
    scope: str,
    condition: str,
    unit_spec: dict[str, Any],
    reducer_name: str,
    n_components: int,
    output_root: Path,
) -> dict[str, Any]:
    """Build a serialisable fit task dict from run args and unit metadata.

    Computes a deterministic ``fit_id`` (SHA-256 of the input signature +
    sample ids), assembles the ``fit_payload`` provenance dict, and determines
    the artifact path.  The returned dict is consumed by
    :func:`_execute_fit_task`.

    Parameters
    ----------
    args:
        Parsed argument namespace.  Expected fields depend on
        ``args.input_mode`` (``"raw"`` or ``"descriptors"``).
    scope, condition, unit_spec, reducer_name, n_components:
        Run coordinates.
    output_root:
        Root directory for all artifacts in this run.
    """
    container = unit_spec["container"]
    if container.ids is None:
        raise ValueError("Dim-reduction fits expect container.ids to be present.")
    ids = np.asarray(container.ids, dtype=object).astype(str)

    filter_specs = [
        {"column": str(col), "values": [str(v) for v in vals]}
        for col, vals in zip(args.filter_col, args.filter_val)
        if vals
    ]
    input_signature: dict[str, Any] = {
        "input_mode": args.input_mode,
        "representation": args.representation,
        "analysis_mode": args.analysis_mode,
        "descriptor_families": list(getattr(args, "descriptor_families", []) or []),
        "filters": filter_specs,
        "balance_target": args.balance_target,
        "balance_strategy": args.balance_strategy if args.balance_target else None,
        "unit_type": unit_spec["unit_type"],
        "unit_name": unit_spec["unit_name"],
        "family": unit_spec.get("family"),
    }
    if args.input_mode == "raw":
        input_signature.update(
            {
                "bids_root": str(Path(args.bids_root).expanduser()),
                "use_derivatives": bool(args.use_derivatives),
                "task": getattr(args, "task", "clinical"),
                "segment_duration": float(args.segment_duration),
                "overlap": float(args.overlap),
                "desc": args.desc,
                "aggregation_unit": getattr(args, "aggregation_unit", None),
            }
        )
    else:
        input_signature.update(
            {
                "descriptor_table_path": str(
                    Path(args.descriptor_table_path).expanduser()
                ),
                "descriptor_feature_columns_path": str(
                    Path(args.descriptor_feature_columns_path).expanduser()
                ),
                "descriptor_max_abs_value": getattr(
                    args, "descriptor_max_abs_value", None
                ),
            }
        )

    sample_ids_sha256 = hashlib.sha256(
        "\0".join(ids.tolist()).encode("utf-8")
    ).hexdigest()[:16]
    fit_id = hashlib.sha256(
        json.dumps(
            {
                "scope": scope,
                "condition": condition,
                "analysis_mode": args.analysis_mode,
                "unit_type": unit_spec["unit_type"],
                "unit_name": unit_spec["unit_name"],
                "family": unit_spec.get("family"),
                "input_signature": input_signature,
                "reducer": reducer_name,
                "n_components": int(n_components),
                "sample_ids_sha256": sample_ids_sha256,
                "n_samples": int(len(ids)),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]

    artifact_stem = "_".join(
        [
            "sub-all",
            "ses-all",
            f"scope-{_slug(scope, max_len=32)}",
            f"cond-{_slug(condition, max_len=32)}",
            f"mode-{_slug(args.analysis_mode, max_len=32)}",
            f"unit-{_slug(unit_spec['unit_key'], max_len=32)}",
            f"reducer-{_slug(reducer_name, max_len=32)}",
            f"components-{int(n_components)}",
            f"fit-{fit_id}",
        ]
    )
    fit_payload: dict[str, Any] = {
        "fit_id": fit_id,
        "scope": scope,
        "condition": condition,
        "analysis_mode": args.analysis_mode,
        "unit_type": unit_spec["unit_type"],
        "unit_name": unit_spec["unit_name"],
        "unit_key": unit_spec["unit_key"],
        "family": unit_spec.get("family"),
        "input_mode": args.input_mode,
        "representation": args.representation,
        "aggregation_unit": getattr(args, "aggregation_unit", None),
        "run_label": getattr(args, "run_label", None),
        "descriptor_families": list(getattr(args, "descriptor_families", []) or []),
        "descriptor_max_abs_value": (
            getattr(args, "descriptor_max_abs_value", None)
            if args.input_mode == "descriptors"
            else None
        ),
        "reducer": reducer_name,
        "n_components": int(n_components),
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(container.X.shape[0]),
        "n_subjects": int(
            pd.Index(
                np.asarray(
                    container.coords.get(args.subject_col, ids), dtype=object
                ).astype(str)
            ).nunique()
        ),
        "loaded_obs": int(container.meta.get("loaded_obs", container.X.shape[0])),
        "samples_used": int(container.meta.get("samples_used", container.X.shape[0])),
        "input_signature": input_signature,
        "artifact_stem": artifact_stem,
    }
    artifact_path = (
        output_root
        / "sub-all"
        / "ses-all"
        / "eeg"
        / "fits"
        / f"scope-{_slug(scope)}"
        / f"cond-{_slug(condition)}"
        / f"input-{_slug(args.input_mode)}"
        / f"mode-{_slug(args.analysis_mode)}"
        / f"unit-{_slug(unit_spec['unit_type'])}"
        / f"name-{_slug(unit_spec['unit_key'])}"
        / f"reducer-{_slug(reducer_name)}"
        / f"components-{int(n_components)}"
        / f"fit-{fit_id}"
    )
    return {
        "fit_payload": fit_payload,
        "container": container,
        "artifact_path": artifact_path,
        "output_root": output_root,
        "overwrite": bool(args.overwrite),
    }


def _execute_fit_task(task: dict[str, Any]) -> dict[str, Any]:
    """Execute a fit task produced by :func:`_build_fit_task`.

    Catches all exceptions, logs them, and returns a failed inventory record
    rather than propagating, so the caller's task batch can continue.
    """
    fit_payload = task["fit_payload"]
    container = task["container"]
    artifact_path = task["artifact_path"]
    output_root = task["output_root"]
    overwrite = task["overwrite"]
    logger.info(
        "Fitting %s/%s/%s/%s/n%d",
        fit_payload["condition"],
        fit_payload["analysis_mode"],
        fit_payload["unit_name"],
        fit_payload["reducer"],
        fit_payload["n_components"],
    )
    try:
        return run_fit(
            fit_payload=fit_payload,
            container=container,
            out_path=artifact_path,
            output_root=output_root,
            overwrite=overwrite,
        )
    except Exception as err:
        logger.exception(
            "Fit failed for %s/%s/%s/n%d",
            fit_payload["condition"],
            fit_payload["unit_name"],
            fit_payload["reducer"],
            fit_payload["n_components"],
        )
        return _build_fit_record(
            fit_payload={**fit_payload, "status": "failed"},
            artifact_path=artifact_path,
            output_root=output_root,
            error=str(err),
        )


def _build_eval_task(
    fit_record: dict[str, Any],
    eval_spec: dict[str, Any],
    container: DataContainer,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Build a serialisable eval task dict from a fit record and eval spec.

    Pre-computes ``eval_id``, the artifact path, and a skeleton eval payload
    (without metrics) that is used as a failure fallback in
    :func:`_execute_eval_task`.
    """
    fit_path = output_root / fit_record["artifact_path"]
    fit_artifact = load_fit_artifact(fit_path)
    selected_ids: list[str] = []
    selected_groups: list[str] = []
    selected_labels: list[str] = []
    try:
        _, selected_ids_array, selected_labels_array, selected_groups_array = (
            _prepare_eval_inputs(
                container=container,
                fit_ids=np.asarray(fit_artifact["ids"], dtype=object).astype(str),
                eval_spec=eval_spec,
            )
        )
        selected_ids = selected_ids_array.tolist()
        selected_labels = selected_labels_array.tolist()
        selected_groups = selected_groups_array.tolist()
    except Exception:
        selected_ids = []

    eval_id = hashlib.sha256(
        json.dumps(
            {
                "fit_id": fit_record["fit_id"],
                "eval_name": eval_spec["name"],
                "target_col": eval_spec["target_col"],
                "group_col": eval_spec["group_col"],
                "filters": eval_spec["filters"],
                "label_map": eval_spec["label_map"],
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]

    artifact_stem = "_".join(
        [
            "sub-all",
            "ses-all",
            f"scope-{_slug(fit_record['scope'], max_len=32)}",
            f"cond-{_slug(fit_record['condition'], max_len=32)}",
            f"unit-{_slug(fit_record['unit_key'], max_len=32)}",
            f"reducer-{_slug(fit_record['reducer'], max_len=32)}",
            f"components-{int(fit_record['n_components'])}",
            f"eval-{_slug(eval_spec['name'], max_len=32)}",
            f"eval-{eval_id}",
        ]
    )
    # Skeleton payload — used as failure fallback if run_eval raises.
    eval_payload: dict[str, Any] = {
        "eval_id": eval_id,
        "fit_id": fit_record["fit_id"],
        "scope": fit_record["scope"],
        "condition": fit_record["condition"],
        "analysis_mode": fit_record["analysis_mode"],
        "unit_type": fit_record["unit_type"],
        "unit_name": fit_record["unit_name"],
        "unit_key": fit_record["unit_key"],
        "family": fit_record.get("family"),
        "eval_name": eval_spec["name"],
        "input_mode": fit_record["input_mode"],
        "representation": fit_record["representation"],
        "aggregation_unit": fit_record.get("aggregation_unit"),
        "run_label": fit_record.get("run_label"),
        "reducer": fit_record["reducer"],
        "n_components": int(fit_record["n_components"]),
        "target_col": eval_spec["target_col"],
        "group_col": eval_spec["group_col"],
        "filters": list(eval_spec["filters"]),
        "label_map": dict(eval_spec["label_map"]),
        "descriptor_families": list(fit_record.get("descriptor_families", [])),
        "descriptor_max_abs_value": fit_record.get("descriptor_max_abs_value"),
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(selected_ids)),
        "n_groups": (
            int(pd.Index(selected_groups).nunique()) if selected_groups else 0
        ),
        "n_labels": (
            int(pd.Index(selected_labels).nunique()) if selected_labels else 0
        ),
        "artifact_stem": artifact_stem,
    }
    artifact_path = (
        output_root
        / "sub-all"
        / "ses-all"
        / "eeg"
        / "evals"
        / f"scope-{_slug(fit_record['scope'])}"
        / f"cond-{_slug(fit_record['condition'])}"
        / f"unit-{_slug(fit_record['unit_key'])}"
        / f"reducer-{_slug(fit_record['reducer'])}"
        / f"components-{int(fit_record['n_components'])}"
        / f"eval-{_slug(eval_spec['name'])}"
        / f"fit-{fit_record['fit_id']}"
        / f"eval-{eval_id}"
    )
    return {
        "fit_record": fit_record,
        "fit_artifact": fit_artifact,
        "eval_spec": eval_spec,
        "container": container,
        "artifact_path": artifact_path,
        "output_root": output_root,
        "overwrite": bool(overwrite),
        "eval_payload": eval_payload,
    }


def _execute_eval_task(task: dict[str, Any]) -> dict[str, Any]:
    """Execute an eval task produced by :func:`_build_eval_task`.

    Catches all exceptions and returns a failed inventory record so the
    caller's task batch can continue.
    """
    fit_record = task["fit_record"]
    fit_artifact = task["fit_artifact"]
    eval_spec = task["eval_spec"]
    container = task["container"]
    artifact_path = task["artifact_path"]
    output_root = task["output_root"]
    overwrite = task["overwrite"]
    eval_payload = task["eval_payload"]
    logger.info(
        "Evaluating %s/%s/%s/%s/n%d [%s]",
        fit_record["condition"],
        fit_record["analysis_mode"],
        fit_record["unit_name"],
        fit_record["reducer"],
        fit_record["n_components"],
        eval_spec["name"],
    )
    try:
        return run_eval(
            fit_payload=fit_artifact["fit"],
            fit_artifact=fit_artifact,
            container=container,
            eval_spec=eval_spec,
            out_path=artifact_path,
            output_root=output_root,
            overwrite=overwrite,
        )
    except Exception as err:
        logger.exception(
            "Eval failed for %s/%s/%s/%s/n%d [%s]",
            fit_record["condition"],
            fit_record["analysis_mode"],
            fit_record["unit_name"],
            fit_record["reducer"],
            fit_record["n_components"],
            eval_spec["name"],
        )
        return _build_eval_record(
            eval_payload={**eval_payload, "status": "failed"},
            artifact_path=artifact_path,
            output_root=output_root,
            error=str(err),
        )
