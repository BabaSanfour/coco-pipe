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
valid_n_components_for_container
    Check whether *n_components* is feasible for a container's matrix shape.
valid_component_sweep
    Filter a list of component counts to feasible values.
prepare_eval_inputs
    Align a DataContainer to saved fit ids and resolve eval labels/groups.
build_fit_request / build_eval_request
    Construct request dictionaries that can be passed to ``run_fit`` and
    ``run_eval``.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coco_pipe.dim_reduction._constants import (
    ARTIFACTS_DIRNAME,
    EVAL_METRIC_COLUMNS,
    EVALS_SUBDIR,
    FIT_METRIC_COLUMNS,
    FIT_PROVENANCE_FIELDS,
    FITS_SUBDIR,
    POOLED_CONDITION,
    STEM_SLUG_MAX_LEN,
    ErrorMode,
)
from coco_pipe.dim_reduction.artifacts import (
    _load_eval_payload,
    build_record,
    load_fit_artifact,
    save_eval_artifact,
    save_fit_artifact,
)
from coco_pipe.dim_reduction.config import DEFAULT_EVAL_GROUP_COL, MISSING_EVAL_VALUES
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding
from coco_pipe.io import DataContainer, fingerprint_container
from coco_pipe.utils import slug, stable_hash

__all__ = [
    "POOLED_CONDITION",
    "build_auto_pooled_eval_spec",
    "build_eval_request",
    "build_fit_request",
    "prepare_eval_inputs",
    "run_eval",
    "run_fit",
    "run_fit_group",
    "supports_nested_components",
    "valid_component_sweep",
    "valid_n_components_for_container",
]

logger = logging.getLogger(__name__)


def _normalize_ids(ids: Sequence[Any]) -> list[str]:
    """Safely cast a sequence of IDs (including tuples/arrays) to a list of strings."""

    def _to_str(v: Any) -> str:
        if isinstance(v, (list, tuple, np.ndarray)):
            return "_".join(str(x) for x in v)
        return str(v)

    return [_to_str(x) for x in ids]


@cache
def supports_nested_components(method: str) -> bool:
    """Whether *method* can synthesise its whole sweep from one max-n fit.

    Nested reducers (PCA family, SVD) decompose once at the largest
    ``n_components`` and slice the smaller sweep values out of that single fit;
    everything else (UMAP, t-SNE, PHATE, Isomap, ICA, …) must fit independently
    per dimension. Callers use this both to fit efficiently
    (:func:`run_fit_group`) and to decide the parallel grain: a non-nested
    reducer's sweep is a set of independent fits that can run as separate tasks
    rather than one serial group. Cached because instantiating ``DimReduction``
    only to read ``capabilities`` is wasteful to repeat per request.
    """
    try:
        return bool(
            DimReduction(method=method, n_components=2).capabilities.get(
                "nested_components", False
            )
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core fit / eval
# ---------------------------------------------------------------------------


def _evr_scalars(explained_variance: np.ndarray | None) -> dict[str, float]:
    """Derive promoted scalar diagnostics from an explained-variance vector.

    Returns ``participation_ratio`` (effective dimensionality,
    ``(Σλ)² / Σλ²``) and ``cumulative_explained_variance`` (the fraction of
    variance captured by the retained components, i.e. ``Σλ``).  Empty when
    *explained_variance* is ``None``/empty so non-PCA reducers contribute no
    such columns.
    """
    if explained_variance is None:
        return {}
    evr = np.asarray(explained_variance, dtype=float)
    if evr.size == 0:
        return {}
    total = float(evr.sum())
    participation_ratio = float(total**2 / float((evr**2).sum())) if total > 0 else 0.0
    return {
        "participation_ratio": participation_ratio,
        "cumulative_explained_variance": total,
    }


def run_fit(
    fit_payload: dict[str, Any],
    container: DataContainer,
    out_path: Path,
    output_root: Path,
    overwrite: bool,
    *,
    errors: ErrorMode = "raise",
) -> dict[str, Any]:
    """Fit one reducer variant and return a fit-runs inventory record.

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
    errors:
        ``"raise"`` (default) propagates exceptions; ``"record"`` catches them,
        logs, and returns a failed inventory record of the same shape.

    Returns
    -------
    dict
        A flat inventory record suitable for passing to :func:`update_runs`.
    """
    if errors not in {"raise", "record"}:
        raise ValueError("errors must be 'raise' or 'record'.")
    try:
        success_marker = out_path / "_SUCCESS"
        if success_marker.exists() and not overwrite:
            artifact = load_fit_artifact(out_path)
            return build_record(
                artifact["fit"],
                out_path,
                output_root,
                FIT_METRIC_COLUMNS,
                artifact["metrics"],
            )

        if overwrite and out_path.exists():
            shutil.rmtree(out_path)

        X = np.asarray(container.X)
        if X.ndim != 2:
            raise ValueError("run_fit expects a 2D matrix.")
        if container.ids is None:
            raise ValueError("Dim-reduction fits expect container.ids to be present.")
        ids = _normalize_ids(container.ids)

        reducer = DimReduction(
            method=fit_payload["reducer"],
            n_components=fit_payload["n_components"],
            params=dict(fit_payload.get("reducer_params") or {}),
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
            explained_variance = np.asarray(explained_variance)
            diagnostics["explained_variance_ratio"] = explained_variance
            fit_payload = {**fit_payload, **_evr_scalars(explained_variance)}

        save_fit_artifact(
            out_path, embedding, ids, fit_payload, metrics_payload, diagnostics
        )
        return build_record(
            fit_payload, out_path, output_root, FIT_METRIC_COLUMNS, metrics_payload
        )
    except Exception as err:
        if errors == "raise":
            raise
        logger.exception(
            "Fit failed for %s/%s/%s/n%s",
            fit_payload.get("condition"),
            fit_payload.get("unit_name"),
            fit_payload.get("reducer"),
            fit_payload.get("n_components"),
        )
        return build_record(
            {**fit_payload, "status": "failed"},
            out_path,
            output_root,
            FIT_METRIC_COLUMNS,
            error=str(err),
        )


def _metrics_payload_from_scores(score_metrics: dict[str, Any]) -> dict[str, Any]:
    """Promote geometry-quality scores to the canonical fit-metric columns.

    Mirrors :func:`run_fit`: each :data:`FIT_METRIC_COLUMNS` entry becomes a
    ``float`` or ``None`` (for missing/NaN values).
    """
    return {
        metric_name: (
            None
            if np.isnan(score_metrics.get(metric_name, np.nan))
            else float(score_metrics.get(metric_name))
        )
        for metric_name in FIT_METRIC_COLUMNS
    }


def run_fit_group(
    requests: list[dict[str, Any]],
    *,
    errors: ErrorMode = "raise",
) -> list[dict[str, Any]]:
    """Fit a group of requests sharing one analysis unit and reducer.

    All *requests* must describe the same container and reducer, differing only
    in ``n_components`` (as produced by :func:`build_fit_request` for one unit's
    sweep).  When the reducer is hierarchically nested, the largest
    ``n_components`` is fitted once and the smaller sweep values are synthesised
    by slicing the embedding, components, and explained-variance arrays —
    avoiding a redundant decomposition per sweep value.  Non-nested reducers
    (or singleton groups) fall back to an independent :func:`run_fit` per request,
    so behaviour is unchanged for UMAP/t-SNE/ICA/etc.

    Returns one inventory record per request, in the input order's resolution
    (resumed first, then synthesised).
    """
    if errors not in {"raise", "record"}:
        raise ValueError("errors must be 'raise' or 'record'.")
    if not requests:
        return []

    method = str(requests[0]["fit_payload"]["reducer"])
    # A nested reducer (PCA family, SVD) lets the sweep be synthesised from one
    # max-n fit; anything else (ICA, UMAP, t-SNE, …) must fit per dimension.
    nested = supports_nested_components(method)
    if len(requests) == 1 or not nested:
        return [run_fit(**request, errors=errors) for request in requests]

    output_root = requests[0]["output_root"]
    records: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for request in requests:
        out_path = request["out_path"]
        if (out_path / "_SUCCESS").exists() and not request["overwrite"]:
            artifact = load_fit_artifact(out_path)
            records.append(
                build_record(
                    artifact["fit"],
                    out_path,
                    output_root,
                    FIT_METRIC_COLUMNS,
                    artifact["metrics"],
                )
            )
        else:
            pending.append(request)
    if not pending:
        return records

    # Fit once at the largest pending target dimension, then slice downward.
    try:
        container = requests[0]["container"]
        X = np.asarray(container.X)
        if X.ndim != 2:
            raise ValueError("run_fit_group expects a 2D matrix.")
        if container.ids is None:
            raise ValueError("Dim-reduction fits expect container.ids to be present.")
        ids = _normalize_ids(container.ids)
        max_n = max(int(r["fit_payload"]["n_components"]) for r in pending)
        reducer = DimReduction(
            method=method,
            n_components=max_n,
            params=dict(requests[0]["fit_payload"].get("reducer_params") or {}),
        )
        full_embedding = np.asarray(reducer.fit_transform(container).X)
        try:
            full_components = np.asarray(reducer.get_components())
        except Exception:
            full_components = None
        full_evr = getattr(reducer.reducer, "explained_variance_ratio_", None)
        full_evr = None if full_evr is None else np.asarray(full_evr)
    except Exception as err:
        if errors == "raise":
            raise
        logger.exception(
            "Grouped fit failed for %s/%s/%s",
            requests[0]["fit_payload"].get("condition"),
            requests[0]["fit_payload"].get("unit_name"),
            method,
        )
        for request in pending:
            records.append(
                build_record(
                    {**request["fit_payload"], "status": "failed"},
                    request["out_path"],
                    output_root,
                    FIT_METRIC_COLUMNS,
                    error=str(err),
                )
            )
        return records

    for request in pending:
        fit_payload = request["fit_payload"]
        out_path = request["out_path"]
        n_components = int(fit_payload["n_components"])
        try:
            if request["overwrite"] and out_path.exists():
                shutil.rmtree(out_path)
            embedding = full_embedding[:, :n_components]
            components = (
                None if full_components is None else full_components[:n_components]
            )
            evr = None if full_evr is None else full_evr[:n_components]
            score_payload = reducer.score(embedding, X=X, metrics=FIT_METRIC_COLUMNS)
            metrics_payload = _metrics_payload_from_scores(
                dict(score_payload["metrics"])
            )

            diagnostics: dict[str, Any] = {"score_payload": score_payload}
            if components is not None:
                diagnostics["components"] = components
            if evr is not None:
                diagnostics["explained_variance_ratio"] = evr
            synthesized_payload = {**fit_payload, **_evr_scalars(evr)}

            save_fit_artifact(
                out_path,
                embedding,
                ids,
                synthesized_payload,
                metrics_payload,
                diagnostics,
            )
            records.append(
                build_record(
                    synthesized_payload,
                    out_path,
                    output_root,
                    FIT_METRIC_COLUMNS,
                    metrics_payload,
                )
            )
        except Exception as err:
            if errors == "raise":
                raise
            logger.exception(
                "Sliced fit failed for %s/%s/%s/n%s",
                fit_payload.get("condition"),
                fit_payload.get("unit_name"),
                method,
                n_components,
            )
            records.append(
                build_record(
                    {**fit_payload, "status": "failed"},
                    out_path,
                    output_root,
                    FIT_METRIC_COLUMNS,
                    error=str(err),
                )
            )
    return records


def _eval_id(fit_id: str, eval_spec: dict[str, Any]) -> str:
    return stable_hash(
        {
            "fit_id": fit_id,
            "eval_name": eval_spec["name"],
            "target_col": eval_spec["target_col"],
            "group_col": eval_spec["group_col"],
            "filters": eval_spec["filters"],
            "label_map": eval_spec["label_map"],
        },
        length=16,
    )


def _eval_artifact_stem(
    fit_payload: dict[str, Any],
    eval_spec: dict[str, Any],
    eval_id: str,
) -> str:
    return "_".join(
        [
            "eval",
            slug(fit_payload["scope"], max_len=STEM_SLUG_MAX_LEN),
            slug(fit_payload["condition"], max_len=STEM_SLUG_MAX_LEN),
            slug(fit_payload["unit_key"], max_len=STEM_SLUG_MAX_LEN),
            slug(fit_payload["reducer"], max_len=STEM_SLUG_MAX_LEN),
            f"n{int(fit_payload['n_components'])}",
            slug(eval_spec["name"], max_len=STEM_SLUG_MAX_LEN),
            str(fit_payload["fit_id"]),
            eval_id,
        ]
    )


def _base_eval_payload(
    fit_payload: dict[str, Any],
    eval_spec: dict[str, Any],
    eval_id: str,
    *,
    artifact_stem: str,
    n_samples: int = 0,
    n_groups: int = 0,
    n_labels: int = 0,
) -> dict[str, Any]:
    payload = {field: fit_payload.get(field) for field in FIT_PROVENANCE_FIELDS}
    payload.update(
        {
            "n_components": int(fit_payload["n_components"]),
            "descriptor_families": list(fit_payload.get("descriptor_families", [])),
            "eval_id": eval_id,
            "eval_name": eval_spec["name"],
            "target_col": eval_spec["target_col"],
            "group_col": eval_spec["group_col"],
            "filters": list(eval_spec["filters"]),
            "label_map": dict(eval_spec["label_map"]),
            "status": "success",
            "timestamp": datetime.now(UTC).isoformat(),
            "n_samples": int(n_samples),
            "n_groups": int(n_groups),
            "n_labels": int(n_labels),
            "artifact_stem": artifact_stem,
        }
    )
    return payload


def run_eval(
    fit_artifact: dict[str, Any],
    container: DataContainer,
    eval_spec: dict[str, Any],
    out_path: Path,
    output_root: Path,
    overwrite: bool,
    *,
    errors: ErrorMode = "raise",
) -> dict[str, Any]:
    """Run one post-hoc evaluation and return an eval-runs inventory record.

    The fit provenance is read from ``fit_artifact["fit"]``.  If ``_SUCCESS``
    already exists in *out_path* and *overwrite* is ``False`` the existing eval
    artifact is loaded and its inventory record is returned immediately
    (checkpoint resume).

    Parameters
    ----------
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
    errors:
        ``"raise"`` (default) propagates exceptions; ``"record"`` catches them,
        logs, and returns a failed inventory record.

    Returns
    -------
    dict
        A flat eval inventory record suitable for passing to
        :func:`update_runs`.
    """
    if errors not in {"raise", "record"}:
        raise ValueError("errors must be 'raise' or 'record'.")
    fit_payload = dict(fit_artifact["fit"])
    try:
        success_marker = out_path / "_SUCCESS"
        if success_marker.exists() and not overwrite:
            eval_payload = _load_eval_payload(out_path)
            return build_record(
                eval_payload,
                out_path,
                output_root,
                EVAL_METRIC_COLUMNS,
                eval_payload.get("metrics"),
            )

        if overwrite and out_path.exists():
            shutil.rmtree(out_path)

        selected_index, selected_ids, labels, groups = prepare_eval_inputs(
            container=container,
            fit_ids=_normalize_ids(fit_artifact["ids"]),
            eval_spec=eval_spec,
        )
        eval_id = _eval_id(str(fit_payload["fit_id"]), eval_spec)

        stored_embedding = fit_artifact["embedding"]
        if isinstance(stored_embedding, DataContainer):
            stored_embedding = stored_embedding.X
        embedding = np.asarray(stored_embedding)[selected_index.to_numpy()]
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
        artifact_stem = _eval_artifact_stem(fit_payload, eval_spec, eval_id)
        eval_payload = {
            **_base_eval_payload(
                fit_payload,
                eval_spec,
                eval_id,
                artifact_stem=artifact_stem,
                n_samples=len(selected_ids),
                n_groups=int(pd.Index(groups).nunique()),
                n_labels=int(pd.Index(labels).nunique()),
            ),
            "metrics": metrics_payload,
            "records": score_payload.get("records", []),
            "metadata": score_payload.get("metadata", {}),
            "artifacts": score_payload.get("artifacts", {}),
        }
        save_eval_artifact(out_path, eval_payload)
        return build_record(
            eval_payload, out_path, output_root, EVAL_METRIC_COLUMNS, metrics_payload
        )
    except Exception as err:
        if errors == "raise":
            raise
        logger.exception(
            "Eval failed for %s/%s/%s/n%s [%s]",
            fit_payload.get("condition"),
            fit_payload.get("unit_name"),
            fit_payload.get("reducer"),
            fit_payload.get("n_components"),
            eval_spec.get("name"),
        )
        try:
            eval_id = _eval_id(str(fit_payload["fit_id"]), eval_spec)
            artifact_stem = _eval_artifact_stem(fit_payload, eval_spec, eval_id)
            eval_payload = _base_eval_payload(
                fit_payload,
                eval_spec,
                eval_id,
                artifact_stem=artifact_stem,
            )
        except Exception:
            eval_payload = {
                "fit_id": fit_payload.get("fit_id"),
                "eval_name": eval_spec.get("name"),
                "status": "failed",
            }
        return build_record(
            {**eval_payload, "status": "failed"},
            out_path,
            output_root,
            EVAL_METRIC_COLUMNS,
            error=str(err),
        )


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def occurrence_aligned_positions(
    container_ids: np.ndarray, fit_ids: np.ndarray
) -> list[int] | None:
    """Row positions in *container_ids* matching *fit_ids* in order.

    A saved fit may cover a differently ordered or smaller subset of the current
    container's observations, and observation ids are **not unique** (many rows
    can share an id). Alignment therefore uses occurrence-count disambiguation:
    the k-th occurrence of an id in *fit_ids* maps to the k-th occurrence of that
    id in *container_ids*. Returns the list of container row positions (one per
    fit id, in fit order), or ``None`` when any fit id occurrence is absent from
    the container.
    """
    container_arr = _normalize_ids(container_ids)
    fit_arr = _normalize_ids(fit_ids)

    key_to_pos: dict[str, int] = {}
    counts: dict[str, int] = {}
    for pos, obs_id in enumerate(container_arr):
        occurrence = counts.get(obs_id, 0)
        counts[obs_id] = occurrence + 1
        key_to_pos.setdefault(f"{obs_id}__{occurrence}", pos)

    positions: list[int] = []
    counts = {}
    for obs_id in fit_arr:
        occurrence = counts.get(obs_id, 0)
        counts[obs_id] = occurrence + 1
        pos = key_to_pos.get(f"{obs_id}__{occurrence}")
        if pos is None:
            return None
        positions.append(pos)
    return positions


def prepare_eval_inputs(
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
    tuple
        A tuple of `(selected_index, selected_ids, labels, groups)`.
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

    container_ids = _normalize_ids(container.ids)
    n_obs = len(container_ids)
    columns: dict[str, np.ndarray] = {"obs_id": container_ids}
    for key, values in container.coords.items():
        arr = np.asarray(values)
        if arr.ndim == 1 and len(arr) == n_obs and key != "feature":
            columns[key] = arr
    if container.y is not None and "y" not in columns:
        columns["y"] = np.asarray(container.y)
    frame = pd.DataFrame(columns)

    positions = occurrence_aligned_positions(container_ids, fit_ids)
    if positions is None:
        raise RuntimeError(
            "Saved fit ids could not be aligned to the current container."
        )
    aligned_frame = frame.iloc[positions].reset_index(drop=True)

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
    labels = labels.mask(labels.str.lower().isin(MISSING_EVAL_VALUES))
    groups = aligned_frame[eval_spec["group_col"]].astype("string").str.strip()
    groups = groups.mask(groups.str.lower().isin(MISSING_EVAL_VALUES))
    valid_mask = labels.notna() & groups.notna()
    if not valid_mask.any():
        raise RuntimeError(f"Eval '{eval_spec['name']}' produced no valid samples.")

    selected_frame = aligned_frame.loc[valid_mask].copy()
    return (
        selected_frame.index,
        _normalize_ids(selected_frame["obs_id"]),
        labels.loc[valid_mask].astype(str).to_numpy(),
        groups.loc[valid_mask].astype(str).to_numpy(),
    )


# ---------------------------------------------------------------------------
# Component sweep helpers
# ---------------------------------------------------------------------------


def valid_n_components_for_container(
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


def valid_component_sweep(
    container: DataContainer, requested: Sequence[int]
) -> list[int]:
    """Filter *requested* to the component counts feasible for *container*.

    Logs a message if any values are skipped.
    """
    valid = [
        int(v) for v in requested if valid_n_components_for_container(container, int(v))
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
) -> dict[str, Any] | None:
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
# Request builders
# ---------------------------------------------------------------------------


def build_fit_request(
    *,
    container: DataContainer,
    scope: str,
    condition: str,
    unit_spec: dict[str, Any],
    reducer: str,
    n_components: int,
    input_signature: dict[str, Any],
    output_root: Path,
    overwrite: bool = False,
    subject_col: str = "subject",
    extra_payload: dict[str, Any] | None = None,
    artifact_path: Path | None = None,
    artifact_path_factory: Callable[[dict[str, Any], Path], Path] | None = None,
    container_signature: dict[str, Any] | None = None,
    reducer_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a request dictionary suitable for passing to :func:`run_fit`.

    The caller owns project-specific input provenance via *input_signature* and
    optional *extra_payload*. coco-pipe owns the deterministic fit id, standard
    fit payload fields, and default flat artifact path. ``reducer_params`` are
    execution controls passed to the reducer but intentionally excluded from the
    fit identity so worker/thread tuning can resume existing scientific outputs.
    """
    if container.ids is None:
        raise ValueError("Dim-reduction fits expect container.ids to be present.")
    ids = _normalize_ids(container.ids)
    reducer_name = str(reducer)
    unit_key = str(unit_spec["unit_key"])
    if container_signature is None:
        container_signature = fingerprint_container(container)
    resolved_reducer_params = dict(reducer_params or {})
    fit_identity = {
        "scope": scope,
        "condition": condition,
        "analysis_mode": input_signature.get("analysis_mode"),
        "unit_type": unit_spec["unit_type"],
        "unit_name": unit_spec["unit_name"],
        "unit_key": unit_key,
        "family": unit_spec.get("family"),
        "subfamily": unit_spec.get("subfamily"),
        "container_signature": container_signature,
        "input_signature": input_signature,
        "reducer": reducer_name,
        "n_components": int(n_components),
        "sample_ids_sha256": hashlib.sha256("\0".join(ids).encode("utf-8")).hexdigest()[
            :16
        ],
        "n_samples": len(ids),
    }
    fit_id = stable_hash(fit_identity, length=16)
    fit_payload: dict[str, Any] = {
        "fit_id": fit_id,
        "scope": scope,
        "condition": condition,
        "analysis_mode": input_signature.get("analysis_mode"),
        "unit_type": unit_spec["unit_type"],
        "unit_name": unit_spec["unit_name"],
        "unit_key": unit_key,
        "family": unit_spec.get("family"),
        "subfamily": unit_spec.get("subfamily"),
        "container_signature": container_signature,
        "input_mode": input_signature.get("input_mode"),
        "representation": input_signature.get("representation"),
        "aggregation_unit": input_signature.get("aggregation_unit"),
        "run_label": input_signature.get("run_label"),
        "descriptor_families": list(
            input_signature.get("descriptor_families", []) or []
        ),
        "descriptor_max_abs_value": input_signature.get("descriptor_max_abs_value"),
        "embedding_model_key": input_signature.get("embedding_model_key"),
        "embedding_representation": input_signature.get("embedding_representation"),
        "embedding_aggregate_by": input_signature.get("embedding_aggregate_by"),
        "reducer": reducer_name,
        "reducer_params": resolved_reducer_params,
        "n_components": int(n_components),
        "status": "success",
        "timestamp": datetime.now(UTC).isoformat(),
        "n_samples": int(container.X.shape[0]),
        "n_subjects": int(
            pd.Index(
                np.asarray(container.coords.get(subject_col, ids), dtype=object).astype(
                    str
                )
            ).nunique()
        ),
        "loaded_obs": int(container.meta.get("loaded_obs", container.X.shape[0])),
        "samples_used": int(container.meta.get("samples_used", container.X.shape[0])),
        "input_signature": input_signature,
    }
    if extra_payload:
        fit_payload.update(extra_payload)
    fit_payload["artifact_stem"] = "_".join(
        [
            "fit",
            slug(fit_payload["scope"], max_len=STEM_SLUG_MAX_LEN),
            slug(fit_payload["condition"], max_len=STEM_SLUG_MAX_LEN),
            slug(fit_payload["unit_key"], max_len=STEM_SLUG_MAX_LEN),
            slug(fit_payload["reducer"], max_len=STEM_SLUG_MAX_LEN),
            f"n{int(fit_payload['n_components'])}",
            str(fit_payload["fit_id"]),
        ]
    )

    if artifact_path is None:
        if artifact_path_factory is not None:
            artifact_path = artifact_path_factory(fit_payload, output_root)
        else:
            stem = fit_payload["artifact_stem"]
            artifact_path = output_root / ARTIFACTS_DIRNAME / FITS_SUBDIR / stem

    return {
        "fit_payload": fit_payload,
        "container": container,
        "out_path": Path(artifact_path),
        "output_root": output_root,
        "overwrite": bool(overwrite),
    }


def build_eval_request(
    *,
    fit_record: dict[str, Any],
    eval_spec: dict[str, Any],
    container: DataContainer,
    output_root: Path,
    overwrite: bool = False,
    fit_artifact: dict[str, Any] | None = None,
    artifact_path: Path | None = None,
    artifact_path_factory: Callable[[dict[str, Any], dict[str, Any], Path], Path]
    | None = None,
) -> dict[str, Any]:
    """Build a request dictionary suitable for passing to :func:`run_eval`.

    By default, the fit artifact is loaded from ``fit_record['artifact_path']``
    relative to *output_root* and the eval artifact is placed under the flat
    ``artifacts/evals`` directory.
    """
    if fit_artifact is None:
        fit_artifact = load_fit_artifact(output_root / fit_record["artifact_path"])
    fit_payload = dict(fit_artifact["fit"])
    eval_id = _eval_id(str(fit_payload["fit_id"]), eval_spec)
    artifact_stem = _eval_artifact_stem(fit_payload, eval_spec, eval_id)
    if artifact_path is None:
        if artifact_path_factory is not None:
            artifact_path = artifact_path_factory(fit_payload, eval_spec, output_root)
        else:
            artifact_path = (
                output_root / ARTIFACTS_DIRNAME / EVALS_SUBDIR / artifact_stem
            )
    return {
        "fit_artifact": fit_artifact,
        "container": container,
        "eval_spec": eval_spec,
        "out_path": Path(artifact_path),
        "output_root": output_root,
        "overwrite": bool(overwrite),
    }
