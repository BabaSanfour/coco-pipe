"""Study-agnostic per-unit runner and sweep orchestration for decoding.

Mirrors :mod:`coco_pipe.dim_reduction.pipeline`: a study enumerates the
independent decoding units of a sweep (one :class:`DecodingUnit` each). Two
layers live here and are deliberately free of any report/visualization imports
(reports are built study-side from the saved ``result.joblib`` artifacts):

* :func:`run_decoding_unit` owns the per-unit lifecycle — resume, run, export,
  record extraction;
* :func:`execute_decoding_sweep` runs many units (optionally in parallel via
  :func:`coco_pipe.utils.run_task_batch`) and persists one flat result table plus
  a resumable ``runs/`` inventory (``sweep_runs.json`` + ``run_summary.json`` +
  ``leaderboard.json``) and a run-status marker.
"""

from __future__ import annotations

import gc
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from coco_pipe.io import read_json, write_json
from coco_pipe.utils import resolve_n_jobs, run_task_batch

from ._constants import (
    CONFIG_SNAPSHOT_FILENAME,
    FAILURES_FILENAME,
    LEADERBOARD_FILENAME,
    RESULTS_FILENAME,
    RUN_SUMMARY_FILENAME,
    RUNS_DIRNAME,
    SWEEP_RUNS_FILENAME,
)
from .configs import ExperimentConfig
from .experiment import Experiment
from .persistence import (
    completed_for_config,
    load_completed_result_records,
    redact_sensitive,
    stamp_primary_metric,
    write_run_status,
)

LOGGER = logging.getLogger(__name__)

PrimaryMask = Callable[[pd.DataFrame], "pd.Series"]

__all__ = [
    "DecodingUnit",
    "allocate_inner_jobs",
    "allocate_outer_inner",
    "build_leaderboard",
    "execute_decoding_sweep",
    "execute_decoding_sweep_streaming",
    "load_sweep_records",
    "run_decoding_unit",
    "write_run_summary",
]


@dataclass
class DecodingUnit:
    """One independent unit of a decoding sweep, ready to run in isolation.

    The unit carries a fully-built :class:`ExperimentConfig` plus the data and
    provenance needed to fit, resume, export, and label it. It is a plain data
    object (picklable) so it can be dispatched to a process pool.
    """

    experiment_config: ExperimentConfig
    X: np.ndarray
    y: np.ndarray
    output_dir: Path
    context: dict[str, Any]
    run_config: dict[str, Any]
    groups: np.ndarray | None = None
    feature_names: Sequence[str] | None = None
    sample_ids: Sequence[Any] | None = None
    sample_metadata: pd.DataFrame | None = None
    observation_level: str = "sample"
    inferential_unit: str | None = None
    overwrite: bool = False
    include_p_values: bool = True


def _unit_records(
    result: Any,
    *,
    context: Mapping[str, Any],
    output_dir: Path,
    include_p_values: bool,
    metrics: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """One flat record per model: run context + summary metrics (+ p-value)."""
    summary = result.summary().reset_index()
    stats = result.get_statistical_assessment() if include_p_values else pd.DataFrame()
    records: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        model = row.get("Model")
        record: dict[str, Any] = {
            **dict(context),
            "model": model,
            "status": "success",
            "output_dir": str(output_dir),
            **{str(key): value for key, value in row.items() if key != "Model"},
        }
        if not stats.empty and {"Model", "PValue"}.issubset(stats.columns):
            model_stats = stats[stats["Model"] == model]
            if "Metric" in model_stats:
                model_stats = model_stats[model_stats["Metric"] == "accuracy"]
            if not model_stats.empty:
                record["p_value"] = float(model_stats.iloc[0]["PValue"])
        records.append(stamp_primary_metric(record, metrics))

    # Surface a clear per-model failure record for any model that did not make
    # it into the summary (e.g. degenerate folds during the observed run).
    # Without this, callers receive an empty list and later see a cryptic
    # KeyError: 'Model' when they try to index into the records.
    succeeded = {r["model"] for r in records}
    for model_name, res in result.raw.items():
        if model_name in succeeded:
            continue
        error_msg = res.get("error", "unknown error") if isinstance(res, dict) else str(res)
        records.append(
            stamp_primary_metric(
                {
                    **dict(context),
                    "model": model_name,
                    "status": "failed",
                    "reason": error_msg,
                    "output_dir": str(output_dir),
                },
                metrics,
            )
        )

    return records


def run_decoding_unit(
    unit: DecodingUnit,
    *,
    errors: Literal["raise", "record"] = "record",
) -> list[dict[str, Any]]:
    """Run one decoding unit and return one record per model.

    Resumes from a matching completed output when ``overwrite`` is false, else
    runs the experiment, exports it (``result.joblib`` + tidy tables), and
    extracts records. A resume config-hash mismatch is intentionally *not*
    caught — it raises so the sweep aborts rather than silently mixing configs.
    A runtime failure becomes a single ``status="failed"`` record when
    ``errors="record"`` (the default for a fault-tolerant sweep).
    """
    output_dir = Path(unit.output_dir)
    metrics = list(unit.experiment_config.metrics or ())
    # Resume check runs outside the try: a hash mismatch must abort the sweep.
    if not unit.overwrite and completed_for_config(output_dir, unit.run_config):
        LOGGER.info("Skipping completed decoding unit: %s", output_dir.name)
        return load_completed_result_records(
            output_dir, context=unit.context, metrics=metrics
        )
    try:
        result = Experiment(unit.experiment_config).run(
            unit.X,
            unit.y,
            groups=unit.groups,
            feature_names=unit.feature_names,
            sample_ids=unit.sample_ids,
            sample_metadata=unit.sample_metadata,
            observation_level=unit.observation_level,
            inferential_unit=unit.inferential_unit,
        )
        result.export(output_dir, config=unit.run_config)
        return _unit_records(
            result,
            context=unit.context,
            output_dir=output_dir,
            include_p_values=unit.include_p_values,
            metrics=metrics,
        )
    except Exception as exc:
        if errors == "raise":
            raise
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "_FAILED").write_text("", encoding="utf-8")
        return [
            {
                **dict(unit.context),
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "output_dir": str(output_dir),
            }
        ]


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------


def allocate_inner_jobs(cfg: ExperimentConfig, n_jobs: int) -> ExperimentConfig:
    """Return a copy of the experiment config with a new inner ``n_jobs``.

    Hands the whole job budget to a lone unit's folds/permutations when only one
    outer worker is active.
    """
    tuning = (
        cfg.tuning.model_copy(update={"n_jobs": n_jobs})
        if cfg.tuning.enabled
        else cfg.tuning
    )
    return cfg.model_copy(update={"n_jobs": n_jobs, "tuning": tuning})


def allocate_outer_inner(total_jobs: int, n_units: int) -> tuple[int, int]:
    """Split a job budget into outer workers and per-unit inner jobs."""
    outer = max(1, min(total_jobs, max(n_units, 1)))
    inner = total_jobs if outer == 1 else 1
    return outer, inner


def _records_to_json_safe(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert records (possibly holding numpy scalars / NaN) to JSON-safe dicts."""
    if not records:
        return []
    return json.loads(pd.DataFrame(list(records)).to_json(orient="records"))


def build_leaderboard(
    records: Sequence[dict[str, Any]],
    group_fields: Sequence[str],
    *,
    primary_metric: str = "primary_metric",
) -> list[dict[str, Any]]:
    """Best successful record per group, ranked by the primary metric."""
    frame = pd.DataFrame(list(records))
    if frame.empty or primary_metric not in frame or "status" not in frame:
        return []
    success = frame[frame["status"].astype(str) == "success"].copy()
    success[primary_metric] = pd.to_numeric(success[primary_metric], errors="coerce")
    success = success[success[primary_metric].notna()]
    if success.empty:
        return []
    groups = [field for field in group_fields if field in success]
    ranked = success.sort_values(primary_metric, ascending=False, kind="stable")
    if groups:
        ranked = ranked.groupby(groups, dropna=False, sort=False).head(1)
    return _records_to_json_safe(ranked.to_dict("records"))


def write_run_summary(
    output_root: Path,
    records: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    *,
    primary_mask: PrimaryMask | None = None,
    run_metadata: dict[str, Any] | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Write ``runs/run_summary.json`` and the top-level status marker.

    Status is ``SUCCESS`` when a primary result succeeded and nothing failed,
    ``PARTIAL`` when there is any success, else ``FAILED`` — matching the
    dim-reduction run-status contract.
    """
    frame = pd.DataFrame(list(records))
    if "status" in frame:
        n_success = int((frame["status"].astype(str) == "success").sum())
    else:
        n_success = 0
    n_failed = len(list(failures))
    primary_success = False
    if primary_mask is not None and not frame.empty:
        primary_success = bool(primary_mask(frame).fillna(False).any())

    if primary_success and n_failed == 0:
        status = "SUCCESS"
    elif n_success > 0:
        status = "PARTIAL"
    else:
        status = "FAILED"

    payload: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "unit_total": len(frame),
        "unit_success": n_success,
        "unit_failed": n_failed,
        "primary_success": primary_success,
        "report_path": str(report_path) if report_path is not None else None,
    }
    if run_metadata:
        payload.update(run_metadata)

    write_json(output_root / RUNS_DIRNAME / RUN_SUMMARY_FILENAME, payload, indent=2)
    write_run_status(output_root, status)
    return payload


def load_sweep_records(output_root: Path) -> list[dict[str, Any]]:
    """Load the persisted per-unit records for a ``--reports-only`` pass."""
    runs_path = Path(output_root) / RUNS_DIRNAME / SWEEP_RUNS_FILENAME
    if not runs_path.exists():
        raise RuntimeError(
            f"No persisted sweep inventory at {runs_path}; run the sweep before "
            "requesting reports-only."
        )
    runs = read_json(runs_path)
    if not isinstance(runs, list):
        raise ValueError(f"Expected a list payload in {runs_path}.")
    return runs


def execute_decoding_sweep(
    units: list[DecodingUnit],
    failures: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    output_root: Path,
    results_filename: str = RESULTS_FILENAME,
    primary_mask: PrimaryMask | None = None,
    leaderboard_group_fields: Sequence[str] = (),
    reallocate_inner_jobs: bool = False,
    frame_post: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    extra_outputs: (
        Callable[[pd.DataFrame, list[dict[str, Any]], Path], None] | None
    ) = None,
    run_metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Run a decoding sweep and persist its results + resumable ``runs/`` inventory.

    Owns the study-agnostic tail shared by classical and foundation decoding:
    job allocation, per-unit execution, the flat result table, the ``runs/``
    inventory, and the run-status marker. Study-specific pre/post steps (p-value
    correction, capability matrices) inject via *frame_post* / *extra_outputs*.
    """
    total_jobs = resolve_n_jobs(int(config["n_jobs"]))
    raw_records = _run_unit_batch(
        units,
        total_jobs=total_jobs,
        reallocate_inner_jobs=reallocate_inner_jobs,
    )
    return _finalize_sweep(
        raw_records,
        failures,
        config=config,
        output_root=output_root,
        results_filename=results_filename,
        primary_mask=primary_mask,
        leaderboard_group_fields=leaderboard_group_fields,
        frame_post=frame_post,
        extra_outputs=extra_outputs,
        run_metadata=run_metadata,
    )


def execute_decoding_sweep_streaming(
    unit_batches: Iterable[Sequence[DecodingUnit]],
    failures: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    output_root: Path,
    results_filename: str = RESULTS_FILENAME,
    primary_mask: PrimaryMask | None = None,
    leaderboard_group_fields: Sequence[str] = (),
    reallocate_inner_jobs: bool = False,
    frame_post: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    extra_outputs: (
        Callable[[pd.DataFrame, list[dict[str, Any]], Path], None] | None
    ) = None,
    run_metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Run a decoding sweep from a lazy stream of per-scope unit batches.

    Behaves like :func:`execute_decoding_sweep` but consumes *unit_batches* one
    batch at a time — running and then releasing each batch (and, via the
    generator, its source data container) before the next is materialized — so
    only one scope's worth of arrays is resident at once. This keeps host memory
    bounded to the largest single scope rather than the whole sweep. *failures*
    is the caller-owned list of enumeration-time skips, extended in place with
    runtime failures (so any *extra_outputs* closure over it sees them all), and
    every batch shares the same job budget/parallelism policy as the one-shot
    variant. Outputs are persisted once, after the stream is exhausted.
    """
    total_jobs = resolve_n_jobs(int(config["n_jobs"]))
    raw_records: list[dict[str, Any]] = []
    for batch in unit_batches:
        raw_records.extend(
            _run_unit_batch(
                batch,
                total_jobs=total_jobs,
                reallocate_inner_jobs=reallocate_inner_jobs,
            )
        )
        # Drop the batch's units (and their X arrays) before pulling the next
        # scope so peak RSS tracks one scope, not the accumulated sweep.
        del batch
        gc.collect()
    return _finalize_sweep(
        raw_records,
        failures,
        config=config,
        output_root=output_root,
        results_filename=results_filename,
        primary_mask=primary_mask,
        leaderboard_group_fields=leaderboard_group_fields,
        frame_post=frame_post,
        extra_outputs=extra_outputs,
        run_metadata=run_metadata,
    )


def _run_unit_batch(
    units: Sequence[DecodingUnit],
    *,
    total_jobs: int,
    reallocate_inner_jobs: bool,
) -> list[dict[str, Any]]:
    """Run one batch of decoding units and return their flat records."""
    units = list(units)
    if not units:
        return []
    outer_workers, inner_n_jobs = allocate_outer_inner(total_jobs, len(units))
    if reallocate_inner_jobs and inner_n_jobs != 1:
        for unit in units:
            unit.experiment_config = allocate_inner_jobs(
                unit.experiment_config, inner_n_jobs
            )
    LOGGER.info(
        "Running %d decoding unit(s) with %d outer worker(s) (inner n_jobs=%d).",
        len(units),
        outer_workers,
        inner_n_jobs,
    )
    return [
        record
        for unit_records in run_task_batch(units, run_decoding_unit, outer_workers)
        for record in unit_records
    ]


def _finalize_sweep(
    raw_records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    output_root: Path,
    results_filename: str,
    primary_mask: PrimaryMask | None,
    leaderboard_group_fields: Sequence[str],
    frame_post: Callable[[pd.DataFrame], pd.DataFrame] | None,
    extra_outputs: Callable[[pd.DataFrame, list[dict[str, Any]], Path], None] | None,
    run_metadata: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Persist the sweep's flat result table + resumable ``runs/`` inventory."""
    failures.extend(
        record for record in raw_records if record.get("status") == "failed"
    )

    frame = pd.DataFrame(raw_records)
    if frame_post is not None and not frame.empty:
        frame = frame_post(frame)
    # Persist and return the post-processed records (e.g. with FDR columns) so a
    # --reports-only pass reproduces byte-identical reports from disk.
    records = _records_to_json_safe(frame.to_dict("records")) if not frame.empty else []

    output_root = Path(output_root)
    runs_dir = output_root / RUNS_DIRNAME
    runs_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / results_filename, index=False)
    pd.DataFrame(failures).to_csv(output_root / FAILURES_FILENAME, index=False)
    (output_root / CONFIG_SNAPSHOT_FILENAME).write_text(
        yaml.safe_dump(redact_sensitive(config), sort_keys=False),
        encoding="utf-8",
    )
    write_json(runs_dir / SWEEP_RUNS_FILENAME, records, indent=2)
    write_json(
        runs_dir / LEADERBOARD_FILENAME,
        build_leaderboard(records, leaderboard_group_fields),
        indent=2,
    )
    if extra_outputs is not None:
        extra_outputs(frame, records, output_root)

    write_run_summary(
        output_root,
        records,
        failures,
        primary_mask=primary_mask,
        run_metadata=run_metadata,
    )
    return records, frame
