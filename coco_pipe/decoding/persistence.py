"""Resumable decoding-run artifact utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coco_pipe.utils import stable_hash

_SENSITIVE_KEY_PARTS = ("token", "password", "secret", "api_key", "apikey")


def resolve_primary_metric_name(
    available: Sequence[str],
    metrics: Sequence[str] | None = None,
) -> str | None:
    """Pick the primary metric name from available ``<metric>_mean`` columns.

    Preference order: ``balanced_accuracy`` (the group-robust default), then the
    first configured metric that has a ``_mean`` column, then the first available
    ``*_mean`` column. Returns ``None`` when no scalar metric column is present.
    """
    mean_metrics = [str(col)[:-5] for col in available if str(col).endswith("_mean")]
    if not mean_metrics:
        return None
    if "balanced_accuracy" in mean_metrics:
        return "balanced_accuracy"
    for metric in metrics or ():
        if metric in mean_metrics:
            return metric
    return mean_metrics[0]


def stamp_primary_metric(
    record: dict[str, Any],
    metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Add ``primary_metric_name`` + ``primary_metric`` to a summary record in place.

    Keeps every downstream consumer (leaderboards, head-to-head, reports) on one
    metric contract instead of re-deriving it. No-op when the record carries no
    scalar ``*_mean`` metric column.
    """
    name = resolve_primary_metric_name(record.keys(), metrics)
    if name is not None:
        record["primary_metric_name"] = name
        record["primary_metric"] = record[f"{name}_mean"]
    return record


def redact_sensitive(value: Any) -> Any:
    """Recursively redact credentials before persisting configs or reports."""
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if any(part in str(key).lower() for part in _SENSITIVE_KEY_PARTS)
                else redact_sensitive(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive(item) for item in value]
    return value


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable short hash for a run configuration."""
    return stable_hash(config, length=16)


def data_digest(*values: Any) -> str:
    """Short SHA-256 over array-likes: dtype, shape and contents (``None`` allowed).

    The decoding runner uses it for a unit's input-data fingerprint (part of the
    resume hash) and for the sample-set signature paired comparisons must share.
    """
    digest = hashlib.sha256()
    for value in values:
        if value is None:
            digest.update(b"<none>\x1e")
            continue
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(f"{array.dtype.str}{array.shape}".encode())
        if array.dtype.kind in "OUS":
            digest.update("\x1f".join(map(str, array.ravel().tolist())).encode())
        else:
            digest.update(array.tobytes())
        digest.update(b"\x1e")
    return digest.hexdigest()[:16]


RUN_MANIFEST_FILENAME = "run_manifest.json"


def make_serializable(obj: Any) -> Any:
    """Recursively convert NumPy types to JSON-safe Python primitives."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.integer)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def write_run_manifest(output_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write a unit's manifest in one step, so it is never seen half-written."""
    path = Path(output_dir) / RUN_MANIFEST_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(make_serializable(dict(manifest)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def read_run_manifest(output_dir: str | Path) -> dict[str, Any] | None:
    """Return a unit's manifest, or None if the unit has not finished."""
    path = Path(output_dir) / RUN_MANIFEST_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def completed_for_config(
    output_dir: str | Path,
    config: Mapping[str, Any],
) -> bool:
    """Return whether ``output_dir`` contains a successful matching run."""
    manifest = read_run_manifest(output_dir)
    if manifest is None or manifest.get("status") != "success":
        return False
    expected = config_hash(config)
    actual = manifest.get("config_hash")
    if actual != expected:
        raise RuntimeError(
            "Config hash mismatch (config or input data changed) for completed "
            f"output {output_dir}: "
            f"existing={actual}, requested={expected}. Use overwrite to replace it."
        )
    return True


def load_completed_result_records(
    output_dir: str | Path,
    context: Mapping[str, Any],
    *,
    metrics: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Rehydrate aggregate rows for a safely resumed experiment."""
    output = Path(output_dir)
    manifest = read_run_manifest(output)
    if manifest is None or "summary" not in manifest:
        raise RuntimeError(f"Completed output has no summary to resume from: {output}")
    statistical = pd.DataFrame(manifest.get("statistical_assessment", []))
    records = []
    for source_row in manifest["summary"]:
        row = dict(source_row)
        model = row.pop("Model")
        p_value = None
        if not statistical.empty and {"Model", "Metric", "PValue"}.issubset(
            statistical.columns
        ):
            model_stats = statistical[
                (statistical["Model"] == model) & (statistical["Metric"] == "accuracy")
            ]
            if not model_stats.empty:
                p_value = float(model_stats.iloc[0]["PValue"])
        records.append(
            stamp_primary_metric(
                {
                    **dict(context),
                    "model": model,
                    "status": "success",
                    "reason": "resumed",
                    "output_dir": str(output),
                    **row,
                    **({"p_value": p_value} if p_value is not None else {}),
                },
                metrics,
            )
        )
    return records


def write_run_status(root: str | Path, status: str) -> Path:
    """Write exactly one run-level success, partial, or failure marker."""
    normalized = status.upper()
    if normalized not in {"SUCCESS", "PARTIAL", "FAILED"}:
        raise ValueError("status must be SUCCESS, PARTIAL, or FAILED.")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("_SUCCESS", "_PARTIAL", "_FAILED"):
        (root / name).unlink(missing_ok=True)
    marker = root / f"_{normalized}"
    marker.write_text("", encoding="utf-8")
    return marker
