"""Resumable decoding-run artifact utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

_SENSITIVE_KEY_PARTS = ("token", "password", "secret", "api_key", "apikey")


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
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def completed_for_config(
    output_dir: str | Path,
    config: Mapping[str, Any],
) -> bool:
    """Return whether ``output_dir`` contains a successful matching run."""
    output = Path(output_dir)
    if not (output / "_SUCCESS").exists():
        return False
    manifest_path = output / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Completed output has no run manifest and cannot be resumed: {output}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = config_hash(config)
    actual = manifest.get("config_hash")
    if actual != expected:
        raise RuntimeError(
            f"Config hash mismatch for completed output {output}: "
            f"existing={actual}, requested={expected}. Use overwrite to replace it."
        )
    return True


def load_completed_result_records(
    output_dir: str | Path,
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Rehydrate aggregate rows for a safely resumed experiment."""
    output = Path(output_dir)
    summary_path = output / "summary.csv"
    if not summary_path.exists():
        raise RuntimeError(f"Completed output has no summary table: {output}")
    summary = pd.read_csv(summary_path)
    if "Model" not in summary.columns:
        raise RuntimeError(f"Completed summary has no Model column: {summary_path}")
    statistical_path = output / "statistical_assessment.csv"
    statistical = (
        pd.read_csv(statistical_path) if statistical_path.exists() else pd.DataFrame()
    )
    records = []
    for source_row in summary.to_dict("records"):
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
            {
                **dict(context),
                "model": model,
                "status": "success",
                "reason": "resumed",
                "output_dir": str(output),
                **row,
                **({"p_value": p_value} if p_value is not None else {}),
            }
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
