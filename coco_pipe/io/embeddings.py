"""BIDS-compatible derivative I/O for foundation-model embeddings."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._constants import (
    ARTIFACT_SUFFIX,
    EMBEDDING_COMBINED_TABLE_LABELS,
    REQUIRED_ARRAYS,
    TOKEN_REQUIRED_ARRAYS,
)
from .structures import DataContainer


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__module__}.{type(value).__name__}: {value!s}>"


def embedding_sidecar_path(path: str | Path) -> Path:
    """Return the JSON sidecar paired with an embedding NPZ."""
    return Path(path).with_suffix(".json")


def validate_embedding_derivative(path: str | Path) -> dict[str, Any]:
    """Validate arrays, shape consistency, and sidecar presence."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    sidecar = embedding_sidecar_path(path)
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing embedding sidecar: {sidecar}")
    with np.load(path, allow_pickle=False) as payload:
        if (
            "token_embeddings" in payload.files
            and "window_embeddings" not in payload.files
        ):
            missing = TOKEN_REQUIRED_ARRAYS.difference(payload.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            tokens = np.asarray(payload["token_embeddings"])
            if tokens.ndim != 3:
                raise ValueError(
                    "token_embeddings must be 3-D (window, token, feature)."
                )
            for key in ("window_start", "window_stop", "window_index"):
                if len(payload[key]) != len(tokens):
                    raise ValueError(f"{key} length does not match token_embeddings.")
        else:
            missing = REQUIRED_ARRAYS.difference(payload.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            windows = np.asarray(payload["window_embeddings"])
            recording = np.asarray(payload["recording_embedding"])
            if windows.ndim != 2:
                raise ValueError("window_embeddings must be 2-D.")
            if recording.ndim != 1 or recording.shape[0] != windows.shape[1]:
                raise ValueError(
                    "recording_embedding must match the embedding dimension."
                )
            for key in ("window_start", "window_stop", "window_index"):
                if len(payload[key]) != len(windows):
                    raise ValueError(f"{key} length does not match window_embeddings.")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected an object in {sidecar}.")
    return metadata


def save_embedding_derivative(
    result: Any,
    path: str | Path,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Atomically write an embedding NPZ and matching JSON sidecar."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Embedding derivative path must end in .npz.")
    tokens = getattr(result, "token_embeddings", None)
    kind = "token" if tokens is not None else "embedding"
    expected_suffix = ARTIFACT_SUFFIX[kind]
    if not path.name.endswith(expected_suffix):
        raise ValueError(
            f"A {kind} artifact must be saved to a '{expected_suffix}' path, "
            f"got {path.name!r}."
        )
    sidecar = embedding_sidecar_path(path)
    if not overwrite and (path.exists() or sidecar.exists()):
        raise FileExistsError(f"Embedding derivative already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = path.with_name(f".{path.name}.tmp.npz")
    tmp_json = sidecar.with_name(f".{sidecar.name}.tmp")

    if tokens is not None:
        tokens = np.asarray(tokens)
        arrays: dict[str, list[str]] = {
            "token_embeddings": ["window", "token", "embedding_feature"],
            "window_start": ["window"],
            "window_stop": ["window"],
            "window_index": ["window"],
        }
        saved = {
            "token_embeddings": tokens,
            "window_start": np.asarray(result.window_start),
            "window_stop": np.asarray(result.window_stop),
            "window_index": np.asarray(result.window_index),
        }
        extra_meta = {"representation": "token", "token_shape": list(tokens.shape)}
    else:
        arrays = {
            "window_embeddings": ["window", "embedding_feature"],
            "recording_embedding": ["embedding_feature"],
            "window_start": ["window"],
            "window_stop": ["window"],
            "window_index": ["window"],
        }
        saved = {
            "window_embeddings": np.asarray(result.window_embeddings),
            "recording_embedding": np.asarray(result.recording_embedding),
            "window_start": np.asarray(result.window_start),
            "window_stop": np.asarray(result.window_stop),
            "window_index": np.asarray(result.window_index),
        }
        extra_meta = {}

    payload_metadata = {
        **dict(getattr(result, "metadata", {}) or {}),
        **dict(metadata or {}),
        **extra_meta,
        "created_at": datetime.now(UTC).isoformat(),
        "arrays": arrays,
    }
    np.savez_compressed(tmp_npz, **saved)
    tmp_json.write_text(
        json.dumps(_json_value(payload_metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_npz, path)
    os.replace(tmp_json, sidecar)
    validate_embedding_derivative(path)
    return path, sidecar


def _manifest_artifacts(root: Path) -> list[Path]:
    """Existing artifact paths recorded in ``run_manifest.json`` (any kind)."""
    manifest = root / "run_manifest.json"
    if not manifest.exists():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return []
    paths: list[Path] = []
    for record in payload.get("records", []):
        if record.get("status") != "success" or not record.get("artifact_path"):
            continue
        path = Path(record["artifact_path"])
        if not path.is_absolute():
            path = root / path
        if path.exists():
            paths.append(path)
    return paths


def discover_embedding_derivatives(
    root: str | Path,
    model_key: str | None = None,
    *,
    kind: str = "embedding",
) -> list[Path]:
    """Discover valid ``"embedding"`` or ``"token"`` NPZ artifacts under a root.

    Artifact kind is identified by filename suffix (see :data:`ARTIFACT_SUFFIX`),
    which ``save_embedding_derivative`` enforces. The run manifest (if present)
    may index either kind; results are filtered to the requested ``kind``, and a
    recursive glob is used as a fallback when the manifest names none of it.
    """
    if kind not in ARTIFACT_SUFFIX:
        raise ValueError(f"kind must be one of {sorted(ARTIFACT_SUFFIX)}.")
    root = Path(root)
    suffix = ARTIFACT_SUFFIX[kind]
    paths = sorted({p for p in _manifest_artifacts(root) if p.name.endswith(suffix)})
    if not paths:
        paths = sorted(root.rglob(f"*{suffix}"))
    if model_key is None:
        return paths
    selected = []
    for path in paths:
        try:
            metadata = json.loads(
                embedding_sidecar_path(path).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if str(metadata.get("model_key", "")).lower() == model_key.lower():
            selected.append(path)
    return selected


def embedding_observation_id(
    metadata: Mapping[str, Any],
    path: str | Path,
    window_position: int | None = None,
) -> str:
    """Return the canonical recording/window ID for pooled or token artifacts."""
    path = Path(path)
    base = metadata.get("recording_id")
    if not base:
        base = path.name
        for suffix in ("_embedding.npz", "_tokens.npz", ".npz"):
            if base.endswith(suffix):
                base = base.removesuffix(suffix)
                break
    epoch = "" if window_position is None else f"_epoch-{window_position:04d}"
    return f"{base}{epoch}"


def load_embedding_derivatives(
    paths: Sequence[str | Path] | str | Path,
    representation: str = "recording",
    aggregate_by: str | None = None,
    model_key: str | None = None,
) -> DataContainer:
    """Load embedding artifacts into a DataContainer.

    ``representation`` selects both the on-disk array and the output rank:

    * ``"recording"`` — the pooled ``recording_embedding`` (2-D ``obs x feature``)
    * ``"epoch"`` — per-epoch ``window_embeddings`` (2-D ``obs x feature``)
    * ``"token"`` — per-epoch ``token_embeddings`` from the separate
      ``*_tokens.npz`` artifacts (3-D ``obs x token x feature``)

    A coarser ``"subject"`` level is produced by the merge step, not here (it
    pools across recordings). Kinds are discovered by filename suffix (see
    :data:`ARTIFACT_SUFFIX`); token and embedding artifacts are separate files
    that join only by shared observation ID.
    """
    if representation not in {"epoch", "recording", "token"}:
        raise ValueError("representation must be 'epoch', 'recording', or 'token'.")
    is_recording = representation == "recording"
    is_token = representation == "token"

    if isinstance(paths, (str, Path)):
        candidate = Path(paths)
        if candidate.is_dir():
            resolved = discover_embedding_derivatives(
                candidate,
                model_key=model_key,
                kind="token" if is_token else "embedding",
            )
        else:
            resolved = [candidate]
    else:
        resolved = [Path(path) for path in paths]
    if not resolved:
        raise FileNotFoundError("No embedding derivatives were found.")

    rows: list[np.ndarray] = []
    ids: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    obs_shape: tuple[int, ...] | None = None
    for path in resolved:
        metadata = validate_embedding_derivative(path)
        with np.load(path, allow_pickle=False) as payload:
            array_key = (
                "token_embeddings"
                if is_token
                else "recording_embedding"
                if is_recording
                else "window_embeddings"
            )
            if array_key not in payload.files:
                raise ValueError(
                    f"{path} has no '{array_key}' array for representation "
                    f"'{representation}'; wrong artifact kind for this path."
                )
            values = np.asarray(payload[array_key])
            if is_recording:
                values = values[None, :]
            if obs_shape is None:
                obs_shape = values.shape[1:]
            elif values.shape[1:] != obs_shape:
                raise ValueError(
                    f"Embedding shapes differ: expected {obs_shape}, "
                    f"got {values.shape[1:]} in {path}."
                )
            for idx, row in enumerate(values):
                rows.append(row)
                position = None if is_recording else idx
                ids.append(embedding_observation_id(metadata, path, position))
                obs_meta = dict(metadata)
                obs_meta["artifact_path"] = str(path)
                obs_meta["representation"] = representation
                if not is_recording:
                    obs_meta.update(
                        {
                            "window_index": int(payload["window_index"][idx]),
                            "window_start": int(payload["window_start"][idx]),
                            "window_stop": int(payload["window_stop"][idx]),
                        }
                    )
                metadata_rows.append(obs_meta)

    X = np.stack(rows) if is_token else np.vstack(rows)
    model_keys = {
        str(row.get("model_key", "")).strip().lower()
        for row in metadata_rows
        if str(row.get("model_key", "")).strip()
    }
    if model_key is not None and model_keys != {model_key.strip().lower()}:
        raise ValueError(
            f"Requested model_key={model_key!r}, but artifacts contain "
            f"{sorted(model_keys)}."
        )
    if len(model_keys) > 1:
        raise ValueError(
            "Embedding derivatives from multiple foundation models cannot be "
            "combined as observations. Pass model_key to load one model at a time."
        )
    coords: dict[str, Any] = {
        "feature": np.asarray(
            [f"embedding_{idx:04d}" for idx in range(X.shape[-1])], dtype=object
        )
    }
    metadata_frame = pd.DataFrame(metadata_rows)
    for column in metadata_frame.columns:
        values = metadata_frame[column]
        if values.map(lambda value: not isinstance(value, (list, dict))).all():
            coords[column] = values.to_numpy(dtype=object)
    dims = ("obs", "token", "feature") if is_token else ("obs", "feature")
    container = DataContainer(
        X=X,
        dims=dims,
        coords=coords,
        ids=np.asarray(ids, dtype=object),
        meta={
            "input_mode": "foundation_embeddings",
            "representation": representation,
            "model_key": next(iter(model_keys), None),
            "artifacts": [str(path) for path in resolved],
        },
    )
    if aggregate_by is not None:
        if aggregate_by not in container.coords:
            raise KeyError(f"aggregate_by coordinate not found: {aggregate_by}")
        container = container.aggregate(by=aggregate_by, stats="mean")
    return container


def combined_embedding_table_path(
    derivative_root: str | Path,
    model_key: str,
    condition: str,
    representation: str = "recording",
) -> Path:
    """Path to the merged per-(model, condition) embedding table.

    Mirrors the ``combined/<model>_<condition>_<representation>_embeddings.parquet``
    layout written by the merge step. ``representation`` is one of
    :data:`~coco_pipe.io.AGGREGATION_LEVELS`.
    """
    if representation not in EMBEDDING_COMBINED_TABLE_LABELS:
        raise ValueError(
            f"representation must be one of {sorted(EMBEDDING_COMBINED_TABLE_LABELS)}."
        )
    label = EMBEDDING_COMBINED_TABLE_LABELS[representation]
    return (
        Path(derivative_root) / "combined" / f"{model_key}_{condition}_{label}.parquet"
    )


def load_combined_embedding_table(
    derivative_root: str | Path,
    model_key: str,
    condition: str,
    representation: str = "recording",
    aggregate_by: str | None = None,
) -> DataContainer:
    """Load one merged per-(model, condition) embedding table as a 2-D container.

    This reads the single parquet the merge step already materialized instead of
    rescanning every per-recording NPZ and filtering — one table read per
    condition, the same access pattern descriptors use. ``representation`` is one
    of :data:`~coco_pipe.io.AGGREGATION_LEVELS`. The table carries id columns
    (subject/session/run/condition/recording_id/model_key[/window_index]) plus
    ``embedding_*`` feature columns; the former become coords, the latter ``X``.
    """
    path = combined_embedding_table_path(
        derivative_root, model_key, condition, representation
    )
    if not path.exists():
        raise FileNotFoundError(f"Combined embedding table not found: {path}")
    frame = pd.read_parquet(path)
    feature_cols = [c for c in frame.columns if str(c).startswith("embedding_")]
    if not feature_cols:
        raise ValueError(f"No embedding_* feature columns in {path}.")
    id_cols = [c for c in frame.columns if c not in feature_cols]

    X = frame[feature_cols].to_numpy(dtype=float)
    coords: dict[str, Any] = {
        "feature": np.asarray([str(c) for c in feature_cols], dtype=object)
    }
    for column in id_cols:
        coords[column] = frame[column].to_numpy(dtype=object)

    if "recording_id" in frame.columns:
        base = frame["recording_id"].astype(str)
        if representation == "epoch" and "window_index" in frame.columns:
            ids = (
                base
                + "_epoch-"
                + frame["window_index"].astype(int).map("{:04d}".format)
            ).to_numpy(dtype=object)
        else:
            ids = base.to_numpy(dtype=object)
    else:
        ids = np.asarray([f"obs-{idx:06d}" for idx in range(len(frame))], dtype=object)

    container = DataContainer(
        X=X,
        dims=("obs", "feature"),
        coords=coords,
        ids=ids,
        meta={
            "input_mode": "foundation_embeddings",
            "representation": representation,
            "model_key": model_key,
            "condition": condition,
            "source": "combined_table",
            "artifacts": [str(path)],
        },
    )
    if aggregate_by is not None:
        if aggregate_by not in container.coords:
            raise KeyError(f"aggregate_by coordinate not found: {aggregate_by}")
        container = container.aggregate(by=aggregate_by, stats="mean")
    return container


def write_embedding_manifest(
    root: str | Path,
    records: Iterable[Mapping[str, Any]],
) -> Path:
    """Write a JSON run manifest indexing successful and failed artifacts."""
    path = Path(root) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "records": [_json_value(dict(record)) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_embedding_dataset_description(
    root: str | Path,
    name: str,
    bids_version: str,
    generated_by: Sequence[Mapping[str, Any]],
    source_datasets: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Write a BIDS derivative dataset_description.json."""
    path = Path(root) / "dataset_description.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Name": name,
        "BIDSVersion": bids_version,
        "DatasetType": "derivative",
        "GeneratedBy": [_json_value(item) for item in generated_by],
    }
    if source_datasets:
        payload["SourceDatasets"] = [_json_value(item) for item in source_datasets]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
