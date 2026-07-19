"""BIDS-compatible derivative I/O for foundation-model embeddings."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from ._constants import (
    ARTIFACT_SUFFIX,
    EMBEDDING_COMBINED_TABLE_LABELS,
    POOLED_ONLY_EMBEDDING_METADATA_KEYS,
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
    metadata: dict[str, Any] | None = None
    with np.load(path, allow_pickle=False) as payload:
        if (
            "token_embeddings" in payload.files
            and "window_embeddings" not in payload.files
        ):
            missing = TOKEN_REQUIRED_ARRAYS.difference(payload.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError(f"Expected an object in {sidecar}.")
            tokens = np.asarray(payload["token_embeddings"])
            if tokens.ndim < 3:
                raise ValueError(
                    "token_embeddings must contain a window axis and at least "
                    "two native feature axes."
                )
            token_axes = metadata.get("token_axes")
            if metadata.get("token_layout") == "native":
                if not isinstance(token_axes, list) or len(token_axes) != tokens.ndim:
                    raise ValueError(
                        "Native token metadata must name every saved tensor axis."
                    )
                if token_axes[0] != "window" or len(set(token_axes)) != len(token_axes):
                    raise ValueError(
                        "Native token axes must be unique and start with 'window'."
                    )
                observation_axes = metadata.get("token_observation_axes")
                feature_axis = metadata.get("token_feature_axis")
                source = metadata.get("token_source")
                if not isinstance(source, str) or not source:
                    raise ValueError(
                        "Native token metadata must identify token_source."
                    )
                if not isinstance(observation_axes, list) or not observation_axes:
                    raise ValueError(
                        "Native token metadata must identify token_observation_axes."
                    )
                if not isinstance(feature_axis, str) or not feature_axis:
                    raise ValueError(
                        "Native token metadata must identify token_feature_axis."
                    )
                if set(token_axes) != {"window", *observation_axes, feature_axis}:
                    raise ValueError(
                        "Native token metadata must assign every non-window axis "
                        "to token_observation_axes or token_feature_axis."
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
    if metadata is None:
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
        result_metadata = dict(getattr(result, "metadata", {}) or {})
        override_metadata = dict(metadata or {})
        token_axes = override_metadata.get(
            "token_axes", result_metadata.get("token_axes")
        )
        if not isinstance(token_axes, list) or len(token_axes) != tokens.ndim:
            raise ValueError(
                "Saving native tokens requires one explicit token_axes label per axis."
            )
        token_contract = {**result_metadata, **override_metadata}
        observation_axes = token_contract.get("token_observation_axes")
        feature_axis = token_contract.get("token_feature_axis")
        source = token_contract.get("token_source")
        if not isinstance(source, str) or not source:
            raise ValueError("Saving native tokens requires token_source.")
        if not isinstance(observation_axes, list) or not observation_axes:
            raise ValueError("Saving native tokens requires token_observation_axes.")
        if not isinstance(feature_axis, str) or not feature_axis:
            raise ValueError("Saving native tokens requires token_feature_axis.")
        if set(token_axes) != {"window", *observation_axes, feature_axis}:
            raise ValueError(
                "Every non-window token axis must be assigned as an observation "
                "axis or the feature axis."
            )
        arrays: dict[str, list[str]] = {
            "token_embeddings": token_axes,
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
        extra_meta = {
            "representation": "token",
            "token_layout": "native",
            "token_layout_version": 1,
            "token_shape": list(tokens.shape),
            "token_dtype": str(tokens.dtype),
        }
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


def save_embedding_outputs(
    result: Any,
    embedding_path: str | Path,
    *,
    token_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    pooled_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path | None]:
    """Write independent pooled and optional token derivatives from *result*.

    ``embedding_path`` and ``token_path`` are independent derivative locations.
    When ``result.token_embeddings`` is present, ``token_path`` is required.
    ``metadata`` applies to both outputs, while ``pooled_metadata`` applies only
    to the pooled derivative. This keeps pooling-specific representation identity
    out of pooling-independent native-token metadata. Complete outputs are left
    untouched unless ``overwrite`` is true; a partial NPZ/sidecar pair is repaired.
    """
    embedding_path = Path(embedding_path)
    embedding_suffix = ARTIFACT_SUFFIX["embedding"]
    if not embedding_path.name.endswith(embedding_suffix):
        raise ValueError(
            f"Embedding output path must end in '{embedding_suffix}', "
            f"got {embedding_path.name!r}."
        )

    def _complete(path: Path) -> bool:
        return path.exists() and embedding_sidecar_path(path).exists()

    def _repair_overwrite(path: Path) -> bool:
        return overwrite or path.exists() or embedding_sidecar_path(path).exists()

    token_embeddings = getattr(result, "token_embeddings", None)
    if token_embeddings is not None and token_path is None:
        raise ValueError("token_path is required when token embeddings are present.")
    resolved_token_path = Path(token_path) if token_path is not None else None
    if (
        token_embeddings is not None
        and resolved_token_path is not None
        and (overwrite or not _complete(resolved_token_path))
    ):
        token_result = SimpleNamespace(
            window_start=result.window_start,
            window_stop=result.window_stop,
            window_index=result.window_index,
            metadata={
                key: value
                for key, value in dict(getattr(result, "metadata", {}) or {}).items()
                if key not in POOLED_ONLY_EMBEDDING_METADATA_KEYS
            },
            token_embeddings=token_embeddings,
        )
        save_embedding_derivative(
            token_result,
            resolved_token_path,
            metadata={
                **{
                    key: value
                    for key, value in dict(metadata or {}).items()
                    if key not in POOLED_ONLY_EMBEDDING_METADATA_KEYS
                },
                "artifact_kind": "window_tokens",
            },
            overwrite=_repair_overwrite(resolved_token_path),
        )

    pooled_result = SimpleNamespace(
        window_embeddings=result.window_embeddings,
        recording_embedding=result.recording_embedding,
        window_start=result.window_start,
        window_stop=result.window_stop,
        window_index=result.window_index,
        metadata=getattr(result, "metadata", None),
        token_embeddings=None,
    )
    if overwrite or not _complete(embedding_path):
        save_embedding_derivative(
            pooled_result,
            embedding_path,
            metadata={
                **dict(metadata or {}),
                **dict(pooled_metadata or {}),
                "artifact_kind": "pooled_embedding",
            },
            overwrite=_repair_overwrite(embedding_path),
        )
    return embedding_path, resolved_token_path if token_embeddings is not None else None


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
    may index either kind. When a requested model is not in that manifest, a
    recursive scan also finds independently materialized model variants.
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

    def select_model(candidates: Iterable[Path]) -> list[Path]:
        selected = []
        for path in candidates:
            try:
                metadata = json.loads(
                    embedding_sidecar_path(path).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            if str(metadata.get("model_key", "")).lower() == model_key.lower():
                selected.append(path)
        return selected

    selected = select_model(paths)
    if selected:
        return selected
    return select_model(sorted(root.rglob(f"*{suffix}")))


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
    condition = str(metadata.get("condition", "")).strip()
    if condition:
        base = f"{base}_condition-{condition}"
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
    * ``"token"`` — native per-epoch feature tensors from separate
      ``*_tokens.npz`` artifacts (arbitrary rank, beginning with ``obs``)

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
    artifact_metadata: dict[str, dict[str, Any]] = {}
    obs_shape: tuple[int, ...] | None = None
    loaded_token_axes: tuple[str, ...] | None = None
    token_feature_axis: str | None = None
    for path in resolved:
        metadata = validate_embedding_derivative(path)
        artifact_metadata[str(path)] = metadata
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
            if is_token:
                axes = metadata.get("token_axes")
                if not isinstance(axes, list) or len(axes) != values.ndim:
                    raise ValueError(
                        "Token derivative has no explicit native axis contract: "
                        f"{path}."
                    )
                current_axes = ("obs", *axes[1:])
                if loaded_token_axes is None:
                    loaded_token_axes = current_axes
                    token_feature_axis = str(metadata.get("token_feature_axis", ""))
                elif current_axes != loaded_token_axes:
                    raise ValueError(
                        f"Token axes differ: expected {loaded_token_axes}, got "
                        f"{current_axes} in {path}."
                    )
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
    feature_dim = token_feature_axis if is_token else "feature"
    if not feature_dim or (
        is_token
        and loaded_token_axes is not None
        and feature_dim not in loaded_token_axes
    ):
        raise ValueError("Token metadata must identify token_feature_axis explicitly.")
    feature_axis = loaded_token_axes.index(feature_dim) if is_token else 1
    coords: dict[str, Any] = {
        feature_dim: np.asarray(
            [f"embedding_{idx:04d}" for idx in range(X.shape[feature_axis])],
            dtype=object,
        )
    }
    metadata_frame = pd.DataFrame(metadata_rows)
    for column in metadata_frame.columns:
        values = metadata_frame[column]
        if values.map(lambda value: not isinstance(value, (list, dict))).all():
            coords[column] = values.to_numpy(dtype=object)
    dims = loaded_token_axes if is_token else ("obs", "feature")
    if dims is None:
        raise RuntimeError("Token axes were not initialized.")
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
            "artifact_metadata": artifact_metadata,
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
