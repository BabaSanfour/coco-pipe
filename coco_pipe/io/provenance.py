"""Deterministic provenance fingerprints for data containers."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from coco_pipe.utils import stable_hash

from .structures import DataContainer

__all__ = ["fingerprint_container"]


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.hasobject:
        return stable_hash(contiguous.tolist())
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def fingerprint_container(container: DataContainer) -> dict[str, Any]:
    """Fingerprint a container's matrix values and non-observation schema.

    The fingerprint is suitable for cache and checkpoint identity. Observation
    identifiers remain a separate part of the calling pipeline's provenance.
    """
    matrix = np.asarray(container.X)
    schema: dict[str, Any] = {
        "dims": list(container.dims),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
    }
    for key, values in sorted(container.coords.items()):
        if key == "obs":
            continue
        if key in container.dims or key.startswith("feature_"):
            schema[key] = np.asarray(values).astype(str).tolist()
    return {
        "matrix_sha256": _array_sha256(matrix),
        "schema_sha256": stable_hash(schema),
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype),
    }
