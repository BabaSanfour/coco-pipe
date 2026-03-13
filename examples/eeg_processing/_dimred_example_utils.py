from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from coco_pipe.dim_reduction import DimReduction
from coco_pipe.io import load_data


def _coerce_input(X: np.ndarray, reducer: DimReduction) -> np.ndarray:
    X_arr = np.asarray(X)
    expected_ndim = reducer.capabilities.get("input_ndim", 2)
    if X_arr.ndim == expected_ndim:
        return X_arr
    if expected_ndim == 2 and X_arr.ndim > 2:
        return X_arr.reshape(X_arr.shape[0], -1)
    raise ValueError(
        f"Cannot coerce input with shape {X_arr.shape} to reducer ndim={expected_ndim}."
    )


def _coords_for_save(container: Any) -> Dict[str, np.ndarray]:
    coords = dict(getattr(container, "coords", {}) or {})
    saved: Dict[str, np.ndarray] = {}
    subject_keys = ("subject", "subjects", "sub")
    time_keys = ("time", "times", "time_segment", "time_segments")

    for key in subject_keys:
        values = coords.get(key)
        if values is not None:
            saved["subjects"] = np.asarray(values)
            break

    for key in time_keys:
        values = coords.get(key)
        if values is not None:
            saved["time_segments"] = np.asarray(values)
            break

    return saved


def execute_reduction(
    *,
    method: str,
    data_path: str | Path,
    type: Optional[str] = None,
    n_components: int = 2,
    output_dir: str | Path = "outputs/dim_reduction",
    output_name: Optional[str] = None,
    benchmark: bool = True,
    score_k: int = 5,
    params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    mode = "bids" if type in {"eeg", "bids"} else type or "auto"
    load_kwargs = dict(kwargs)
    if mode == "bids" and "loading_mode" not in load_kwargs and "mode" in load_kwargs:
        load_kwargs["loading_mode"] = load_kwargs.pop("mode")

    container = load_data(data_path, mode=mode, **load_kwargs)
    reducer = DimReduction(method, n_components=n_components, params=params or {})
    coords = dict(getattr(container, "coords", {}) or {})

    X_input = _coerce_input(container.X, reducer)
    embedding = reducer.fit_transform(X_input, y=getattr(container, "y", None))
    score_payload = {}
    if benchmark:
        score_payload = reducer.score(
            embedding,
            X=X_input,
            n_neighbors=score_k,
            labels=getattr(container, "y", None),
            times=coords.get("time"),
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_name or f"{method.lower()}_reduction.npz"
    if not filename.endswith(".npz"):
        filename = f"{filename}.npz"
    output_path = output_dir / filename

    save_payload = {
        "reduced": embedding,
        "ids": getattr(container, "ids", None),
        "labels": getattr(container, "y", None),
        "method": np.array([method]),
        "metrics_json": np.array([json.dumps(score_payload, default=str)]),
    }
    save_payload.update(_coords_for_save(container))
    np.savez_compressed(output_path, **save_payload)
    return str(output_path)
