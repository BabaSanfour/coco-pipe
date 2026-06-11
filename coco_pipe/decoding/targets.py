"""Target-label resolution and grouped-cross-validation fold helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from coco_pipe.io import DataContainer


def prepare_target(
    container: DataContainer,
    target_spec: Mapping[str, Any],
    group_col: str = "patient_group_id",
) -> tuple[DataContainer, np.ndarray, np.ndarray, pd.DataFrame]:
    """Resolve labels and leakage-safe groups from a ``DataContainer``.

    Binary targets require an explicit ``positive_class``. This prevents
    alphabetical class ordering from silently changing ROC-AUC, precision,
    recall, and F1 semantics.
    """
    frame = container.observation_frame()
    target_col = str(target_spec["target_col"])
    if target_col not in frame:
        raise KeyError(f"Target column '{target_col}' is not available.")

    labels = frame[target_col].astype("string")
    label_map = {
        str(key): str(value)
        for key, value in (target_spec.get("label_map") or {}).items()
    }
    if label_map:
        labels = labels.map(lambda value: label_map.get(str(value), str(value)))
    valid = labels.notna() & ~labels.astype(str).str.lower().isin(
        {"", "nan", "none", "null", "<na>"}
    )
    selected = container.isel(obs=np.flatnonzero(valid.to_numpy()))
    frame = frame.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].astype(str).to_numpy()
    classes = [str(value) for value in pd.unique(labels)]
    if len(classes) < 2:
        raise ValueError("Decoding requires at least two target classes.")

    class_order = target_spec.get("class_order")
    positive_class = target_spec.get("positive_class")
    if len(classes) == 2:
        if positive_class is None:
            raise ValueError(
                "Binary target specifications must define positive_class explicitly."
            )
        positive = str(positive_class)
        if positive not in classes:
            raise ValueError(
                f"positive_class={positive!r} is absent from target classes {classes}."
            )
        negative = next(value for value in classes if value != positive)
        ordered = [negative, positive]
    else:
        if class_order is None:
            raise ValueError(
                "Multiclass target specifications must define class_order explicitly."
            )
        ordered = [str(value) for value in class_order]
        if set(ordered) != set(classes) or len(ordered) != len(classes):
            raise ValueError(
                f"class_order={ordered} does not match target classes {classes}."
            )

    encoded = {label: index for index, label in enumerate(ordered)}
    y = np.asarray([encoded[label] for label in labels], dtype=int)
    effective_group_col = str(target_spec.get("group_col") or group_col)
    if effective_group_col not in frame:
        raise KeyError(
            f"Leakage-safe group column '{effective_group_col}' is not available."
        )
    groups = frame[effective_group_col].astype(str).to_numpy()
    frame["target_label"] = labels
    frame["target_encoded"] = y
    frame["group_id"] = groups
    frame.attrs["label_encoding"] = encoded
    frame.attrs["positive_class"] = str(positive_class) if positive_class else None
    return selected, y, groups, frame


def safe_group_n_splits(
    y: Sequence[Any],
    groups: Sequence[Any],
    requested: int = 5,
    stratified: bool = True,
) -> int:
    """Return the largest viable grouped fold count."""
    frame = pd.DataFrame({"y": np.asarray(y), "group": np.asarray(groups)})
    if frame.empty:
        raise ValueError("Cannot determine grouped CV folds from an empty target.")
    if stratified:
        counts = (
            frame.drop_duplicates(["y", "group"])
            .groupby("y", dropna=False)["group"]
            .nunique()
        )
        available = int(counts.min()) if not counts.empty else 0
    else:
        available = int(frame["group"].nunique())
    n_splits = min(int(requested), available)
    if n_splits < 2:
        raise ValueError(
            "Grouped decoding requires at least two independent groups "
            "in every class."
        )
    return n_splits
