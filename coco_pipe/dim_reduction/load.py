#!/usr/bin/env python3
"""
coco_pipe/dim_reduction/load.py

Load and reshape EEG embeddings stored in BIDS‑style subject folders.
"""
import logging
import pickle
from pathlib import Path
from typing import Tuple, Union, List

import numpy as np
from tqdm import tqdm

from coco_pipe.dim_reduction.config import DEFAULT_MAX_SEG

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_embeddings(
    embeddings_root: Union[str, Path],
    task: str,
    run: str,
    processing: str,
    subjects: Union[int, List[int], None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and extract embeddings stored in BIDS‑style subject folders.

    Each folder under `embeddings_root` named sub-XXXX/ contains pickle files:
      sub-XXXX_task-{task}_run-{run}_embeddings{processing}.pkl

    Args:
        embeddings_root: Root directory containing sub-*/ folders.
        task:           BIDS task identifier (e.g. "RESTING").
        run:            BIDS run identifier (e.g. "01").
        processing:     Suffix after "embeddings" in filename (e.g. "zscoreaxis0seg10").
        subjects:       If int → process only first N subjects;
                        if List[int] → process only those IDs;
                        if None → process all.

    Returns:
        embeddings_array:   (n_samples, sensors, time, features)
        subjects_array:     (n_samples,)
        time_segments_array:(n_samples,)
    """
    root = Path(embeddings_root)
    subs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-"))

    # filter by `subjects` param
    def sid_from_path(p: Path) -> Union[int, None]:
        try:
            return int(p.name.split("-")[1])
        except Exception:
            return None

    if subjects is not None:
        if isinstance(subjects, int):
            subs = subs[:subjects]
        else:
            subs = [p for p in subs if sid_from_path(p) in subjects]

    pattern = f"sub-*_task-{task}_run-{run}_embeddings{processing}.pkl"
    emb_list, subj_list, ts_list = [], [], []

    for sub_dir in subs:
        sid = sid_from_path(sub_dir)
        files = list(sub_dir.glob(pattern))
        if not files:
            logger.warning(f"No matching files in {sub_dir.name} for pattern {pattern}")
            continue

        for fpath in files:
            try:
                with fpath.open("rb") as f:
                    emb_dict = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load {fpath.name}: {e}")
                continue

            for t_idx, emb in emb_dict.items():
                if t_idx > DEFAULT_MAX_SEG:
                    break
                emb_list.append(emb)
                subj_list.append(sid)
                ts_list.append(t_idx)

    embeddings_array    = np.stack(emb_list, axis=0)
    subjects_array      = np.array(subj_list, dtype=int)
    time_segments_array = np.array(ts_list, dtype=int)

    logger.info(
        f"Loaded embeddings: {embeddings_array.shape}, "
        f"subjects: {subjects_array.shape}, "
        f"time segments: {time_segments_array.shape}"
    )
    return embeddings_array, subjects_array, time_segments_array


def reshape_embeddings(
    embeddings_array: np.ndarray,
    sensorwise: bool = False
) -> np.ndarray:
    """
    Reshape embeddings for downstream processing.

    - sensorwise=False → flatten to (n_samples, -1).
    - sensorwise=True  → (n_samples, sensors, time*features).
    """
    n_samples = embeddings_array.shape[0]
    if sensorwise:
        n_sensors, n_time, n_feat = embeddings_array.shape[1:]
        return embeddings_array.reshape(n_samples, n_sensors, n_time * n_feat)
    else:
        return embeddings_array.reshape(n_samples, -1)