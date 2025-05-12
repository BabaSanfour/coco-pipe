#!/usr/bin/env python3
"""
coco_pipe/io/embeddings.py

Load and flatten EEG embeddings stored in BIDS‑style folders.
"""
import logging
import pickle
from pathlib import Path
from typing import Union, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Configure module‐level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_embeddings(
    embeddings_root: Union[str, Path],
    task: str,
    run: str,
    processing: str,
    subjects: Optional[Union[int, List[int]]] = None,
    max_seg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings for subjects from BIDS‑style directories.

    Args:
      embeddings_root: Path to folder containing `sub-XXXX/` subfolders.
      task, run, processing: used to build the filename pattern
        e.g. sub-0001_task-RESTING_run-01_embeddingszscoreaxis0seg10.pkl
      subjects: if int, only first N; if list, only those IDs; else all
      max_seg: if set, only include time segments <= max_seg

    Returns:
      embeddings_array:      shape (n_samples, sensors, time, features)
      subjects_array:        shape (n_samples,)
      time_segments_array:   shape (n_samples,)
    """
    root = Path(embeddings_root)
    # find all sub- dirs
    subs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-"))

    def sid_from_path(p: Path) -> Optional[int]:
        try:
            return int(p.name.split("-")[1])
        except Exception:
            return None

    # filter by subjects parameter
    if subjects is not None:
        if isinstance(subjects, int):
            subs = subs[:subjects]
        else:
            subs = [p for p in subs if sid_from_path(p) in subjects]

    pattern = f"sub-*_task-{task}_run-{run}_embeddings{processing}.pkl"
    emb_list, subj_list, ts_list = [], [], []

    for sub_dir in tqdm(subs, desc="Subjects", leave=True):
        sid = sid_from_path(sub_dir) or 0
        matches = list(sub_dir.glob(pattern))
        if not matches:
            logger.warning(f"No matches in {sub_dir.name} for pattern {pattern}")
            continue

        for fpath in matches:
            try:
                with fpath.open("rb") as f:
                    emb_dict = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load {fpath.name}: {e}")
                continue

            for t_idx, emb in tqdm(emb_dict.items(), desc=f"Segs {sub_dir.name}", leave=False):
                if max_seg is not None and t_idx > max_seg:
                    break
                emb_list.append(emb)
                subj_list.append(sid)
                ts_list.append(t_idx)

    if not emb_list:
        raise RuntimeError(f"No embeddings loaded from {embeddings_root}")

    embeddings_array      = np.stack(emb_list, axis=0)
    subjects_array        = np.array(subj_list, dtype=int)
    time_segments_array   = np.array(ts_list, dtype=int)

    logger.info(f"Loaded embeddings: {embeddings_array.shape}")
    return embeddings_array, subjects_array, time_segments_array


def flatten_embeddings(
    embeddings_array: np.ndarray,
    sensorwise: bool = False,
) -> np.ndarray:
    """
    Flatten the embeddings array.

    - sensorwise=False → flatten to (n_samples, -1)
    - sensorwise=True  → (n_samples, sensors, time * features)
    """
    n_samples = embeddings_array.shape[0]
    if sensorwise:
        n_sensors, n_time, n_feat = embeddings_array.shape[1:]
        return embeddings_array.reshape(n_samples, n_sensors, n_time * n_feat)
    return embeddings_array.reshape(n_samples, -1)