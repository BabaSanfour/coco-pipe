#!/usr/bin/env python3
"""
Load and reshape EEG embeddings for coco-pipe dim_reduction module.
"""
import logging
from pathlib import Path
import pickle
from coco_pipe.dim_reduction.config import DEFAULT_SEGMENT_DUR, DEFAULT_Z_SCORE, DEFAULT_AXIS, DEFAULT_MAX_SEG
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

from coco_pipe.utils.config import embeddings_dir

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_embeddings(
    n_subjects: int,
    segment_duration: int = 10,
    z_score: bool = True,
    z_score_axis: int = 1,
    n_time_segments: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and extract embeddings for all subjects.

    Args:
        n_subjects: Number of subjects to load.
        segment_duration: Duration of each segment in seconds.
        z_score: Whether embeddings were z-scored.
        z_score_axis: Axis used for z-scoring in the saved file.
        n_time_segments: Maximum time segments to include per subject.

    Returns:
        embeddings_array: np.ndarray of shape (n_samples, sensors, time, features)
        subjects_array: np.ndarray of shape (n_samples,)
        time_segments_array: np.ndarray of shape (n_samples,)
    """
    embeddings_dict = {}

    # Load per-subject pickle files
    for subject_id in tqdm(range(1, n_subjects + 1), desc="Loading embeddings"):
        file_name = (
            f"embeddings_sub-{subject_id}"
            f"_dur-{segment_duration}s"
            f"_zscore-{z_score}"
            f"_axis-{z_score_axis}.pkl"
        )
        file_path = os.path.join(embeddings_dir, file_name)
        try:
            with open(file_path, 'rb') as f:
                embeddings_dict[subject_id] = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    embeddings_list = []
    subjects_list = []
    time_list = []

    # Extract and stack embeddings
    for subject_id, embed_map in tqdm(embeddings_dict.items(), desc="Processing embeddings"):
        for t_idx, emb in embed_map.items():
            if t_idx > n_time_segments:
                break
            embeddings_list.append(emb)
            subjects_list.append(subject_id)
            time_list.append(t_idx)

    embeddings_array = np.array(embeddings_list)
    subjects_array = np.array(subjects_list)
    time_segments_array = np.array(time_list)

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

    Args:
        embeddings_array: np.ndarray of shape (n_samples, sensors, time, features).
        sensorwise: If True, reshape to (n_samples, sensors, time*features). Else flatten to (n_samples, -1).

    Returns:
        reshaped_array: np.ndarray
    """
    if sensorwise:
        # Keep sensor dimension: merge time and feature dims
        n_samples, n_sensors, n_time, n_feat = embeddings_array.shape
        return embeddings_array.reshape(n_samples, n_sensors, n_time * n_feat)

    # Flatten sensors, time, and feature dims
    n_samples = embeddings_array.shape[0]
    return embeddings_array.reshape(n_samples, -1)