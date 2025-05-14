#!/usr/bin/env python3
"""
run_dim_reduction.py

DimReductionPipeline: A class to orchestrate dimensionality reduction 
process for M/EEG, CSV, or Embedding data.
"""
import json
import logging
from pathlib import Path

import numpy as np

from coco_pipe.dim_reduction.config import METHODS, METHODS_DICT
from coco_pipe.dim_reduction.reducers.base import BaseReducer
from coco_pipe.io.load import load

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class DimReductionPipeline:
    """
    DimReductionPipeline: A class to orchestrate the dimensionality reduction
    process for M/EEG, CSV, or Embedding data.
    """

    def __init__(
        self,
        type: str,
        method: str,
        data_path: Path,
        task: str = None,
        run: str = None,
        processing: str = None,
        subjects: list = None,
        max_seg: int = None,
        flatten: bool = False,
        sensorwise: bool = False,
        n_components: int = 2,
        reducer_kwargs: dict = None,
    ):
        self.type = type.lower()
        if self.type not in {"embeddings", "csv", "meg", "eeg"}:
            raise ValueError(f"Unknown data '{type}', choose from 'embeddings', 'csv', 'meg', 'eeg'")

        self.method = method.upper()
        if self.method not in METHODS:
            raise ValueError(f"Unknown method '{method}', choose from {METHODS}")

        ReducerCls = METHODS_DICT[self.method]
        self.reducer: BaseReducer = ReducerCls(
            n_components=n_components,
            **(reducer_kwargs or {})
        )

        self.data_path = Path(data_path)
        self.task = task
        self.run = run
        self.processing = processing
        self.subjects = subjects
        self.max_seg = max_seg
        self.flatten = flatten
        self.sensorwise = sensorwise
        self.n_components = n_components
        self.reducer_kwargs = reducer_kwargs or {}

        # Prepare output paths
        self.output_dir = Path(data_path).parent / "dim_reduction"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base = f"{self.type}_dimred-{self.method}_{self.n_components}d_task-{self.task}_run-{self.run}_proc-{self.processing}"

    
        self.reducer_path = self.output_dir / f"{self.base}_reducer.joblib"
        self.meta_path = self.output_dir / f"{self.base}_meta.json"

    def fit(self, X: np.ndarray, y: np.ndarray = None, save: bool = True) -> None:
        """
        Fit the reducer on the data.
        """
        if self.reducer_path.exists():
            logger.info(f"Loading reducer from {self.reducer_path}")
            self.reducer = BaseReducer.load(str(self.reducer_path))
            return
        logger.info(f"Fitting {self.method} on data shape {X.shape}")
        self.reducer.fit(X, y=y)
        if save:
            logger.info(f"Saving fitted reducer to {self.reducer_path}")
            self.reducer.save(str(self.reducer_path))

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, save: bool = True) -> np.ndarray:
        # Load existing reducer if present
        if self.reducer_path.exists():
            logger.info(f"Loading reducer from {self.reducer_path}")
            self.reducer = BaseReducer.load(str(self.reducer_path))
            return self.reducer.transform(X)

        # Fit and save reducer
        logger.info(f"Fitting {self.method} on data shape {X.shape}")
        reduced = self.reducer.fit_transform(X, y=y)
        if save:
            logger.info(f"Saving fitted reducer to {self.reducer_path}")
            self.reducer.save(str(self.reducer_path))
        return reduced

    def save_outputs(self, reduced: np.ndarray, subjects: np.ndarray, times: np.ndarray):
        logger.info(f"Saving reduced data to {self.output_dir}")
        np.savez_compressed(
            self.output_dir / f"{self.base}.npz",
            reduced=reduced,
            subjects=subjects,
            time_segments=times,
        )
        meta = {
            "type":             self.type,
            "method":           self.method,
            "task":             self.task,
            "run":              self.run,
            "processing":       self.processing,
            "subjects":         self.subjects,
            "max_seg":          self.max_seg,
            "sensorwise":       self.sensorwise,
            "n_components":     self.n_components,
            "reducer_kwargs":   self.reducer_kwargs,
            "reducer_file":     self.reducer_path.name,
            "output_file":      self.base + ".npz",
        }
        logger.info(f"Writing metadata to {self.meta_path}")
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def execute(self) -> Path:
        X, subj, times = load(self.type, self.data_path, self.task, self.run, self.processing, self.subjects, self.max_seg, self.flatten, self.sensorwise)
        reduced = self.fit_transform(X)
        self.save_outputs(reduced, subj, times)
        logger.info("Pipeline complete.")
        # Return the output file path where reduced data was saved
        return self.data_path / f"{self.base}.npz"