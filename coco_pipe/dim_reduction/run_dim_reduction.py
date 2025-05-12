#!/usr/bin/env python3
"""
run_dim_reduction.py

YAML‑driven CLI to run dimensionality reduction jobs using an embedded
DimReductionPipeline class. Supports EEG, M/EEG embeddings and CSV loaders,
multiple reducers (sequential or parallel), and outputs a JSON summary.
"""
import argparse
import ast
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
import numpy as np
import pandas as pd

from coco_pipe.io.embeddings import load_embeddings, reshape_embeddings
from coco_pipe.dim_reduction.config import METHODS, METHODS_DICT, RESULTS_DIR
from coco_pipe.dim_reduction.reducers.base import BaseReducer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class DimReductionPipeline:
    """
    Orchestrates loading, DR, and saving for embeddings, M/EEG or CSV data.
    """

    def __init__(
        self,
        loader: str,
        method: str,
        data_path: Path,
        task: str = None,
        run: str = None,
        processing: str = None,
        subjects: list = None,
        max_seg: int = None,
        sensorwise: bool = False,
        n_components: int = 2,
        reducer_kwargs: dict = None,
    ):
        self.loader = loader.lower()
        if self.loader not in {"embeddings", "csv", "meg", "eeg", "meeg"}:
            raise ValueError(f"Unknown loader '{loader}', choose from 'embeddings', 'csv', 'meg', 'eeg', or 'meeg'")

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
        self.sensorwise = sensorwise
        self.n_components = n_components
        self.reducer_kwargs = reducer_kwargs or {}

        # Prepare output paths
        self.output_dir = Path(RESULTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        base = f"{self.method}"
        if self.loader == "embeddings":
            base += f"_{task}_run-{run}_{processing}"
        elif self.loader == "meeg":
            base += "_meeg"
        else:
            base += "_csv"

        self.reducer_path = self.output_dir / f"{base}_reducer.joblib"
        self.embedding_out_path = self.output_dir / f"{base}.npz"
        self.meta_path = self.output_dir / f"{base}_meta.json"


    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Load existing reducer if present
        if self.reducer_path.exists():
            logger.info(f"Loading reducer from {self.reducer_path}")
            self.reducer = BaseReducer.load(str(self.reducer_path))
            return self.reducer.transform(X)

        # Fit and save reducer
        logger.info(f"Fitting {self.method} on data shape {X.shape}")
        reduced = self.reducer.fit_transform(X)
        logger.info(f"Saving reducer to {self.reducer_path}")
        self.reducer.save(str(self.reducer_path))
        return reduced

    def save_outputs(self, reduced: np.ndarray, subjects: np.ndarray, times: np.ndarray):
        logger.info(f"Saving reduced data to {self.embedding_out_path}")
        np.savez_compressed(
            self.embedding_out_path,
            reduced=reduced,
            subjects=subjects,
            time_segments=times,
        )
        meta = {
            "loader":           self.loader,
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
            "output_file":      self.embedding_out_path.name,
        }
        logger.info(f"Writing metadata to {self.meta_path}")
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def run(self) -> Path:
        X, subj, times = self.load_and_preprocess()
        reduced = self.fit_transform(X)
        self.save_outputs(reduced, subj, times)
        logger.info("Pipeline complete.")
        return self.embedding_out_path


def run_job(cfg: dict) -> dict:
    pipeline = DimReductionPipeline(**cfg)
    out = pipeline.run()
    return {"method": pipeline.method, "output": str(out)}


def main():
    parser = argparse.ArgumentParser("run_dim_reduction")
    parser.add_argument(
        "--config", "-c", required=True, type=Path, help="Path to YAML config file"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = cfg.get("data", {})
    reducers_cfg = cfg.get("reducers", [])
    parallel = cfg.get("parallel", False)
    summary_path = Path(cfg.get("summary_path", "dr_summary.json"))

    if not reducers_cfg:
        logger.error("No reducers defined under 'reducers' in config")
        return

    # Build job configs
    jobs = []
    for r in reducers_cfg:
        job = {
            "loader":         data_cfg.get("loader", "embeddings"),
            "method":         r["method"],
            "data_path":      data_cfg["data_path"],
            "n_components":   r.get("n_components", 2),
            "reducer_kwargs": r.get("reducer_kwargs", {}),
            "task":           data_cfg.get("task"),
            "run":            data_cfg.get("run"),
            "processing":     data_cfg.get("processing"),
            "subjects":       data_cfg.get("subjects"),
            "max_seg":        data_cfg.get("max_seg"),
            "sensorwise":     data_cfg.get("sensorwise", False),
        }
        jobs.append(job)

    results = []
    if parallel and len(jobs) > 1:
        logger.info(f"Running {len(jobs)} jobs in parallel")
        with ProcessPoolExecutor() as exe:
            future_to_job = {exe.submit(run_job, job): job for job in jobs}
            for fut in as_completed(future_to_job):
                job = future_to_job[fut]
                try:
                    res = fut.result()
                    logger.info(f"{res['method']} → {res['output']}")
                except Exception as e:
                    logger.error(f"Job {job['method']} failed: {e}")
                    res = {"method": job["method"], "error": str(e)}
                results.append(res)
    else:
        logger.info(f"Running {len(jobs)} jobs sequentially")
        for job in jobs:
            try:
                res = run_job(job)
                logger.info(f"{res['method']} → {res['output']}")
            except Exception as e:
                logger.error(f"Job {job['method']} failed: {e}")
                res = {"method": job["method"], "error": str(e)}
            results.append(res)

    logger.info(f"Writing summary to {summary_path}")
    summary_path.write_text(json.dumps(results, indent=2))
    logger.info("All tasks complete.")


if __name__ == "__main__":
    main()
