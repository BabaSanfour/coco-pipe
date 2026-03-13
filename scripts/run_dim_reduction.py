#!/usr/bin/env python3
"""
Run Dimensionality Reduction Pipeline (Enhanced)
================================================

This script provides a command-line interface to the dim_reduction module.
It supports configuration via YAML files, parallel processing, benchmarking,
and automated visualization.

Usage:
    python run_dim_reduction.py --config configs/dim_reduction.yaml --benchmark --plot
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

from coco_pipe.dim_reduction import DimReduction
from coco_pipe.io import load_data
from coco_pipe.viz.dim_reduction import plot_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _infer_mode(config_type):
    if config_type in {"eeg", "bids"}:
        return "bids"
    if config_type in {"tabular", "embedding"}:
        return config_type
    return "auto"


def _coerce_input(X, reducer):
    X_arr = np.asarray(X)
    expected_ndim = reducer.capabilities.get("input_ndim", 2)
    if X_arr.ndim == expected_ndim:
        return X_arr
    if expected_ndim == 2 and X_arr.ndim > 2:
        return X_arr.reshape(X_arr.shape[0], -1)
    raise ValueError(
        f"Cannot coerce input with shape {X_arr.shape} to reducer ndim={expected_ndim}."
    )


def _metadata_from_container(container):
    return dict(getattr(container, "coords", {}) or {})


def _build_output_path(config):
    output_dir = Path(config.get("output_dir", "outputs/dim_reduction"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = config.get("output_name") or f"{config['method'].lower()}_reduction"
    if not str(output_name).endswith(".npz"):
        output_name = f"{output_name}.npz"
    return output_dir / output_name


def run_pipeline(config, benchmark=False, plot=False):
    """
    Run a single pipeline instance.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    benchmark : bool
        Whether to compute quality metrics.
    plot : bool
        Whether to generate plots.
    """
    try:
        mode = _infer_mode(config.get("type"))
        load_kwargs = dict(config)
        method = load_kwargs.pop("method")
        n_components = load_kwargs.pop("n_components", 2)
        params = load_kwargs.pop("params", {})
        output_path = _build_output_path(config)
        load_kwargs.pop("output_dir", None)
        load_kwargs.pop("output_name", None)
        load_kwargs.pop("type", None)
        load_kwargs.pop("benchmark", None)
        load_kwargs.pop("report", None)
        score_k = load_kwargs.pop("score_k", 5)
        data_path = load_kwargs.pop("data_path")
        if (
            mode == "bids"
            and "loading_mode" not in load_kwargs
            and "mode" in load_kwargs
        ):
            load_kwargs["loading_mode"] = load_kwargs.pop("mode")

        container = load_data(data_path, mode=mode, **load_kwargs)
        reducer = DimReduction(method, n_components=n_components, params=params)
        metadata = _metadata_from_container(container)

        X_input = _coerce_input(container.X, reducer)
        X_emb = reducer.fit_transform(X_input, y=getattr(container, "y", None))
        score_payload = {}

        if benchmark:
            logger.info("Running benchmarks...")
            score_payload = reducer.score(
                X_emb,
                X=X_input,
                n_neighbors=score_k,
                labels=getattr(container, "y", None),
                times=metadata.get("time"),
            )

        save_payload = {
            "reduced": X_emb,
            "ids": getattr(container, "ids", None),
            "labels": getattr(container, "y", None),
            "method": np.array([method]),
            "metrics_json": np.array([json.dumps(score_payload, default=str)]),
        }
        coords = metadata or {}
        for key in ("subject", "subjects", "sub"):
            if key in coords:
                save_payload["subjects"] = np.asarray(coords[key])
                break
        for key in ("time", "times", "time_segment", "time_segments"):
            if key in coords:
                save_payload["time_segments"] = np.asarray(coords[key])
                break
        np.savez_compressed(output_path, **save_payload)
        output_dir = output_path.parent

        if plot:
            logger.info("Generating plots...")
            # 2D Scatter
            if X_emb.shape[1] in [2, 3]:
                plot_embedding(
                    X_emb,
                    labels=getattr(container, "y", None),
                    title=f"{method} Embedding",
                    save_path=output_dir / "embedding_plot.png",
                )
                logger.info(f"Saved embedding plot to {output_dir}/embedding_plot.png")
            else:
                logger.warning(
                    f"Skipping plot: Embedding has {X_emb.shape[1]} dims (need 2 or 3)."
                )

        return str(output_path)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="Run dimensionality reduction pipeline."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Number of parallel jobs."
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Calculate quality metrics."
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    # If config is a list, run multiple jobs
    if isinstance(configs, list):
        job_list = configs
    elif isinstance(configs, dict):
        job_list = [configs]
    else:
        logger.error("Invalid config format. Must be dict or list of dicts.")
        sys.exit(1)

    logger.info(f"Found {len(job_list)} pipeline configurations.")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_pipeline)(cfg, args.benchmark, args.plot) for cfg in job_list
    )

    logger.info("All pipelines completed successfully.")
    for res in results:
        print(res)


if __name__ == "__main__":
    main()
