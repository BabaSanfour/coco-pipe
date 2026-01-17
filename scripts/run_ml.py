#!/usr/bin/env python3
import argparse
import logging
import os
from copy import deepcopy

import pandas as pd
import yaml

from coco_pipe.io import TabularDataset
from coco_pipe.ml.pipeline import MLPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_analysis(X, y, analysis_cfg):
    """Run a single analysis with the given config, passing through the new `mode`."""
    # scikit-learn pipelines expect numpy arrays
    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y
    # ADD blabla blaq
    # Build the MLPipeline config dict
    pipeline_config = {
        "task": analysis_cfg.get("task"),
        "analysis_type": analysis_cfg.get("analysis_type"),
        "models": analysis_cfg.get("models"),
        "metrics": analysis_cfg.get("metrics"),
        "cv_strategy": analysis_cfg.get("cv_kwargs", {}).get("cv_strategy"),
        "n_splits": analysis_cfg.get("cv_kwargs", {}).get("n_splits"),
        "cv_kwargs": analysis_cfg.get("cv_kwargs"),
        "n_features": analysis_cfg.get("n_features"),
        "direction": analysis_cfg.get("direction"),
        "search_type": analysis_cfg.get("search_type"),
        "n_iter": analysis_cfg.get("n_iter"),
        "scoring": analysis_cfg.get("scoring"),
        "n_jobs": analysis_cfg.get("n_jobs"),
        "save_intermediate": analysis_cfg.get("save_intermediate"),
        "results_dir": analysis_cfg.get("results_dir"),
        "results_file": analysis_cfg.get("results_file"),
        # **NEW** univariate vs. multivariate mode
        "mode": analysis_cfg.get("mode"),
    }

    # strip out any None so pipeline defaults apply
    pipeline_config = {k: v for k, v in pipeline_config.items() if v is not None}

    logger.info(
        f"Launching {pipeline_config['task']} pipeline "
        f"({pipeline_config.get('mode','multivariate')}) – "
        f"{pipeline_config['analysis_type']} on "
        f"{X_arr.shape[0]}×{X_arr.shape[1]} data"
    )

    pipeline = MLPipeline(X=X_arr, y=y_arr, config=pipeline_config)
    results = pipeline.run()

    logger.info(f"Analysis {analysis_cfg['id']} completed")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", required=True, help="YAML file with defaults+analyses"
    )
    args = parser.parse_args()

    # 0) Load config & data
    # 0) Load data
    cfg = yaml.safe_load(open(args.config))
    # Using TabularDataset directly to get full container
    # Assuming config provides necessary kwargs for reshaping if applicable
    data_path = cfg["data_path"]
    load_kwargs = cfg.get("loader_kwargs", {})

    # Check if we should reshape based on config hint or defaults
    # For now, simplistic loading unless config specifies columns_to_dims
    ds = TabularDataset(data_path, **load_kwargs)
    full_container = ds.load()

    all_results = {}
    defaults = cfg.get("defaults", {})

    for analysis in cfg["analyses"]:
        # merge defaults + specific
        analysis_cfg = deepcopy(defaults)
        analysis_cfg.update(analysis)

        # 1) Select relevant data using DataContainer
        # Map generic 'select_features' config to DataContainer.select()
        # "spatial_units" -> usually 'channel' dimension
        # "feature_names" -> 'feature' dimension
        # "covariates" -> handled at load time or aux selection

        selection_query = {}

        if "spatial_units" in analysis_cfg and analysis_cfg["spatial_units"] != "all":
            # Assuming 'channel' dim exists if reshaping happened
            # If strictly flat 2D but logical spatial units exist,
            # user should have loaded with columns_to_dims=['channel', 'feature']
            if "channel" in full_container.dims:
                selection_query["channel"] = analysis_cfg["spatial_units"]

        if "feature_names" in analysis_cfg and analysis_cfg["feature_names"] != "all":
            if "feature" in full_container.dims:
                selection_query["feature"] = analysis_cfg["feature_names"]

        # Row filters (e.g. subjects)
        if "row_filter" in analysis_cfg:
            # This requires parsing row_filter dicts to container.select queries
            # Simplified: assume simple kwargs for now
            # row_filter: {'group': 1}
            pass

        if "target_columns" in analysis_cfg:
            # If target was not set at load time, it might be in X
            # But DataContainer should ideally handle y.
            # If y IS loaded, we effectively just use it.
            # If we need to SWAP target (rare), we'd need to manipulate container.
            pass

        # Apply Selection
        # If no query, we get the whole container
        sub_container = (
            full_container.select(**selection_query)
            if selection_query
            else full_container
        )

        X = sub_container.X
        y = sub_container.y

        logger.info(
            f"Analysis {analysis['id']} selected shape {X.shape}, "
            f"target available: {y is not None}"
        )

        # 1.5) Concatenate covariates if requested and separated?
        # If covariates are in coords, we might need to add them back to X for some ML pipelines?
        # Or MLPipeline handles coords? For now assume standard X, y.

        # 2) Run
        # ensure results_dir/file come from global defaults if not overwritten
        analysis_cfg["results_dir"] = cfg.get(
            "results_dir", analysis_cfg.get("results_dir")
        )
        analysis_cfg["results_file"] = cfg.get(
            "results_file", analysis_cfg.get("results_file")
        )

        results = run_analysis(sub_container.X, sub_container.y, analysis_cfg)
        all_results[analysis["id"]] = results

    # 3) Save all results
    out_path = os.path.join(cfg["results_dir"], f"{cfg['global_experiment_id']}.pkl")
    logger.info(f"Saving aggregated results to {out_path}")
    pd.to_pickle(all_results, out_path)


if __name__ == "__main__":
    main()
