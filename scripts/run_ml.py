#!/usr/bin/env python3
import argparse
import logging
import os
import json
import yaml
import numpy as np
import pandas as pd
from copy import deepcopy

from coco_pipe.io import load, select_features
from coco_pipe.ml.pipeline import MLPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_analysis(X, y, analysis_cfg):
    """Run a single analysis with the given config, passing through the new `mode`."""
    # scikit-learn pipelines expect numpy arrays
    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y

    # Build the MLPipeline config dict
    pipeline_config = {
        "task":           analysis_cfg.get("task"),
        "analysis_type":  analysis_cfg.get("analysis_type"),
        "models":         analysis_cfg.get("models"),
        "metrics":        analysis_cfg.get("metrics"),
        "cv_strategy":    analysis_cfg.get("cv_kwargs", {}).get("cv_strategy"),
        "n_splits":       analysis_cfg.get("cv_kwargs", {}).get("n_splits"),
        "cv_kwargs":      analysis_cfg.get("cv_kwargs"),
        "n_features":     analysis_cfg.get("n_features"),
        "direction":      analysis_cfg.get("direction"),
        "search_type":    analysis_cfg.get("search_type"),
        "n_iter":         analysis_cfg.get("n_iter"),
        "scoring":        analysis_cfg.get("scoring"),
        "n_jobs":         analysis_cfg.get("n_jobs"),
        "save_intermediate": analysis_cfg.get("save_intermediate"),
        "results_dir":    analysis_cfg.get("results_dir"),
        "results_file":   analysis_cfg.get("results_file"),
        "mode":           analysis_cfg.get("mode"),
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
    cfg = yaml.safe_load(open(args.config))
    df = load("tabular", cfg["data_path"])
    all_results = {}

    defaults = cfg.get("defaults", {})

    for analysis in cfg["analyses"]:
        # merge defaults + specific
        analysis_cfg = deepcopy(defaults)
        analysis_cfg.update(analysis)

        # 1) Select features & target
        X, y = select_features(
            df,
            target_columns=analysis_cfg["target_columns"],
            covariates=analysis_cfg.get("covariates"),
            spatial_units=analysis_cfg.get("spatial_units"),
            feature_names=analysis_cfg.get("feature_names", "all"),
            row_filter=analysis_cfg.get("row_filter"),
        )

        logger.info(
            f"Analysis {analysis['id']} selected {X.shape[1]} features, "
            f"{y.shape[0]} samples, target '{y.name}'"
        )
        # show first few features
        feat_list = X.columns.tolist()
        logger.info(
            f"First features: {feat_list[:5]}{'...' if len(feat_list)>5 else ''}"
        )

        # 2) Run
        # ensure results_dir/file come from global defaults if not overwritten
        anal
