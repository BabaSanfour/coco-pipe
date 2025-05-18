import yaml
import numpy as np
import pandas as pd
from coco_pipe.io import load, select_features
from coco_pipe.ml.pipeline import MLPipeline
from copy import deepcopy

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_analysis(X, y, analysis_cfg):
    """Run a single analysis with given config"""
    X = X.values
    y = y.values
    
    # Extract MLPipeline specific arguments
    pipeline_args = {
        'X': X,
        'y': y,
        'config': {
            'task': analysis_cfg.get('task'),
            'analysis_type': analysis_cfg.get('analysis_type'),
            'models': analysis_cfg.get('models'),
            'metrics': analysis_cfg.get('metrics'),
            'cv_strategy': analysis_cfg.get('cv_kwargs', {}).get('strategy'),
            'n_splits': analysis_cfg.get('cv_kwargs', {}).get('n_splits'),
            'n_features': analysis_cfg.get('n_features'),
            'direction': analysis_cfg.get('direction'),
            'search_type': analysis_cfg.get('search_type'),
            'n_iter': analysis_cfg.get('n_iter'),
            'scoring': analysis_cfg.get('scoring'),
            'n_jobs': analysis_cfg.get('n_jobs'),
            'results_dir': analysis_cfg.get('results_dir'),
            'results_file': analysis_cfg.get('results_file'),
            'cv_kwargs': analysis_cfg.get('cv_kwargs')
        }
    }
    
    # Remove None values from config
    pipeline_args['config'] = {k: v for k, v in pipeline_args['config'].items() if v is not None}
    
    pipeline = MLPipeline(**pipeline_args)
    results = pipeline.run()
    logger.info(f"Analysis {analysis_cfg['id']} completed")
    return results

def main():
    # 0) Load config & data
    cfg = yaml.safe_load(open("../configs/toy_ml_config.yml"))
    df = load("tabular", cfg["data_path"])
    all_results = {}

    # Get default parameters
    defaults = cfg["defaults"]

    # Run different analyses based on config
    for analysis in cfg["analyses"]:
        # Create analysis config starting with defaults
        analysis_cfg = deepcopy(defaults)
        
        # Update with analysis-specific settings
        analysis_cfg.update(analysis)

        # 1) Select features & target based on analysis config
        X, y = select_features(
            df,
            target_columns=analysis_cfg["target_columns"],
            covariates=analysis_cfg.get("covariates"),
            spatial_units=analysis_cfg.get("spatial_units"),
            feature_names=analysis_cfg.get("feature_names", "all"),
            row_filter=analysis_cfg.get("row_filter"),
        )

        # 1.1) Print the shape of the selected features and target
        logger.info(f"Analysis {analysis['id']} selected {X.shape[1]} features and {y.shape[0]} samples")
        logger.info(f"Analysis {analysis['id']} selected features: {X.columns.tolist()}")
        logger.info(f"Analysis {analysis['id']} selected target: {y.name}")

        # 2) Run analysis
        analysis_cfg["results_dir"] = cfg["results_dir"]
        analysis_cfg["results_file"] = cfg["results_file"]
        results = run_analysis(X, y, analysis_cfg)
        
        # Store results with analysis identifier
        all_results[analysis["id"]] = results

    # 3) Save all results
    results_file = f"{cfg['results_dir']}/{cfg['global_experiment_id']}.pkl"
    logger.info(f"Saving results to {results_file}")
    pd.to_pickle(all_results, results_file)

if __name__ == "__main__":
    main()