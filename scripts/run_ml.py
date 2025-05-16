#!/usr/bin/env python3
"""
run_ml.py

Command-line interface for running various machine learning analyses based on
a YAML configuration file with flexible feature grouping definitions.

This script is a high-level interface to the coco_pipe.ml module, supporting:
- Classification and regression tasks (auto-detected or specified)
- Multiple analysis types (baseline, feature selection, hyperparameter search)
- Single target and multivariate tasks
- Flexible feature grouping and global features
- Results persistence in multiple formats
- Multiple cross-validation strategies
"""

import os
import sys
import argparse
import logging
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Adjust sys.path to include the coco_pipe folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coco_pipe.ml.pipeline import run_ml_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def load_config(config_path):
    """
    Load YAML configuration from the specified file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_config):
    """
    Load and preprocess the dataset from the configuration.
    
    Parameters
    ----------
    data_config : dict
        Data configuration containing:
        - file: path to data file (CSV/Parquet)
        - target: target column name(s)
        - feature_columns: optional, specific features to use
        - index_column: optional, index column name
        - preprocessing: optional, preprocessing steps
        
    Returns
    -------
    tuple
        (X, y) tuple of features and target(s)
    """
    # Load data file
    data_file = data_config.get("file")
    if not data_file:
        raise ValueError("Data file path must be provided in the config (data:file).")
    if not os.path.isabs(data_file):
        data_file = os.path.abspath(data_file)
    if not os.path.exists(data_file):
        raise ValueError(f"Data file does not exist: {data_file}")
    
    # Read data based on file format
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    elif data_file.endswith('.parquet'):
        df = pd.read_parquet(data_file)
    else:
        raise ValueError(f"Unsupported file format: {data_file}")
    
    # Handle index column if specified
    index_col = data_config.get("index_column")
    if index_col:
        if index_col in df.columns:
            df.set_index(index_col, inplace=True)
        else:
            logging.warning(f"Index column '{index_col}' not found in data.")
    
    # Extract target(s)
    target = data_config.get("target")
    if not target:
        raise ValueError("Target column(s) must be specified in the config.")
    
    if isinstance(target, str):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data.")
        y = df[target]
    elif isinstance(target, list):
        missing = [col for col in target if col not in df.columns]
        if missing:
            raise ValueError(f"Target columns not found in data: {missing}")
        y = df[target]
    else:
        raise ValueError("Target must be a string or list of strings.")
    
    # Extract features
    feature_cols = data_config.get("feature_columns")
    if feature_cols:
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in data: {missing}")
        X = df[feature_cols]
    else:
        # Use all columns except target(s)
        exclude_cols = [target] if isinstance(target, str) else target
        X = df.drop(columns=exclude_cols)
    
    # Apply preprocessing if specified
    preprocessing = data_config.get("preprocessing", {})
    if preprocessing:
        X = preprocess_data(X, preprocessing)
    
    return X, y

def preprocess_data(X, preprocessing_config):
    """
    Apply preprocessing steps to the feature matrix.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    preprocessing_config : dict
        Preprocessing configuration
        
    Returns
    -------
    pd.DataFrame
        Preprocessed feature matrix
    """
    # Handle missing values
    missing_strategy = preprocessing_config.get("missing_values", "drop")
    if missing_strategy == "drop":
        X = X.dropna()
    elif missing_strategy == "mean":
        X = X.fillna(X.mean())
    elif missing_strategy == "median":
        X = X.fillna(X.median())
    elif missing_strategy == "zero":
        X = X.fillna(0)
    
    # Handle categorical variables
    categorical_strategy = preprocessing_config.get("categorical", "drop")
    if categorical_strategy == "drop":
        X = X.select_dtypes(exclude=['object'])
    elif categorical_strategy == "encode":
        X = pd.get_dummies(X)
    
    # Handle scaling
    scaling = preprocessing_config.get("scaling")
    if scaling == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    elif scaling == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    
    return X

def save_results(results, output_dir, analysis_id, format="all"):
    """
    Save analysis results in multiple formats.
    
    Parameters
    ----------
    results : dict
        Analysis results to save
    output_dir : str
        Directory to save results
    analysis_id : str
        Identifier for the analysis
    format : str, optional
        Output format(s): "yaml", "json", "all"
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save in requested format(s)
    if format in ["yaml", "all"]:
        yaml_file = output_dir / f"{analysis_id}_{timestamp}_results.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(results, f)
        logging.info(f"Results saved as YAML: {yaml_file}")
    
    if format in ["json", "all"]:
        # Save full results
        json_file = output_dir / f"{analysis_id}_{timestamp}_results.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved as JSON: {json_file}")
        
        # Extract and save metrics separately
        metrics = {}
        for name, result in results.items():
            if isinstance(result, dict) and 'results' in result:
                metrics[name] = {
                    model.get('model_name', 'unknown'): model.get('metrics', {})
                    for model in result['results']
                }
        
        metrics_file = output_dir / f"{analysis_id}_{timestamp}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Metrics saved separately: {metrics_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Run ML analyses based on a YAML configuration file."
    )
    parser.add_argument("--config", required=True,
                       help="Path to YAML configuration file")
    parser.add_argument("--output-dir", "-o", default="results",
                       help="Directory to save results")
    parser.add_argument("--format", "-f", default="all",
                       choices=["yaml", "json", "all"],
                       help="Output format for results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = load_config(args.config)
        analysis_id = config.get("id", "analysis")
        
        # Load data
        X, y = load_data(config["data"])
        logging.info(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
        
        # Run analysis
        results = run_ml_analysis(
            X=X,
            y=y,
            task_type=config.get("task_type"),  # Auto-detect if not specified
            analysis_config=config.get("analyses", []),
            feature_groups=config.get("feature_groups"),
            global_features=config.get("global_features"),
            cv_strategy=config.get("cv_strategy", "stratified"),
            random_state=config.get("random_state", 42),
            n_jobs=config.get("n_jobs", -1)
        )
        
        # Save results
        save_results(
            results=results,
            output_dir=args.output_dir,
            analysis_id=analysis_id,
            format=args.format
        )
        
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Error running analysis: {str(e)}")
        if args.debug:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()