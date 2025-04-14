#!/usr/bin/env python3
"""
run_ml.py

Command-line interface for running various machine learning analyses based on
a YAML configuration file with flexible feature grouping definitions.

This script supports cases ranging from multiple sensors (each with several features)
to edge cases (e.g., mixed behavioural measures) where no explicit grouping is provided.
The analyses use an ML pipeline defined in ml.py (located in the coco_pipe folder)
with the following supported modes and subset options:

Analysis Modes and Subset Options:
  - Baseline:
      * all_features_all_groups: Analyze on the entire feature set.
      * all_features_per_group:   For each group, analyze the subset of features whose
                                  column names contain the group name.
      * single_feature_all_groups: For each feature (as defined in the mapping),
                                   analyze over all matching columns.
      * single_feature_per_group:  For each group, analyze each individual feature.
  - Feature Selection (FS):
      * (Same subset options as Baseline; for single-feature analyses, num_features is set to 1)
  - Hyperparameter Search (HP search):
      * (Same subset options as Baseline)
  - Combined FS + HP Search:
      * (Same subset options as Baseline)

Results for every subanalysis are saved into their own pickle file with names:
    <ID>_<AnalysisName>_<AnalysisType>_results.pickle

Usage:
    python run_ml.py --config path/to/config.yml

Adapted YAML configuration example:
-----------------------------------------------------------
ID: "example_analysis_01"
data:
  file: "data/dataset.csv"
  target: "target"
  # Provide feature_groups mapping. If not provided, the entire feature set is used as a single global group.
  features_groups:
    groups: ["sensor1", "sensor2"]
    features: ["feat1", "feat2", "feat3", "feat4", "feat5"]
    global:
      - "sensor1.feat1"
      - "sensor1.feat2"
      - "sensor1.feat3"
      - "sensor2.feat4"
      - "sensor2.feat5"
analysis:
  - name: "Baseline Global"
    type: "baseline"
    subset: "all_features_all_groups"
    models: "all"
    scoring: "accuracy"
  - name: "Baseline Per Group"
    type: "baseline"
    subset: "all_features_per_group"
    models: "all"
    scoring: "accuracy"
  - name: "Single Feature Global"
    type: "baseline"
    subset: "single_feature_all_groups"
    models: "all"
    scoring: "f1-score"
  - name: "Feature Selection Per Group"
    type: "fs"
    subset: "single_feature_per_group"
    num_features: 1
    models: ["Random Forest", "SVC"]
    scoring: "auc"
  - name: "FS + HP Search Global"
    type: "fs_hp"
    subset: "all_features_all_groups"
    num_features: 3
    models: ["Logistic Regression"]
    scoring: "accuracy"
output: "results"
-----------------------------------------------------------

Note:
  If neither group information nor features are provided in the config,
  all columns are used as a single “global” group.
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import pickle

# Adjust sys.path to include the coco_pipe folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "coco_pipe")))

# Import ML pipeline wrappers from ml.py.
from ml import (
    pipeline_baseline,
    pipeline_feature_selection,
    pipeline_HP_search,
    pipeline_feature_selection_HP_search,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(config_path):
    """Load YAML configuration from the specified file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_config):
    """
    Load the dataset and determine the feature mapping.

    Expects:
      - 'file': CSV file path.
      - 'target': Target column name.
      - Optionally, 'features_groups': mapping with keys:
           "groups": list of group names,
           "features": list of feature names,
           "global": list of all feature column names formatted so that group names are embedded.
    If the mapping is absent, returns a default mapping using all columns.

    Returns:
      X: DataFrame of features.
      y: Series of target values.
      feature_mapping: dict with keys "groups", "features", "global".
    """
    data_file = data_config.get("file")
    if not data_file:
        raise ValueError("Data file path must be provided in the config (data:file).")
    if not os.path.isabs(data_file):
        data_file = os.path.abspath(data_file)
    if not os.path.exists(data_file):
        raise ValueError(f"Data file does not exist: {data_file}")
    if not data_file:
        raise ValueError("Data file path must be provided in the config (data:file).")
    df = pd.read_csv(data_file, index_col=0)
    df.drop(columns=["subject"], inplace=True, errors="ignore")
    target_col = data_config.get("target")
    if not target_col or target_col not in df.columns:
        raise ValueError("A valid target column must be specified and must exist in the data.")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def save_results_pickle(results, output_file):
    """Save the given results to a pickle file."""
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    logging.info(f"Results saved to {output_file}")


def run_baseline(X, y, subset, analysis_name, models, scoring, feature_mapping):
    """
    Run baseline analysis based on the chosen subset option.

    The expected feature_mapping dictionary should include:
      - 'groups': list of group names (e.g., sensors/regions).
      - 'features': list of feature names.
      - 'global': list of all column names in the dataset; ideally, group names are embedded
                  in these column names (e.g., "sensor1.feat1", "sensor2.feat4").

    Subset options:
      - "all_features_all_groups": Analyze the entire feature set.
      - "all_features_per_group":   For each group, analyze the subset of features whose column names contain the group name.
      - "single_feature_all_groups": For each feature (from feature_mapping['features']), analyze over all matching columns.
      - "single_feature_per_group":  For each group and for each matching column, analyze that single feature.

    Edge case: If the mapping is missing or incomplete, all columns are treated as global.
    """
    # Fallback: if mapping is missing or incomplete.
    feature_mapping["global"] = feature_mapping.get("global", list(X.columns))
    # if feature_mapping is None or not (feature_mapping.get("groups") and feature_mapping.get("global")):
    #     logging.info("No complete feature mapping provided; using all columns as global.")
    #     feature_mapping = {
    #         "groups": [],
    #         "features": list(X.columns),
    #         "global": list(X.columns)
    #     }

    results = {}

    if subset == "all_features_all_groups":
        logging.info(f"{analysis_name}: Running baseline analysis on the entire feature set.")
        results["global"] = pipeline_baseline(X, y, scoring=scoring, models=models)

    elif subset == "all_features_per_group":
        if feature_mapping["groups"]:
            logging.info(f"{analysis_name}: Running baseline analysis per group.")
            for group in feature_mapping["groups"]:
                logging.info(f"Processing group '{group}'.")
                cols = [col for col in feature_mapping["global"] if group in col]
                if not cols:
                    logging.warning(f"No columns found for group '{group}'. Skipping.")
                    continue
                results[group] = pipeline_baseline(X[cols], y, scoring=scoring, models=models)
        else:
            logging.info(f"{analysis_name}: No groups defined; running baseline analysis globally.")
            results["global"] = pipeline_baseline(X, y, scoring=scoring, models=models)

    elif subset == "single_feature_all_groups":
        logging.info(f"{analysis_name}: Running single feature baseline analysis (global).")
        for feat in feature_mapping.get("features", X.columns):
            logging.info(f"Processing feature '{feat}'.")
            matched = [col for col in feature_mapping["global"] if feat in col]
            if not matched and feat in X.columns:
                matched = [feat]
            if not matched:
                logging.warning(f"Feature '{feat}' not found. Skipping.")
                continue
            results[feat] = pipeline_baseline(X[matched], y, scoring=scoring, models=models)

    elif subset == "single_feature_per_group":
        if feature_mapping["groups"]:
            logging.info(f"{analysis_name}: Running single feature baseline analysis per group.")
            for group in feature_mapping["groups"]:
                group_cols = [col for col in feature_mapping["global"] if group in col]
                if not group_cols:
                    logging.warning(f"No columns found for group '{group}'. Skipping.")
                    continue
                for col in group_cols:
                    key = f"{group}_{col}"
                    logging.info(f"Processing group '{group}', feature '{col}'.")
                    results[key] = pipeline_baseline(X[[col]], y, scoring=scoring, models=models)
        else:
            logging.info(f"{analysis_name}: No groups defined; running single feature baseline analysis globally.")
            for col in feature_mapping["global"]:
                logging.info(f"Processing feature '{col}'.")
                results[col] = pipeline_baseline(X[[col]], y, scoring=scoring, models=models)

    else:
        raise ValueError(f"Unknown subset option '{subset}' for baseline analysis.")

    return results


def run_feature_selection(X, y, subset, analysis_name, models, scoring, num_features, feature_mapping):
    """
    Run feature selection analysis based on the chosen subset option.
    For single-feature analyses, num_features is overridden to 1.

    Expected feature_mapping has keys:
      - 'groups': list of group names.
      - 'features': list of feature names.
      - 'global': list of all feature column names.

    Subset options are the same as in run_baseline.
    """
    # Fallback if mapping is missing or incomplete.
    feature_mapping["global"] = feature_mapping.get("global", list(X.columns))

    # if feature_mapping is None or not (feature_mapping.get("groups") and feature_mapping.get("global")):
    #     logging.info("No complete feature mapping provided; using all columns as global.")
    #     feature_mapping = {
    #         "groups": [],
    #         "features": list(X.columns),
    #         "global": list(X.columns)
    #     }

    results = {}

    if subset in ["all_features_all_groups"]:
        logging.info(f"{analysis_name}: Running feature selection on the entire feature set.")
        results["global"] = pipeline_feature_selection(X, y, num_features=num_features, scoring=scoring, models=models)

    elif subset == "all_features_per_group":
        if feature_mapping["groups"]:
            logging.info(f"{analysis_name}: Running feature selection per group.")
            for group in feature_mapping["groups"]:
                logging.info(f"Processing group '{group}'.")
                cols = [col for col in feature_mapping["global"] if group in col]
                if not cols:
                    logging.warning(f"No columns found for group '{group}'. Skipping.")
                    continue
                results[group] = pipeline_feature_selection(X[cols], y, num_features=num_features, scoring=scoring, models=models)
        else:
            logging.info(f"{analysis_name}: No groups defined; running global feature selection.")
            results["global"] = pipeline_feature_selection(X, y, num_features=num_features, scoring=scoring, models=models)

    elif subset == "single_feature_all_groups":
        logging.info(f"{analysis_name}: Running single feature feature selection (global).")
        for feat in feature_mapping.get("features", X.columns):
            logging.info(f"Processing single feature '{feat}'.")
            matched = [col for col in feature_mapping["global"] if feat in col]
            if not matched and feat in X.columns:
                matched = [feat]
            if not matched:
                logging.warning(f"Feature '{feat}' not found. Skipping.")
                continue
            results[feat] = pipeline_feature_selection(X[matched], y, num_features=1, scoring=scoring, models=models)

    elif subset == "single_feature_per_group":
        if feature_mapping["groups"]:
            logging.info(f"{analysis_name}: Running single feature feature selection per group.")
            for group in feature_mapping["groups"]:
                group_cols = [col for col in feature_mapping["global"] if group in col]
                if not group_cols:
                    logging.warning(f"No columns found for group '{group}'. Skipping.")
                    continue
                for col in group_cols:
                    key = f"{group}_{col}"
                    logging.info(f"Processing group '{group}', feature '{col}'.")
                    results[key] = pipeline_feature_selection(X[[col]], y, num_features=1, scoring=scoring, models=models)
        else:
            logging.info(f"{analysis_name}: No groups defined; running single feature selection globally.")
            for col in feature_mapping["global"]:
                logging.info(f"Processing single feature '{col}'.")
                results[col] = pipeline_feature_selection(X[[col]], y, num_features=1, scoring=scoring, models=models)

    else:
        raise ValueError(f"Unknown subset option '{subset}' for feature selection analysis.")

    return results


def run_hp_search(X, y, subset, analysis_name, models, scoring, feature_mapping):
    """
    Run hyperparameter search analysis based on the chosen subset option.

    Expected feature_mapping has keys:
      - 'groups': list of group names.
      - 'features': list of feature names.
      - 'global': list of all feature column names.

    Subset options are the same as in run_baseline.
    """
    feature_mapping["global"] = feature_mapping.get("global", list(X.columns))

    # if feature_mapping is None or not (feature_mapping.get("groups") and feature_mapping.get("global") and feature_mapping.get("features")):
    #     logging.info("No complete feature mapping provided; using all columns as global.")
    #     feature_mapping = {
    #         "groups": [],
    #         "features": list(X.columns),
    #         "global": list(X.columns)
    #     }

    results = {}

    if subset == "all_features_all_groups":
        logging.info(f"{analysis_name}: Running HP search on the entire feature set (global).")
        results["global"] = pipeline_HP_search(X, y, models=models, scoring=scoring)

    elif subset == "all_features_per_group":
        if feature_mapping["groups"]:
            logging.info(f"{analysis_name}: Running HP search per group.")
            for group in feature_mapping["groups"]:
                logging.info(f"Processing group '{group}'.")
                cols = [col for col in feature_mapping["global"] if group in col]
                if not cols:
                    logging.warning(f"No columns found for group '{group}'. Skipping.")
                    continue
                results[group] = pipeline_HP_search(X[cols], y, models=models, scoring=scoring)
        else:
            logging.info(f"{analysis_name}: No groups defined; running HP search globally.")
            results["global"] = pipeline_HP_search(X, y, models=models, scoring=scoring)

    elif subset == "single_feature_all_groups":
        logging.info(f"{analysis_name}: Running HP search for each single feature (global).")
        for feat in feature_mapping["features"]:
            logging.info(f"Processing single feature '{feat}'.")
            matched = [col for col in feature_mapping["global"] if feat in col]
            if not matched and feat in X.columns:
                matched = [feat]
            if not matched:
                logging.warning(f"Feature '{feat}' not found. Skipping.")
                continue
            results[feat] = pipeline_HP_search(X[matched], y, models=models, scoring=scoring)

    elif subset == "single_feature_per_group":
        if feature_mapping["groups"]:
            logging.info(f"{analysis_name}: Running HP search for each single feature per group.")
            for group in feature_mapping["groups"]:
                group_cols = [col for col in feature_mapping["global"] if group in col]
                if not group_cols:
                    logging.warning(f"No columns found for group '{group}'. Skipping.")
                    continue
                for col in group_cols:
                    key = f"{group}_{col}"
                    logging.info(f"Processing group '{group}', feature '{col}'.")
                    results[key] = pipeline_HP_search(X[[col]], y, models=models, scoring=scoring)
        else:
            logging.info(f"{analysis_name}: No groups defined; running HP search for each single feature globally.")
            for col in feature_mapping["global"]:
                logging.info(f"Processing single feature '{col}'.")
                results[col] = pipeline_HP_search(X[[col]], y, models=models, scoring=scoring)

    else:
        raise ValueError(f"Unknown subset option '{subset}' for HP search analysis.")

    return results


def run_fs_hp_search(X, y, subset, analysis_name, models, scoring, num_features, feature_mapping):
    """
    Run combined feature selection and hyperparameter search (FS + HP search) analysis.

    Expected feature_mapping keys:
      - 'groups': list of group names.
      - 'features': list of feature names.
      - 'global': list of all feature column names.
      
    Subset options are the same as in the other functions.
    """
    feature_mapping["global"] = feature_mapping.get("global", list(X.columns))

    # if feature_mapping is None or not (feature_mapping.get("groups") and feature_mapping.get("global") and feature_mapping.get("features")):
    #     logging.info("No complete feature mapping provided; using all columns as global.")
    #     feature_mapping = {
    #         "groups": [],
    #         "features": list(X.columns),
    #         "global": list(X.columns)
    #     }

    results = {}

    if subset == "all_features_all_groups":
        logging.info(f"{analysis_name}: Running FS + HP search on the entire feature set (global).")
        results["global"] = pipeline_feature_selection_HP_search(X, y, num_features=num_features, models=models, scoring=scoring)

    elif subset == "all_features_per_group":
        if feature_mapping["groups"]:
            logging.info(f"{analysis_name}: Running FS + HP search per group.")
            for group in feature_mapping["groups"]:
                logging.info(f"Processing group '{group}'.")
                cols = [col for col in feature_mapping["global"] if group in col]
                if not cols:
                    logging.warning(f"No columns found for group '{group}'. Skipping.")
                    continue
                results[group] = pipeline_feature_selection_HP_search(X[cols], y, num_features=num_features, models=models, scoring=scoring)
        else:
            logging.info(f"{analysis_name}: No groups defined; running FS + HP search globally.")
            results["global"] = pipeline_feature_selection_HP_search(X, y, num_features=num_features, models=models, scoring=scoring)

    elif subset == "single_feature_all_groups":
        logging.info(f"{analysis_name}: Running FS + HP search for each single feature (global).")
        for feat in feature_mapping["features"]:
            logging.info(f"Processing single feature '{feat}'.")
            matched = [col for col in feature_mapping["global"] if feat in col]
            if not matched and feat in X.columns:
                matched = [feat]
            if not matched:
                logging.warning(f"Feature '{feat}' not found. Skipping.")
                continue
            results[feat] = pipeline_feature_selection_HP_search(X[matched], y, num_features=1, models=models, scoring=scoring)

    elif subset == "single_feature_per_group":
        if feature_mapping["groups"]:
            logging.info(f"{analysis_name}: Running FS + HP search for each single feature per group.")
            for group in feature_mapping["groups"]:
                group_cols = [col for col in feature_mapping["global"] if group in col]
                if not group_cols:
                    logging.warning(f"No columns found for group '{group}'. Skipping.")
                    continue
                for col in group_cols:
                    key = f"{group}_{col}"
                    logging.info(f"Processing group '{group}', single feature '{col}'.")
                    results[key] = pipeline_feature_selection_HP_search(X[[col]], y, num_features=1, models=models, scoring=scoring)
        else:
            logging.info(f"{analysis_name}: No groups defined; running FS + HP search for each single feature globally.")
            for col in feature_mapping["global"]:
                logging.info(f"Processing single feature '{col}'.")
                results[col] = pipeline_feature_selection_HP_search(X[[col]], y, num_features=1, models=models, scoring=scoring)

    else:
        raise ValueError(f"Unknown subset option '{subset}' for FS + HP search analysis.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ML analyses based on a YAML configuration file with flexible feature group definitions."
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    # Load configuration and data.
    config = load_config(args.config)
    # X, y, feature_mapping = load_data(config["data"])
    data = config["data"]
    X, y = load_data(data)
    feature_mapping = data.get("features_groups", {})
    analyses = config.get("analysis", [])
    all_combined_results = {}
    analysis_id = config.get("ID", "defaultID")
    output_dir = config.get("output_dir", "results")
    for analysis in analyses:
        analysis_name = analysis.get("name", "UnnamedAnalysis")
        analysis_type = analysis.get("type")
        subset = analysis.get("subset", "all_features_all_groups")
        models = analysis.get("models", "all")
        scoring = analysis.get("scoring", "accuracy")

        logging.info(f"Starting analysis: {analysis_name} (Type: {analysis_type}, Subset: {subset})")

        if analysis_type == "baseline":
            result = run_baseline(X, y, subset, analysis_name, models, scoring, feature_mapping)
        elif analysis_type == "fs":
            num_feats = analysis.get("num_features", 5)
            result = run_feature_selection(X, y, subset, analysis_name, models, scoring, num_feats, feature_mapping)
        elif analysis_type == "hp_search":
            result = run_hp_search(X, y, subset, analysis_name, models, scoring, feature_mapping)
        elif analysis_type == "fs_hp":
            num_feats = analysis.get("num_features", 5)
            result = run_fs_hp_search(X, y, subset, analysis_name, models, scoring, num_feats, feature_mapping)
        else:
            logging.error(f"Unknown analysis type: {analysis_type} for {analysis_name}")
            continue

        # Save the subanalysis result in its own pickle file.
        pickle_filename = f"{analysis_id}_{analysis_name}_{analysis_type}_results.pickle"
        output_file = os.path.join(output_dir, pickle_filename)
        save_results_pickle(result, output_file)

        all_combined_results[analysis_name] = result
        logging.info(f"Completed analysis: {analysis_name}")

    # Save combined results as YAML.

    combined_results_file = config.get("output", f"{analysis_id}_results.yaml")
    output_file = os.path.join(output_dir, combined_results_file)
    with open(output_file, "w") as outfile:
        yaml.dump(all_combined_results, outfile)
    logging.info(f"Combined analysis results saved to: {combined_results_file}")

# TODO fix single single

if __name__ == "__main__":
    main()