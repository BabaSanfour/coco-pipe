#!/usr/bin/env python3
"""
Command-line script to run ML pipelines for various classification and clustering analyses.

Usage:
    python run_ml.py --analysis 1 --analysis_type all_ages
"""

import os
import sys
import argparse
import pickle
import logging
import pandas as pd

# Ensure the package is importable. Adjust the path as needed.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import wrapper functions from the package.
from coco_pipe.ml import (
    pipeline_baseline,
    pipeline_feature_selection,
    pipeline_HP_search,
    pipeline_feature_selection_HP_search,
    pipeline_unsupervised
)

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configuration for directories. You may replace these with your own config.
try:
    from utils.config import data_dir, results_dir
except ImportError:
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

def load_and_preprocess_features(analysis_type):
    """
    Load the features CSV file, filter by age group if required, and drop unneeded columns.
    """
    features_file = os.path.join(data_dir, "csv", "features_with_age_group.csv")
    features = pd.read_csv(features_file)
    
    if analysis_type == "adolescent":
        logging.info("Running analysis for adolescent group.")
        features = features[features["age_group"] == "adolescent"]
    elif analysis_type == "child":
        logging.info("Running analysis for child group.")
        features = features[features["age_group"] == "child"]
    else:
        logging.info("Running analysis for all ages.")
    
    drop_cols = ["dataset", "id", "age", "sex", "task", "subject", "age_group"]
    features = features.drop(columns=drop_cols)
    features["group"] = features["group"].replace({"PAT": 1, "CTR": 0})
    return features

def get_clean_features(columns):
    """Return a list of simplified feature names."""
    seen = set()
    clean_features = []
    for col in columns:
        feat = col.split(".spaces-")[0].replace("feature-", "")
        if feat not in seen and feat != "group":
            seen.add(feat)
            clean_features.append(feat)
    return clean_features

def get_columns_for_feature(columns, feature):
    """Return a list of columns that include the given feature substring."""
    return [col for col in columns if feature in col]

def save_results(results, fname):
    """Save results as a pickle file."""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(os.path.join(results_dir, fname), "wb") as f:
        pickle.dump(results, f)

# Analysis functions (examples)

def run_analysis0(features, analysis_type):
    """Baseline: run a separate baseline per sensor (i.e. per feature)."""
    X = features.drop("group", axis=1)
    y = features["group"]
    feature_results = {}
    for feature in X.columns:
        logging.info(f"Running baseline pipeline for feature: {feature}")
        result = pipeline_baseline(X[[feature]], y)
        feature_results[feature] = result["results"]
    fname = f"single_feature_baseline_results_{analysis_type}.pkl"
    save_results(feature_results, fname)
    logging.info("Completed analysis 0!")

def run_analysis1(features, analysis_type):
    """HP search for each sensor (per feature)."""
    X = features.drop("group", axis=1)
    y = features["group"]
    model_results = {}
    for model in ["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors"]:
        logging.info(f"Running HP search for model: {model}")
        feature_results = {}
        for feature in X.columns:
            logging.info(f"Running HP search for feature: {feature}")
            result = pipeline_HP_search(X[[feature]], y, model)
            feature_results[feature] = result["best_score"]
        model_results[model] = feature_results
    fname = f"single_feature_HP_results_{analysis_type}.pkl"
    save_results(model_results, fname)
    logging.info("Completed analysis 1!")

def run_analysis9(features, analysis_type):
    """Unsupervised learning using all features."""
    X = features.drop("group", axis=1)
    clusters_results = {}
    for n_clusters in range(2, 11):
        logging.info(f"Running unsupervised pipeline for {n_clusters} clusters")
        silhouette, cluster_labels = pipeline_unsupervised(X, n_clusters)
        clusters_results[n_clusters] = {
            "silhouette": silhouette,
            "cluster_labels": cluster_labels,
        }
    fname = f"unsupervised_results_{analysis_type}.pkl"
    save_results(clusters_results, fname)
    logging.info("Completed analysis 9!")

# Additional analysis functions (e.g., run_analysis2, run_analysis3, etc.) can be added following a similar pattern.

def main():
    parser = argparse.ArgumentParser(description="Run ML pipelines analyses.")
    parser.add_argument("--analysis", type=int, default=0, help="Analysis number to run (e.g., 0, 1, 9).")
    parser.add_argument("--analysis_type", type=str, default="all_ages", help="Analysis type (adolescent, child, all_ages).")
    args = parser.parse_args()
    
    analysis = args.analysis
    analysis_type = args.analysis_type
    features = load_and_preprocess_features(analysis_type)
    
    # Map analysis numbers to functions. Extend as needed.
    analysis_map = {
        0: run_analysis0,
        1: run_analysis1,
        9: run_analysis9,
    }
    
    if analysis in analysis_map:
        analysis_map[analysis](features, analysis_type)
    else:
        logging.error("Invalid analysis number provided. Choose a valid analysis.")

if __name__ == "__main__":
    main()
