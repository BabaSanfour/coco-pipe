"""
Unified regression pipeline supporting single target and multivariate tasks.

This module provides a single, comprehensive regression function that detects
the appropriate pipeline based on the target variable(s) and runs the pipeline.

Key Features:
- Automatic task type detection
- Support for single target and multivariate regression
- Consistent interface across regression types
"""

import logging
import numpy as np
from typing import Union
import pandas as pd
from .single_target_regression import SingleTargetRegressionPipeline
from .multivariate_regression import MultivariateRegressionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def regression(
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        type: str = "baseline",
        models: Union[str, list] = "all",
        metrics: Union[str, list] = None,
        random_state: int = 42,
        cv_strategy: str = "stratified",
        n_features: int = None,
        direction: str = "forward",
        search_type: str = "grid",
        n_iter: int = 100,
        scoring: str = None,
        n_jobs: int = -1
    ) -> dict:
    """
    Automatically detect and run the appropriate regression pipeline.

    This function examines the target variable(s) and determines whether to run
    single target or multivariate regression. It then creates and executes the
    appropriate pipeline.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target vector(s)
    type : str, optional (default="baseline")
        Type of analysis to run:
        - "baseline": Basic model evaluation
        - "feature_selection": Feature selection
        - "hp_search": Hyperparameter optimization
    models : str or list, optional (default="all")
        Models to include in the pipeline
    metrics : str or list, optional (default=None)
        Metrics to evaluate
    random_state : int, optional (default=42)
        Random state for reproducibility
    cv_strategy : str, optional (default="stratified")
        Cross-validation strategy
    n_features : int, optional (default=None)
        Number of features to select
    direction : str, optional (default="forward")
        Direction of feature selection
    search_type : str, optional (default="grid")
        Search type for hyperparameter optimization
    n_iter : int, optional (default=100)
        Number of iterations for hyperparameter optimization
    scoring : str, optional (default=None)
        Scoring metric for hyperparameter optimization
    n_jobs : int, optional (default=-1)
        Number of parallel jobs

    Returns
    -------
    dict
        Dictionary containing:
        - pipeline: The fitted pipeline object
        - results: Results from pipeline execution including:
            - metrics: Performance metrics
            - predictions: Model predictions
            - feature_importances: Feature importance scores (if available)
            - selected_features: Selected features (if feature_selection)
            - best_params: Best parameters (if hp_search)
    """
    if type == "baseline":
        kwargs = {
            "models": models,
            "metrics": metrics,
            "random_state": random_state,
            "cv_strategy": cv_strategy,
            "n_jobs": n_jobs
        }
    elif type == "feature_selection":
        kwargs = {
            "models": models,
            "metrics": metrics,
            "random_state": random_state,
            "cv_strategy": cv_strategy,
            "n_jobs": n_jobs,
            "n_features": n_features,
            "direction": direction,
            "scoring": scoring
        }
    elif type == "hp_search":
        kwargs = {
            "models": models,
            "metrics": metrics,
            "random_state": random_state,
            "cv_strategy": cv_strategy,
            "n_jobs": n_jobs,
            "n_iter": n_iter,
            "search_type": search_type,
            "scoring": scoring
        }
    else:
        raise ValueError(f"Unknown analysis type: {type}")
        
    # Determine regression type
    if len(np.array(y).shape) == 2:
        pipeline_class = MultivariateRegressionPipeline
        task_type = "multivariate"
    else:
        pipeline_class = SingleTargetRegressionPipeline
        task_type = "single_target"
    
    # Create and run pipeline
    pipeline = pipeline_class(
        X=X,
        y=y,
        **kwargs
    )
    
    # Execute pipeline and get results
    results = pipeline.execute(type=type)
    
    return {
        'pipeline': pipeline,
        'results': results,
        'task_type': task_type,
        'type': type
    } 