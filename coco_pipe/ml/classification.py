"""
Unified classification pipeline supporting binary, multiclass, and multivariate tasks.

This module provides a single, comprehensive classification function that detects
the appropriate pipeline based on the target variable(s) and runs the pipeline.

Key Features:
- Automatic task type detection
"""

import logging
import numpy as np
from typing import Union
import pandas as pd
from .binary_classification import BinaryClassificationPipeline
from .multiclass_classification import MultiClassPipeline
from .multivariate_classification import MultivariateClassificationPipeline
    
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def classification(
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        type: str = "baseline",
        models: Union[str, list] = "all",
        metrics: Union[str, list] = "accuracy",
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
    Automatically detect and run the appropriate classification pipeline.

    This function examines the target variable(s) and determines whether to run
    binary, multiclass, or multivariate classification. It then creates and
    executes the appropriate pipeline.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target vector
    models : str or list, optional (default="all")
        Models to include in the pipeline
    metrics : str or list, optional (default="accuracy")
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
        
    # Determine classification type
    if len(np.array(y).shape) == 2:
        pipeline_class = MultivariateClassificationPipeline
        task_type = "multivariate"
    else:
        n_classes = len(np.unique(y))
        if n_classes == 2:
            pipeline_class = BinaryClassificationPipeline
            task_type = "binary"
        else:
            pipeline_class = MultiClassPipeline
            task_type = "multiclass"
    
    # Create and run pipeline
    pipeline = pipeline_class(
        X=X,
        y=y,
        **kwargs
    )
    
    # Execute pipeline and get results
    results = pipeline.execute()
    
    return {
        'pipeline': pipeline,
        'results': results,
        'task_type': task_type,
        'type': type
    } 