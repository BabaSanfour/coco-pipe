"""
Single target regression pipeline with specialized functionality.

This module extends the base regression pipeline with single-target specific
functionality, metrics, and optimizations.
"""

import logging
import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    make_scorer
)
from .base_regression import BaseRegressionPipeline, REGRESSION_METRICS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Add single-target specific metrics
SINGLE_TARGET_METRICS = {
    **REGRESSION_METRICS,
    "explained_variance": make_scorer(explained_variance_score),
    "max_error": make_scorer(max_error, greater_is_better=False),
}

class SingleTargetRegressionPipeline(BaseRegressionPipeline):
    """
    Pipeline specifically for single target regression tasks.
    
    This class extends BaseRegressionPipeline with single-target specific:
    - Metrics (explained variance, max error)
    - Model configurations optimized for single target
    - Specialized evaluation methods
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target vector
    models : str or list, optional (default="all")
        Models to include in the pipeline
    metrics : str or list, optional (default=None)
        Metrics to evaluate
    random_state : int, optional (default=42)
        Random state for reproducibility
    n_jobs : int, optional (default=-1)
        Number of parallel jobs
    """
    
    def __init__(self, X, y, models="all", metrics=None, random_state=42, n_jobs=-1):
        super().__init__(X, y, models, metrics, random_state, n_jobs)
        self._validate_single_target()
        self._setup_single_target_models()
    
    def _validate_single_target(self):
        """Ensure target is 1D array."""
        if len(self.y.shape) != 1:
            raise ValueError(
                f"Target must be 1D array for single target regression. "
                f"Shape is {self.y.shape}"
            )
    
    def _validate_metrics(self):
        """Validate that all requested metrics are available."""
        invalid_metrics = [m for m in self.metrics if m not in SINGLE_TARGET_METRICS]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. "
                           f"Available metrics: {list(SINGLE_TARGET_METRICS.keys())}")
    
    def _setup_single_target_models(self):
        """Add single-target specific model configurations."""
        # Add any single-target specific model configurations
        if "Random Forest" in self.models:
            self.models["Random Forest"]["params"].update({
                "min_impurity_decrease": [0.0, 0.1, 0.2],
            })
        
        if "Gradient Boosting" in self.models:
            self.models["Gradient Boosting"]["params"].update({
                "validation_fraction": [0.1, 0.2],
                "n_iter_no_change": [5, 10],
            })
    
    def _compute_single_target_metrics(self, results):
        """
        Compute single-target specific metrics.
        """
        metrics_single = {}
        y_true = results['predictions']['y_true']
        y_pred = results['predictions']['y_pred']
        
        # Compute additional metrics
        if 'explained_variance' in self.metrics:
            metrics_single['explained_variance'] = explained_variance_score(y_true, y_pred)
            logging.info(f"Explained Variance: {metrics_single['explained_variance']:.4f}")
        
        if 'max_error' in self.metrics:
            metrics_single['max_error'] = max_error(y_true, y_pred)
            logging.info(f"Max Error: {metrics_single['max_error']:.4f}")
        
        # Add metrics to results
        results['metrics'].update(metrics_single)
        return results
    
    def baseline(self, estimator=None, X=None, y=None):
        """
        Run baseline evaluation for single target regression.
        """
        results = super().baseline(estimator, X, y)
        return self._compute_single_target_metrics(results)

    def feature_selection(self, model_name=None, n_features=None, direction="forward", scoring=None):
        """
        Perform feature selection optimized for single target regression.
        
        Extends the base method to:
        - Use single target metrics
        - Compute additional metrics for selected features
        
        Parameters and returns are the same as the base method.
        """
        results = super().feature_selection(
            model_name=model_name,
            n_features=n_features,
            direction=direction,
            scoring=scoring
        )
        return self._compute_single_target_metrics(results)

    def hp_search(self, model_name, param_grid=None, search_type='grid', 
                 n_iter=100, scoring=None):
        """
        Perform hyperparameter optimization for single target regression.
        
        Extends the base method to:
        - Use single target metrics
        - Include single target specific parameters
        - Compute additional metrics for best parameters
        
        Parameters and returns are the same as the base method.
        """
        results = super().hp_search(
            model_name=model_name,
            param_grid=param_grid,
            search_type=search_type,
            n_iter=n_iter,
            scoring=scoring
        )
        return self._compute_single_target_metrics(results)
    
    def execute(self, type='baseline', **kwargs):
        """
        Run the pipeline.
        """
        return super().execute(type, **kwargs) 