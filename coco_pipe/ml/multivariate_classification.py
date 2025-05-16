"""
Multivariate classification pipeline with specialized functionality.

This module extends the base classification pipeline with multivariate-specific
functionality for handling multiple target variables.
"""

import logging
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    make_scorer, accuracy_score, f1_score
)
from .base_classification import BaseClassificationPipeline, CLASSIFICATION_METRICS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Add multivariate-specific metrics (aggregated across targets)
MULTIVARIATE_METRICS = {
    **CLASSIFICATION_METRICS,
    "mean_accuracy": make_scorer(lambda y_true, y_pred: np.mean([
        accuracy_score(y_true[:, i], y_pred[:, i])
        for i in range(y_true.shape[1])
    ])),
    "mean_f1": make_scorer(lambda y_true, y_pred: np.mean([
        f1_score(y_true[:, i], y_pred[:, i], average='weighted')
        for i in range(y_true.shape[1])
    ])),
}

class MultivariateClassificationPipeline(BaseClassificationPipeline):
    """
    Pipeline specifically for multivariate classification tasks.
    
    This class extends BaseClassificationPipeline with multivariate-specific:
    - Metrics (mean accuracy, mean F1 across targets)
    - Model configurations optimized for multiple targets
    - Per-target evaluation methods
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples, n_targets)
        Target matrix where each column is a classification target
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
        # Initialize with base metrics if none provided
        if metrics is None:
            metrics = ['mean_accuracy', 'mean_f1']
        
        super().__init__(X, y, models, metrics, random_state, n_jobs)
        self._validate_multivariate_target()
        self._setup_multivariate_models()
        self.n_targets_ = y.shape[1]
    
    def _validate_multivariate_target(self):
        """Ensure target is 2D array."""
        if len(self.y.shape) != 2:
            raise ValueError(
                f"Target must be 2D array for multivariate classification. "
                f"Shape is {self.y.shape}"
            )
    
    def _validate_metrics(self):
        """Validate that all requested metrics are available."""
        invalid_metrics = [m for m in self.metrics if m not in MULTIVARIATE_METRICS]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. "
                           f"Available metrics: {list(MULTIVARIATE_METRICS.keys())}")
    
    def _setup_multivariate_models(self):
        """Add multivariate-specific model configurations."""
        # Wrap all estimators with MultiOutputClassifier
        for name in self.models:
            base_estimator = self.models[name]["estimator"]
            if not isinstance(base_estimator, MultiOutputClassifier):
                self.models[name]["estimator"] = MultiOutputClassifier(
                    base_estimator,
                    n_jobs=self.n_jobs
                )
    
    def _compute_multivariate_metrics(self, results):
        """
        Compute multivariate-specific metrics.
        """
        metrics_multivariate = {}
        y_true = results['predictions']['y_true']
        y_pred = results['predictions']['y_pred']
        
        # Compute mean metrics across targets
        for metric in self.metrics:
            if metric == 'mean_accuracy':
                scores = [accuracy_score(y_true[:, i], y_pred[:, i])
                         for i in range(self.n_targets_)]
                metrics_multivariate['mean_accuracy'] = np.mean(scores)
                logging.info(f"Mean Accuracy: {metrics_multivariate['mean_accuracy']:.4f}")
            
            elif metric == 'mean_f1':
                scores = [f1_score(y_true[:, i], y_pred[:, i], average='weighted')
                         for i in range(self.n_targets_)]
                metrics_multivariate['mean_f1'] = np.mean(scores)
                logging.info(f"Mean F1: {metrics_multivariate['mean_f1']:.4f}")
        
        # Add metrics to results
        results['metrics'].update(metrics_multivariate)
        return results
    
    def baseline(self, estimator=None, X=None, y=None):
        """
        Run baseline evaluation for multivariate classification.
        """
        results = super().baseline(estimator, X, y)
        return self._compute_multivariate_metrics(results)
    
    def feature_selection(self, estimator=None, n_features=None, direction="forward", scoring=None):
        """
        Perform feature selection optimized for multivariate classification.
        
        Extends the base method to:
        - Use multivariate metrics (mean accuracy, mean F1)
        - Select features that work well for all targets
        - Compute per-target metrics for selected features
        
        Parameters and returns are the same as the base method.
        """
        results = super().feature_selection(
            estimator=estimator,
            n_features=n_features,
            direction=direction,
            scoring=scoring
        )
        return self._compute_multivariate_metrics(results)
    
    def hp_search(self, model_name, param_grid=None, search_type='grid', 
                 n_iter=100, scoring=None, cv=5):
        """
        Perform hyperparameter optimization for multivariate classification.
        
        Extends the base method to:
        - Use multivariate metrics (mean accuracy, mean F1)
        - Optimize parameters that work well for all targets
        - Compute per-target metrics for best parameters
        
        Parameters and returns are the same as the base method.
        """
        results = super().hp_search(
            model_name=model_name,
            param_grid=param_grid,
            search_type=search_type,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv
        )
        return self._compute_multivariate_metrics(results)
    
    def execute(self, type='baseline', **kwargs):
        """
        Execute the pipeline.
        """
        return super().execute(type, **kwargs) 