"""
Binary classification pipeline with specialized functionality.

This module extends the base classification pipeline with binary-specific
functionality, metrics, and optimizations.
"""

import logging
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    make_scorer
)
from .base_classification import BaseClassificationPipeline, CLASSIFICATION_METRICS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Add binary-specific metrics
BINARY_METRICS = {
    **CLASSIFICATION_METRICS,
    "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
    "average_precision": make_scorer(average_precision_score, needs_proba=True)
}

class BinaryClassificationPipeline(BaseClassificationPipeline):
    """
    Pipeline specifically for binary classification tasks.
    
    This class extends BaseClassificationPipeline with binary-specific:
    - Metrics (ROC AUC, Average Precision)
    - Model configurations optimized for binary tasks
    - Specialized evaluation methods
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Binary target vector (0/1 or -1/1)
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
        self._validate_binary_target()
        self._setup_binary_models()
    
    def _validate_binary_target(self):
        """Ensure target is binary."""
        unique_classes = np.unique(self.y)
        if len(unique_classes) != 2:
            raise ValueError(
                f"Target must be binary. Found {len(unique_classes)} classes: {unique_classes}"
            )
    
    def _validate_metrics(self):
        """Validate that all requested metrics are available."""
        invalid_metrics = [m for m in self.metrics if m not in BINARY_METRICS]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. "
                           f"Available metrics: {list(BINARY_METRICS.keys())}")
    
    def _setup_binary_models(self):
        """Add binary-specific model configurations."""
        # Modify existing models for binary classification
        if "Logistic Regression" in self.models:
            self.models["Logistic Regression"]["estimator"].set_params(
                class_weight='balanced'
            )
            self.models["Logistic Regression"]["params"]["class_weight"] = [
                'balanced', None
            ]
        
        if "Random Forest" in self.models:
            self.models["Random Forest"]["estimator"].set_params(
                class_weight='balanced'
            )
            self.models["Random Forest"]["params"]["class_weight"] = [
                'balanced', 'balanced_subsample', None
            ]
        
        if "SVC" in self.models:
            self.models["SVC"]["estimator"].set_params(
                class_weight='balanced'
            )
            self.models["SVC"]["params"]["class_weight"] = [
                'balanced', None
            ]
    def _compute_binary_metrics(self, results):
        """
        Compute binary-specific metrics.
        """
        metrics_binary = {}
        y = results['predictions']['y_true']
        y_proba = results['predictions']['y_proba']
        if 'roc_auc' in self.metrics:
            metrics_binary['roc_auc'] = roc_auc_score(y, y_proba[:, 1])
            logging.info(f"ROC AUC: {metrics_binary['roc_auc']:.4f}")
        if 'average_precision' in self.metrics:
            metrics_binary['average_precision'] = average_precision_score(y, y_proba[:, 1], pos_label=1)
            logging.info(f"Average Precision: {metrics_binary['average_precision']:.4f}")
        # add metrics to results
        results['metrics'].update(metrics_binary)
        return results
    
    def baseline(self, estimator=None, X=None, y=None):
        """
        Run baseline evaluation for binary classification.
        """
        results = super().baseline(estimator, X, y)
        # compute ROC AUC and average precision based on predictions if requested
        return self._compute_binary_metrics(results)

    def feature_selection(self, estimator=None, n_features=None, direction="forward", scoring=None):
        """
        Perform feature selection optimized for binary classification.
        
        Extends the base method to:
        - Use probability-based metrics if available
        - Balance classes during selection
        - Compute binary-specific metrics for selected features
        
        Parameters and returns are the same as the base method.
        """
        
        # Call base method with binary-specific scoring
        results = super().feature_selection(
            estimator=estimator,
            n_features=n_features,
            direction=direction,
            scoring=scoring, 
            cv=self.cv_strategy
        )
        return self._compute_binary_metrics(results)

    def hp_search(self, model_name, param_grid=None, search_type='grid', 
                 n_iter=100, scoring=None, cv=5):
        """
        Perform hyperparameter optimization for binary classification.
        
        Extends the base method to:
        - Use probability-based metrics if available
        - Include class balancing parameters
        - Compute binary-specific metrics for best parameters
        
        Parameters and returns are the same as the base method.
        """
        # Add class balancing parameters if not in grid
        if param_grid is None:
            param_grid = self.models[model_name]["params"]
            if "class_weight" not in param_grid:
                param_grid["class_weight"] = ['balanced', None]
        
        # Call base method with binary-specific setup
        results = super().hp_search(
            model_name=model_name,
            param_grid=param_grid,
            search_type=search_type,
            n_iter=n_iter,
            scoring=scoring,
            cv=self.cv_strategy
        )
        return self._compute_binary_metrics(results)
    
    def execute(self, type='baseline', **kwargs):
        """
        Run the pipeline.
        """
        self.results = []
        for model_name in self.models.keys():
            if type == 'feature_selection':
                results = self.feature_selection(model_name, **kwargs)
            elif type == 'hp_search':
                results = self.hp_search(model_name, **kwargs)
            elif type == 'baseline':
                results = self.baseline(model_name, **kwargs)
            self.results.append(results)
        return self.results