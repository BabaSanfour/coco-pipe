"""
Multivariate regression pipeline with specialized functionality.

This module extends the base regression pipeline with multi-target specific
functionality, metrics, and optimizations.
"""

import logging
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    make_scorer
)
from .base_regression import BaseRegressionPipeline, REGRESSION_METRICS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Add multivariate-specific metrics
def mean_r2_score(y_true, y_pred):
    """Compute mean R² score across all targets."""
    return np.mean([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])

def mean_mse(y_true, y_pred):
    """Compute mean MSE across all targets."""
    return np.mean([mean_squared_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])

def mean_mae(y_true, y_pred):
    """Compute mean MAE across all targets."""
    return np.mean([mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])

MULTIVARIATE_METRICS = {
    **REGRESSION_METRICS,
    "mean_r2": make_scorer(mean_r2_score),
    "mean_mse": make_scorer(mean_mse, greater_is_better=False),
    "mean_mae": make_scorer(mean_mae, greater_is_better=False),
}

class MultivariateRegressionPipeline(BaseRegressionPipeline):
    """
    Pipeline specifically for multivariate regression tasks.
    
    This class extends BaseRegressionPipeline with multi-target specific:
    - Metrics (mean R², mean MSE, mean MAE across targets)
    - Model configurations optimized for multiple targets
    - Per-target evaluation methods
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples, n_targets)
        Target matrix
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
            metrics = ['mean_r2', 'mean_mse']
        
        super().__init__(X, y, models, metrics, random_state, n_jobs)
        self._validate_multivariate_target()
        self._setup_multivariate_models()
        self.n_targets_ = y.shape[1]
    
    def _validate_multivariate_target(self):
        """Ensure target is 2D array."""
        if len(self.y.shape) != 2:
            raise ValueError(
                f"Target must be 2D array for multivariate regression. "
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
        # Wrap all estimators with MultiOutputRegressor
        for name in self.models:
            base_estimator = self.models[name]["estimator"]
            if not isinstance(base_estimator, MultiOutputRegressor):
                self.models[name]["estimator"] = MultiOutputRegressor(
                    base_estimator,
                    n_jobs=self.n_jobs
                )
    
    def _compute_multivariate_metrics(self, results):
        """
        Compute multivariate-specific metrics.
        """
        metrics_multi = {}
        y_true = results['predictions']['y_true']
        y_pred = results['predictions']['y_pred']
        
        # Compute mean metrics across targets
        if 'mean_r2' in self.metrics:
            metrics_multi['mean_r2'] = mean_r2_score(y_true, y_pred)
            logging.info(f"Mean R²: {metrics_multi['mean_r2']:.4f}")
            
            # Also compute per-target R²
            r2_scores = [r2_score(y_true[:, i], y_pred[:, i]) 
                        for i in range(self.n_targets_)]
            metrics_multi['r2_per_target'] = r2_scores
            for i, score in enumerate(r2_scores):
                logging.info(f"Target {i} R²: {score:.4f}")
        
        if 'mean_mse' in self.metrics:
            metrics_multi['mean_mse'] = mean_mse(y_true, y_pred)
            logging.info(f"Mean MSE: {metrics_multi['mean_mse']:.4f}")
            
            # Also compute per-target MSE
            mse_scores = [mean_squared_error(y_true[:, i], y_pred[:, i])
                         for i in range(self.n_targets_)]
            metrics_multi['mse_per_target'] = mse_scores
            for i, score in enumerate(mse_scores):
                logging.info(f"Target {i} MSE: {score:.4f}")
        
        if 'mean_mae' in self.metrics:
            metrics_multi['mean_mae'] = mean_mae(y_true, y_pred)
            logging.info(f"Mean MAE: {metrics_multi['mean_mae']:.4f}")
            
            # Also compute per-target MAE
            mae_scores = [mean_absolute_error(y_true[:, i], y_pred[:, i])
                         for i in range(self.n_targets_)]
            metrics_multi['mae_per_target'] = mae_scores
            for i, score in enumerate(mae_scores):
                logging.info(f"Target {i} MAE: {score:.4f}")
        
        # Add metrics to results
        results['metrics'].update(metrics_multi)
        return results
    
    def baseline(self, estimator=None, X=None, y=None):
        """
        Run baseline evaluation for multivariate regression.
        """
        results = super().baseline(estimator, X, y)
        return self._compute_multivariate_metrics(results)

    def feature_selection(self, model_name=None, n_features=None, direction="forward", scoring=None):
        """
        Perform feature selection optimized for multivariate regression.
        
        Extends the base method to:
        - Use multivariate metrics
        - Compute per-target metrics for selected features
        - Balance feature importance across targets
        
        Parameters and returns are the same as the base method.
        """
        results = super().feature_selection(
            model_name=model_name,
            n_features=n_features,
            direction=direction,
            scoring=scoring
        )
        return self._compute_multivariate_metrics(results)

    def hp_search(self, model_name, param_grid=None, search_type='grid', 
                 n_iter=100, scoring=None):
        """
        Perform hyperparameter optimization for multivariate regression.
        
        Extends the base method to:
        - Use multivariate metrics
        - Include multi-target specific parameters
        - Compute per-target metrics for best parameters
        
        Parameters and returns are the same as the base method.
        """
        results = super().hp_search(
            model_name=model_name,
            param_grid=param_grid,
            search_type=search_type,
            n_iter=n_iter,
            scoring=scoring
        )
        return self._compute_multivariate_metrics(results)
    
    def execute(self, type='baseline', **kwargs):
        """
        Run the pipeline.
        """
        return super().execute(type, **kwargs) 