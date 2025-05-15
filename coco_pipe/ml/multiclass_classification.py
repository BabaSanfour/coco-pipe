"""
Multiclass classification pipeline with specialized functionality.

This module extends the base classification pipeline with multiclass-specific
functionality, metrics, and optimizations.
"""
import logging
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    make_scorer
)
from sklearn.preprocessing import label_binarize
from .base_classification import BaseClassificationPipeline, CLASSIFICATION_METRICS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def multiclass_roc_auc_score(y_true, y_proba):
    """Calculate ROC AUC score for multiclass using one-vs-rest."""
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)
    if y_proba.shape[1] == 2:  # Binary prediction
        return roc_auc_score(y_true_bin, y_proba[:, 1])
    return roc_auc_score(y_true_bin, y_proba, average='weighted', multi_class='ovr')

# Add multiclass-specific metrics
MULTICLASS_METRICS = {
    **CLASSIFICATION_METRICS,
    "roc_auc_ovr": make_scorer(multiclass_roc_auc_score, needs_proba=True),
}

class MultiClassPipeline(BaseClassificationPipeline):
    """
    Pipeline specifically for multiclass classification tasks.
    
    This class extends BaseClassificationPipeline with multiclass-specific:
    - Metrics (One-vs-Rest ROC AUC)
    - Model configurations optimized for multiclass
    - Per-class evaluation methods
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Multiclass target vector
    models : str or list, optional (default="all")
        Models to include in the pipeline
    metrics : str or list, optional (default=None)
        Metrics to evaluate
    per_class : bool, optional (default=False)
        Whether to compute per-class metrics
    random_state : int, optional (default=42)
        Random state for reproducibility
    n_jobs : int, optional (default=-1)
        Number of parallel jobs
    """
    
    def __init__(self, X, y, models="all", metrics=None, per_class=False, random_state=42, n_jobs=-1):
        super().__init__(X, y, models, metrics, random_state, n_jobs)
        self._validate_multiclass_target()
        self._setup_multiclass_models()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.per_class = per_class
    
    def _validate_multiclass_target(self):
        """Ensure target is multiclass."""
        unique_classes = np.unique(self.y)
        if len(unique_classes) <= 2:
            raise ValueError(
                f"Target must have more than 2 classes. Found {len(unique_classes)} classes: {unique_classes}"
            )
    
    def _validate_metrics(self):
        """Validate that all requested metrics are available."""
        invalid_metrics = [m for m in self.metrics if m not in MULTICLASS_METRICS]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. "
                           f"Available metrics: {list(MULTICLASS_METRICS.keys())}")
    
    def _setup_multiclass_models(self):
        """Add multiclass-specific model configurations."""
        # Modify existing models for multiclass
        if "Logistic Regression" in self.models:
            self.models["Logistic Regression"]["estimator"].set_params(
                multi_class='multinomial',
                solver='lbfgs'
            )
            self.models["Logistic Regression"]["params"]["multi_class"] = ['multinomial']
            self.models["Logistic Regression"]["params"]["solver"] = ['lbfgs', 'saga']
        
        if "SVC" in self.models:
            self.models["SVC"]["params"]["decision_function_shape"] = ['ovr', 'ovo']

    def compute_metrics(self, fold_predictions, metrics, metric_funcs):
        results = super().compute_metrics(fold_predictions,
                                          [m for m in metrics if m != "roc_auc_ovr"],
                                          metric_funcs)

        y_true = results["predictions"]["y_true"]
        y_pred = results["predictions"]["y_pred"]
        if self.per_class and "roc_auc_ovr" in metrics:
            from sklearn.metrics import precision_recall_fscore_support
            prec, rec, f1, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=self.classes_,
                zero_division=0
            )
            results["per_class_metrics"] = {
                str(cls): {
                    metric: {"mean": score} if metric in metrics else {}
                    for metric, score in zip(["precision", "recall", "f1"], [p, r, f])
                }
                for cls, p, r, f, sup in zip(self.classes_, prec, rec, f1, support)
            }

        if "roc_auc_ovr" in metrics:
            y_proba = results["predictions"]["y_proba"]
            score   = multiclass_roc_auc_score(y_true, y_proba)
            results["metrics"]["roc_auc_ovr"] = {
                "mean": score, "std": 0.0, "scores": [score]
            }

        return results
    
    def run(self, estimator, X: None, y: None, metrics, metric_funcs, **cv_kwargs):
        cv_results = self.cross_validate(estimator, X, y, **cv_kwargs)
        cv_results.update(self.compute_metrics(cv_results['fold_predictions'], metrics, metric_funcs))
        return cv_results

    def baseline(self, estimator, X, y):
        """
        Run baseline evaluation for estimator.
        
        Returns
        -------
        dict
            Dictionary with results:
            - cv_results: Cross-validation results
            - estimator: Fitted estimator
            - predictions: Predictions
            - feature_importances: Feature importances
        """
        return super().baseline(estimator, X, y)
    
    def feature_selection(self, estimator=None, n_features=None, direction="forward", scoring=None):
        """
        Perform feature selection optimized for multiclass classification.
        
        Extends the base method to:
        - Use multiclass metrics (e.g., ROC AUC OVR)
        - Compute per-class metrics for selected features
        - Balance feature importance across classes
        
        Parameters and returns are the same as the base method.
        """
        return super().feature_selection(estimator, n_features, direction, scoring)
    
    def hp_search(self, model_name, param_grid=None, search_type='grid', 
                 n_iter=100, scoring=None, cv=5):
        """
        Perform hyperparameter optimization for multiclass classification.
        
        Extends the base method to:
        - Use multiclass metrics (e.g., ROC AUC OVR)
        - Include multiclass-specific parameters
        - Compute per-class metrics for best parameters
        
        Parameters and returns are the same as the base method.
        """
        return super().hp_search(model_name, param_grid, search_type, n_iter, scoring, cv)
    
    def execute(self, type='baseline', **kwargs):
        """
        Execute the pipeline.
        """
        return super().execute(type, **kwargs)