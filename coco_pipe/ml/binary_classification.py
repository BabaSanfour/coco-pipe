"""
Binary classification.

This module extends the base pipeline with binary-specific
functionality, metrics, and optimizations.
"""

import logging
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    make_scorer
)
from .base import BasePipeline
from .config import BINARY_METRICS, BINARY_MODELS, DEFAULT_CV

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class BinaryClassificationPipeline(BasePipeline):
    """
    Pipeline specifically for binary classification tasks.
        
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
    cv_kwargs : dict, optional (default=None)
        Cross-validation parameters
    groups : array-like of shape (n_samples,), optional (default=None)
        Group labels for the samples used while splitting the dataset into
        train/test set.
    """
    
    def __init__(
        self,
        X,
        y,
        models="all",
        metrics=None,
        random_state=42,
        n_jobs=-1,
        cv_kwargs=None,
        groups=None,
    ):
        
        self._validate_binary_target(y)

        # Build metric_funcs and defaults
        metric_funcs = BINARY_METRICS
        default_metrics = [metrics] if isinstance(metrics, str) else (metrics or ["accuracy"])

        # Build model_configs
        base = BINARY_MODELS
        # filter models if requested
        if models == "all":
            model_configs = base
        elif isinstance(models, str):
            model_configs = {models: base[models]}
        else:
            model_configs = {m: base[m] for m in models}

        cv = cv_kwargs or DEFAULT_CV

        super().__init__(
            X=X,
            y=y,
            metric_funcs=metric_funcs,
            model_configs=model_configs,
            default_metrics=default_metrics,
            cv_kwargs=cv,
            groups=groups,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _validate_binary_target(self, y):
        """Ensure target is binary."""
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(
                f"Target must be binary. Found {len(unique_classes)} classes: {unique_classes}"
            )
    
    def compute_metrics(self, fold_preds, metrics, funcs):
        results = super().compute_metrics(fold_preds, metrics, funcs)
        y_true = results["predictions"]["y_true"]
        y_proba = results["predictions"]["y_proba"]
        extra = {}
        if "roc_auc" in self.metrics:
            extra["roc_auc"] = {"mean": roc_auc_score(y_true, y_proba[:,1]),
                                "std": 0.0}
            logging.info(f"ROC AUC: {extra['roc_auc']['mean']:.4f}")

        if "average_precision" in self.metrics:
            extra["average_precision"] = {"mean": average_precision_score(y_true, y_proba[:,1]),
                                          "std": 0.0}
            logging.info(f"Avg Precision: {extra['average_precision']['mean']:.4f}")
        results["metrics"].update(extra)
        return results    