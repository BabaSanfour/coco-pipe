"""
Multiclass classification.

This module extends the base classification pipeline with multiclass-specific
functionality, metrics, and optimizations.
"""
import logging
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

from .base import BasePipeline
from .config import CLASSIFICATION_METRICS, DEFAULT_CV, MULTICLASS_MODELS

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def multiclass_roc_auc_score(y_true, y_proba):
    """Compute OVR ROC AUC for multiclass."""
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)
    # if binary-probabilities, pad to 2 columns
    if y_proba.shape[1] == 2:
        return roc_auc_score(y_true_bin, y_proba[:,1])
    return roc_auc_score(y_true_bin, y_proba, average='weighted', multi_class='ovr')


class MultiClassPipeline(BasePipeline):
    """
    Pipeline specifically for multiclass classification tasks.
        
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
        per_class=False,
        random_state=42,
        n_jobs=-1,
        cv_kwargs=None,
        groups=None,
    ):
        self._validate_multiclass_target(y)

        metric_funcs = {
            **CLASSIFICATION_METRICS,
            'roc_auc_ovr': multiclass_roc_auc_score
        }
        default_metrics = [metrics] if isinstance(metrics, str) else (metrics or ["accuracy"])

        super().__init__(
            X=X, 
            y=y, 
            model_configs=models, 
            metric_funcs=metric_funcs, 
            random_state=random_state, 
            n_jobs=n_jobs,
            cv_kwargs=cv_kwargs,
            groups=groups)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.per_class = per_class
        
        base = MULTICLASS_MODELS
        if models == 'all':
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
        self.per_class = per_class
        self.classes_ = np.unique(y)

    def _validate_multiclass_target(self, y):
        """Ensure target is multiclass."""
        unique_classes = np.unique(y)
        if len(unique_classes) <= 2:
            raise ValueError(
                f"Target must have more than 2 classes. Found {len(unique_classes)} classes: {unique_classes}"
            )

    def compute_metrics(self, fold_preds, metrics, funcs):
        # base metrics
        results = super().compute_metrics(fold_preds, [m for m in metrics if m != "roc_auc_ovr"], funcs)
        y_true = results['predictions']['y_true']
        y_proba = results['predictions']['y_proba']
        # multiclass ROC AUC OVR
        if 'roc_auc_ovr' in self.metrics:
            score = multiclass_roc_auc_score(y_true, y_proba)
            results['metrics']['roc_auc_ovr'] = {'mean': score, 'std': 0.0, 'scores': [score]}
            logger.info(f"ROC AUC OVR: {score:.4f}")
        # per-class precision/recall/f1
        if self.per_class:
            y_pred = results['predictions']['y_pred']
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=self.classes_, zero_division=0
            )
            pcm = {}
            for cls, p, r, f in zip(self.classes_, prec, rec, f1):
                pcm[int(cls)] = {'precision': p, 'recall': r, 'f1': f}
            results['per_class_metrics'] = pcm
        return results