"""
Multi-output (multi-label) classification.

This module wraps single-output estimators in MultiOutputClassifier,
performs cross-validation, and computes specified multi-output metrics.
"""
import logging
import numpy as np
from sklearn.multioutput import MultiOutputClassifier

from .base import BasePipeline
from .config import MULTIOUTPUT_METRICS, BINARY_MODELS, DEFAULT_CV

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class MultiOutputClassificationPipeline(BasePipeline):
    """
    Pipeline for multi-output classification.
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
        self._validate_multivariate_target(y)

        # Metric functions and defaults
        metric_funcs = MULTIOUTPUT_METRICS
        if isinstance(metrics, str):
            default_metrics = [metrics]
        else:
            default_metrics = metrics or list(MULTIOUTPUT_METRICS.keys())

        # Model configs selection
        model_configs = self._setup_multivariate_models(models, n_jobs)

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

    def _validate_multivariate_target(self, y):
        """Ensure target is multivariate."""
        if not (hasattr(y, "ndim") and y.ndim == 2):
            raise ValueError(f"Target must be 2D for multi-output; got shape {getattr(y, 'shape', None)}")

    def _setup_multivariate_models(self, models, n_jobs):
        """Setup multivariate models."""
        base = BINARY_MODELS
        if models == "all":
            model_configs = base
        elif isinstance(models, str):
            model_configs = {models: base[models]}
        else:
            model_configs = {m: base[m] for m in models}
        for name, cfg in model_configs.items():
            est = cfg["estimator"]
            if not isinstance(est, MultiOutputClassifier):
                cfg["estimator"] = MultiOutputClassifier(est, n_jobs=n_jobs)
        return model_configs
    
    def compute_metrics(self, fold_preds, metrics, funcs):
        """
        Compute metrics using BasePipeline, then add per-output breakdown if specified.
        """
        results = super().compute_metrics(fold_preds, metrics, funcs, multioutput=True)

        y_true = results["predictions"]["y_true"]
        y_pred = results["predictions"]["y_pred"]

        # Per-output precision/recall/f1 samples (if requested)
        from sklearn.metrics import precision_score, recall_score, f1_score
        per_output = {}
        for i in range(y_true.shape[1]):
            out = {}
            if "precision_samples" in self.metrics:
                out["precision"] = float(
                    precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
                )
            if "recall_samples" in self.metrics:
                out["recall"] = float(
                    recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
                )
            if "f1_samples" in self.metrics:
                out["f1"] = float(
                    f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
                )
            if out:
                per_output[i] = out
        if per_output:
            results["per_output_metrics"] = per_output

        return results
