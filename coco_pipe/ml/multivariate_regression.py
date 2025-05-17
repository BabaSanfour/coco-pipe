"""
Multi‐output regression.
"""

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from .base import BasePipeline
from .config import MULTIOUTPUT_METRICS_REGRESSION, MULTIOUTPUT_MODELS_REGRESSION, DEFAULT_CV

class MultiOutputRegressionPipeline(BasePipeline):
    """
    Pipeline for multi‐output regression tasks.
    
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
    cv_kwargs : dict, optional (default=None)
        Keyword arguments for cross-validation
    groups : array-like of shape (n_samples,), optional (default=None)
        Group labels for the samples used while splitting the dataset into train/test set
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models="all",
        metrics=None,
        random_state=42,
        n_jobs=-1,
        cv_kwargs=None,
        groups=None,
    ):
        self._validate_multioutput_target(y)
        model_configs = self._setup_multioutput_models(models, n_jobs)

        metric_funcs = MULTIOUTPUT_METRICS_REGRESSION
        default_metrics = [metrics] if isinstance(metrics, str) else (metrics or ["mean_r2"])

        cv = dict(DEFAULT_CV) if cv_kwargs is None else dict(cv_kwargs)

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

    def _validate_multioutput_target(self, y):
        """Ensure target is 2D array."""
        if len(y.shape) != 2:
            raise ValueError(
                f"Target must be 2D array for multivariate regression. "
                f"Shape is {y.shape}"
            )
    
    def _setup_multioutput_models(self, models, n_jobs):
        """Add multi‐output specific model configurations."""

        base = MULTIOUTPUT_MODELS_REGRESSION
        if models == "all":
            model_configs = base
        elif isinstance(models, str):
            model_configs = {models: base[models]}
        else:
            model_configs = {m: base[m] for m in models}

        for name, cfg in model_configs.items():
            cfg = cfg.copy()
            est = cfg["estimator"]
            if not isinstance(est, MultiOutputRegressor):
                cfg["estimator"] = MultiOutputRegressor(est, n_jobs=n_jobs)
            model_configs[name] = cfg
            
        return model_configs