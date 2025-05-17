
"""
Single‐target regression pipeline.
"""


import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import numpy as np

from .base import BasePipeline
from .config import REGRESSION_METRICS, REGRESSION_MODELS, DEFAULT_CV

class SingleOutputRegressionPipeline(BasePipeline):
    """
    Pipeline for single‐output regression tasks.

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
        self._validate_single_target(y)

        metric_funcs = REGRESSION_METRICS
        default_metrics = [metrics] if isinstance(metrics, str) else (metrics or ["r2"])

        base = REGRESSION_MODELS
        if models == "all":
            model_configs = base
        elif isinstance(models, str):
            model_configs = {models: base[models]}
        else:
            model_configs = {m: base[m] for m in models}

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

    def _validate_single_target(self, y):
        """Ensure target is 1D array."""
        if len(y.shape) != 1:
            raise ValueError(
                f"Target must be 1D array for single target regression. "
                f"Shape is {y.shape}"
            )