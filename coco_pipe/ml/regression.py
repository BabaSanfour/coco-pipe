# coco_pipe/ml/regression_pipeline.py
"""
Wrapper for regression pipelines supporting single-target and multi-output tasks.
"""
import logging
import pickle
import numpy as np
from sklearn.utils.multiclass import type_of_target

from .base import BasePipeline
from .config import REGRESSION_METRICS, REGRESSION_MODELS, DEFAULT_CV, MULTIOUTPUT_MODELS_REGRESSION, MULTIOUTPUT_METRICS_REGRESSION
from sklearn.multioutput import MultiOutputRegressor

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
    

class RegressionPipeline:
    """
    Wrapper that selects and runs the appropriate regression pipeline.
    """
    def __init__(
        self,
        X,
        y,
        type="baseline",
        models="all",
        metrics=None,
        random_state=42,
        cv_strategy="stratified",
        n_splits=5,
        n_features=None,
        direction="forward",
        search_type="grid",
        n_iter=100,
        scoring=None,
        n_jobs=-1,
        save_intermediate=False,
        results_dir="results",
        results_file="results",
    ):
        self.X = X
        self.y = y
        self.type = type
        self.models = models
        self.metrics = metrics
        self.random_state = random_state
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.n_features = n_features
        self.direction = direction
        self.search_type = search_type
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.save_intermediate = save_intermediate
        self.results_dir = results_dir
        self.results_file = results_file

        self.pipeline = None
        self.results = {}

    def save(self, name, res):
        """
        Save intermediate results to a pickle file.
        """
        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(res, f)
        logger.info(f"Saved results to {name}.pkl")

    def run(self):
        """
        Detect regression task, instantiate pipeline, run operation across all models, 
        save if requested, and return results dict.
        """
        # choose pipeline
        if hasattr(self.y, "ndim") and self.y.ndim == 2:
            PipelineClass = MultiOutputRegressionPipeline
            task = "multioutput"
            logger.info("Detected multi-output regression task")
        else:
            PipelineClass = SingleOutputRegressionPipeline
            task = "singleoutput"
            logger.info("Detected single-output regression task")

        # prepare cv_kwargs
        cv_kwargs = dict(DEFAULT_CV)
        cv_kwargs["strategy"] = self.cv_strategy
        cv_kwargs["random_state"] = self.random_state
        cv_kwargs["n_splits"] = self.n_splits
        # instantiate
        self.pipeline = PipelineClass(
            X=self.X,
            y=self.y,
            models=self.models,
            metrics=self.metrics,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            cv_kwargs=cv_kwargs,
        )

        base_name = f"{self.results_file}_{task}_{self.type}_rs{self.random_state}"
        if self.type == "feature_selection":
            base_name += f"_nfeat{self.n_features}_dir{self.direction}"
        if self.type == "hp_search":
            base_name += f"_niter{self.n_iter}_search{self.search_type}"

        # run for each model
        for name in self.pipeline.model_configs:
            if self.type == "baseline":
                res = self.pipeline.baseline(name)
            elif self.type == "feature_selection":
                res = self.pipeline.feature_selection(
                    name,
                    n_features=self.n_features,
                    direction=self.direction,
                    scoring=self.scoring,
                )
            elif self.type == "hp_search":
                res = self.pipeline.hp_search(
                    name,
                    param_grid=None,
                    search_type=self.search_type,
                    n_iter=self.n_iter,
                    scoring=self.scoring,
                )
            elif self.type == "hp_search_fs":
                res = self.pipeline.hp_search_fs(
                    name,
                    param_grid=None,
                    search_type=self.search_type,
                    n_features=self.n_features,
                    direction=self.direction,
                    n_iter=self.n_iter,
                    scoring=self.scoring,
                )
            else:
                raise ValueError(f"Unknown pipeline type: {self.type}")

            if self.save_intermediate:
                self.save(f"{name}_{base_name}", res)

            self.results[name] = res

        # final save
        self.save(base_name, self.results)
        return self.results
