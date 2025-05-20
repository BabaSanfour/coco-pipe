"""
coco_pipe/ml/regression.py
----------------
Wrapper for regression pipelines supporting single-target and multi-output tasks.

Author: Hamza Abdelhedi <hamza.abdelhedii@gmail.com>
Date: 2025-05-18
Version: 0.0.1
License: TBD
"""
import logging
import pickle
import datetime
import json
import os

from .base import BasePipeline
from .config import (
    REGRESSION_METRICS,
    REGRESSION_MODELS,
    DEFAULT_CV,
    MULTIOUTPUT_MODELS_REGRESSION,
    MULTIOUTPUT_METRICS_REGRESSION,
)

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
        :X: np.ndarray, Feature matrix, array-like of shape (n_samples, n_features)
        :y: np.ndarray, Target vector, array-like of shape (n_samples,)
        :models: str or list, optional (default="all"), Models to include in the pipeline
        :metrics: str or list, optional (default=None), Metrics to evaluate
        :random_state: int, optional (default=42), Random state for reproducibility
        :n_jobs: int, optional (default=-1), Number of parallel jobs
        :cv_kwargs: dict, optional (default=None), Keyword arguments for cross-validation
        :groups: np.ndarray, array-like of shape (n_samples,), optional (default=None), 
            Group labels for the samples used while splitting the dataset into train/test set
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
        self._validate_single_target(y)

        metric_funcs = REGRESSION_METRICS
        default_metrics = [metrics] if isinstance(metrics, str) else (metrics or ["r2"])

        base = REGRESSION_MODELS.copy()
        if models == "all":
            model_configs = base.copy()
        elif isinstance(models, str):
            model_configs = {models: base[models].copy()}
        else:
            model_configs = {m: base[m].copy() for m in models}

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
        if len(getattr(y, 'shape', ())) != 1:
            raise ValueError(
                f"Target must be 1D array for single target regression. Shape is {getattr(y, 'shape', None)}"
            )


class MultiOutputRegressionPipeline(BasePipeline):
    """
    Pipeline for multi‐output regression tasks.
    
    Parameters
    ----------
        :X: np.ndarray, Feature matrix, array-like of shape (n_samples, n_features)
        :y: np.ndarray, Target matrix, array-like of shape (n_samples, n_targets)
        :models: str or list, optional (default="all"), Models to include in the pipeline
        :metrics: str or list, optional (default=None), Metrics to evaluate
        :random_state: int, optional (default=42), Random state for reproducibility
        :n_jobs: int, optional (default=-1), Number of parallel jobs
        :cv_kwargs: dict, optional (default=None), Keyword arguments for cross-validation
        :groups: np.ndarray, array-like of shape (n_samples,), optional (default=None), 
            Group labels for the samples used while splitting the dataset into train/test set
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
        self._validate_multioutput_target(y)

        base = MULTIOUTPUT_MODELS_REGRESSION.copy()
        if models == "all":
            model_configs = base.copy()
        elif isinstance(models, str):
            model_configs = {models: base[models].copy()}
        else:
            model_configs = {m: base[m].copy() for m in models}

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
        if len(getattr(y, 'shape', ())) != 2:
            raise ValueError(
                f"Target must be 2D array for multivariate regression. Shape is {getattr(y, 'shape', None)}"
            )


class RegressionPipeline:
    """
    Wrapper that selects and runs the appropriate regression pipeline.

    Parameters
    ----------
        :X: np.ndarray, Feature matrix, array-like of shape (n_samples, n_features)
        :y: np.ndarray, Target matrix, array-like of shape (n_samples, n_targets)
        :type: str, Type of analysis to perform, one of ["baseline", "feature_selection", "hp_search", "hp_search_fs"]
        :models: str or list, optional (default="all"), Models to include in the pipeline
        :metrics: str or list, optional (default=None), Metrics to evaluate
        :random_state: int, optional (default=42), Random state for reproducibility
        :cv_strategy: str, optional (default="stratified"), Cross-validation strategy
        :n_splits: int, optional (default=5), Number of cross-validation splits
        :n_features: int, optional (default=None), Number of features to select
        :direction: str, optional (default="forward"), Direction for feature selection
        :search_type: str, optional (default="grid"), Type of hyperparameter search
        :n_iter: int, optional (default=100), Number of iterations for hyperparameter search
        :scoring: str, optional (default=None), Scoring metric for model selection
        :n_jobs: int, optional (default=-1), Number of parallel jobs
        :save_intermediate: bool, optional (default=False), Whether to save intermediate results
        :results_dir: str, optional (default="results"), Directory to save results
        :results_file: str, optional (default="results"), Base filename for results

    The pipeline can:
    - Run baseline model evaluation with cross-validation
    - Perform automated feature selection
    - Conduct hyperparameter tuning
    - Support multiple evaluation metrics
    - Save intermediate results during training

    Returns:
        :results: Dictionary containing model performances, predictions, and evaluation metrics
    """
    def __init__(
        self,
        X,
        y,
        analysis_type="baseline",
        models="all",
        metrics=None,
        random_state=42,
        cv_strategy="kfold",
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
        cv_kwargs=None,
    ):
        self.X = X
        self.y = y
        analysis_type = analysis_type.lower()
        if analysis_type not in ["baseline", "feature_selection", "hp_search", "hp_search_fs"]:
            raise ValueError(f"Invalid analysis type: {analysis_type}")
        self.analysis_type = analysis_type
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
        self.cv_kwargs = cv_kwargs
        self.pipeline = None
        
        if hasattr(self.y, "ndim") and self.y.ndim == 2:
            PipelineClass = MultiOutputRegressionPipeline
            self.task = "multioutput"
            logger.info("Detected multi-output regression task")
        else:
            PipelineClass = SingleOutputRegressionPipeline
            self.task = "singleoutput"
            logger.info("Detected single-output regression task")

        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

        cv_kwargs = dict(DEFAULT_CV)
        if self.cv_kwargs is not None:
            cv_kwargs.update(self.cv_kwargs)
        cv_kwargs["cv_strategy"] = self.cv_strategy
        cv_kwargs["random_state"] = self.random_state
        cv_kwargs["n_splits"] = self.n_splits

        self.pipeline = PipelineClass(
            X=self.X,
            y=self.y,
            models=self.models,
            metrics=self.metrics,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            cv_kwargs=cv_kwargs,
        )

    def save(self, name, res):
        """
        Save intermediate or final results as a pickle.
        """
        filepath = os.path.join(self.results_dir, f"{name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(res, f)
        logger.info(f"Saved results to {filepath}")

    def run(self):
        """
        run across models, save and return results.
        """
        results = {}

        base_name = f"{self.results_file}_{self.task}_{self.analysis_type}_rs{self.random_state}"
        if self.analysis_type == "feature_selection":
            base_name += f"_nfeat{self.n_features}_dir{self.direction}"
        if self.analysis_type == "hp_search":
            base_name += f"_niter{self.n_iter}_search{self.search_type}"
        if self.analysis_type == "hp_search_fs":
            base_name += (
                f"_nfeat{self.n_features}_dir{self.direction}"
                f"_niter{self.n_iter}_search{self.search_type}"
            )

        metadata = {
            "task": self.task,
            "analysis_type": self.analysis_type,
            "models": self.models,
            "metrics": self.metrics,
            "random_state": self.random_state,
            "cv_strategy": self.cv_strategy,
            "n_splits": self.n_splits,
            "n_features": self.n_features,
            "direction": self.direction,
            "search_type": self.search_type,
            "n_iter": self.n_iter,
            "n_jobs": self.n_jobs,
            "X_shape": self.X.shape,
            "y_shape": getattr(self.y, 'shape', (len(self.y),)),
            "start_time": datetime.datetime.now().isoformat(),
            "completed_models": [],
            "failed_models": [],
            "status": "running",
        }

        for name in self.pipeline.model_configs:
            try:
                if self.analysis_type == "baseline":
                    res = self.pipeline.baseline(name)
                elif self.analysis_type == "feature_selection":
                    res = self.pipeline.feature_selection(
                        name,
                        n_features=self.n_features,
                        direction=self.direction,
                        scoring=self.scoring,
                    )
                elif self.analysis_type == "hp_search":
                    res = self.pipeline.hp_search(
                        name,
                        param_grid=None,
                        search_type=self.search_type,
                        n_iter=self.n_iter,
                        scoring=self.scoring,
                    )
                elif self.analysis_type == "hp_search_fs":
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
                    raise ValueError(f"Unknown pipeline type: {self.analysis_type}")

                if self.save_intermediate:
                    self.save(f"{name}_{base_name}", res)

                results[name] = res
                metadata["completed_models"].append(name)
            except Exception as e:
                logger.error(f"Failed to run {name}: {e}")
                metadata["failed_models"].append({"model": name, "error": str(e)})

        metadata["end_time"] = datetime.datetime.now().isoformat()
        metadata["status"] = (
            "completed" if not metadata["failed_models"] else "partial_failure"
        )
        metadata["total_models"] = len(self.pipeline.model_configs)
        metadata["successful_models"] = len(metadata["completed_models"])
        metadata["failed_models_count"] = len(metadata["failed_models"])

        self.save(base_name, results)
        with open(os.path.join(self.results_dir, f"{base_name}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {os.path.join(self.results_dir, f'{base_name}_metadata.json')}" )

        return results
