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

import datetime
import logging
import pickle
import os
import json
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd
import numpy as np
from .base import BasePipeline
from .config import (
    REGRESSION_METRICS,
    REGRESSION_MODELS,
    DEFAULT_CV,
    MULTIOUTPUT_REG_METRICS,
    MULTIOUTPUT_REG_MODELS,
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
    X : np.ndarray
        Feature matrix, array-like of shape (n_samples, n_features).
    y : np.ndarray
        Target vector, array-like of shape (n_samples,).
    models : str or list, optional (default="all")
        Models to include in the pipeline. If "all", all available models are used.
    metrics : str or list, optional (default=None)
        Metrics to evaluate. If None, default metrics are used.
    use_scaler: bool = True,
        Whether to use a scaler (StandardScaler) in the pipeline.
        If True, a StandardScaler will be applied before the model.
    random_state : int, optional (default=42)
        Random state for reproducibility.
    n_jobs : int, optional (default=-1)
        Number of parallel jobs to run. -1 means using all processors.
    cv_kwargs : dict, optional (default=None)
        Keyword arguments for cross-validation, such as 'cv_strategy' and 'n_splits'.
    groups : np.ndarray, optional (default=None)
        Group labels for the samples used while splitting the dataset into train/test set.
    
    Raises
    ------
    ValueError
        If the target `y` is not a 1D array.
    
    Notes
    -----   
    This pipeline supports single-output regression tasks, allowing for model evaluation,
    hyperparameter tuning, and feature selection. It uses predefined regression models and metrics.
    The models and metrics can be specified or defaulted to all available options.
    The pipeline can handle cross-validation and supports parallel
    execution with the `n_jobs` parameter.
    The `cv_kwargs` parameter allows customization of the cross-validation strategy.
    The `groups` parameter can be used to specify group labels for grouped cross-validation.
    The pipeline is designed to be flexible and extensible, allowing for easy integration
    with other components of the coco_pipe framework.

    Examples
    --------
    >>> from coco_pipe.ml import SingleOutputRegressionPipeline
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = pd.DataFrame(np.random.rand(100, 10), columns=[f"feature_{i}" for i in range(10)])
    >>> y = pd.Series(np.random.rand(100))
    >>> pipeline = SingleOutputRegressionPipeline(X=X, y=y, models=["linear_regression", "random_forest"])
    >>> results = pipeline.execute()
    """
    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        models: Union[str, Sequence[str]] = "all",
        metrics: Union[str, Sequence[str]] = None,
        use_scaler: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
        cv_kwargs: Optional[Dict[str, Any]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        verbose: bool = False,
    ):
        self.verbose = verbose
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

        cv = dict(DEFAULT_CV)
        if cv_kwargs:
            cv.update(cv_kwargs)

        super().__init__(
            X=X,
            y=y,
            metric_funcs=metric_funcs,
            model_configs=model_configs,
            use_scaler=use_scaler,
            default_metrics=default_metrics,
            cv_kwargs=cv,
            groups=groups,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _validate_single_target(self, y):
        """Ensure target is 1D array."""
        if not hasattr(y, 'ndim') or y.ndim != 1:
            raise ValueError(
                f"Target must be 1D array for single-output regression. Shape is {getattr(y,'shape',None)}"
            )


class MultiOutputRegressionPipeline(BasePipeline):
    """
    Pipeline for multi‐output regression tasks.
    
    Parameters
    ----------

    X : np.ndarray
        Feature matrix, array-like of shape (n_samples, n_features).
    y : np.ndarray
        Target matrix, array-like of shape (n_samples, n_targets).
    models : str or list, optional (default="all")
        Models to include in the pipeline. If "all", all available models are used.
    metrics : str or list, optional (default=None)
        Metrics to evaluate. If None, default metrics are used.
    use_scaler: bool = True,
        Whether to use a scaler (StandardScaler) in the pipeline.
        If True, a StandardScaler will be applied before the model.
    random_state : int, optional (default=42)
        Random state for reproducibility.
    n_jobs : int, optional (default=-1)
        Number of parallel jobs to run. -1 means using all processors.
    cv_kwargs : dict, optional (default=None)
        Keyword arguments for cross-validation, such as 'cv_strategy' and 'n_splits'.
    groups : np.ndarray, optional (default=None)
        Group labels for the samples used while splitting the dataset into train/test set.  

    Raises
    ------
    ValueError
        If the target `y` is not a 2D array.

    Notes
    -----
    This pipeline supports multi-output regression tasks, allowing for model evaluation,
    hyperparameter tuning, and feature selection. It uses predefined multi-output regression
    models and metrics. The models and metrics can be specified or defaulted to all available options.
    The pipeline can handle cross-validation and supports parallel execution with the `n_jobs` parameter.
    The `cv_kwargs` parameter allows customization of the cross-validation strategy.
    The `groups` parameter can be used to specify group labels for grouped cross-validation.
    The pipeline is designed to be flexible and extensible, allowing for easy integration
    with other components of the coco_pipe framework.

    Examples
    --------
    >>> from coco_pipe.ml import MultiOutputRegressionPipeline
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = pd.DataFrame(np.random.rand(100, 10), columns=[f"feature_{i}" for i in range(10)])
    >>> y = pd.DataFrame(np.random.rand(100, 3), columns=[f"target_{i}" for i in range(3)])
    >>> pipeline = MultiOutputRegressionPipeline(X=X, y=y, models=["linear_regression", "random_forest"])
    >>> results = pipeline.execute()
    """
    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        models: Union[str, Sequence[str]] = "all",
        metrics: Union[str, Sequence[str]] = None,
        use_scaler: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
        cv_kwargs: Optional[Dict[str, Any]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self._validate_multioutput_target(y)

        metric_funcs = MULTIOUTPUT_REG_METRICS
        default_metrics = [metrics] if isinstance(metrics, str) else (metrics or ["mean_r2"])

        base = MULTIOUTPUT_REG_MODELS.copy()
        if models == "all":
            model_configs = base.copy()
        elif isinstance(models, str):
            model_configs = {models: base[models].copy()}
        else:
            model_configs = {m: base[m].copy() for m in models}

        cv = dict(DEFAULT_CV)
        if cv_kwargs:
            cv.update(cv_kwargs)

        super().__init__(
            X=X,
            y=y,
            metric_funcs=metric_funcs,
            model_configs=model_configs,
            use_scaler=use_scaler,
            default_metrics=default_metrics,
            cv_kwargs=cv,
            groups=groups,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _validate_multioutput_target(self, y):
        """Ensure target is 2D array."""
        if not hasattr(y, 'ndim') or y.ndim != 2:
            raise ValueError(
                f"Target must be 2D array for multi-output regression. Shape is {getattr(y,'shape',None)}"
            )


class RegressionPipeline:
    """
    Wrapper that selects and runs the appropriate regression pipeline.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix, array-like of shape (n_samples, n_features).
    y : pd.Series or np.ndarray
        Target vector or matrix, array-like of shape (n_samples,) for single-output
        or (n_samples, n_targets) for multi-output regression.
    analysis_type : str, optional (default="baseline")
        Type of analysis to perform. Options are "baseline", "feature_selection",
        "hp_search", or "hp_search_fs".
    models : str or list, optional (default="all")
        Models to include in the pipeline. If "all", all available models are used.
    metrics : str or list, optional (default=None)
        Metrics to evaluate. If None, default metrics are used.
    use_scaler: bool = True,
        Whether to use a scaler (StandardScaler) in the pipeline.
        If True, a StandardScaler will be applied before the model.
    random_state : int, optional (default=42)
        Random state for reproducibility.
    cv_strategy : str, optional (default="kfold")
        Cross-validation strategy to use. Options include "kfold", "stratified_kfold",
        "group_kfold", etc.
    n_splits : int, optional (default=5)
        Number of splits for cross-validation.
    n_features : int, optional (default=None)
        Number of features to select in feature selection analysis. If None, all features are used.
    direction : str, optional (default="forward")
        Direction for feature selection. Options are "forward", "backward", or "both".  
    search_type : str, optional (default="grid")
        Search type for hyperparameter tuning. Options are "grid" or "random".
    n_iter : int, optional (default=100)
        Number of iterations for random search in hyperparameter tuning.
    scoring : str, optional (default=None)
        Scoring metric for hyperparameter tuning. If None, default scoring is used.
    n_jobs : int, optional (default=-1)
        Number of parallel jobs to run. -1 means using all processors.
    save_intermediate : bool, optional (default=False)
        Whether to save intermediate results during the analysis.
    results_dir : str, optional (default="results")
        Directory to save results. If it does not exist, it will be created.
    results_file : str, optional (default="results")
        Base name for the results file. The full filename will include the task type and analysis type.
    cv_kwargs : dict, optional (default=None)
        Additional keyword arguments for cross-validation, such as 'groups' for grouped CV.

    Raises
    ------
    ValueError
        If the `analysis_type` is not one of the supported types.
        If the target `y` is not a 1D or 2D array depending on the task type.

    Notes
    -----
    This wrapper class is designed to handle both single-output and multi-output regression tasks.
    It automatically selects the appropriate pipeline class based on the shape of the target variable `y`.
    The `analysis_type` parameter determines the type of analysis to perform, such as baseline evaluation,
    feature selection, or hyperparameter search. The results are saved in the specified directory,
    and metadata about the analysis is stored in a JSON file alongside the results. 

    Examples
    --------
    >>> from coco_pipe.ml import RegressionPipeline
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = pd.DataFrame(np.random.rand(100, 10), columns=[f"feature_{i}" for i in range(10)])
    >>> y = pd.Series(np.random.rand(100))  # Single-output regression
    >>> pipeline = RegressionPipeline(X=X, y=y, analysis_type="baseline", models=["linear_regression", "random_forest"])
    >>> results = pipeline.run()
    """
    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        analysis_type: str = "baseline",
        models: Union[str, Sequence[str]] = "all",
        metrics: Union[str, Sequence[str]] = None,
        use_scaler: bool = True,
        random_state: int = 42,
        cv_strategy: str = "kfold",
        n_splits: int = 5,
        n_features: Optional[int] = None,
        direction: str = "forward",
        search_type: str = "grid",
        n_iter: int = 100,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
        save_intermediate: bool = False,
        results_dir: str = "results",
        results_file: str = "results",
        cv_kwargs: Optional[Dict[str, Any]] = None,
        model_configs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        self.X = X
        self.y = y
        self.verbose = verbose
        analysis_type = analysis_type.lower()
        if analysis_type not in ["baseline", "feature_selection", "hp_search", "hp_search_fs"]:
            raise ValueError(f"Invalid analysis type: {analysis_type}")
        self.analysis_type = analysis_type
        self.models = models
        self.metrics = metrics
        self.use_scaler = use_scaler
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
        self.results = {}
        self.groups = groups
        self.new_model_configs = model_configs
        if self.new_model_configs is not None:
            self.update_configs = True

        # pick pipeline class based on target dimension
        if hasattr(self.y, 'ndim') and self.y.ndim == 2:
            PipelineClass = MultiOutputRegressionPipeline
            self.task = "multioutput"
        else:
            PipelineClass = SingleOutputRegressionPipeline
            self.task = "singleoutput"

        os.makedirs(self.results_dir, exist_ok=True)

        cvk = dict(DEFAULT_CV)
        if cv_kwargs:
            cvk.update(cv_kwargs)
        cvk.update({
            "cv_strategy": cv_strategy,
            "n_splits": n_splits,
            "random_state": random_state,
        })

        self.pipeline = PipelineClass(
            X=self.X,
            y=self.y,
            groups=self.groups,
            models=self.models,
            metrics=self.metrics,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            cv_kwargs=cvk,
            verbose=self.verbose,
        )

    def save(self, name: str, res: Dict[str, Any]):
        filepath = os.path.join(self.results_dir, f"{name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(res, f)
        logger.info(f"Saved results to {filepath}")

    def run(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        
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
            "use_scaler": self.use_scaler,
            "random_state": self.random_state,
            "cv_strategy": self.cv_strategy,
            "n_splits": self.n_splits,
            "n_features": self.n_features,
            "direction": self.direction,
            "search_type": self.search_type,
            "n_iter": self.n_iter,
            "n_jobs": self.n_jobs,
            "X_shape": getattr(self.X, 'shape', None),
            "y_shape": getattr(self.y, 'shape', (len(self.y),)),
            "start_time": datetime.datetime.now().isoformat(),
            "completed_models": [],
            "failed_models": [],
            "status": "running",
        }
                # update model configs using
        if self.update_configs:
            logger.info("Updating model configurations with provided configs.")
            for model_name, (default_config, params) in self.new_model_configs.items():
                self.pipeline.update_model_config(
                    model_name,
                    default_params=default_config,
                    params=params,
                )

        if self.verbose:
            logger.info("Starting regression analysis with the following configuration:")
            logger.info(metadata)

        for name in self.pipeline.model_configs:
            try:
                if self.analysis_type == "baseline":
                    res = self.pipeline.baseline_evaluation(name)
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
                    raise ValueError(f"Unknown analysis type: {self.analysis_type}")

                if self.save_intermediate:
                    self.save(f"{name}_{base_name}", res)

                results[name] = res
                metadata["completed_models"].append(name)
                if self.verbose:
                    logger.info(f"Completed analysis for model: {name}")
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

        # save final results and metadata
        self.save(base_name, results)
        meta_path = os.path.join(self.results_dir, f"{base_name}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {meta_path}")

        return results
