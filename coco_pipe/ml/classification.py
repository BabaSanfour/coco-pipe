#!/usr/bin/env python3
"""
coco_pipe/ml/classification.py
----------------
Wrapper for classification pipelines supporting binary, multiclass, and multioutput tasks.

Author: Hamza Abdelhedi <hamza.abdelhedii@gmail.com>
Date: 2025-05-18
Version: 0.0.1
License: TBD
"""

import datetime
import logging
import pickle
import os
import json
import pandas as pd
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support
)
from sklearn.utils.multiclass import type_of_target

from .base import BasePipeline
from .config import (
    BINARY_METRICS, BINARY_MODELS, 
    MULTICLASS_METRICS, MULTICLASS_MODELS,
    MULTIOUTPUT_CLASS_METRICS, MULTIOUTPUT_CLASS_MODELS, 
    DEFAULT_CV
)



logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class BinaryClassificationPipeline(BasePipeline):
    """
    Binary classification pipeline.
        
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, array-like of shape (n_samples, n_features).
    y : np.ndarray
        Binary target vector (0/1 or -1/1), array-like of shape (n_samples,).
    models : str or list, optional
        Models to include. Can be "all", a model name, or a list of model names.
        Default is "all".
    metrics : str or list, optional
        Metrics to evaluate. If None, uses default metrics for binary classification.
        Default is None.
    random_state : int, optional
        Random state for reproducibility. Default is 42.
    n_jobs : int, optional
        Number of parallel jobs. Default is -1.
    cv_kwargs : dict, optional
        Cross-validation parameters. If None, uses `DEFAULT_CV`. Default is None.
    groups : np.ndarray, optional
        Group labels for samples, for group-based CV. Shape (n_samples,). Default is None.
    """
    
    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        models: Union[str, Sequence[str]] = "all",
        metrics: Union[str, Sequence[str]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        cv_kwargs: Optional[Dict[str, Any]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
    ):
        self._validate_target(y)

        metric_funcs =  BINARY_METRICS
        default_metrics = [metrics] if isinstance(metrics, str) else (metrics or ["accuracy"])

        base = BINARY_MODELS
        if models == "all":
            model_configs = base
        elif isinstance(models, str):
            model_configs = {models: base[models]}
        else:
            model_configs = {m: base[m] for m in models}

        cv = dict(DEFAULT_CV)
        if cv_kwargs:
            cv.update(cv_kwargs)

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

    def _validate_target(self, y):
        unique = np.unique(y)
        if unique.size != 2:
            raise ValueError(f"Target must be binary. Found {unique.size} classes: {unique}")


class MultiClassClassificationPipeline(BasePipeline):
    """
    Multiclass classification pipeline.
        
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, array-like of shape (n_samples, n_features).
    y : np.ndarray
        Multiclass target vector, array-like of shape (n_samples,).
    models : str or list, optional
        Models to include. Default is "all".
    metrics : str or list, optional
        Metrics to evaluate. Default is None.
    per_class : bool, optional
        Whether to compute per-class metrics (precision, recall, F1). Default is False.
    random_state : int, optional
        Random state for reproducibility. Default is 42.
    n_jobs : int, optional
        Number of parallel jobs. Default is -1.
    cv_kwargs : dict, optional
        Cross-validation parameters. Default is None.
    groups : np.ndarray, optional
        Group labels for samples. Default is None.
    """
    
    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        models: Union[str, Sequence[str]] = "all",
        metrics: Union[str, Sequence[str]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        cv_kwargs: Optional[Dict[str, Any]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        per_class: bool = False,
    ):
        metric_funcs = MULTICLASS_METRICS
        if isinstance(metrics, str):
            default_metrics = [metrics]
        else:
            default_metrics = metrics or list(MULTICLASS_METRICS.keys())

        base = MULTICLASS_MODELS.copy()
        if models == "all":
            model_configs = base.copy()
        elif isinstance(models, str):
            model_configs = {models: base[models].copy()}
        else:
            model_configs = {m: base[m].copy() for m in models}
        super().__init__(
            X=X,
            y=y,
            model_configs=model_configs,
            default_metrics=metrics,
            metric_funcs=metric_funcs,
            random_state=random_state,
            n_jobs=n_jobs,
            cv_kwargs=cv_kwargs,
            groups=groups,
        )
        self._validate_target(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.per_class = per_class

    def _validate_target(self, y):
        """Ensure target is multiclass."""
        classes = np.unique(y)
        if classes.size <= 2:
            raise ValueError(f"Multiclass target requires >2 classes, got: {classes}")


    def _aggregate(self, fold_preds):
        agg = super()._aggregate(fold_preds)
        # add multiclass ROC-AUC if requested
        if 'roc_auc' in self.metrics:
            from .config import multiclass_roc_auc_score
            proba = agg["predictions"].get("y_proba")
            if proba is not None:
                score = multiclass_roc_auc_score(
                    y_true=agg["predictions"]["y_true"],
                    y_proba=proba
                )
                agg["metrics"]["roc_auc"] = {"scores":[score], "mean":score, "std":0.0}
        # per-class precision/recall/f1
        if self.per_class:
            yt = agg["predictions"]["y_true"]
            yp = agg["predictions"]["y_pred"]
            prec, rec, f1, _ = precision_recall_fscore_support(
                yt, yp, labels=np.unique(yt), zero_division=0
            )
            pcm = {}
            for cls, p, r, f in zip(np.unique(yt), prec, rec, f1):
                pcm[int(cls)] = {"precision":float(p),"recall":float(r),"f1":float(f)}
            agg["per_class_metrics"] = pcm
        return agg


class MultiOutputClassificationPipeline(BasePipeline):
    """
    Pipeline for multi-output classification.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, array-like of shape (n_samples, n_features).
    y : np.ndarray
        Multi-output target matrix, array-like of shape (n_samples, n_outputs).
    models : str or list, optional
        Models to include. Can be "all", a model name, or a list of model names.
    metrics : str or list, optional
        Metrics to evaluate. If None, uses default metrics for multi-output classification.
    random_state : int, optional
        Random state for reproducibility. Default is 42.
    n_jobs : int, optional
        Number of parallel jobs. Default is -1.
    cv_kwargs : dict, optional
        Cross-validation parameters. If None, uses `DEFAULT_CV`. Default is None.
    groups : np.ndarray, optional
        Group labels for samples, for group-based CV. Shape (n_samples,). Default is None.
    """
    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        models: Union[str, Sequence[str]] = "all",
        metrics: Union[str, Sequence[str]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        cv_kwargs: Optional[Dict[str, Any]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
    ):

        metric_funcs = MULTIOUTPUT_CLASS_METRICS
        if isinstance(metrics, str):
            default_metrics = [metrics]
        else:
            default_metrics = metrics or list(MULTIOUTPUT_CLASS_METRICS.keys())

        base = MULTIOUTPUT_CLASS_MODELS.copy()
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
            default_metrics=default_metrics,
            cv_kwargs=cv,
            groups=groups,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _validate_target(self, y):
        """Ensure target is multioutput."""
        if not (hasattr(y, "ndim") and y.ndim == 2):
            raise ValueError(f"Target must be 2D for multi-output; got shape {getattr(y, 'shape', None)}")


    def _aggregate(self, fold_preds):
        agg = super()._aggregate(fold_preds)
        yt = agg["predictions"]["y_true"]
        yp = agg["predictions"]["y_pred"]
        # per-output metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        pom = {}
        for i in range(yt.shape[1]):
            out = {}
            if "precision_samples" in self.metrics:
                out["precision"] = float(precision_score(
                    yt[:, i], yp[:, i], zero_division=0))
            if "recall_samples" in self.metrics:
                out["recall"] = float(recall_score(
                    yt[:, i], yp[:, i], zero_division=0))
            if "f1_samples" in self.metrics:
                out["f1"] = float(f1_score(
                    yt[:, i], yp[:, i], zero_division=0))
            if out:
                pom[i] = out
        if pom:
            agg["per_output_metrics"] = pom
        return agg


class ClassificationPipeline:
    """
    Wrapper that selects and runs the appropriate classification pipeline.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, array-like of shape (n_samples, n_features).
    y : np.ndarray
        Target vector, array-like of shape (n_samples,) or (n_samples, n_outputs) for multi-output.
    analysis_type : str, optional
        Type of analysis to perform. Can be "baseline", "feature_selection", "hp_search", or "hp_search_fs".
    models : str or list, optional
        Models to include. Can be "all", a model name, or a list of model names.
    metrics : str or list, optional
        Metrics to evaluate. If None, uses default metrics for the task type.
    random_state : int, optional
        Random state for reproducibility. Default is 42.
    cv_strategy : str, optional
        Cross-validation strategy to use. Default is "stratified".
    n_splits : int, optional
        Number of splits for cross-validation. Default is 5.
    n_features : int, optional
        Number of features to select in feature selection. Default is None (select all).
    direction : str, optional
        Direction for feature selection. Can be "forward", "backward", or "both". Default is "forward".
    search_type : str, optional
        Type of hyperparameter search to perform. Can be "grid" or "random". Default is "grid".
    n_iter : int, optional
        Number of iterations for random search. Default is 100.
    scoring : str, optional
        Scoring metric for hyperparameter search. If None, uses default metric for the task type.
    n_jobs : int, optional
        Number of parallel jobs to run. Default is -1 (use all available cores).
    save_intermediate : bool, optional
        Whether to save intermediate results during the run. Default is False.
    results_dir : str, optional
        Directory to save results. Default is "results".
    results_file : str, optional
        Base name for results files. Default is "results".
    cv_kwargs : dict, optional
        Additional cross-validation parameters. If None, uses `DEFAULT_CV`.

    Raises
    ------
    ValueError
        If `analysis_type` is not one of the supported types.
    ValueError
        If `y` is not a valid target for classification (e.g., not binary, multiclass, or multioutput).

    Notes
    -----
    This class automatically detects the type of classification task (binary, multiclass, or multioutput)
    based on the shape and content of `y`. It then instantiates the appropriate pipeline class
    (`BinaryClassificationPipeline`, `MultiClassClassificationPipeline`, or `MultiOutputClassificationPipeline`)
    and runs the specified analysis type (baseline evaluation, feature selection, hyperparameter search,
    or hyperparameter search with feature selection).

    Example
    -------
    >>> from coco_pipe.ml.classification import ClassificationPipeline
    >>> import numpy as np
    >>> X = np.random.rand(100, 20)  # 100 samples, 20 features
    >>> y = np.random.randint(0, 2, size=100)  # Binary target
    >>> pipeline = ClassificationPipeline(X, y, analysis_type="baseline", models="all")
    >>> results = pipeline.run()
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        analysis_type: str = "baseline",
        models: Union[str, Sequence[str]] = "all",
        metrics: Union[str, Sequence[str]] = "accuracy",
        random_state: int = 42,
        cv_strategy: str = "stratified",
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
        self.results = {}

        # pick pipeline class
        if hasattr(y, "ndim") and y.ndim == 2:
            PipelineClass = MultiOutputClassificationPipeline
        else:
            t = type_of_target(y)
            PipelineClass = (
                BinaryClassificationPipeline
                if t == "binary"
                else MultiClassClassificationPipeline
            )
        if hasattr(y, "ndim") and y.ndim == 2:
            self.task = "multioutput"
        else:
            self.task = "binary" if type_of_target(y) == "binary" else "multiclass"

        cvk = dict(DEFAULT_CV)
        if cv_kwargs:
            cvk.update(cv_kwargs)
        cvk.update({
            "cv_strategy": self.cv_strategy,
            "n_splits": self.n_splits,
            "random_state": self.random_state
        })

        self.pipeline = PipelineClass(
            X=X, y=y,
            models=models, metrics=metrics,
            random_state=random_state, n_jobs=n_jobs,
            cv_kwargs=cvk
        )
        os.makedirs(self.results_dir, exist_ok=True)

    def save(self, name, res):
        """
        Save results to a pickle file in the results directory.

        Parameters
        ----------
        name : str
            Name of the results file (without extension).
        res : dict
            Results dictionary to save.

        Raises
        ------
        IOError
            If there is an error saving the results file.

        Notes
        -----
        The results are saved as a pickle file in the specified results directory.
        The file is named `<name>.pkl` where `name` is the provided name parameter.
        The results directory is created if it does not exist.
        """
        filepath = os.path.join(self.results_dir, f"{name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(res, f)
        logger.info(f"Saved results to {filepath}")

    def run(self):
        """
        Run the selected analysis type on the classification pipeline.
        This method executes the specified analysis type (baseline evaluation,
        feature selection, hyperparameter search, or hyperparameter search with feature selection)
        and saves the results to the specified results directory.
        It also saves metadata about the run, including the task type, analysis type,
        models used, metrics evaluated, random state, cross-validation strategy,
        number of splits, number of features (if applicable), direction (if applicable),
        search type (if applicable), number of iterations (if applicable), and the shapes of
        the input data `X` and target `y`.
        It returns a dictionary containing the results of the analysis.

        Parameters
        ----------
        None

        Returns
        dict
            A dictionary containing the results of the analysis.

        Raises
        ValueError
            If the analysis type is not recognized or if there is an error during the run.
        
        Notes
        The results are saved in the specified results directory with a filename
        formatted as `<results_file>_<task>_<analysis_type>_rs<random_state>.pkl`.
        The metadata is saved in a separate JSON file with the same base name
        but with `_metadata.json` appended.
        The metadata includes information about the task, analysis type, models,
        metrics, random state, cross-validation strategy, number of splits,
        number of features (if applicable), direction (if applicable),
        search type (if applicable), number of iterations (if applicable),
        number of jobs, shapes of `X` and `y`, start time, completed models,
        failed models, status, total models, successful models, and failed models count.
        The metadata is saved in JSON format with indentation for readability.
        The results are saved in pickle format for easy loading later.
        The results dictionary contains the results of the analysis for each model
        run, including the predictions, metrics, and any additional information
        specific to the analysis type performed.

        Example
        -------
        >>> pipeline = ClassificationPipeline(
            X=X, y=y,
            analysis_type="baseline",
            models="all",
            metrics=["accuracy", "f1"],
            random_state=42,
            cv_strategy="stratified",
            n_splits=5,
            n_jobs=-1,
            save_intermediate=True,
            results_dir="results",
            results_file="classification_results"
        )
        >>> results = pipeline.run()
        """


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
                    raise ValueError(f"Unknown pipeline type: {self.analysis_type}")

                if self.save_intermediate:
                    self.save(f"{name}_{base_name}", res)

                self.results[name] = res
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

        self.save(base_name, self.results)
        with open(os.path.join(self.results_dir, f"{base_name}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {os.path.join(self.results_dir, f'{base_name}_metadata.json')}" )

        return self.results


def main():
    """
    Example usage of the ClassificationPipeline.
    This function demonstrates how to create and run a classification pipeline
    with synthetic data.
    """
    # Generate synthetic data
    X = np.random.rand(100, 20)  # 100 samples, 20 features
    y = np.random.randint(0, 2, size=100)  # Binary target

    # Create and run the classification pipeline
    pipeline = ClassificationPipeline(
        X=X,
        y=y,
        analysis_type="baseline",
        models="all",
        metrics=["accuracy", "f1"],
        random_state=42,
        cv_strategy="stratified",
        n_splits=5,
        n_jobs=-1,
        save_intermediate=True,
        results_dir="results",
        results_file="classification_results"
    )
    
    results = pipeline.run()
    print("Results:", results)
if __name__ == "__main__":
    main()
