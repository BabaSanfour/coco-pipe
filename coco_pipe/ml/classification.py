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
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.multiclass import type_of_target

from .base import BasePipeline
from .config import BINARY_METRICS, BINARY_MODELS, DEFAULT_CV, MULTICLASS_MODELS, MULTIOUTPUT_METRICS, MULTICLASS_METRICS, MULTIOUTPUT_MODELS


logger = logging.getLogger(__name__)
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
        :X: np.ndarray, Feature matrix, array-like of shape (n_samples, n_features)
        :y: np.ndarray, Binary target vector (0/1 or -1/1), array-like of shape (n_samples,)
        :models: str or list, optional (default="all"), Models to include in the pipeline
        :metrics: str or list, optional (default=None), Metrics to evaluate
        :random_state: int, optional (default=42), Random state for reproducibility
        :n_jobs: int, optional (default=-1), Number of parallel jobs
        :cv_kwargs: dict, optional (default=None), Cross-validation parameters
        :groups: np.ndarray, array-like of shape (n_samples,), optional (default=None), 
            Group labels for the samples used while splitting the dataset into train/test set.
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



class MultiClassClassificationPipeline(BasePipeline):
    """
    Pipeline specifically for multiclass classification tasks.
        
    Parameters
    ----------
        :X: np.ndarray, Feature matrix, array-like of shape (n_samples, n_features)
        :y: np.ndarray, Multiclass target vector, array-like of shape (n_samples,)
        :models: str or list, optional (default="all"), Models to include in the pipeline
        :metrics: str or list, optional (default=None), Metrics to evaluate
        :per_class: bool, optional (default=False), Whether to compute per-class metrics
        :random_state: int, optional (default=42), Random state for reproducibility
        :n_jobs: int, optional (default=-1), Number of parallel jobs
        :cv_kwargs: dict, optional (default=None), Cross-validation parameters
        :groups: np.ndarray, array-like of shape (n_samples,), optional (default=None), 
            Group labels for the samples used while splitting the dataset into train/test set.
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

        metric_funcs = MULTICLASS_METRICS
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
        results = super().compute_metrics(fold_preds, [m for m in metrics if m != "roc_auc"], funcs)
        y_true = results['predictions']['y_true']
        y_proba = results['predictions']['y_proba']
        # multiclass ROC AUC OVR
        if 'roc_auc' in self.metrics:
            from .config import multiclass_roc_auc_score
            score = multiclass_roc_auc_score(y_true=y_true, y_proba=y_proba)
            results['metrics']['roc_auc'] = {'mean': score, 'std': 0.0, 'scores': [score]}
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


class MultiOutputClassificationPipeline(BasePipeline):
    """
    Pipeline for multi-output classification.

    Parameters
    ----------
        :X: np.ndarray, Feature matrix, array-like of shape (n_samples, n_features)
        :y: np.ndarray, Multivariate target vector, array-like of shape (n_samples, n_outputs)
        :models: str or list, optional (default="all"), Models to include in the pipeline
        :metrics: str or list, optional (default=None), Metrics to evaluate
        :random_state: int, optional (default=42), Random state for reproducibility
        :n_jobs: int, optional (default=-1), Number of parallel jobs
        :cv_kwargs: dict, optional (default=None), Cross-validation parameters
        :groups: np.ndarray, array-like of shape (n_samples,), optional (default=None), 
            Group labels for the samples used while splitting the dataset into train/test set.
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
        base = MULTIOUTPUT_MODELS.copy()
        if models == "all":
            model_configs = base.copy()
        elif isinstance(models, str):
            model_configs = {models: base[models].copy()}
        else:
            model_configs = {m: base[m].copy() for m in models}

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

class ClassificationPipeline:
    """
    Wrapper that selects and runs the appropriate classification pipeline.

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
        metrics="accuracy",
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
        self.results = {}

        if hasattr(self.y, "ndim") and self.y.ndim == 2:
            PipelineClass  = MultiOutputClassificationPipeline
            self.task = "multioutput"
            logger.info("Detected multi-output classification task")
        else:
            target_type = type_of_target(self.y)
            if target_type == "binary":
                PipelineClass = BinaryClassificationPipeline
                self.task = "binary"
                logger.info("Detected binary classification task")
            else:
                PipelineClass = MultiClassClassificationPipeline
                self.task = "multiclass"
                logger.info("Detected multiclass classification task")
        
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

        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

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
        Detect task, instantiate pipeline, run across models, save and return results.
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
