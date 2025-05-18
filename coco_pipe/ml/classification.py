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
    Wrapper that selects and runs the appropriate pipeline based on y.

    Parameters:
    ---------
        :X: Input features as a numpy array or pandas DataFrame
        :y: Target variable(s) as a numpy array or pandas Series/DataFrame
        :analysis_type: Type of analysis to perform, one of ["baseline", "feature_selection", "hyperparameter_tuning"]. Default is "baseline"
        :models: List of models to evaluate or "all" to try all available models. Default is "all"
        :metrics: Metrics to evaluate models on. Can be a string or list of strings. Default is "accuracy"
        :random_state: Random seed for reproducibility. Default is 42
        :cv_strategy: Cross-validation strategy, one of ["stratified", "kfold"]. Default is "stratified"
        :n_splits: Number of cross-validation folds. Default is 5
        :n_features: Number of features to select when doing feature selection. Default is None
        :direction: Direction for feature selection, one of ["forward", "backward"]. Default is "forward"
        :search_type: Type of hyperparameter search, one of ["grid", "random"]. Default is "grid"
        :n_iter: Number of iterations for random search. Default is 100
        :scoring: Scoring metric for model selection. Default is None
        :n_jobs: Number of parallel jobs. Default is -1 (use all processors)
        :save_intermediate: Whether to save intermediate results. Default is False
        :results_dir: Directory to save results. Default is "results"
        :results_file: Base filename for results. Default is "results"

    The pipeline can:
    - Run baseline model evaluation with cross-validation
    - Perform automated feature selection
    - Conduct hyperparameter tuning
    - Handle both binary and multi-class classification
    - Support multiple evaluation metrics
    - Save intermediate results during training
    - Generate comprehensive performance reports

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
        if analysis_type.lower() not in ["baseline", "feature_selection", "hp_search", "hp_search_fs"]:
            raise ValueError(f"Invalid analysis type: {analysis_type}")
        self.analysis_type = analysis_type.lower()
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
        # placeholders
        self.pipeline = None
        self.results = {}

    def save(self, name, res):
        """
        Save intermediate results for each model in a pickle file as well as the final results in a pickle file.
        """
        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(res, f)
        logger.info(f"Saved results for {name}")

    def run(self):
        """
        Detect task type, build pipeline, run all models, and store results.
        Returns:
            dict of model_name -> results dict
        """
        # Determine task
        if hasattr(self.y, "ndim") and self.y.ndim == 2:
            PipelineClass = MultiOutputClassificationPipeline
            task = "multioutput"
            logger.info("Detected multi-output task")
        else:
            target_type = type_of_target(self.y)
            if target_type == "binary":
                PipelineClass = BinaryClassificationPipeline
                task = "binary"
                logger.info("Detected binary classification task")
            else:
                PipelineClass = MultiClassClassificationPipeline
                task = "multiclass"
                logger.info("Detected multiclass classification task")

        # Prepare cv_kwargs
        cv_kwargs = dict(DEFAULT_CV)
        if self.cv_kwargs is not None:
            cv_kwargs.update(self.cv_kwargs)
        cv_kwargs["cv_strategy"] = self.cv_strategy
        cv_kwargs["random_state"] = self.random_state
        cv_kwargs["n_splits"] = self.n_splits

        # Instantiate pipeline
        self.pipeline = PipelineClass(
            X=self.X,
            y=self.y,
            models=self.models,
            metrics=self.metrics,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            cv_kwargs=cv_kwargs,
        )
        file_name = f"{self.results_file}_{task}_{self.analysis_type}_rs{self.random_state}"
        if self.analysis_type == "feature_selection":
            file_name += f"_nfeatures{self.n_features}_direction{self.direction}"
        if self.analysis_type == "hp_search":
            file_name += f"_niter{self.n_iter}_search{self.search_type}"

        # Initialize metadata
        import json
        metadata = {
            "task": task,
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
            "y_shape": self.y.shape if hasattr(self.y, "shape") else (len(self.y),),
            "start_time": datetime.datetime.now().isoformat(),
            "completed_models": [],
            "failed_models": [],
            "status": "running"
        }

        # Execute for each model
        for model_name in self.pipeline.model_configs:
            try:
                if self.analysis_type == "baseline":
                    res = self.pipeline.baseline(model_name)
                    if self.save_intermediate:
                        self.save(f"{model_name}_{file_name}", res)
                elif self.analysis_type == "feature_selection":
                    res = self.pipeline.feature_selection(
                        model_name,
                        n_features=self.n_features,
                        direction=self.direction,
                        scoring=self.scoring,
                    )
                    if self.save_intermediate:
                        self.save(f"{model_name}_{file_name}", res)
                elif self.analysis_type == "hp_search":
                    res = self.pipeline.hp_search(
                        model_name,
                        param_grid=None,
                        search_type=self.search_type,
                        n_iter=self.n_iter,
                        scoring=self.scoring,
                    )
                    if self.save_intermediate:
                        self.save(f"{model_name}_{file_name}", res)
                elif self.analysis_type == "hp_search_fs":
                    res = self.pipeline.hp_search_fs(
                        model_name,
                        param_grid=None,
                        search_type=self.search_type,
                        n_features=self.n_features,
                        direction=self.direction,
                        n_iter=self.n_iter,
                        scoring=self.scoring,
                    )
                    if self.save_intermediate:
                        self.save(f"{model_name}_{file_name}", res)
                else:
                    raise ValueError(f"Unknown pipeline type: {self.analysis_type}")

                self.results[model_name] = res
                metadata["completed_models"].append(model_name)
            except Exception as e:
                logger.error(f"Failed to run {model_name}: {str(e)}")
                metadata["failed_models"].append({"model": model_name, "error": str(e)})

        # Update final metadata
        metadata["end_time"] = datetime.datetime.now().isoformat()
        metadata["status"] = "completed" if not metadata["failed_models"] else "partial_failure"
        metadata["total_models"] = len(self.pipeline.model_configs)
        metadata["successful_models"] = len(metadata["completed_models"])
        metadata["failed_models_count"] = len(metadata["failed_models"])

        # Save final results and metadata
        self.save(file_name, self.results)
        with open(f"{file_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {file_name}_metadata.json")

        return self.results
