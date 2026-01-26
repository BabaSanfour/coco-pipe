"""
Decoding Core
=============
This module is responsible for:
1. Orchestrating the Cross-Validation loop.
2. Managing Estimator lifecycles (instantiation, fitting, prediction).
3. Computing metrics dynamically based on task type.
4. Aggregating results for downstream analysis.
"""

import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_classif, f_regression
from tempfile import mkdtemp
from shutil import rmtree
import atexit

from .configs import ExperimentConfig
from .registry import get_estimator_cls
from .utils import get_cv_splitter, get_scorer

logger = logging.getLogger(__name__)

class Experiment:
    """
    Main executor for decoding experiments.

    Parameters
    ----------
    config : ExperimentConfig
        The complete configuration for the experiment.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: Dict[str, Any] = {}
        self._validate_config()

    def _validate_config(self):
        """
        Perform comprehensive runtime validation of the configuration.
        
        Logic
        -----
        1. **Tuning Consistency**: Warns if `tuning.enabled` but no `grids` are provided.
        2. **Task vs Metrics**: Checks if metrics match the task (e.g. no 'accuracy' for regression).
           Raises ValueError if incompatible.
        3. **Task vs CV**: Checks if CV strategy matches task (e.g. no 'stratified' for regression).
           Raises ValueError if incompatible.
        4. **Task vs Model**: Heuristic check for model type (e.g. no Regressor for Classification).
           Raises ValueError if incompatible.
           
        Raises
        ------
        ValueError
            If configuration contains incompatible settings.
        """
        task = self.config.task
        
        # 1. Tuning Consistency
        if self.config.tuning.enabled and not self.config.grids:
             logger.warning("Hyperparameter tuning is enabled but no 'grids' are defined in the config.")

        # 2. Task vs Metrics
        # Define forbidden substrings for each task
        forbidden_metrics = {
            "classification": ["r2", "squared_error", "absolute_error"],
            "regression": ["accuracy", "roc_auc", "f1", "precision", "recall", "log_loss"]
        }
        
        for metric in self.config.metrics:
            # Check internal sklearn/scorer names
            if any(bad in metric for bad in forbidden_metrics.get(task, [])):
                raise ValueError(
                    f"Metric '{metric}' is incompatible with task '{task}'. "
                    f"Please choose suitable metrics (e.g., {forbidden_metrics['regression'] if task == 'classification' else forbidden_metrics['classification']}...)"
                )

        # 3. Task vs CV Strategy
        if task == "regression":
            if "stratified" in self.config.cv.strategy:
                raise ValueError(
                    f"CV strategy '{self.config.cv.strategy}' implies stratification, "
                    f"which is invalid for regression tasks."
                )

        # 4. Task vs Model Type
        # We infer type from the config class name or method string
        for name, model_cfg in self.config.models.items():
            method_name = model_cfg.method.lower()
            
            if task == "classification":
                if "regressor" in method_name or "regression" in method_name:
                    # Exception: LogisticRegression is a classifier
                    if "logistic" not in method_name:
                        raise ValueError(
                            f"Model '{name}' ({model_cfg.method}) appears to be a Regressor, "
                            f"but task is 'classification'."
                        )
            
            elif task == "regression":
                if "classifier" in method_name or "svc" in method_name or "logistic" in method_name:
                    # SVR is valid, SVC is not (usually)
                    raise ValueError(
                        f"Model '{name}' ({model_cfg.method}) appears to be a Classifier, "
                        f"but task is 'regression'."
                    )

    def _prepare_estimator(self, model_name: str, model_config: Any) -> BaseEstimator:
        """
        Orchestrate the creation of the full Estimator Pipeline.
        
        Steps
        -----
        1. **Instantiation**: Calls `_instantiate_model` to get the base estimator (handling recursion).
        2. **Scaling**: If `use_scaler=True`, prepends a StandardScaler.
        3. **Feature Selection**: If enabled, prepends the FS step (Filter or Wrapper).
        4. **Pipeline Construction**: wraps steps in `sklearn.pipeline.Pipeline`.
           - Enables caching if FS + Tuning are both active.
        5. **Tuning Wrapper**: If tuning is enabled for this model, wraps the Pipeline 
           in GridSearchCV/RandomizedSearchCV via `_wrap_with_tuning`.
           
        Parameters
        ----------
        model_name : str
            Friendly name from config (used for grid lookup).
        model_config : Any
            Pydantic configuration object for the model.
            
        Returns
        -------
        BaseEstimator
            Final ready-to-run estimator (Pipeline or SearchCV).
        """
        # 1. Instantiate the Core Estimator
        full_est = self._instantiate_model(model_name, model_config)

        # 2. Build Pipeline Steps
        steps = []
        
        # Scaling
        if self.config.use_scaler:
            steps.append(("scaler", StandardScaler()))

        # Feature Selection
        if self.config.feature_selection.enabled:
            fs_step = self._create_fs_step(full_est)
            if fs_step:
                steps.append(fs_step)

        # Final Estimator
        steps.append(("clf", full_est))

        # 3. Create Pipeline with Caching if needed
        if self.config.feature_selection.enabled and self.config.tuning.enabled and self.config.grids:
             cachedir = mkdtemp()
             atexit.register(lambda: rmtree(cachedir, ignore_errors=True))
             est = Pipeline(steps, memory=cachedir)
        else:
             est = Pipeline(steps)
        
        # 4. Wrap with Tuning if enabled
        if self.config.tuning.enabled and self.config.grids and model_name in self.config.grids:
            est = self._wrap_with_tuning(est, model_name)
            
        return est

    def _instantiate_model(self, name: str, config: Any) -> BaseEstimator:
        """
        Instantiate a raw estimator from its configuration object.
        
        Logic
        -----
        1. **Registry Lookup**: Resolves class from `config.method`.
        2. **Recursion**: If config implies a meta-estimator (has `base_estimator`),
           recursively calls `_prepare_estimator` for the child.
        3. **Parameter Injection**: passed config fields as kwargs to `__init__`.
           - Automatically filters out invalid parameters if `TypeError` occurs 
             (robustness for mismatched config/class versions).
             
        Returns
        -------
        BaseEstimator
            The instantiated model (e.g., LogisticRegression instance) without pipeline wrappers.
        """
        # 1. Get Class
        est_cls = get_estimator_cls(config.method)
        
        # 2. Extract Params
        params = config.model_dump(exclude={"method"})

        # 3. Recursive Preparation (for Sliding/Generalizing internal 'base_estimator')
        if "base_estimator" in params:
             base_conf = params["base_estimator"]
             params["base_estimator"] = self._prepare_estimator(f"{name}_base", base_conf)

        # 4. Instantiate with Parameter Filtering
        try:
            return est_cls(**params)
        except TypeError:
            # Fallback: Filter invalid params (e.g. metadata fields in config)
            valid_params = est_cls().get_params().keys()
            filtered = {k: v for k, v in params.items() if k in valid_params}
            dropped = set(params) - set(filtered)
            if dropped:
                logger.debug(f"[{name}] Dropping invalid params: {dropped}")
            return est_cls(**filtered)

    def _create_fs_step(self, estimator: BaseEstimator) -> Optional[tuple]:
        """
        Create a Feature Selection step for the pipeline.
        
        Logic
        -----
        - **Filter (k_best)**: Fast. selected before training the classifier based on statistical test.
          No inner CV loop required.
        - **Wrapper (sfs)**: Slow but accurate. Wraps the estimator in a SequentialFeatureSelector.
          This runs an **Inner CV Loop** (size = config.feature_selection.cv) to validate feature subsets.
          
        If used inside Hyperparameter Tuning, this step is part of the Pipeline, ensuring 
        features are re-selected for every fold and every parameter combination (Nested Simplification).
        
        Returns
        -------
        tuple or None
            ("fs", Transformer) step for sklearn Pipeline.
        """
        fs_conf = self.config.feature_selection

        if fs_conf.method == "k_best":
            score_func = f_classif if self.config.task == "classification" else f_regression
            return ("fs", SelectKBest(score_func=score_func, k=fs_conf.n_features or 10))

        elif fs_conf.method == "sfs":
            inner_cv = fs_conf.cv or 3
            return ("fs", SequentialFeatureSelector(
                estimator=clone(estimator),
                n_features_to_select=fs_conf.n_features,
                direction=fs_conf.direction,
                cv=inner_cv,
                n_jobs=self.config.n_jobs
            ))
        return None

    def _wrap_with_tuning(self, estimator: BaseEstimator, name: str) -> BaseEstimator:
        """
        Wrap the estimator (or pipeline) in a Hyperparameter Search object.
        
        This implements **Nested Cross-Validation** (Middle Loop):
        1. **Input**: A Pipeline (Scaler + FS + Classifier).
        2. **Search**: Creates a GridSearchCV / RandomizedSearchCV.
        3. **Process**:
           - For each fold of the *tuning* CV (defined by config.cv):
             - Train the Pipeline (including FS!) on the tuning train set.
             - Evaluate on the tuning validation set.
           - Finds the best (Hyperparameters + Features) combination.
           - Refits on the entire training set provided by the Outer Loop.
           
        This ensures simultaneous optimization of Preprocessing (FS) and Modeling parameters.
        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        grid = self.config.grids[name]
        
        new_grid = {}
        for k, v in grid.items():
            if "__" in k:
                new_grid[k] = v # trusted user input
            else:
                new_grid[f"clf__{k}"] = v
        grid = new_grid

        cv_splitter = get_cv_splitter(self.config.cv) 
        # Note: We don't pass groups here; they are passed to fit()

        search_kwargs = {
            "estimator": estimator,
            "param_grid" if self.config.tuning.search_type == "grid" else "param_distributions": grid,
            "cv": cv_splitter,
            "scoring": self.config.tuning.scoring or self.config.metrics[0],
            "n_jobs": self.config.tuning.n_jobs,
            "refit": True,
        }

        if self.config.tuning.search_type == "grid":
            return GridSearchCV(**search_kwargs)
        else:
            return RandomizedSearchCV(
                n_iter=self.config.tuning.n_iter,
                **search_kwargs
            )

    def run(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "ExperimentResult":
        """
        Execute the full experiment pipeline.
        
        This is the main entry point. It orchestrates:
        1. **Data Validation**: Checks input shapes and types.
        2. **Model Loop**: Iterates through all configured models.
        3. **Preparation**: Instantiates models -> Builds Pipelines (Scaler/FS) -> Wraps in Tuning.
        4. **Validation**: Runs the Outer Cross-Validation loop (optionally parallelized).
        5. **Aggregation**: Collects scores, predictions, and importances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (2D) or Time-Series data (3D).
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target labels or values.
        groups : array-like of shape (n_samples,), optional
            Group labels for splitting (e.g., subject-specific splits).

        Returns
        -------
        ExperimentResult
            Object containing results with methods to export to Tidy DataFrames.
        """
        start_time = time.time()
        logger.info(f"Starting Experiment: Task={self.config.task}")

        # 1. Validate Inputs
        X = np.asarray(X)
        y = np.asarray(y)
        if len(X) == 0:
            raise ValueError("Input X is empty.")
        if len(y) != len(X):
            raise ValueError(f"Length mismatch: X has {len(X)} samples, y has {len(y)}.")
            
        if groups is not None:
            groups = np.asarray(groups)
            if len(groups) != len(X):
                raise ValueError(f"Length mismatch: groups has {len(groups)}, X has {len(X)}.")

        # 2. Check Task Consistency (Classification vs Regression)
        target_type = type_of_target(y)
        if self.config.task == "classification" and target_type == "continuous":
             raise ValueError(
                 f"Task is 'classification' but target type is '{target_type}'. "
                 "Please check your labels or switch task to 'regression'."
             )

        # 3. Main Loop over Configured Models
        for friendly_name, model_cfg in self.config.models.items():
            logger.info(f"Evaluating Model: {friendly_name} ({model_cfg.method})")
            
            try:
                # A. Prepare (Instantiate + Scale + FS + Tune Wrapper)
                estimator = self._prepare_estimator(friendly_name, model_cfg)
                
                # B. Execute Cross-Validation
                # Note: Parallelism is handled inside _cross_validate if config.n_jobs > 1
                cv_results = self._cross_validate(estimator, X, y, groups)
                
                # C. Store Results
                self.results[friendly_name] = cv_results
            
            except Exception as e:
                logger.error(f"Failed to evaluate model '{friendly_name}': {e}", exc_info=True)
                self.results[friendly_name] = {
                    "error": str(e),
                    "status": "failed"
                }

        total_time = time.time() - start_time
        logger.info(f"Experiment Completed in {total_time:.2f}s")
        
        return ExperimentResult(self.results)


    def save_results(self, path: Optional[Union[str, Path]] = None):
        """
        Serialize results, configuration, and metadata to disk.

        Parameters
        ----------
        path : str or Path, optional
            Path to save the results. If None, uses config.output_dir.
            If both are None, raises ValueError.
        """
        if path is None:
            path = self.config.output_dir
            if path is None:
                raise ValueError("No output path specified in config or arguments.")

        path = Path(path)
        
        # 1. Prepare Metadata
        meta = {
            "timestamp": datetime.now().isoformat(),
            "tag": self.config.tag,
            "coco_pipe_version": "0.0.1", # TODO: Get from package
        }

        # 2. Bundle
        payload = {
            "config": self.config.model_dump(),
            "results": self.results,
            "meta": meta
        }

        # 3. Create Directory
        # If path is a directory (no extension), append filename
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
            filename = f"{self.config.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            target = path / filename
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            target = path

        # 4. Save
        logger.info(f"Saving results to {target}")
        joblib.dump(payload, target)
        return target

    @staticmethod
    def load_results(path: Union[str, Path]) -> "ExperimentResult":
        """
        Load a saved experiment payload and wrap it in ExperimentResult.

        Returns
        -------
        ExperimentResult
            The loaded results wrapper.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")
        
        payload = joblib.load(path)
        # Handle backward compatibility or raw load
        results = payload.get("results", payload) 
        return ExperimentResult(results)

    def _cross_validate(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Execute the Outer Cross-Validation Loop (Evaluation).
        
        This is the **Level 1 (Top Level)** Splits:
        - Splits the entire dataset into K folds (defined by config.cv).
        - For each fold:
          1. **Training Data**: 80% (if 5-fold). Passed to `estimator.fit()`.
             - If `estimator` is a GridSearch (Tuning Enabled), it will internally split this 80% 
               again for validation (Level 2 Split).
          2. **Test Data**: 20%. Used strictly for final `estimator.predict()` evaluation.
        
        Parallelization
        ---------------
        If `config.n_jobs > 1`, these folds run in parallel processes to speed up execution.
        """
        cv = get_cv_splitter(self.config.cv, groups=groups)
        
        # Prepare CV iterator
        splits = list(cv.split(X, y, groups))
        
        # Parallel Execution
        # We use n_jobs from config.
        n_jobs_outer = self.config.n_jobs
        
        # OVERSUBSCRIPTION PROTECTION
        # If outer loop is parallel, force inner estimators to run sequentially.
        # Otherwise, we might spawn N_outer * N_inner threads, crashing the system.
        parallel_estimator = clone(estimator)
        if n_jobs_outer != 1:
            parallel_estimator = self._force_serial_execution(parallel_estimator)
        
        parallel = joblib.Parallel(n_jobs=n_jobs_outer, verbose=self.config.verbose)
        
        results = parallel(
            joblib.delayed(self._fit_and_score_fold)(
                clone(parallel_estimator), X, y, train_idx, test_idx
            )
            for train_idx, test_idx in splits
        )
        
        # Unpack Results
        fold_scores = defaultdict(list)
        fold_preds = []
        fold_indices = []
        fold_importances = []
        fold_metadata = []
        
        for res in results:
            fold_indices.append(res["test_idx"])
            fold_preds.append(res["preds"])
            fold_importances.append(res["importance"])
            fold_metadata.append(res.get("metadata", {}))
            
            for m, s in res["scores"].items():
                fold_scores[m].append(s)

        # Aggregate Metrics
        metrics_summary = {
            m: {
                "mean": np.nanmean(s),
                "std": np.nanstd(s),
                "folds": s
            }
            for m, s in fold_scores.items()
        }
        
        # Aggregate Importances
        valid_imps = [f for f in fold_importances if f is not None]
        aggregated_importances = None
        if valid_imps:
            try:
                # Check consistency
                first_shape = valid_imps[0].shape
                if all(imp.shape == first_shape for imp in valid_imps):
                    stack = np.vstack(valid_imps)
                    aggregated_importances = {
                        "mean": np.mean(stack, axis=0),
                        "std": np.std(stack, axis=0),
                        "raw": stack
                    }
            except Exception:
                pass 

        return {
            "metrics": metrics_summary,
            "predictions": fold_preds, 
            "indices": fold_indices,
            "importances": aggregated_importances,
            "metadata": fold_metadata
        }

    def _fit_and_score_fold(
        self, 
        estimator: BaseEstimator, 
        X: np.ndarray, 
        y: np.ndarray, 
        train_idx: np.ndarray, 
        test_idx: np.ndarray
    ) -> Dict[str, Any]:
        """
        Execute a single Cross-Validation fold: Fit, Predict, and Score.
        
        Optimized for:
        - **Standard Estimators**: (N, F) input -> (N,) output.
        - **Sliding Estimators**: (N, F, T) input -> (N, T) output (Diagonal Decoding).
        
        Returns
        -------
        dict
            Contains 'test_idx', 'preds' (y_pred, y_true, y_proba), 
            'scores' (dict of metric values), and 'importance'.
        """
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 1. Fit
        estimator.fit(X_train, y_train)
        
        # 2. Predict (Standard or Temporal)
        y_pred = estimator.predict(X_test)
        fold_data = {"y_true": y_test, "y_pred": y_pred}

        # 3. Predict Proba (if available and needed)
        # Optimization: We always check/compute this if available, as 'roc_auc' is common.
        if hasattr(estimator, "predict_proba"):
            try:
                fold_data["y_proba"] = estimator.predict_proba(X_test)
            except Exception:
                pass # Some estimators have the method but fail if not calibrated/supported correctly
        
        # 4. Extract Feature Importances
        imp = None
        try:
            imp = self._extract_feature_importances(estimator)
        except Exception:
            pass

        # 5. Compute Metrics
        scores = {}
        is_multiclass = (type_of_target(y_test) == "multiclass")
        
        for metric_name in self.config.metrics:
            scorer = get_scorer(metric_name)
            try:
                # Determine if we should use Proba or Predictions
                use_proba = (metric_name in ["roc_auc", "log_loss"] and "y_proba" in fold_data)
                
                if use_proba:
                    val = self._compute_metric_safe(scorer, y_test, fold_data["y_proba"], is_multiclass, is_proba=True)
                else:
                    val = self._compute_metric_safe(scorer, y_test, y_pred, is_multiclass, is_proba=False)
                    
                scores[metric_name] = val
            except Exception as e:
                logger.warning(f"Metric '{metric_name}' failed in CV fold: {e}")
                scores[metric_name] = np.nan

        # 6. Extract Metadata (Best Params, Selected Features)
        meta = {}
        try:
            meta = self._extract_metadata(estimator)
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")

        return {
            "test_idx": test_idx,
            "preds": fold_data,
            "scores": scores,
            "importance": imp,
            "metadata": meta
        }
    
    @staticmethod
    def _extract_metadata(estimator: BaseEstimator) -> Dict[str, Any]:
        """
        Extract training metadata like best Hyperparameters and Selected Features.
        """
        meta = {}
        
        # 1. Best Params (from GridSearchCV/RandomizedSearchCV)
        if hasattr(estimator, "best_params_"):
            meta["best_params"] = estimator.best_params_
            # Unwrap best estimator for feature selection
            search_best = estimator.best_estimator_
        else:
            search_best = estimator

        # 2. Selected Features (from Pipeline step 'fs')
        if isinstance(search_best, Pipeline):
            fs_step = search_best.named_steps.get("fs")
            if fs_step and hasattr(fs_step, "get_support"):
                 meta["selected_features"] = fs_step.get_support()
                 
        return meta

    @staticmethod
    def _compute_metric_safe(scorer, y_true, y_est, is_multiclass, is_proba=False):
        """
        Compute metric handling standard and temporal (diagonal) shapes.
        
        Shapes Handled
        --------------
        - **Standard**: y_est is (N,) or (N, C)
        - **Generalizing (Matrix)**:
          - y_pred: (N, T_train, T_test) -> Score each (T_train, T_test) pair.
          - y_proba: (N, C, T_train, T_test) -> Score each (T_train, T_test) pair.
        """
        # 1. Temporal / Sliding Case (Extra Dimension)
        # Check for (N, T) predictions or (N, C, T) probabilities
        is_temporal = (y_est.ndim == 2 and not is_proba and y_true.ndim == 1) or (y_est.ndim == 3)
        
        if is_temporal:
             # Case A: Binary/Regression Predictions (N, T)
             if y_est.ndim == 2: 
                 # Iterate over time (dim 1)
                 return np.array([scorer(y_true, y_est[:, t]) for t in range(y_est.shape[1])])
             
             # Case B: Probabilities (N, C, T) or Generalizing (N, T_train, T_test)
             if y_est.ndim == 3:
                 # Logic: 
                 # - If input is NOT proba, (N, T, T) implies Generalizing Predictions.
                 # - If input IS proba, (N, C, T) implies Sliding Probabilities.
                 
                 if not is_proba:
                     # Generalizing Predictions (N, T_train, T_test)
                     n_train = y_est.shape[1]
                     n_test = y_est.shape[2]
                     matrix_scores = np.zeros((n_train, n_test))
                     
                     for t_tr in range(n_train):
                         for t_te in range(n_test):
                             y_slice = y_est[:, t_tr, t_te]
                             matrix_scores[t_tr, t_te] = scorer(y_true, y_slice)
                     return matrix_scores
                 
                 # Sliding Probabilities (N, C, T)
                 n_times = y_est.shape[2]
                 scores = []
                 for t in range(n_times):
                     slice_y = y_est[:, :, t] # (N, C)
                     
                     if not is_multiclass:
                         if slice_y.shape[1] == 2:
                             slice_y = slice_y[:, 1]
                     
                     kwargs = {"multi_class": "ovr"} if is_multiclass else {}
                     scores.append(scorer(y_true, slice_y, **kwargs))
                 return np.array(scores)

             # Case C: GenEst Probabilities (N, C, T_train, T_test) -> 4D
             if y_est.ndim == 4:
                 n_train = y_est.shape[2]
                 n_test = y_est.shape[3]
                 matrix_scores = np.zeros((n_train, n_test))

                 for t_tr in range(n_train):
                     for t_te in range(n_test):
                         slice_y = y_est[:, :, t_tr, t_te] # (N, C)
                         
                         if not is_multiclass:
                             if slice_y.shape[1] == 2:
                                 slice_y = slice_y[:, 1]
                                 
                         kwargs = {"multi_class": "ovr"} if is_multiclass else {}
                         matrix_scores[t_tr, t_te] = scorer(y_true, slice_y, **kwargs)
                 return matrix_scores

        # 2. Standard Case (N,) or (N, C)
        kwargs = {}
        if is_proba:
             if is_multiclass:
                 kwargs = {"multi_class": "ovr"}
             elif y_est.ndim == 2 and y_est.shape[1] == 2:
                 # Standard Binary Probabilities -> Take Positive Class
                 y_est = y_est[:, 1]
        
        return scorer(y_true, y_est, **kwargs)

    def _force_serial_execution(self, estimator: BaseEstimator) -> BaseEstimator:
        """
        Recursively set n_jobs=1 for the estimator and its sub-components.
        Used when the outer loop is already parallelized to avoid oversubscription.
        """
        # 1. Get all parameters
        params = estimator.get_params()
        
        # 2. Identify keys ending in 'n_jobs'
        updates = {}
        for key, value in params.items():
            if key.endswith("n_jobs") and value is not None and value != 1:
                 updates[key] = 1
                 
        # 3. Apply updates
        if updates:
            estimator.set_params(**updates)
            
        return estimator

    @staticmethod
    def _extract_feature_importances(estimator: BaseEstimator) -> Optional[np.ndarray]:
        """
        Extract feature importances or coefficients from a fitted estimator.
        Handles Pipelines and Feature Selection.
        """
        # 1. Unwrap Pipeline
        if isinstance(estimator, Pipeline):
            # Check for FS step
            fs_step = estimator.named_steps.get("fs")
            clf_step = estimator.named_steps.get("clf")
            
            # Get raw importances from classifier
            raw_imp = Experiment._extract_feature_importances(clf_step)
            if raw_imp is None:
                return None
            
            # Map back if FS was used
            if fs_step:
                support = fs_step.get_support() # bool mask of selected features
                # We need to reconstruct the full importance array with zeros (or NaNs) for unselected
                full_imp = np.zeros_like(support, dtype=float)
                full_imp[support] = raw_imp
                return full_imp
            
            return raw_imp

        # 2. Extract from Base Estimator
        if hasattr(estimator, "feature_importances_"):
            return estimator.feature_importances_
        if hasattr(estimator, "coef_"):
            # Handle multi-class coefs (n_classes, n_features) -> take magnitude/mean?
            # For strict "importance", usually mean of abs(coefs) across classes
            if estimator.coef_.ndim > 1:
                return np.mean(np.abs(estimator.coef_), axis=0)
            return np.abs(estimator.coef_)
        
        return None

class ExperimentResult:
    """
    Unified Container for Experiment Results.
    Provides Tidy Data views for easier analysis.
    """
    def __init__(self, raw_results: Dict[str, Any]):
        self.raw = raw_results

    def summary(self) -> pd.DataFrame:
        """
        Get a high-level summary of performance (Mean/Std across folds).
        
        Returns
        -------
        pd.DataFrame
            Index: Model Name
            Columns: Metric Mean/Std
        """
        rows = []
        for model, res in self.raw.items():
            if "error" in res:
                continue
            
            row = {"Model": model}
            for metric, stats in res["metrics"].items():
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
            rows.append(row)
            
        return pd.DataFrame(rows).set_index("Model")

    def get_detailed_scores(self) -> pd.DataFrame:
        """
        Get fold-level scores for all models in long format.
        
        Returns
        -------
        pd.DataFrame
            Columns: Model, Fold, Metric, Value
        """
        rows = []
        for model, res in self.raw.items():
            if "error" in res:
                continue
                
            metrics_data = res["metrics"]
            # Assume all metrics have same number of folds
            n_folds = len(next(iter(metrics_data.values()))["folds"])
            
            for fold_idx in range(n_folds):
                for metric, stats in metrics_data.items():
                    rows.append({
                        "Model": model,
                        "Fold": fold_idx,
                        "Metric": metric,
                        "Value": stats["folds"][fold_idx]
                    })
        return pd.DataFrame(rows)

    def get_predictions(self) -> pd.DataFrame:
        """
        Get concatenated predictions for all models.
        
        Returns
        -------
        pd.DataFrame
            Columns: Model, Fold, y_true, y_pred, (y_proba if available)
        """
        dfs = []
        for model, res in self.raw.items():
            if "error" in res:
                continue
                
            for fold_idx, preds in enumerate(res["predictions"]):
                # preds is dict: y_true, y_pred, y_proba
                df = pd.DataFrame({
                    "y_true": preds["y_true"],
                    "y_pred": preds["y_pred"]
                })
                df["Model"] = model
                df["Fold"] = fold_idx
                
                if "y_proba" in preds:
                    # Handle proba columns (might be multi-class)
                    proba = preds["y_proba"]
                    if proba.ndim == 1:
                        df["y_proba"] = proba
                    elif proba.ndim == 2:
                        for c in range(proba.shape[1]):
                            df[f"y_proba_{c}"] = proba[:, c]
                            
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
            
        return pd.concat(dfs, ignore_index=True)

    def get_best_params(self) -> pd.DataFrame:
        """
        Get the best hyperparameters selected per fold (if Tuning was enabled).
        
        Returns
        -------
        pd.DataFrame
            Columns: Model, Fold, Param, Value
        """
        rows = []
        for model_name, res in self.raw.items():
            if "error" in res:
                continue
            
            # Check if metadata exists (handling backward compatibility)
            if "metadata" in res:
                 for fold_idx, meta in enumerate(res["metadata"]):
                     if "best_params" in meta:
                         for p_name, p_val in meta["best_params"].items():
                             rows.append({
                                 "Model": model_name,
                                 "Fold": fold_idx,
                                 "Param": p_name,
                                 "Value": p_val
                             })
                             
        return pd.DataFrame(rows)

    def get_feature_stability(self) -> pd.DataFrame:
        """
        Analyze feature selection stability across folds.
        
        Returns
        -------
        pd.DataFrame
            Index: Feature Index/Name
            Columns: Selection Frequency (0.0 - 1.0)
        """
        rows = []
        for model_name, res in self.raw.items():
            if "error" in res:
                continue
                
            if "metadata" in res:
                # Collect masks
                masks = []
                for meta in res["metadata"]:
                    if "selected_features" in meta:
                        masks.append(meta["selected_features"])
                
                if masks:
                    # Stack: (n_folds, n_features)
                    stack = np.vstack(masks)
                    stability = np.mean(stack, axis=0) # 0 to 1
                    
                    for feat_idx, freq in enumerate(stability):
                         rows.append({
                             "Model": model_name,
                             "Feature": feat_idx, 
                             "Frequency": freq
                         })

        if not rows:
            return pd.DataFrame()
            
        return pd.DataFrame(rows)

    def get_generalization_matrix(self, metric: str = None) -> pd.DataFrame:
        """
        Get Generalization Matrix (Train Time x Test Time) averaged across folds.
        
        Parameters
        ----------
        metric : str, optional
            The metric to retrieve (e.g., 'accuracy', 'roc_auc'). 
            Defaults to the first metric found in results.
            
        Returns
        -------
        pd.DataFrame
            Index: Train Time
            Columns: Test Time
            Values: Average Score
        """
        # 1. Collect all matrices for the metric
        for model_name, res in self.raw.items():
            if "error" in res:
                continue
            
            metrics_data = res["metrics"]
            if metric is None:
                metric = list(metrics_data.keys())[0]
                
            if metric not in metrics_data:
                continue
                
            fold_scores = metrics_data[metric]["folds"]
            # Check if scores are matrices (2D arrays)
            valid_matrices = [s for s in fold_scores if isinstance(s, np.ndarray) and s.ndim == 2]
            
            if valid_matrices:
                 # Stack and Mean -> (n_folds, n_train, n_test) -> (n_train, n_test)
                 stack = np.stack(valid_matrices)
                 mean_matrix = np.mean(stack, axis=0)
                 return pd.DataFrame(mean_matrix)
        
        return pd.DataFrame()
