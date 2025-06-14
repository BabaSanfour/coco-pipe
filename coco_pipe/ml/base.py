#!/usr/bin/env python3
"""
coco_pipe/ml/base.py
----------------
Core functionality for machine learning pipelines, including cross-validation,
metric computation, feature selection, and hyperparameter search.

Author: Hamza Abdelhedi <hamza.abdelhedii@gmail.com>
Date: 2025-05-18
Version: 0.0.1
License: TBD
"""

import logging
import warnings
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Type

import numpy as np
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass, field
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline

from coco_pipe.ml.config import DEFAULT_CV
from coco_pipe.ml.utils import get_cv_splitter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

@dataclass
class ModelConfig:
    """
    Configuration holder for a single model: estimator instance, initial params,
    and hyperparameter grid. Automatically stores original values for resets.
    """
    estimator: Union[Type[BaseEstimator], BaseEstimator]
    init_params: Dict[str, Any] = field(default_factory=dict)
    param_grid: Dict[str, Sequence[Any]] = field(default_factory=dict)

    # these get set by __post_init__
    original_estimator: BaseEstimator = field(init=False)
    original_init_params: Dict[str, Any] = field(init=False)
    original_param_grid: Dict[str, Sequence[Any]] = field(init=False)

    def __post_init__(self):
        # If user passed in a class, instantiate it with init_params
        if isinstance(self.estimator, type):
            try:
                # Try to instantiate with init_params first
                est_instance = self.estimator(**self.init_params)
            except TypeError:
                # If that fails, try with no args
                try:
                    est_instance = self.estimator()
                except TypeError as e:
                    raise ValueError(
                        f"Cannot instantiate {self.estimator.__name__}. "
                        f"Either provide required parameters in init_params or pass "
                        f"an already instantiated estimator. Error: {e}"
                    )
        else:
            est_instance = self.estimator

        # Store clones of the "pure" estimator and config
        self.original_estimator = clone(est_instance)
        self.original_init_params = dict(self.init_params)
        self.original_param_grid = dict(self.param_grid)

        # Now apply init_params to a fresh clone for use
        fresh = clone(self.original_estimator)
        if self.init_params:
            fresh.set_params(**self.init_params)
        self.estimator = fresh    


    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key, returning default if not found.
        
        Parameters
        ----------
        key : str
            The key to retrieve from the configuration.
        default : Any, optional
            Default value to return if key is not found.

        Returns
        -------
        Any
            The value associated with the key, or default if not found.
        """
        return getattr(self, key, default)


class BasePipeline(ABC):
    """
    Abstract base class for ML pipelines: provides CV, metrics, feature selection,
    and hyperparameter search functionality.
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        metric_funcs: Dict[str, Any],
        model_configs: Dict[str, Any],
        use_scaler: bool = True,
        columns: Optional[Sequence[str]] = None,
        default_metrics: Optional[Sequence[str]] = None,
        cv_kwargs: Dict[str, Any] = DEFAULT_CV,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        verbose: bool = True  # new verbose flag
    ):
        """
        Initialize pipeline with data, metrics, and configuration.

        This constructor sets up the ML pipeline with the provided data, metrics,
        and configuration parameters for cross-validation and model evaluation.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix. If DataFrame, column names will be preserved.
            Shape (n_samples, n_features).
        y : Series or ndarray
            Target array. Shape (n_samples,) or (n_samples, n_targets) for multi-output.
        metric_funcs : dict
            Mapping from metric names to callable functions with signature
            f(y_true, y_pred) or f(y_true, y_proba) for probability-based metrics.
        model_configs : dict
            Mapping from model names to dicts with keys:
            - 'estimator': sklearn-compatible estimator instance
            - 'params': optional dict of hyperparameter grids for tuning
        use_scaler : bool, default=True
            Whether to use a scaler (StandardScaler) in the pipeline.
            If True, a StandardScaler will be applied before the model.
        columns : sequence of str, optional
            Names of feature matrix columns. If None, and X is a DataFrame we will use the DataFrame's columns.
            If X is an ndarray, default names will be generated as 'feature_0', 'feature_1', etc.
        default_metrics : sequence of str, optional
            Names of metrics to compute. Must be keys in metric_funcs.
            If None, no metrics will be calculated by default.
        cv_kwargs : dict, optional
            Cross-validation settings passed to get_cv_splitter().
            Common keys: 'cv_strategy', 'n_splits', 'shuffle'.
            Will not be mutated.
        groups : Series or ndarray, optional
            Group labels for GroupKFold or other group-based CV strategies.
            Shape (n_samples,).
        n_jobs : int, default=-1
            Number of parallel jobs for cross-validation and feature selection.
            -1 means using all processors.
        random_state : int, optional
            Random seed for reproducibility of CV splits and model initialization.
        verbose : bool, default=False
            Whether to print detailed progress messages.

        Raises
        ------
        ValueError
            If X and y have different numbers of samples, if groups length does not match X,
            if invalid metrics or model configurations are provided.,

        Notes
        -----
        The constructor performs validation on inputs through _validate_input()
        and _validate_metrics() methods.

        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.metrics import accuracy_score, roc_auc_score
        >>> metrics = {'accuracy': accuracy_score, 'roc_auc': roc_auc_score}
        >>> models = {'rf': {'estimator': RandomForestClassifier()}}
        >>> pipeline = BasePipeline(X_train, y_train, metrics, models, 
        ...                         default_metrics=['accuracy'])
        """

        self.X = X
        self.y = y
        self.metric_funcs = metric_funcs
        self.metrics = list(default_metrics) if default_metrics else []
        self.use_scaler = use_scaler
        self.cv_kwargs = deepcopy(cv_kwargs)  # shallow copy
        self.groups = groups
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose  

        self.model_configs: Dict[str, ModelConfig] = {}
        for name, cfg in model_configs.items():
            est = cfg['estimator']
            init = cfg.get('default_params', {})
            grid = cfg.get('hp_search_params', cfg.get('params', {}))
            self.model_configs[name] = ModelConfig(
                estimator=est,
                init_params=dict(init),
                param_grid=dict(grid)
            )

        self._validate_input()
        self._validate_metrics()
        self.feature_names = columns or self._get_feature_names(X)

    def _validate_input(self) -> None:
        if not isinstance(self.X, (pd.DataFrame, np.ndarray)):
            raise ValueError("X must be a DataFrame or numpy array.")
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of samples.")
        if self.groups is not None and len(self.groups) != len(self.X):
            raise ValueError("If groups are provided, they must match the number of samples in X.")

    def _validate_metrics(self) -> None:
        invalid = [m for m in self.metrics if m not in self.metric_funcs]
        if invalid:
            raise ValueError(f"Unknown metrics: {invalid}")

    @staticmethod
    def _get_feature_names(
        X: Union[pd.DataFrame, np.ndarray]
    ) -> List[str]:
        """
        Retrieve feature names for DataFrame or generate defaults for ndarray.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix.

        Returns
        -------
        names : list of str
            Column names if DataFrame, else 'feature_i'.
        """
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        return [f"feature_{i}" for i in range(X.shape[1])]

    @staticmethod
    def _select_columns(
        X: Union[pd.DataFrame, np.ndarray],
        mask: np.ndarray
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Select columns by boolean mask.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix.
        mask : ndarray
            Boolean mask for selecting columns.
            Shape (n_features,).

        Returns
        -------
        X : DataFrame or ndarray
            Feature matrix with selected columns.
            If input was DataFrame, output is also DataFrame.
            If input was ndarray, output is also ndarray.
        Raises
        ------
        ValueError
            If mask length does not match number of features in X.
        """
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, mask]
        return X[:, mask]
        
    @staticmethod
    def _extract_feature_importances(
        estimator: BaseEstimator
    ) -> Optional[np.ndarray]:
        """
        Extract feature importances or coefficients from a (possibly nested) estimator.

        This will recursively unwrap pipelines, selectors, and meta-estimators
        to find either:
        - `feature_importances_` (tree-based models, ensembles)
        - `coef_` (linear models), averaging multiclass outputs if needed
        - A combination of sub-estimators (e.g. VotingClassifier)

        Parameters
        ----------
        estimator : BaseEstimator
            A fitted scikit-learn estimator, which may be:
            - A tree-based or ensemble model with `.feature_importances_`
            - A linear model with `.coef_`
            - A `Pipeline` (`.named_steps`)
            - A `SequentialFeatureSelector` (`.estimator_`)
            - A meta-estimator with `.estimators_`

        Returns
        -------
        importances : ndarray or None
            1D array of importances (length = n_features), or
            None if no importances/coefs are found.
        """
        # 1) Tree‐based or ensemble
        if hasattr(estimator, 'feature_importances_'):
            return estimator.feature_importances_
        if hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            if coef.ndim == 1:
                return coef
            return np.mean(coef, axis=0)
        if hasattr(estimator, 'named_steps'):
            last = list(estimator.named_steps.values())[-1]
            return BasePipeline._extract_feature_importances(last)
        if hasattr(estimator, 'estimator_'):
            return BasePipeline._extract_feature_importances(estimator.estimator_)
        if hasattr(estimator, 'best_estimator_'):
            return BasePipeline._extract_feature_importances(estimator.best_estimator_)
        if hasattr(estimator, 'estimators_'):
            imps = []
            for sub in estimator.estimators_:
                imp = BasePipeline._extract_feature_importances(sub)
                if imp is not None:
                    imps.append(imp)
            if imps:
                return np.mean(imps, axis=0)
        return None

    def _aggregate(self, fold_preds, fold_scores, fold_importances, freq=None):
        """Aggregate predictions and compute metrics across folds."""
        # Concatenate all predictions for the predictions output
        y_true = np.concatenate([fp["y_true"] for fp in fold_preds])
        y_pred = np.concatenate([fp["y_pred"] for fp in fold_preds])
        predictions = {"y_true": y_true, "y_pred": y_pred}
        
        if all("y_proba" in fp for fp in fold_preds):
            try:
                shapes = [fp["y_proba"].shape for fp in fold_preds]
                if all(s[1] == shapes[0][1] for s in shapes):
                    y_proba = np.vstack([fp["y_proba"] for fp in fold_preds])
                    predictions["y_proba"] = y_proba
            except Exception:
                pass
        
        predictions["fold_preds"] = fold_preds
        metrics = {}
        for metric_name in self.metrics:
            fold_score = fold_scores[metric_name]
            metrics[metric_name] = {
                "mean": np.mean(fold_score),
                "std": np.std(fold_score),
                "fold_scores": fold_score
            }
            if self.verbose:
                logger.info("Metric '%s' - Mean: %.4f, Std: %.4f", 
                            metric_name, np.mean(fold_score), np.std(fold_score))
        feature_importances = {}
        if fold_importances:
            for feat_name, fi in fold_importances.items():
                mean = np.mean(fi)
                feat_freq = freq.get(feat_name, 1) if freq else 1
                std = np.std(fi)
                feature_importances[feat_name] = {
                    "mean": mean,
                    "std": std,
                    "weighted_mean": mean * feat_freq,
                    "weighted_std": std * feat_freq,
                    "fold_importances": fi
                }
                if self.verbose:
                    logger.info("Feature '%s' - Mean importance: %.4f", feat_name, mean)
        return predictions, metrics, feature_importances


    def get_model_params(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get parameters for all models in the pipeline or a specific model.
        This method retrieves the parameters of the model's estimator, initial
        parameters, and hyperparameter grid used for tuning.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to retrieve parameters for.
            If None, returns parameters for all models in the pipeline.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the model parameters.
            If model_name is specified, returns parameters for that model only.

        Raises
        ------
        KeyError
            If the specified model name does not exist in the model_configs.

        Notes
        -------
        This method is useful for inspecting the current configuration of models
        in the pipeline.
        It returns a dictionary with the following keys:
        - 'estimator_type': Type of the model's estimator (e.g., 'RandomForestClassifier').
        - 'estimator_params': Parameters of the model's estimator.
        - 'init_params': Initial parameters used to create the model.
        - 'param_grid': Hyperparameter grid used for tuning the model.
        
        Examples
        --------
        >>> pipeline.get_model_params('random_forest')
        {
            'estimator_type': 'RandomForestClassifier',
            'estimator_params': {'n_estimators': 100, ...},
            'init_params': {'max_depth': None, ...},
            'param_grid': {'n_estimators': [100, 200], ...}
        }
        """
        def _single_params(mc: ModelConfig) -> Dict[str, Any]:
            est = mc.estimator
            return {
                'estimator_type': type(est).__name__,
                'estimator_params': est.get_params(deep=False),
                'init_params': dict(mc.init_params),
                'param_grid': dict(mc.param_grid)
            }

        if model_name:
            if model_name not in self.model_configs:
                raise KeyError(f"Model '{model_name}' not found")
            return _single_params(self.model_configs[model_name])
        return {name: _single_params(mc) for name, mc in self.model_configs.items()}

    def update_model_params(
        self,
        model_name: str,
        params: Dict[str, Any],
        update_estimator: bool = True,
        update_config: bool = True,
        param_type: str = 'default'
    ) -> None:
        """
        Update parameters for a specific model in the pipeline.
        This method allows updating the initial parameters of the model's
        estimator or the hyperparameter grid used for tuning.

        Parameters
        ----------
        model_name : str
            The name of the model to update.
        params : dict
            The parameters to update.
        update_estimator : bool
            Whether to update the model's estimator.
        update_config : bool
            Whether to update the model's configuration.
        param_type : str
            The type of parameters to update ('default' or 'hp_search').

        Raises
        ------
        KeyError
            If the specified model name does not exist in the model_configs.
        ValueError
            If the specified parameter type is invalid.

        Notes
        This method modifies the model's estimator and/or hyperparameter grid
        based on the provided parameters. It can be used to adjust model behavior
        or prepare for hyperparameter tuning.

        Examples
        --------
        >>> pipeline.update_model_params('random_forest', {'n_estimators': 200})
        """
        if model_name not in self.model_configs:
            raise KeyError(f"Model '{model_name}' not found")
        mc = self.model_configs[model_name]

        if param_type not in ['default', 'hp_search']:
            raise ValueError("param_type must be 'default' or 'hp_search'")

        if update_estimator and param_type == 'default':
            temp = clone(mc.original_estimator)
            temp.set_params(**params)
            mc.init_params.update(params)
            mc.estimator = temp
            logger.info(f"Updated init parameters for '{model_name}': {params}")
        if update_config and param_type == 'hp_search':
            for k, v in params.items():
                if not isinstance(v, (list, tuple, np.ndarray)):
                    raise ValueError(f"Grid values for '{k}' must be a sequence")
            mc.param_grid.update(params)
            logger.info(f"Updated param_grid for '{model_name}': {params}")

    def reset_model_params(self, model_name: str) -> None:
        """
        Reset parameters for a specific model to their original state.

        This method restores the model's estimator, initial parameters, and
        hyperparameter grid to their original values.

        Parameters
        ----------
        model_name : str
            The name of the model to reset.

        Raises
        ------
        KeyError
            If the specified model name does not exist in the model_configs.

        Notes
        -----
        This method is useful for reverting any changes made to a model's
        parameters or configuration.

        Examples
        --------
        >>> pipeline.reset_model_params('random_forest')
        >>> print(pipeline.get_model_params('random_forest'))
        {'estimator_type': 'RandomForestClassifier', ...}
        """
        if model_name not in self.model_configs:
            raise KeyError(f"Model '{model_name}' not found")
        mc = self.model_configs[model_name]
        mc.estimator = clone(mc.original_estimator)
        mc.init_params = dict(mc.original_init_params)
        mc.param_grid = dict(mc.original_param_grid)
        logger.info(f"Reset parameters for '{model_name}' to original state")

    def list_models(self) -> Dict[str, str]:
        """
        List all models in the pipeline with their estimator types.

        Returns
        -------
        dict
            Mapping of model names to their estimator types.

        Notes
        -----
        This method provides a quick overview of the models configured in the
        pipeline.

        Examples
        --------
        >>> models = pipeline.list_models()
        """
        return {name: type(mc.estimator).__name__
                for name, mc in self.model_configs.items()}

    def _set_inner_cv(self, estimator, inner_splits):
        """
        Recursively replace any .cv on an estimator (or sub‐estimator)
        with our list of (train_idx, test_idx) tuples.
        """
        # 1) If it’s a search‐CV, overwrite its cv
        if isinstance(estimator, (GridSearchCV, RandomizedSearchCV)):
            estimator.cv = inner_splits

        # 2) If it has a .cv attribute (e.g. SequentialFeatureSelector)
        elif hasattr(estimator, 'cv'):
            estimator.cv = inner_splits

        # 3) If it’s a Pipeline, descend into each step
        if isinstance(estimator, Pipeline):
            for step in estimator.named_steps.values():
                self._set_inner_cv(step, inner_splits)

        return estimator

    def cross_val(
        self,
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        # 1) Optional leakage-safe scaling
        if getattr(self, 'use_scaler', False):
            from sklearn.preprocessing import StandardScaler
            if isinstance(estimator, Pipeline):
                estimator = Pipeline(
                    [('scaler', StandardScaler())] + estimator.steps
                )
            else:
                estimator = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf',    estimator)
                ])

        # 2) Prepare data arrays & groups
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
        groups_arr = (
            self.groups.values if isinstance(self.groups, pd.Series)
            else self.groups
        )

        # 3) Outer CV with groups
        cv_conf = deepcopy(self.cv_kwargs)
        cv_conf.setdefault('random_state', self.random_state)
        outer_cv = get_cv_splitter(**cv_conf, groups=groups_arr)
        outer_splits = list(outer_cv.split(X_arr, y_arr, groups_arr))

        # 4) Figure out which metrics need proba
        proba_required = {
            m for m, fn in self.metric_funcs.items()
            if hasattr(fn, '__name__') and 'proba' in fn.__name__
        }

        # 5) Containers for fold‐level results
        fold_predictions = []
        fold_scores_list = []  # list of {metric:score} dicts
        fold_importances = {f: [] for f in self.feature_names}
        fold_estimators = []

        # 6) Manual nested loop
        for train_idx, test_idx in outer_splits:
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_te, y_te = X_arr[test_idx],  y_arr[test_idx]
            grp_tr     = groups_arr[train_idx] if groups_arr is not None else None

            # 6a) Build inner CV on this train‐set (with its own groups)
            inner_kwargs = deepcopy(self.cv_kwargs)
            if grp_tr is not None:
                inner_kwargs['groups'] = grp_tr
            inner_cv = get_cv_splitter(**inner_kwargs)
            inner_splits = list(inner_cv.split(X_tr, y_tr, grp_tr))

            # 6b) Clone & patch every .cv inside the estimator
            est = clone(estimator)
            est = self._set_inner_cv(est, inner_splits)

            # 6c) Fit on train
            fitted = est.fit(X_tr, y_tr)

            # 6d) Predict on test
            y_pred = fitted.predict(X_te)
            fold_pred = {'y_true': y_te, 'y_pred': y_pred}

            if hasattr(fitted, 'predict_proba'):
                try:
                    fold_pred['y_proba'] = fitted.predict_proba(X_te)
                except Exception:
                    pass

            fold_predictions.append(fold_pred)

            # 6e) Compute metrics
            scores = {}
            for m in self.metrics:
                fn = self.metric_funcs[m]
                if m in proba_required and 'y_proba' in fold_pred:
                    scores[m] = fn(y_te, fold_pred['y_proba'])
                else:
                    scores[m] = fn(y_te, y_pred)
            fold_scores_list.append(scores)

            # 6f) Extract importances
            imp = self._extract_feature_importances(fitted)
            if imp is not None:
                for i, feat in enumerate(self.feature_names):
                    if i < len(imp):
                        fold_importances[feat].append(imp[i])

            fold_estimators.append(fitted)

        # — unwrap any Pipelines around a search‐CV so best_params_/best_estimator_ survive —
        raw_ests = fold_estimators
        cv_fold_estimators = []
        for est in raw_ests:
            if isinstance(est, Pipeline):
                inner = est.named_steps.get('clf')
                if isinstance(inner, (GridSearchCV, RandomizedSearchCV)):
                    cv_fold_estimators.append(inner)
                else:
                    cv_fold_estimators.append(est)
            else:
                cv_fold_estimators.append(est)
        fold_estimators = cv_fold_estimators
        # 7) Pack into arrays/dicts for aggregation
        cv_fold_scores = {
            m: np.array([fs[m] for fs in fold_scores_list])
            for m in self.metrics
        }
        cv_fold_importances = {
            f: np.array(vals)
            for f, vals in fold_importances.items()
            if vals
        }

        return {
            'cv_fold_predictions': fold_predictions,
            'cv_fold_scores':      cv_fold_scores,
            'cv_fold_importances': cv_fold_importances,
            'cv_fold_estimators':  fold_estimators,
        }

    def baseline_evaluation(
        self,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model via cross-validation and return evaluation metrics, predictions, and feature importances.

        This method performs cross-validation on a specified model extracted from model_configs.
        It collects the scoring metrics, per-fold predictions, and extracts feature importances from each fold,
        aggregating the results into a comprehensive evaluation summary. Additionally, it refits the final model
        on the complete dataset using the original parameters for potential deployment.

        Parameters
        ----------
        model_name : str
            Name of the model in model_configs to evaluate.

        Returns
        -------
        dict
            Dictionary containing:
            - 'model_name': Name of the evaluated model.
            - 'metric_scores': Scoring metrics aggregated across folds with keys 'mean', 'std', and 'fold_scores'.
            - 'feature_importances': Dictionary of feature importance statistics (mean, std, weighted values) or None.
            - 'predictions': Dictionary containing concatenated y_true, y_pred, and optionally y_proba from cross-validation.
            - 'params': The initial model parameters used during evaluation.
            - 'folds_estimators': List of fitted estimator instances from each fold of cross-validation.

        Raises
        ------
        KeyError
            If model_name is not found in model_configs.

        Notes
        -----
        This method follows a two-step process:
          1. Perform cross-validation to assess the model's performance, score consistency, and extract feature importances.
          2. Refit the model on the full dataset using the original parameters for final evaluation and deployment.
        This design aligns with common scikit-learn practices for model evaluation and ensuring reproducible results.

        Examples
        --------
        >>> results = pipeline.baseline_evaluation('random_forest')
        >>> print(f"Accuracy: {results['metric_scores']['accuracy']['mean']:.4f}")
        >>> print(f"Feature Importances: {results['feature_importances']}")
        """
        if self.verbose:
            logger.info("Starting baseline evaluation for model '%s'...", model_name)
        if model_name not in self.model_configs:
            raise KeyError(f"Model '{model_name}' not found in model_configs")

        mc = self.model_configs[model_name]
        clf = clone(mc.original_estimator).set_params(**mc.init_params)

        results_ = self.cross_val(clf, self.X, self.y)
        predictions, metric_scores, feature_importances = self._aggregate(
            results_["cv_fold_predictions"], 
            results_["cv_fold_scores"],
            results_["cv_fold_importances"]
        )
        if self.verbose:
            logger.info("Baseline Evaluation Metrics: %s", metric_scores)
        results = {
            'model_name': model_name,
            'metric_scores': metric_scores,
            'feature_importances': feature_importances,
            'predictions': predictions,
            'params': mc.get('default_params', {}),
            'folds_estimators': results_['cv_fold_estimators'],
        }
        return results

    def _build_sfs_pipeline(
        self, 
        model_name: str,
        n_features: int,
        direction: str,
        scoring: str
    ) -> Tuple[Pipeline, np.ndarray, str]:
        """
        Build a scikit-learn Pipeline with Sequential Feature Selection.

        This helper method constructs a pipeline that performs feature selection
        followed by model fitting. The pipeline has two steps:
        1. 'sfs': SequentialFeatureSelector that selects a subset of features
        2. 'clf': The base estimator that is trained on the selected features

        Parameters
        ----------
        model_name : str
            Name of the model in self.model_configs to use as the base estimator.
            This model is cloned twice - once for feature selection and once for final fitting.
        n_features : int or None
            Number of features to select. If None, defaults to half of the available features.
            This value is passed to SequentialFeatureSelector's n_features_to_select parameter.
        direction : str
            Direction for feature selection, either 'forward' or 'backward'.
            - 'forward': Start with no features and add one at a time
            - 'backward': Start with all features and remove one at a time
        scoring : str or None
            Metric name to use for evaluating feature importance during selection.
            If None, defaults to the first metric in self.metrics.
            Must be a key in self.metric_funcs.

        Returns
        -------
        pipe : Pipeline
            Scikit-learn Pipeline with sequential feature selection and base estimator steps.
        feat_names : ndarray
            Array of feature names from the input data (self.X).
        metric : str
            The metric name that will be used for feature selection scoring.

        Notes
        -----
        - The base estimator is cloned twice to ensure independent instances for selection and fitting.
        - The inner cross-validation strategy is determined by self.cv_kwargs.
        - This method is primarily used internally by the feature_selection method.
        - The sequential feature selector uses cross-validation internally to evaluate features.

        See Also
        --------
        sklearn.feature_selection.SequentialFeatureSelector : Underlying feature selector
        sklearn.pipeline.Pipeline : Pipeline implementation used to chain operations
        """
        n_sel = n_features or (self.X.shape[1] // 2)
        metric = scoring or self.metrics[0]
        scorer = make_scorer(self.metric_funcs[metric])
        mc = self.model_configs[model_name]
        base = clone(mc.original_estimator).set_params(**mc.init_params)
        inner_cv = get_cv_splitter(**self.cv_kwargs)
        pipe = Pipeline([
            ('sfs', SequentialFeatureSelector(
                clone(base), n_features_to_select=n_sel,
                direction=direction, scoring=scorer,
                cv=inner_cv, n_jobs=self.n_jobs)),
            ('clf', clone(base))
        ])
        feat_names = np.array(self._get_feature_names(self.X))
        if self.verbose:
            logger.info("Built SFS pipeline for model '%s' with %d features using '%s' direction.",
                        model_name, n_sel, direction)
        return pipe, feat_names, metric

    def _extract_sfs_selected_features(
        self, 
        cv_fold_estimators: List, 
        feat_names: np.ndarray, 
    ) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
        """
        Extract selected features, and selection frequency from cross-validation results.
        
        This helper method processes cross-validation results from a Sequential Feature Selection
        pipeline to extract:
          - A dictionary mapping fold indices to the list of feature names selected in that fold.
          - A frequency dictionary for each feature representing the proportion of folds in which it was selected.
          - A set of all features that were selected in at least one fold.
        
        Parameters
        ----------
        cv_fold_estimators : dict
            Dictionary containing cross-validation fold estimators, where each estimator is a Pipeline
            with a 'sfs' step that is a SequentialFeatureSelector.
        feat_names : ndarray
            Array of feature names corresponding to the columns in the input data.
            
        Returns
        -------
        selected_per_fold : dict
            Dictionary mapping fold indices to lists of feature names selected in each fold.
        selected_all : set
            Set of all features that were selected in at least one fold.
        freq : dict
            Dictionary mapping each feature name to its selection frequency (a float between 0 and 1),
            calculated as the number of folds in which the feature was selected divided by the total number of folds.
            
        Notes
        -----
        This method is used as a helper for feature_selection to:
          1. Extract which features were selected in each cross-validation fold.
          2. Prepare a structured representation of per-fold feature selection results.
          3. Format the data for analysis of feature selection stability.
        
        It assumes that each estimator is a Pipeline containing a 'sfs'
        step with a SequentialFeatureSelector.
        
        See Also
        --------
        sklearn.feature_selection.SequentialFeatureSelector.get_support
        """
        n_folds = len(cv_fold_estimators)
        masks = [est.named_steps['sfs'].get_support() for est in cv_fold_estimators]
        selected_per_fold = {
            i: feat_names[mask].tolist()
            for i, mask in enumerate(masks)
        }
        selected_all = set([feat for mask in masks for feat in feat_names[mask]])
        freq = {
            f: sum(mask[i] for mask in masks) / n_folds
            for i, f in enumerate(feat_names)
        }
        if self.verbose:
            logger.info("Extracted selected features and frequency across folds.")
        return selected_per_fold, selected_all, freq

    def feature_selection(
        self,
        model_name: str,
        n_features: Optional[int] = None,
        direction: str = 'forward',
        scoring: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform feature selection via sequential feature selection (SFS) with cross-validation.

        This method selects and returns the most informative features along with their importance values
        and selection frequencies using an inner cross-validation loop for feature selection and an outer
        cross-validation loop for evaluation. The updated logic now returns the selection direction and the
        corresponding feature names (values) along with detailed importance statistics.

        The process follows these steps:
          1. Build a pipeline with SequentialFeatureSelector and a base estimator using _build_sfs_pipeline.
          2. Perform outer cross-validation to evaluate feature selection performance.
          3. Extract feature selection masks and selected feature names from CV estimators via _extract_sfs_selected_features.
          4. Aggregate results to determine stable features (those selected in the majority of folds),
             compute selection frequency, and compile detailed importance statistics.
          5. Get weighted importances as the product of the mean importance and the selection frequency.

        Parameters
        ----------
        model_name : str
            Name of the model in model_configs to use for feature selection.
        n_features : int, optional
            Number of features to select. If None, uses half of the available features.
        direction : str, default='forward'
            Direction for sequential feature selection ('forward' or 'backward'):
              - 'forward': Start with no features and add one at a time.
              - 'backward': Start with all features and remove one at a time.
        scoring : str, optional
            Metric to use for evaluating feature selection. If None, uses the first metric in self.metrics.

        Returns
        -------
        dict
            Dictionary containing:
            model_name : str
                Name of the model used for feature selection.
            metric_scores : dict
                Scoring metrics aggregated across folds with keys:
                - 'mean': Mean score across folds.
                - 'std': Standard deviation of scores.
                - 'fold_scores': Scores per fold.
            selected_features : set
                The combined set of features selected across all CV folds.
            feature_frequency : dict
                Mapping of each feature name to its selection frequency (a float between 0 and 1).
            feature_importances : dict
                Feature importance statistics with keys:
                - 'mean': Mean importance across folds.
                - 'weighted_mean': Mean importance weighted by selection frequency.
                - 'std': Standard deviation of importances.
                - 'weighted_std': Standard deviation of importances weighted by selection frequency.
                - 'fold_importances': Importances per fold.
            predictions : dict
                Dictionary containing concatenated y_true, y_pred, and optionally y_proba from cross-validation
                and fold-level predictions.
            selected_per_fold : dict
                Dictionary mapping fold indices to lists of feature names selected in each fold.
            best_fold : dict
                Information about the best-performing fold, including:
                'fold': Index of the best fold.
                'features': Features selected in that fold.
                <metric>: Metric score for that fold.
                'estimator': The fitted estimator for the best fold.
            folds_estimators : list
                List of fitted estimator instances from each fold of cross-validation.
            fs parameters : dict
                Parameters used for feature selection, including:
                - 'n_features': Number of features selected.
                - 'direction': Direction of feature selection ('forward' or 'backward').
                - 'scoring': Metric used for feature selection.


        See Also
        --------
        _build_sfs_pipeline : Constructs the SFS pipeline.
        _extract_sfs_selected_features : Extracts selected feature masks and names from CV estimators.
        _aggregate : Aggregates predictions, scores, and importances from fold-level results.
        cross_val : Performs outer cross-validation.

        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model_configs = {'rf': {'estimator': RandomForestClassifier()}}
        >>> result = pipeline.feature_selection('rf', n_features=5, direction='forward', scoring='accuracy')
        >>> print(f"Selected features: {result['selected_features']}")
        >>> print("Weighted importances:")
        >>> for feat, imp in sorted(result['weighted_importances'].items(), key=lambda x: x[1], reverse=True)[:5]:
        ...     print(f"  {feat}: {imp:.4f}")
        """
        if self.verbose:
            logger.info("Starting feature selection for model '%s'...", model_name)

        pipe, feat_names, metric = self._build_sfs_pipeline(
            model_name, n_features, direction, scoring
        )

        cv_res = self.cross_val(pipe, self.X, self.y)
        selected_per_fold, selected_all, freq = self._extract_sfs_selected_features(cv_res["cv_fold_estimators"], feat_names)
        predictions, metric_scores, feature_importances = self._aggregate(
            cv_res["cv_fold_predictions"],
            cv_res["cv_fold_scores"],
            cv_res["cv_fold_importances"],
            freq=freq
        )
        
        best_fold_idx = np.argmax(metric_scores[metric]['mean'])
        best_fold = {
            'fold': best_fold_idx,
            'features': selected_per_fold[best_fold_idx],
            metric: metric_scores[metric]["fold_scores"][best_fold_idx],
            "estimator": cv_res['cv_fold_estimators'][best_fold_idx]
        }
        if self.verbose:
            logger.info("Feature selection results: Selected features %s", selected_all)
        return {
            'model_name': model_name,
            'metric_scores': metric_scores,
            'selected_features': selected_all,
            'feature_frequency': freq,
            'feature_importances': feature_importances,
            'predictions': predictions,
            'selected_per_fold': selected_per_fold,
            'best_fold': best_fold,
            'folds_estimators': cv_res['cv_fold_estimators'],
            'fs parameters': {
                'n_features': n_features,
                'direction': direction,
                'scoring': metric
            },
        }

    def _build_search_estimator(
        self,
        model_name: str,
        search_type: str,
        param_grid: Optional[Dict[str, Any]],
        n_iter: int,
        scoring: str
    ) -> Tuple[Union[GridSearchCV, RandomizedSearchCV], str]:
        """
        Build a search estimator for hyperparameter tuning.

        This helper method constructs either a GridSearchCV or RandomizedSearchCV 
        estimator for hyperparameter tuning based on the specified search type.

        Parameters
        ----------
        model_name : str
            Name of the model in model_configs to use as the base estimator.
        search_type : str
            Type of search to perform, either 'grid' or 'random'.
        param_grid : dict or None
            Dictionary with parameter names as keys and lists of parameter values.
            If None, uses the 'params' from model_configs.
        n_iter : int
            Number of parameter settings sampled in RandomizedSearchCV.
            Ignored for GridSearchCV.
        scoring : str
            Metric name to use for evaluating hyperparameter configurations.
            Must be a key in self.metric_funcs.

        Returns
        -------
        search_est : GridSearchCV or RandomizedSearchCV
            Configured search estimator ready for cross-validation.
        metric : str
            The metric name that will be used for search scoring.
        """
        metric = scoring or self.metrics[0]
        scorer = make_scorer(self.metric_funcs[metric])
        mc = self.model_configs[model_name]
        grid = param_grid or mc.param_grid
        base_est = clone(mc.original_estimator).set_params(**mc.init_params)
        inner_cv = get_cv_splitter(**self.cv_kwargs)

        if hasattr(base_est, 'random_state') and self.random_state is not None:
            base_est.random_state = self.random_state
        
        if search_type.lower() == 'grid':
            search_est = GridSearchCV(
                base_est,
                grid,
                scoring=scorer,
                cv=inner_cv,
                n_jobs=self.n_jobs,
                refit=True
            )
        else:
            search_est = RandomizedSearchCV(
                base_est,
                grid,
                n_iter=n_iter,
                scoring=scorer,
                cv=inner_cv,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                refit=True
            )

        if self.verbose:
            logger.info("Built %s search estimator for model '%s'.", search_type, model_name)
        return search_est, metric

    def _extract_hp_search_params(
        self,
        cv_fold_estimators: List,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        best_params_per_fold = {
            i: est.best_params_
            for i, est in enumerate(cv_fold_estimators)
        }
        n_folds = len(best_params_per_fold)
        param_counts = {}
        for params in best_params_per_fold.values():
            for k, v in params.items():
                v_hashable = tuple(v) if isinstance(v, list) else v
                param_counts.setdefault(k, {}).setdefault(v_hashable, 0)
                param_counts[k][v_hashable] += 1
        
        best_params = {}
        for k, val_counts in param_counts.items():
            best_val, _ = max(val_counts.items(), key=lambda item: item[1])
            best_params[k] = list(best_val) if isinstance(best_val, tuple) else best_val
        
        param_frequency = {
            k: {(list(v) if isinstance(v, tuple) else v): cnt / n_folds 
                for v, cnt in vals.items()}
            for k, vals in param_counts.items()
        }
        if self.verbose:
            logger.info("Extracted hyperparameter search parameters from folds.")
        return best_params_per_fold, best_params, param_frequency

    def hp_search(
        self,
        model_name: str,
        search_type: str = 'grid',
        param_grid: Optional[Dict[str, Any]] = None,
        n_iter: int = 50,
        scoring: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter search using nested cross-validation.
        
        This method performs hyperparameter tuning using an inner cross-validation loop for
        parameter selection and an outer cross-validation loop for model evaluation. It
        which collectively assess parameter stability across folds and identify the most robust parameter
        settings via majority voting.
        
        The process follows these steps:
        1. Build a search estimator (GridSearchCV or RandomizedSearchCV) using _build_search_estimator.
        2. Execute outer cross-validation via cross_val to evaluate hyperparameter search performance.
        3. Extract best parameters per fold and their frequencies.
        4. Aggregate predictions, metric scores, and feature importances.
        5. Identify the best-performing fold based on the defined metric.
        
        Parameters
        ----------
        model_name : str
            Name of the model in model_configs to tune.
        search_type : str, default='grid'
            Type of search to perform, either 'grid' or 'random'.
        param_grid : dict, optional
            Dictionary with parameter names as keys and lists of parameter values.
            If None, uses the 'params' from model_configs.
        n_iter : int, default=50
            Number of parameter settings sampled in RandomizedSearchCV.
            Ignored for GridSearchCV.
        scoring : str, optional
            Metric to use for hyperparameter selection. If None, uses the first metric in self.metrics.
            
        Returns
        -------
        dict
            Dictionary containing:
            
            model_name : str
                Name of the model that was tuned.
            metric_scores : dict
                Dictionary mapping metric names to cross-validation scores; each entry contains the mean, 
                standard deviation, and per-fold scores.
            feature_importances : dict
                Feature importance statistics aggregated across CV folds (if available).
            best_params : dict
                Aggregated best parameter settings determined by majority voting across folds.
            param_frequency : dict
                Frequency mapping of each hyperparameter value across CV folds.
            predictions : dict
                Aggregated predictions from all folds including y_true, y_pred, and optionally y_proba.
            best_params_per_fold : dict
                Per-fold best parameter configurations.
            best_fold : dict
                Information for the best-performing fold including its index, best parameters,
                metric score, and the fitted estimator.
            folds_estimators : list
                List of fitted search estimator instances from each CV fold.
            hp search parameters : dict
                Meta-information on the hyperparameter search containing:
                    - search type: The type of search performed ('grid' or 'random').
                    - param grid: The parameter grid provided (or from model_configs if None).
                    - scoring: The metric used for evaluation.
                    - n_iter: The number of parameter settings sampled (for randomized search).
                
        Notes
        -----
        Parameter stability is assessed by analyzing selection frequency across folds. The final parameter 
        setting is obtained by majority voting over per-fold best parameters, providing more robust 
        hyperparameter tuning than a single GridSearchCV or RandomizedSearchCV run.
        
        The best fold information can be used to inspect which specific parameter combination achieved the highest
        performance in cross-validation.
        
        See Also
        --------
        _build_search_estimator : Builds the search estimator.
        _extract_hp_search_params : Extracts best parameter settings from each CV fold.
        _aggregate : Aggregates predictions, metric scores, and feature importances across folds.
        cross_val : Performs cross-validation.
        
        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model_configs = {'rf': {'estimator': RandomForestClassifier(),
        ...                         'params': {'max_depth': [3, 5, None]}}}
        >>> result = pipeline.hp_search('rf', search_type='grid')
        >>> print(f"Best parameters: {result['best_params']}")
        >>> print(f"Parameter frequency: {result['param_frequency']}")
        """
        if self.verbose:
            logger.info("Starting hyperparameter search for model '%s'...", model_name)

        search_est, metric = self._build_search_estimator(
            model_name, search_type, param_grid, n_iter, scoring
        )

        cv_res = self.cross_val(search_est, self.X, self.y)

        best_params_per_fold, best_params, param_frequency = self._extract_hp_search_params(cv_res['cv_fold_estimators'])

        predictions, metric_scores, feature_importances = self._aggregate(
            cv_res["cv_fold_predictions"],
            cv_res["cv_fold_scores"],
            cv_res["cv_fold_importances"],
            freq=None
        )
        
        best_fold_idx = np.argmax(metric_scores[metric]['mean'])
        best_fold = {
            'fold': best_fold_idx,
            'params': best_params_per_fold[best_fold_idx],
            metric: metric_scores[metric]["fold_scores"][best_fold_idx],
            "estimator": cv_res['cv_fold_estimators'][best_fold_idx]
        }
        if self.verbose:
            logger.info("Hyperparameter search results: Best parameters %s", best_params)
        return {
            'model_name': model_name,
            'metric_scores': metric_scores,
            'feature_importances': feature_importances,
            'best_params': best_params,
            'param_frequency': param_frequency,
            'predictions': predictions,
            'best_params_per_fold': best_params_per_fold,
            'best_fold': best_fold,
            'folds_estimators': cv_res['cv_fold_estimators'],
            'hp search parameters': {
                'search type': search_type,
                'param grid': param_grid,
                'scoring': metric,
                'n_iter': n_iter
            },
        }

    def _build_combined_fs_hp_pipeline(
        self,
        model_name: str,
        search_type: str,
        param_grid: Optional[Dict[str, Any]],
        n_features: int,
        direction: str,
        n_iter: int,
        scoring: str
    ) -> Tuple[Union[GridSearchCV, RandomizedSearchCV], np.ndarray, str]:
        """
        Build a pipeline for combined feature selection and hyperparameter search.

        This helper method constructs a pipeline that performs sequential feature selection
        followed by model fitting, then wraps it in a hyperparameter search estimator.

        Parameters
        ----------
        model_name : str
            Name of the model in model_configs to use as the base estimator.
        search_type : str
            Type of search to perform, either 'grid' or 'random'.
        param_grid : dict or None
            Dictionary with parameter names as keys and lists of parameter values.
            If None, uses the 'params' from model_configs.
        n_features : int
            Number of features to select via SequentialFeatureSelector.
        direction : str
            Direction for feature selection, either 'forward' or 'backward'.
        n_iter : int
            Number of parameter settings sampled in RandomizedSearchCV.
            Ignored for GridSearchCV.
        scoring : str
            Metric name to use for evaluating feature selection and hyperparameters.

        Returns
        -------
        search_est : GridSearchCV or RandomizedSearchCV
            Configured search estimator with nested feature selection pipeline.
        feat_names : ndarray
            Array of feature names from the input data.
        metric : str
            The metric name that will be used for scoring.
        """
        # Set up defaults and scoring
        metric = scoring or self.metrics[0]
        scorer = make_scorer(self.metric_funcs[metric])
        mc = self.model_configs[model_name]
        raw_grid = param_grid or mc.param_grid
        inner_cv_fs = get_cv_splitter(**self.cv_kwargs)
        inner_cv_hp = get_cv_splitter(**self.cv_kwargs)
        
        base_est = clone(mc.get('estimator'))
        if hasattr(base_est, 'random_state') and self.random_state is not None:
            base_est.random_state = self.random_state
        
        fs_pipe = Pipeline([
            ('sfs', SequentialFeatureSelector(
                clone(base_est),
                n_features_to_select=n_features,
                direction=direction,
                scoring=scorer,
                cv=inner_cv_fs,
                n_jobs=self.n_jobs
            )),
            ('clf', clone(base_est))
        ])
        grid = {f'clf__{param}': values for param, values in raw_grid.items()}
        if search_type.lower() == 'grid':
            search_est = GridSearchCV(
                fs_pipe,
                grid,
                scoring=scorer,
                cv=inner_cv_hp,
                n_jobs=self.n_jobs,
                return_train_score=True,
                refit=True
            )
        else:
            search_est = RandomizedSearchCV(
                fs_pipe,
                grid,
                n_iter=n_iter,
                scoring=scorer,
                cv=inner_cv_hp,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                return_train_score=True,
                refit=True
            )
        feat_names = np.array(self._get_feature_names(self.X))
        if self.verbose:
            logger.info("Built combined FS and HP search pipeline for model '%s'.", model_name)
        return search_est, feat_names, metric            

    def hp_search_fs(
        self,
        model_name: str,
        search_type: str = 'grid',
        param_grid: Optional[Dict[str, Any]] = None,
        n_features: Optional[int] = None,
        direction: str = 'forward',
        n_iter: int = 50,
        scoring: Optional[str] = None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Perform combined feature selection and hyperparameter tuning with nested cross-validation.

        This method integrates sequential feature selection and hyperparameter optimization
        into a single nested cross-validation framework. The inner CV loops optimize both
        the feature subset and the hyperparameter configuration, while the outer CV loop evaluates
        the overall performance on held-out data. Specifically, the pipeline is constructed with
        a SequentialFeatureSelector followed by an estimator, which is then wrapped in a hyperparameter
        search estimator (either GridSearchCV or RandomizedSearchCV). After performing outer CV,
        the function extracts the selected features and best hyperparameters from each fold and aggregates
        the results to identify stable features and parameters.

        Parameters
        ----------
        model_name : str
            Name of the model in model_configs to use.
        search_type : str, default='grid'
            Type of hyperparameter search to perform, either 'grid' or 'random'.
        param_grid : dict, optional
            Dictionary with parameter names as keys and lists of parameter values.
            If None, uses the 'params' from model_configs.
        n_features : int, optional
            Number of features to select. If None, defaults to half of the available features.
        direction : str, default='forward'
            Direction for sequential feature selection ('forward' or 'backward').
        n_iter : int, default=50
            Number of parameter settings sampled in RandomizedSearchCV. Ignored for GridSearchCV.
        scoring : str, optional
            Metric to use for evaluation. If None, uses the first metric in self.metrics.
        X : DataFrame or ndarray, optional
            Feature matrix. If provided, overrides self.X.
        y : Series or ndarray, optional
            Target vector. If provided, overrides self.y.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - model_name : str
              Name of the model used.
            - metric_scores : dict
              Aggregated scoring metrics from cross-validation with keys:
              'mean', 'std', and 'fold_scores' (list of scores per fold).
            - selected_features : set
              The union of features selected across all outer CV folds (stable features).
            - feature_frequency : dict
              Mapping of each feature name to its selection frequency across folds (0.0 to 1.0).
            - feature_importances : dict
              Feature importance statistics with keys for each feature including:
                  'mean', 'std', and 'fold_importances'.
            - best_params : dict
              Aggregated best hyperparameters selected via majority voting across folds.
            - param_frequency : dict
              Frequency mapping of hyperparameter values across folds.
            - predictions : dict
              Aggregated predictions from all folds, including 'y_true', 'y_pred', and optionally 'y_proba'.
            - selected_per_fold : dict
              Mapping of each fold index to the list of features selected in that fold.
            - best_params_per_fold : dict
              Per-fold best hyperparameter configurations.
            - best_fold : dict
              Information about the best-performing fold, including:
              'fold' (index), 'features' (features selected in that fold), 'params' (hyperparameters),
              the metric score for that fold, and the corresponding fitted estimator.
            - folds_estimators : list
              List of fitted search estimator instances from each outer CV fold.
            - hp search and fs parameters : dict
              Metadata for the combined search including:
                  'n_features' : number of features targeted for selection,
                  'direction'  : direction used in feature selection,
                  'search type': type of hyperparameter search ('grid' or 'random'),
                  'param grid' : the parameter grid used,
                  'n_iter'     : number of iterations for randomized search,
                  'scoring'    : metric used for evaluation.

        Notes
        -----
        This combined approach provides a comprehensive assessment of both feature subset stability
        and hyperparameter robustness by leveraging nested cross-validation. It yields detailed
        fold-level insights along with aggregated statistics for model performance, making it easier
        to understand which features and parameter settings are consistently beneficial.

        See Also
        --------
        _build_combined_fs_hp_pipeline : Constructs the combined pipeline.
        _extract_sfs_selected_features : Extracts the selected features from cross-validation estimators.
        _extract_hp_search_params : Extracts best hyperparameter configurations from each fold.
        _aggregate : Aggregates predictions, scores, and feature importances across folds.
        feature_selection : Performs feature selection only.
        hp_search : Performs hyperparameter tuning only.

        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model_configs = {
        ...     'rf': {
        ...         'estimator': RandomForestClassifier(),
        ...         'params': {'max_depth': [3, 5, None], 'min_samples_split': [2, 4, 6]}
        ...     }
        ... }
        >>> result = pipeline.hp_search_fs('rf', search_type='grid', n_features=5, direction='forward')
        >>> print(f"Stable selected features: {result['selected_features']}")
        >>> print(f"Aggregated best parameters: {result['best_params']}")
        """
        if self.verbose:
            logger.info("Starting combined feature selection and HP search for model '%s'...", model_name)
        # Set default n_features if not provided
        n_sel = n_features or (self.X.shape[1] // 2)

        search_est, feat_names, metric = self._build_combined_fs_hp_pipeline(
            model_name, search_type, param_grid, n_sel, direction, n_iter, scoring
        )
        
        cv_res = self.cross_val(search_est, self.X, self.y)
        estimators = [est.best_estimator_ for est in cv_res["cv_fold_estimators"]]
        selected_per_fold, selected_all, freq = self._extract_sfs_selected_features(estimators, feat_names)
        
        best_params_per_fold, best_params, param_frequency = self._extract_hp_search_params(cv_res["cv_fold_estimators"])
        
        predictions, metric_scores, feature_importances = self._aggregate(
            cv_res["cv_fold_predictions"],
            cv_res["cv_fold_scores"],
            cv_res["cv_fold_importances"],
            freq=freq
        )
        best_fold_idx = np.argmax(metric_scores[metric]['mean'])
        best_fold = {
            'fold': best_fold_idx,
            'features': selected_per_fold[best_fold_idx],
            'params': best_params_per_fold[best_fold_idx],
            metric: metric_scores[metric]["fold_scores"][best_fold_idx],
            "estimator": cv_res['cv_fold_estimators'][best_fold_idx]
        }
        if self.verbose:
            logger.info("Combined FS and HP search results: Selected features %s and best parameters %s", selected_all, best_params)
        return {
            'model_name': model_name,
            'metric_scores': metric_scores,
            'selected_features': selected_all,
            'feature_frequency': freq,
            'feature_importances': feature_importances,
            'best_params': best_params,
            'param_frequency': param_frequency,
            'predictions': predictions,
            'selected_per_fold': selected_per_fold,
            'best_params_per_fold': best_params_per_fold,
            'best_fold': best_fold,
            'folds_estimators': cv_res['cv_fold_estimators'],
            'hp search and fs parameters': {
                'n_features': n_features,
                'direction': direction,
                'search type': search_type,
                'param grid': param_grid,
                'n_iter': n_iter,
                'scoring': metric,
            },
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Dispatch execution to the appropriate pipeline method based on type parameter.
        
        This method serves as a unified entry point for running different analysis methods
        of the pipeline. It dynamically dispatches the call to one of the primary pipeline 
        methods based on the 'type' parameter provided in kwargs.
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass to the target method. Must include a 'type' parameter
            that specifies which pipeline method to execute. The remaining parameters are
            passed directly to the selected method.
            
            Special parameters:
            - type : {'baseline', 'feature_selection', 'hp_search', 'hp_search_fs'}, default='baseline'
                The type of analysis to perform:
                * 'baseline': Run baseline model evaluation
                * 'feature_selection': Perform sequential feature selection
                * 'hp_search': Perform hyperparameter search
                * 'hp_search_fs': Perform combined feature selection and hyperparameter search
                
        Returns
        -------
        dict
            The results dictionary from the called method. Contents vary depending on the
            specific method called, but typically include:
            
            For baseline:
                Model performance metrics, predictions, and feature importances
            
            For feature_selection:
                Selected features, feature frequencies, feature importances, and fold results
                
            For hp_search:
                Best parameters, parameter frequencies, and fold results
                
            For hp_search_fs:
                Selected features, best parameters, and combined analysis results
        
        Raises
        ------
        ValueError
            If the specified 'type' is not one of the supported methods.
        
        See Also
        --------
        baseline_evaluation : Run baseline model evaluation
        feature_selection : Perform sequential feature selection
        hp_search : Perform hyperparameter search
        hp_search_fs : Perform combined feature selection and hyperparameter search
        
        Examples
        --------
        >>> # Run baseline evaluation
        >>> results = pipeline.execute(type='baseline', model_name='random_forest')
        >>> 
        >>> # Run feature selection
        >>> results = pipeline.execute(
        ...     type='feature_selection', 
        ...     model_name='random_forest',
        ...     n_features=5
        ... )
        >>> 
        >>> # Run hyperparameter search
        >>> results = pipeline.execute(
        ...     type='hp_search',
        ...     model_name='random_forest',
        ...     search_type='grid'
        ... )
        """
        # Extract execution type from kwargs
        method_name = kwargs.pop('type', 'baseline')
        valid_methods = ['baseline', 'feature_selection', 'hp_search', 'hp_search_fs']
        if method_name not in valid_methods:
            raise ValueError(
                f"Invalid execution type '{method_name}'. "
                f"Must be one of: {', '.join(valid_methods)}"
            )
        method_map = {
            'baseline': 'baseline_evaluation',
            'feature_selection': 'feature_selection',
            'hp_search': 'hp_search',
            'hp_search_fs': 'hp_search_fs'
        }
        if self.verbose:
            logger.info("Executing method '%s' with parameters: %s", method_name, kwargs)
        try:
            method = getattr(self, method_map[method_name])
            results = method(**kwargs)
            if self.verbose:
                logger.info("Execution results: %s", results.get('metric_scores', {}))
            return results
        except Exception as e:
            logger.error(f"Error during {method_name} execution: {str(e)}")
            raise