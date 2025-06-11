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
        columns: Optional[Sequence[str]] = None,
        default_metrics: Optional[Sequence[str]] = None,
        cv_kwargs: Dict[str, Any] = DEFAULT_CV,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
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
        self.cv_kwargs = deepcopy(cv_kwargs)  # shallow copy
        self.groups = groups
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.feature_names = columns or self._get_feature_names(X)

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

        # 2) Linear models
        if hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            # single‐output
            if coef.ndim == 1:
                return coef
            # multiclass: average over classes
            return np.mean(coef, axis=0)

        # 3) Pipeline: unwrap last step
        if hasattr(estimator, 'named_steps'):
            last = list(estimator.named_steps.values())[-1]
            return BasePipeline._extract_feature_importances(last)

        # 4) SequentialFeatureSelector: use its underlying estimator_
        if hasattr(estimator, 'estimator_'):
            return BasePipeline._extract_feature_importances(estimator.estimator_)

        # 5) GridSearchCV or RandomizedSearchCV: use best_estimator_
        if hasattr(estimator, 'best_estimator_'):
            return BasePipeline._extract_feature_importances(estimator.best_estimator_)

        # 5) Voting or other meta-estimators: average sub-estimators
        if hasattr(estimator, 'estimators_'):
            imps = []
            for sub in estimator.estimators_:
                imp = BasePipeline._extract_feature_importances(sub)
                if imp is not None:
                    imps.append(imp)
            if imps:
                return np.mean(imps, axis=0)

        # Nothing found
        return None

    def _aggregate(self, fold_preds, fold_scores, fold_importances, freq=None):
        """Aggregate predictions and compute metrics across folds."""
        # Concatenate all predictions for the predictions output
        y_true = np.concatenate([fp["y_true"] for fp in fold_preds])
        y_pred = np.concatenate([fp["y_pred"] for fp in fold_preds])
        
        predictions = {"y_true": y_true, "y_pred": y_pred}
        
        # Add probabilities if available - handle shape mismatches
        if all("y_proba" in fp for fp in fold_preds):
            try:
                # Check if all probability arrays have the same shape
                shapes = [fp["y_proba"].shape for fp in fold_preds]
                if all(s[1] == shapes[0][1] for s in shapes):
                    # All have same number of columns, safe to concatenate
                    y_proba = np.vstack([fp["y_proba"] for fp in fold_preds])
                    predictions["y_proba"] = y_proba
                else:
                    # Shape mismatch - skip probability concatenation
                    pass
            except Exception:
                # If concatenation fails for any reason, skip probabilities
                pass
        
        # add fold predictions to predictions
        predictions["fold_preds"] = fold_preds
        # Compute metrics per fold, then average
        metrics = {}
        for metric_name in self.metrics:
            metric_func = self.metric_funcs[metric_name]
            # fold_scores = []
            fold_score = fold_scores[metric_name]
            # for fs in fold_scores[metric_name]:

                # DONT know why I am using this but lets skip it for now and use scores
                # try:
                #     # Handle probability-based metrics
                #     if metric_name in ["roc_auc", "average_precision"] and "y_proba" in fp:
                #         if fp["y_proba"].ndim == 2 and fp["y_proba"].shape[1] >= 2:
                #             # Use positive class probabilities for binary classification
                #             score = metric_func(fp["y_true"], fp["y_proba"][:, 1])
                #         else:
                #             score = metric_func(fp["y_true"], fp["y_proba"])
                #     else:
                #         # Use predictions for other metrics
                #         score = metric_func(fp["y_true"], fp["y_pred"])
                #     fold_scores.append(score)
                # except Exception as e:
                #     # Skip this fold if metric calculation fails
                #     continue
            
            # if fold_score:
            metrics[metric_name] = {
                "mean": np.mean(fold_score),
                "std": np.std(fold_score),
                "fold_scores": fold_score
            }

        # Aggregate feature importances if available now it is a dict feat_name: array of importances!
        # get mean std of each feature importance and also keep fold importances
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

        # Choose target
        if param_type not in ['default', 'hp_search']:
            raise ValueError("param_type must be 'default' or 'hp_search'")

        # 1) update estimator defaults
        if update_estimator and param_type == 'default':
            temp = clone(mc.original_estimator)
            temp.set_params(**params)  # will raise if invalid
            mc.init_params.update(params)
            mc.estimator = temp
            logger.info(f"Updated init parameters for '{model_name}': {params}")

        # 2) update config grid
        if update_config and param_type == 'hp_search':
            # validate grid values are list-like
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

    def cross_val(
        self,
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Perform cross-validation using scikit-learn's `cross_validate`.

        Wraps `metric_funcs` into `scoring`, collects per-fold predictions,
        scores, estimators, and feature importances.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator to evaluate.
        X : DataFrame or ndarray
            Feature matrix.
        y : Series or ndarray
            Target vector or matrix.

        Returns
        -------
        results : dict
            - 'cv_fold_scores': dict mapping metric names to arrays of fold scores,
                array shape (n_splits,).
            - 'cv_fold_importances': dict mapping feature names to arrays of importances
                (one array per feature), or empty dict if unsupported,
                array shape (n_splits,).
            - 'cv_fold_predictions': list of dicts, each with keys:
                'y_true', 'y_pred', optional 'y_proba' if estimator supports it.
            - 'cv_fold_estimators': list of fitted estimators from each fold.
        Raises
        ------
        ValueError
            If X and y have different numbers of samples or if no metrics are specified.
        Notes
        -----
        This method performs cross-validation on the provided estimator using
        the configured cross-validation strategy. It collects per-fold predictions,
        scores, and feature importances.
        """

        cv_conf = deepcopy(self.cv_kwargs)
        cv_conf.setdefault('random_state', self.random_state)
        cv = get_cv_splitter(**cv_conf)

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
        groups_arr = self.groups.values if isinstance(self.groups, pd.Series) else self.groups

        # build scoring dictionary
        proba_required = {m for m, fn in self.metric_funcs.items() if hasattr(fn, '__name__') and 'proba' in fn.__name__}
        scoring = {
            m: make_scorer(
            self.metric_funcs[m],
            **({'needs_proba': True} if m in proba_required else {})
            )
            for m in self.metrics
        }

        # run CV
        cv_results = cross_validate(
            estimator=estimator,
            X=X_arr, y=y_arr, groups=groups_arr,
            scoring=scoring,
            cv=cv,
            n_jobs=self.n_jobs,
            return_estimator=True,
            return_train_score=False,
            error_score='raise'
        )

        # retrieve splits
        splits = list(cv.split(X_arr, y_arr, groups_arr))

        # collect fold-level predictions
        fold_predictions = []
        for idx, est_fold in enumerate(cv_results['estimator']):
            _, val_idx = splits[idx]
            y_true, y_pred = y_arr[val_idx], est_fold.predict(X_arr[val_idx])
            fp = {'y_true': y_true, 'y_pred': y_pred}
            if hasattr(est_fold, 'predict_proba'):
                try:
                    fp['y_proba'] = est_fold.predict_proba(X_arr[val_idx])
                except Exception:
                    logger.warning(
                        "predict_proba failed for fold %d with %s",
                        idx, type(est_fold).__name__
                    )
            fold_predictions.append(fp)

        # extract feature importances for each fold
        fold_importances = [
            self._extract_feature_importances(est_fold) for est_fold in cv_results['estimator']
        ]
        
        # per-metric arrays
        cv_fold_scores = {m: np.array(cv_results.get(f'test_{m}', [])) for m in self.metrics}
        # per feature importances
        cv_fold_importances = {}


        for i, feature in enumerate(self.feature_names):
            # Only collect importances where the index is within bounds
            values = []
            for imp in fold_importances:
                if imp is not None and i < len(imp):
                    values.append(imp[i])
    
            # Only add features that have at least one valid importance value
            if values:
                cv_fold_importances[feature] = np.array(values)
        return {
            'cv_fold_scores': cv_fold_scores,
            'cv_fold_importances': cv_fold_importances,
            'cv_fold_predictions': fold_predictions,
            'cv_fold_estimators': cv_results['estimator'],
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
        if model_name not in self.model_configs:
            raise KeyError(f"Model '{model_name}' not found in model_configs")

        mc = self.model_configs[model_name]
        clf = clone(mc.original_estimator).set_params(**mc.init_params)

        results_ = self.cross_val(clf, self.X, self.y)
        predictions, metric_scores, feature_importances = self._aggregate(results_["cv_fold_predictions"], 
                                                    results_["cv_fold_scores"],
                                                    results_["cv_fold_importances"])
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
        # Flatten all selected features from each fold into a set
        selected_all = set([feat for mask in masks for feat in feat_names[mask]])
        # Compute frequency as the number of folds in which each feature was selected divided by total folds
        freq = {
            f: sum(mask[i] for mask in masks) / n_folds
            for i, f in enumerate(feat_names)
        }
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
        pipe, feat_names, metric = self._build_sfs_pipeline(
            model_name, n_features, direction, scoring
        )

        cv_res = self.cross_val(pipe, self.X, self.y)

        selected_per_fold, selected_all, freq = self._extract_sfs_selected_features(cv_res["cv_fold_estimators"], feat_names)

        predictions, metric_scores, feature_importances = self._aggregate(cv_res["cv_fold_predictions"],
                                                    cv_res["cv_fold_scores"],
                                                    cv_res["cv_fold_importances"],
                                                    freq=freq)
        
        # get best fold and its selected features
        best_fold_idx = np.argmax(metric_scores[metric]['mean'])
        best_fold = {
            'fold': best_fold_idx,
            'features': selected_per_fold[best_fold_idx],
            metric: metric_scores[metric]["fold_scores"][best_fold_idx],
            "estimator": cv_res['cv_fold_estimators'][best_fold_idx]
        }

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
        
        # Build appropriate search estimator
        if search_type.lower() == 'grid':
            search_est = GridSearchCV(
                base_est,
                grid,
                scoring=scorer,
                cv=inner_cv,
                n_jobs=self.n_jobs,
                refit=True
            )
        else:  # randomized search
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

        return search_est, metric

    def _extract_hp_search_params(
        self,
        cv_fold_estimators: List,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract hyperparameter search selected parameters and their selection frequency from cross-validation.

        This helper method processes the best parameters from each fold to determine
        parameter selection frequency and identify stable parameters (selected in
        majority of folds).
        
        Parameters
        ----------
        cv_fold_estimators : list
            List of fitted search estimators from each fold of cross-validation.
            Each estimator should have a 'best_params_' attribute containing the best parameters
            selected for that fold.
            
        Returns
        -------
        best_params_per_fold : dict
            Dictionary mapping fold indices to the best parameters selected in that fold.
            Example: {0: {'max_depth': 5, 'min_samples_split': 2}, ...}
        best_params : dict
            Aggregated best parameters selected across all folds by majority voting.
            Example: {'max_depth': 5, 'min_samples_split': 2}
        param_frequency : dict
            Dictionary mapping parameter names to dictionaries of value frequencies.
            Example: {'max_depth': {3: 0.6, 5: 0.4}, 'min_samples_split': {2: 1.0}}                    
        """
        best_params_per_fold = {
            i: est.best_params_
            for i, est in enumerate(cv_fold_estimators)
        }

        n_folds = len(best_params_per_fold)

        # Count parameter value occurrences across folds
        param_counts = {}
        for params in best_params_per_fold.values():
            for k, v in params.items():
                # Handle various parameter types (including non-hashable ones like lists)
                v_hashable = tuple(v) if isinstance(v, list) else v
                param_counts.setdefault(k, {}).setdefault(v_hashable, 0)
                param_counts[k][v_hashable] += 1
        
        # Select best parameters by majority vote
        best_params = {}
        for k, val_counts in param_counts.items():
            # Find the value with the highest count
            best_val, _ = max(val_counts.items(), key=lambda item: item[1])
            # Convert tuple back to list if needed
            best_params[k] = list(best_val) if isinstance(best_val, tuple) else best_val
        
        # Compute frequency of each parameter value
        param_frequency = {
            k: {(list(v) if isinstance(v, tuple) else v): cnt / n_folds 
                for v, cnt in vals.items()}
            for k, vals in param_counts.items()
        }
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
        search_est, metric = self._build_search_estimator(
            model_name, search_type, param_grid, n_iter, scoring
        )

        cv_res = self.cross_val(search_est, self.X, self.y)

        best_params_per_fold, best_params, param_frequency = self._extract_hp_search_params(cv_res['cv_fold_estimators'])

        predictions, metric_scores, feature_importances = self._aggregate(cv_res["cv_fold_predictions"],
                                                    cv_res["cv_fold_scores"],
                                                    cv_res["cv_fold_importances"],
                                                    freq=None)
        
        best_fold_idx = np.argmax(metric_scores[metric]['mean'])
        best_fold = {
            'fold': best_fold_idx,
            'params': best_params_per_fold[best_fold_idx],
            metric: metric_scores[metric]["fold_scores"][best_fold_idx],
            "estimator": cv_res['cv_fold_estimators'][best_fold_idx]
        }

        # Return comprehensive results
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

        # Get model config and parameter grid
        cfg = self.model_configs[model_name]
        raw_grid = param_grid or cfg.get('params', {})
        
        # Get inner CV splitters for both feature selection and hyperparameter tuning
        inner_cv_fs = get_cv_splitter(**self.cv_kwargs)
        inner_cv_hp = get_cv_splitter(**self.cv_kwargs)
        
        # Create base estimator
        base_est = clone(cfg.get('estimator'))
        if hasattr(base_est, 'random_state') and self.random_state is not None:
            base_est.random_state = self.random_state
        
        # Build feature selection pipeline
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
        # Wrap in hyperparameter search
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
        else:  # randomized search
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
        
        # Get feature names
        feat_names = np.array(self._get_feature_names(self.X))
        return search_est, feat_names, metric

    def _extract_combined_results(
        self,
        cv_res: Dict[str, Any],
        feat_names: np.ndarray,
        metric: str
    ) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        """
        Extract feature selection and hyperparameter results from cross-validation.
        
        This helper method processes cross-validation results from the combined feature
        selection and hyperparameter search pipeline to extract selected features,
        best parameters, and test scores for each fold.
        
        Parameters
        ----------
        cv_res : dict
            Cross-validation results from the cross_val method.
        feat_names : ndarray
            Array of feature names corresponding to the columns in the input data.
        metric : str
            Name of the metric used for evaluation.
            
        Returns
        -------
        outer_results : list of dict
            List of dictionaries, one per fold, with keys:
            - 'fold': Fold index
            - 'selected_features': List of feature names selected in this fold
            - 'best_params': Best parameters selected for this fold
            - 'test_scores': Dict of metric scores achieved on this fold
            - 'feature_importances': Feature importance values for this fold (if available)
        all_selected : list of str
            Combined list of all selected features across all folds (with duplicates).
        all_params : list of dict
            List of best parameter dictionaries, one per fold.
        """
        estimators = cv_res['cv_fold_estimators']
        scores = cv_res['cv_fold_scores']
        importances = cv_res.get('cv_fold_importances', {})
        
        outer_results, all_selected, all_params = [], [], []

        for fold_idx, est in enumerate(estimators):
            # Extract feature selection mask and selected features
            best_pipe = est.best_estimator_
            mask = best_pipe.named_steps['sfs'].get_support()
            selected = feat_names[mask].tolist()
            
            # Extract best parameters
            params = est.best_params_
            
            # Extract feature importances for this fold (if available)
            fold_importances = {
                feat: float(importances[feat][fold_idx])
                for feat in selected if feat in importances
            }
            
            # Collect selected features and parameters
            all_selected.extend(selected)
            all_params.append(params)
            
            # Extract all metrics for this fold
            test_scores = {m: float(scores[m][fold_idx]) for m in self.metrics}
            
            # Create detailed result entry for this fold
            outer_results.append({
                'fold': fold_idx,
                'selected_features': selected,
                'best_params': params,
                'test_scores': test_scores,
                'feature_importances': fold_importances
            })
        
        return outer_results, all_selected, all_params

    def _aggregate_combined_results(
        self,
        outer_results: List[Dict[str, Any]],
        all_selected: List[str],
        all_params: List[Dict[str, Any]],
        feat_names: np.ndarray,
        metric: str
    ) -> Tuple[List[str], Dict[str, float], Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, float]]:
        """
        Aggregate feature selection and hyperparameter results across folds.
        
        This helper method processes the selected features and best parameters from each fold
        to determine selection frequency, identify stable features and parameters, find
        the best-performing fold, and compile feature importance statistics.
        
        Parameters
        ----------
        outer_results : list of dict
            List of dictionaries with per-fold results.
        all_selected : list of str
            Combined list of all selected features across all folds (with duplicates).
        all_params : list of dict
            List of best parameter dictionaries, one per fold.
        feat_names : ndarray
            Array of feature names corresponding to the columns in the input data.
        metric : str
            Name of the metric to use for identifying the best fold.
            
        Returns
        -------
        selected_features : list of str
            Features selected in more than half of the cross-validation folds.
        feature_frequency : dict
            Dictionary mapping feature names to their selection frequency (0.0 to 1.0).
        best_params : dict
            Aggregated best parameters based on majority voting across folds.
        param_frequency : dict
            Dictionary mapping parameter names to dictionaries of value frequencies.
        best_fold : dict
            Information about the best-performing fold.
        feature_importances : dict
            Feature importance statistics with mean, std, and values for each feature.
        weighted_importances : dict
            Feature importances weighted by selection frequency.
        """
        n_folds = len(outer_results)
        
        # 1. Aggregate features by frequency
        from collections import Counter
        feat_count = Counter(all_selected)
        selected_features = [f for f, cnt in feat_count.items() if cnt > n_folds / 2]
        selected_features.sort(key=lambda f: feat_count[f], reverse=True)
        feature_frequency = {f: feat_count.get(f, 0) / n_folds for f in feat_names}
        
        # 2. Aggregate parameters by majority vote
        param_results = [{'best_params': params, 'test_scores': {metric: 0}} for params in all_params]
        best_params, param_frequency, _ = self._aggregate_hp_search_results(param_results, metric)
        
        # 3. Find the best-performing fold
        fold_scores = [res['test_scores'][metric] for res in outer_results]
        best_idx = int(np.argmax(fold_scores))
        best_fold = {
            'fold': best_idx,
            'features': outer_results[best_idx]['selected_features'],
            'params': outer_results[best_idx]['best_params'],
            'score': fold_scores[best_idx],
            'all_scores': outer_results[best_idx]['test_scores']
        }      

        # 4. Compile feature importance statistics - REUSING SFS METHOD
        # Convert feature importances to format expected by _compile_sfs_importances
        fold_imps = {}
        for feature in feat_names:
            values = []
            for result in outer_results:
                if feature in result.get('feature_importances', {}):
                    values.append(result['feature_importances'][feature])
            if values:
                fold_imps[feature] = np.array(values)
        
        # Use the existing _compile_sfs_importances method
        feature_importances, weighted_importances = self._compile_sfs_importances(
            fold_imps, feat_names, feature_frequency, selected_features
        )
        
        return selected_features, feature_frequency, best_params, param_frequency, best_fold, feature_importances, weighted_importances

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
        within a nested cross-validation framework. The inner cross-validation loops handle
        both feature selection and hyperparameter tuning, while the outer loop evaluates
        the combined approach on held-out data.
        
        The process follows these steps:
        1. Build a pipeline with SequentialFeatureSelector followed by the base estimator
        2. Wrap this pipeline in a hyperparameter search estimator (GridSearchCV or RandomizedSearchCV)
        3. Perform outer cross-validation to evaluate the combined approach
        4. Extract selected features and best parameters from each fold
        5. Aggregate results to find stable features and parameters
        
        Parameters
        ----------
        model_name : str
            Name of the model in model_configs to use.
        search_type : str, default='grid'
            Type of search to perform, either 'grid' or 'random'.
        param_grid : dict, optional
            Dictionary with parameter names as keys and lists of parameter values.
            If None, uses the 'params' from model_configs.
        n_features : int, optional
            Number of features to select. If None, uses half of available features.
        direction : str, default='forward'
            Direction for sequential feature selection ('forward' or 'backward').
        n_iter : int, default=50
            Number of parameter settings sampled in RandomizedSearchCV.
            Ignored for GridSearchCV.
        scoring : str, optional
            Metric to use for evaluation. If None, uses first metric in self.metrics.
            
        Returns
        -------
        dict
            Dictionary containing:
            
            model_name : str
                Name of the model used.
            n_features : int
                Number of features selected in the inner feature selection step.
            direction : str
                Direction used for feature selection ('forward' or 'backward').
            scoring : str
                Metric used for evaluation.
            search_type : str
                Type of search performed ('grid' or 'random').
            selected_features : list
                Features selected in majority of cross-validation folds (stable features).
            feature_frequency : dict
                Mapping of feature names to selection frequency (0.0 to 1.0) across folds.
            best_params : dict
                Best parameters selected based on majority voting across folds.
            param_frequency : dict
                Dictionary mapping parameter names to dictionaries of value frequencies.
                Example: {'max_depth': {3: 0.6, 5: 0.4}, 'min_samples_split': {2: 1.0}}
            feature_importances : dict
                Feature importance statistics with keys:
                - {feature_name}: dict with 'mean', 'std', and 'values' across folds
            weighted_importances : dict
                Feature importances weighted by selection frequency for more robust ranking.
            best_fold : dict
                Information about the best-performing fold:
                - 'fold': Index of the best fold
                - 'features': Features selected in the best fold
                - 'params': Best parameters selected for that fold
                - 'score': Score achieved by the best fold
            outer_results : list
                Per-fold results with selected features, best parameters, and test scores.                
        Notes
        -----
        This approach combines the benefits of feature selection and hyperparameter tuning
        in a single nested cross-validation framework, providing a comprehensive assessment
        of model performance and stability.
        
        The method returns both the stable features (selected in majority of folds) and
        stable parameters (most frequently selected across folds), as well as detailed
        information about each fold.
        
        See Also
        --------
        _build_combined_fs_hp_pipeline : Builds the combined pipeline
        _extract_combined_results : Extracts features and parameters from cross-validation
        _aggregate_combined_results : Aggregates results across folds
        feature_selection : Performs feature selection only
        hp_search : Performs hyperparameter search only
        
        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model_configs = {'rf': {'estimator': RandomForestClassifier(),
        ...                         'params': {'max_depth': [3, 5, None]}}}
        >>> result = pipeline.hp_search_fs('rf', search_type='grid', n_features=5)
        >>> print(f"Selected features: {result['selected_features']}")
        >>> print(f"Best parameters: {result['best_params']}")
        """
        
        # Set default n_features if not provided
        n_sel = n_features or (self.X.shape[1] // 2)

        search_est, feat_names, metric = self._build_combined_fs_hp_pipeline(
            model_name, search_type, param_grid, n_sel, direction, n_iter, scoring
        )
        
        cv_res = self.cross_val(search_est, self.X, self.y)

        # Extract results from each fold
        outer_results, all_selected, all_params = self._extract_combined_results(
            cv_res, feat_names, metric
        )
        
        # Aggregate results across folds
        selected_features, feature_frequency, best_params, param_frequency, best_fold, feature_importances, weighted_importances = self._aggregate_combined_results(
            outer_results, all_selected, all_params, feat_names, metric
        )
        predictions, metric_scores = self._aggregate(cv_res["cv_fold_predictions"])

        # Return comprehensive results
        return {
            'model_name': model_name,
            **cv_res,
            'n_features': n_sel,
            'direction': direction,
            'scoring': metric,
            'search_type': search_type,
            'selected_features': selected_features,
            'feature_frequency': feature_frequency,
            'best_params': best_params, 
            'param_frequency': param_frequency,
            'feature_importances': feature_importances,
            'weighted_importances': weighted_importances,
            'best_fold': best_fold,
            'predictions': predictions,
            'metric_scores': metric_scores,
            'outer_results': outer_results,
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
        
        # Validate method type
        valid_methods = ['baseline', 'feature_selection', 'hp_search', 'hp_search_fs']
        if method_name not in valid_methods:
            raise ValueError(
                f"Invalid execution type '{method_name}'. "
                f"Must be one of: {', '.join(valid_methods)}"
            )
        
        # Map execution type to method name
        method_map = {
            'baseline': 'baseline_evaluation',
            'feature_selection': 'feature_selection',
            'hp_search': 'hp_search',
            'hp_search_fs': 'hp_search_fs'
        }
                
        # Get the method and execute it with the provided kwargs
        try:
            method = getattr(self, method_map[method_name])
            results = method(**kwargs)
            return results
        except Exception as e:
            logger.error(f"Error during {method_name} execution: {str(e)}")
            raise