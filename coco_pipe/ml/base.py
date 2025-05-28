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
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import make_scorer

from coco_pipe.ml.config import DEFAULT_CV
from coco_pipe.ml.utils import get_cv_splitter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
            If X and y have different numbers of samples or if invalid metrics
            are specified.

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
        self.model_configs = deepcopy(model_configs)
        self.metrics = list(default_metrics) if default_metrics else []
        self.cv_kwargs = deepcopy(cv_kwargs)
        self.groups = groups
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._validate_input()
        self._validate_metrics()

    def _validate_input(self) -> None:
        if not isinstance(self.X, (pd.DataFrame, np.ndarray)):
            raise ValueError("X must be a DataFrame or numpy array.")
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of samples.")

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
            If DataFrame, names are taken from columns.
            If ndarray, names are generated as 'feature_0', 'feature_1', etc.

        Returns
        -------
        list of str
            Feature names.
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
        Select columns by boolean mask for DataFrame or ndarray.

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
    def _select_rows(
        X: Union[pd.DataFrame, np.ndarray, pd.Series],
        indices: np.ndarray
    ) -> Union[pd.DataFrame, np.ndarray, pd.Series]:
        """
        Select rows by indices for DataFrame, Series or ndarray.
        
        Parameters
        ----------
        X : DataFrame, Series or ndarray
            Data matrix or vector.
        indices : ndarray
            Indices of rows to select.
            
        Returns
        -------
        X : DataFrame, Series or ndarray
            Data with selected rows.
            Same type as input.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.iloc[indices]
        return X[indices]

    @staticmethod
    def _extract_feature_importances(estimator: BaseEstimator) -> Optional[np.ndarray]:
        """
        Extract feature importances or coefficients from a fitted estimator.

        Parameters
        ----------
        estimator : BaseEstimator
            A fitted scikit-learn compatible estimator.

        Returns
        -------
        Optional[np.ndarray]
            A numpy array of feature importances or coefficients, or None if not available.
        """
        feat_imp = None
        if hasattr(estimator, 'feature_importances_'):
            feat_imp = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            if coef.ndim == 1:
                feat_imp = coef
            elif coef.ndim == 2 and coef.shape[0] == 1: 
                feat_imp = coef[0]
            elif coef.ndim == 2: 
                logger.info(f"Coefficient array has shape {coef.shape}. Taking mean over axis 0 for feature importance.")
                feat_imp = np.mean(coef, axis=0)
            else: 
                logger.warning(f"Coefficient array has {coef.ndim} dimensions. Cannot simply extract feature importances.")
                feat_imp = None
        return feat_imp


    def cross_val(
        self,
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Perform cross-validation using scikit-learn's `cross_validate`.

        Wraps raw metric functions in `make_scorer`, collects per-fold
        predictions, scores, estimators, and computes final feature importances.

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
            - 'fold_predictions' : list of dicts, each with keys
                - 'y_true', 'y_pred', optional 'y_proba'
            - 'fold_scores' : list of dicts, per-fold metric scores
            - 'fold_estimators' : list of fitted estimators from each fold
            - 'fold_feature_importances' : list of arrays (one per fold) or [] if unsupported
            - 'estimator' : final estimator fit on the full dataset
            - 'feature_importances' : array or None, from final estimator
            - 'cv_fold_scores' : dict mapping each metric name to array of fold scores
        """

        cv_conf = deepcopy(self.cv_kwargs)
        cv_conf.setdefault('random_state', self.random_state)
        cv = get_cv_splitter(**cv_conf)

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
        groups_arr = self.groups.values if isinstance(self.groups, pd.Series) else self.groups

        # 2) Build scorer dict for cross validate
        scoring = {
            m: make_scorer(self.metric_funcs[m])
            for m in self.metric_funcs
        }

        # 3) Run cross_validate
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

        # 4) Split indices to re‐invoke train/test splits
        splits = list(cv.split(X_arr, y_arr, groups_arr))

        # 5) Collect per‐fold predictions and importances
        fold_predictions = []
        fold_feature_importances = []
        for i, est_fold in enumerate(cv_results['estimator']):
            _, val_idx = splits[i]
            y_true = y_arr[val_idx]
            y_pred = est_fold.predict(X_arr[val_idx])

            fp: Dict[str, Any] = {'y_true': y_true, 'y_pred': y_pred}
            if hasattr(est_fold, 'predict_proba'):
                try:
                    fp['y_proba'] = est_fold.predict_proba(X_arr[val_idx])
                except Exception:
                    logger.warning(
                        "predict_proba failed for fold %d with estimator %s",
                        i, type(est_fold).__name__
                    )
            fold_predictions.append(fp)

            # feature importances
            fi = self._extract_feature_importances(est_fold)
            if fi is not None:
                fold_feature_importances.append(fi)

        # 6) Build per‐fold scores dicts
        fold_scores = []
        for i in range(len(fold_predictions)):
            d = {}
            for m in self.metrics:
                key = f'test_{m}'
                if key in cv_results:
                    d[m] = cv_results[key][i]
            fold_scores.append(d)

        # 7) Collect arrays of fold‐scores per metric
        cv_fold_scores = {
            m: np.array(cv_results[f'test_{m}']) if f'test_{m}' in cv_results else np.array([])
            for m in self.metrics
        }

        # 8) Final refit on full data
        if fit_all:
            logger.info("Fitting final estimator on full dataset")
            final_est = clone(estimator)
            if hasattr(final_est, 'random_state') and self.random_state is not None:
                setattr(final_est, 'random_state', self.random_state)
            final_est.fit(X_arr, y_arr)
            final_imp = self._extract_feature_importances(final_est)

        return {
            'fold_predictions': fold_predictions,
            'fold_scores': fold_scores,
            'fold_estimators': cv_results['estimator'],
            'fold_feature_importances': fold_feature_importances,
            'final_estimator': final_est if fit_all else None,
            'final_feature_importances': final_imp,
            'cv_fold_scores': cv_fold_scores
        }


    def baseline(
        self,
        model_name: str,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        best_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single model using cross-validation and return metrics and feature importances.
        
        This method performs cross-validation on a specified model, computes metrics on
        the validation folds, and returns a comprehensive evaluation summary including
        the final model fitted on all data.
        
        Parameters
        ----------
        model_name : str
            Name of the model in model_configs to evaluate.
        X : DataFrame or ndarray, optional
            Feature matrix. If None, uses self.X.
            Shape (n_samples, n_features).
        y : Series or ndarray, optional
            Target array. If None, uses self.y.
            Shape (n_samples,) or (n_samples, n_targets).
        best_params : dict, optional
            Parameters to set on the estimator before evaluation.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'model': Fitted estimator on full dataset
            - 'feature_importances': Feature importance values or None
            - 'metrics': Dict of metrics with mean, std, and fold scores
            - 'predictions': Dict with y_true, y_pred, and optionally y_proba
            
        Raises
        ------
        KeyError
            If model_name is not found in model_configs.
        ValueError
            If best_params contains parameters incompatible with the estimator.
            
        Notes
        -----
        This follows scikit-learn's pattern of first evaluating performance via
        cross-validation, then fitting a final model on all data for deployment.
        
        Examples
        --------
        >>> results = pipeline.baseline('random_forest')
        >>> print(f"Accuracy: {results['metrics']['accuracy']['mean']:.4f}")
        >>> feature_imp = results['feature_importances']
        """
        if model_name not in self.model_configs:
            raise KeyError(f"Model '{model_name}' not found in model_configs")
            
        cfg = self.model_configs[model_name]
        estimator = clone(cfg['estimator'])
        
        if best_params:
            estimator.set_params(**best_params)

        X_use = X if X is not None else self.X
        y_use = y if y is not None else self.y
        
        logger.info(f"Running baseline evaluation for {model_name}")
        results = self.cross_val(estimator, X_use, y_use)
        
        return {
            'model': model_name,
            **results,
        }

    def feature_selection(
        self,
        model_name: str,
        n_features: Optional[int] = None,
        direction: str = 'forward',
        scoring: Optional[str] = None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Perform sequential feature selection followed by baseline evaluation.
        
        Parameters
        ----------
        model_name : str
            Name of the model in model_configs to use for feature selection.
        n_features : int, optional
            Number of features to select. If None, uses half of available features.
        direction : str, default='forward'
            Direction for sequential feature selection ('forward' or 'backward').
        scoring : str, optional
            Metric to use for feature selection. If None, uses first metric in self.metrics.
        X : DataFrame or ndarray, optional
            Feature matrix. If None, uses self.X.
        y : Series or ndarray, optional
            Target array. If None, uses self.y.
            
        Returns
        -------
        dict
            Dictionary containing baseline results plus:
            - 'selected_features': List of selected feature names
        """
        X_use = X if X is not None else self.X
        y_use = y if y is not None else self.y
        if scoring is None:
            scoring = self.metrics[0]
        scorer = make_scorer(self.metric_funcs[scoring], needs_proba=False)
        n_sel = n_features or (X_use.shape[1] // 2)
        sfs = SequentialFeatureSelector(
            self.model_configs[model_name]['estimator'],
            n_features_to_select=n_sel,
            direction=direction,
            scoring=scorer,
            n_jobs=self.n_jobs,
            cv=get_cv_splitter(**self.cv_kwargs)
        )
        sfs.fit(
            X_use.values if isinstance(X_use, pd.DataFrame) else X_use,
            y_use.values if isinstance(y_use, (pd.Series, pd.DataFrame)) else y_use
        )
        mask = sfs.get_support()
        Xs = self._select_columns(X_use, mask)
        out = self.baseline(model_name, Xs, y_use)
        feat_names = np.array(self._get_feature_names(X_use))[mask].tolist()
        logger.info(f"Selected features: {feat_names}")
        out['selected_features'] = feat_names
        return out

    def nested_feature_selection(
        self,
        model_name: str,
        n_features: Optional[int] = None,
        direction: str = 'forward',
        scoring: Optional[str] = None,
        outer_cv_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform nested cross-validation for feature selection.
        
        Uses an outer CV loop to evaluate the feature selection process,
        and an inner CV loop to perform the actual feature selection.
        
        Parameters
        ----------
        model_name : str
            Name of the model in model_configs to use.
        n_features : int, optional
            Number of features to select. If None, uses half of available features.
        direction : str, default='forward'
            Direction for sequential feature selection ('forward' or 'backward').
        scoring : str, optional
            Metric to use for feature selection. If None, uses first metric in self.metrics.
        outer_cv_kwargs : dict, optional
            CV settings for outer loop. If None, uses self.cv_kwargs.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'feature_counts': How many times each feature was selected
            - 'top_features': Features selected in majority of outer folds
            - 'metrics': Performance metrics across outer folds
            - 'outer_results': Detailed results from each outer fold
        """
        X_use = self.X
        y_use = self.y
        
        # Configure outer CV
        outer_cv_conf = deepcopy(outer_cv_kwargs or self.cv_kwargs)
        outer_cv_conf.setdefault('random_state', self.random_state)
        outer_cv = get_cv_splitter(**outer_cv_conf)
        
        # Prepare data for CV
        X_arr = X_use.values if isinstance(X_use, pd.DataFrame) else X_use
        y_arr = y_use.values if isinstance(y_use, (pd.Series, pd.DataFrame)) else y_use
        groups_arr = (self.groups.values if isinstance(self.groups, pd.Series) 
                     else self.groups)
        
        # Determine splitting args
        if outer_cv_conf.get('cv_strategy') in ['leave_p_out', 'group_kfold']:
            if groups_arr is None:
                raise ValueError("'groups' required for this CV strategy.")
            split_args = (X_arr, y_arr, groups_arr)
        else:
            split_args = (X_arr, y_arr)
        
        # Track selected features and performance
        feature_names = self._get_feature_names(X_use)
        n_features_total = len(feature_names)
        feature_counts = {name: 0 for name in feature_names}
        outer_metrics = []
        outer_results = []
        
        # Run outer CV loop
        logger.info(f"Starting nested CV feature selection with {outer_cv.get_n_splits()} outer folds")
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(*split_args)):
            logger.info(f"Outer fold {fold_idx+1}/{outer_cv.get_n_splits()}")
            
            # Get train/test split for this outer fold
            X_train = self._select_rows(X_use, train_idx)
            X_test = self._select_rows(X_use, test_idx)
            y_train = self._select_rows(y_use, train_idx)
            y_test = self._select_rows(y_use, test_idx)
            
            # Inner CV: Run feature selection on training data
            fs_result = self.feature_selection(
                model_name=model_name,
                n_features=n_features,
                direction=direction,
                scoring=scoring,
                X=X_train,
                y=y_train
            )
            
            selected_features = fs_result['selected_features']
            
            # Track selected features
            for feature in selected_features:
                feature_counts[feature] += 1
            
            # Create mask for selected features
            mask = np.array([fn in selected_features for fn in feature_names])
            
            # Test on outer fold's test set using selected features
            X_test_selected = self._select_columns(X_test, mask)
            test_result = self.baseline(model_name, X_test_selected, y_test)
            
            # Store fold results
            outer_metrics.append(test_result['metrics'])
            outer_results.append({
                'fold': fold_idx,
                'selected_features': selected_features,
                'test_metrics': test_result['metrics']
            })
        
        # Aggregate results
        feature_selection_counts = [(name, count) for name, count in feature_counts.items() if count > 0]
        feature_selection_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Features selected in majority of folds
        n_folds = outer_cv.get_n_splits()
        majority_threshold = n_folds // 2 + 1
        top_features = [name for name, count in feature_selection_counts if count >= majority_threshold]
        
        # Aggregate metrics across folds
        aggregated_metrics = {}
        for metric_name in self.metrics:
            values = [fold['test_metrics'][metric_name]['mean'] for fold in outer_results]
            aggregated_metrics[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }
            logger.info(f"Outer CV {metric_name}: {aggregated_metrics[metric_name]['mean']:.4f} (±{aggregated_metrics[metric_name]['std']:.4f})")
        
        return {
            'feature_counts': dict(feature_selection_counts),
            'top_features': top_features,
            'metrics': aggregated_metrics,
            'outer_results': outer_results
        }

    def hp_search(
        self,
        model_name: str,
        search_type: str = 'grid',
        param_grid: Optional[Dict[str, Any]] = None,
        n_iter: int = 50,
        scoring: Optional[str] = None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Hyperparameter search (grid or randomized) followed by baseline evaluation.
        """
        X_use = X if X is not None else self.X
        y_use = y if y is not None else self.y
        if scoring is None:
            scoring = self.metrics[0]
        scorer = make_scorer(self.metric_funcs[scoring], needs_proba=('roc' in scoring))
        cfg = self.model_configs[model_name]
        grid = param_grid or cfg.get('params', {})
        CV = get_cv_splitter(**self.cv_kwargs)
        if search_type == 'grid':
            search = GridSearchCV(
                cfg['estimator'], grid, scoring=scorer,
                cv=CV, n_jobs=self.n_jobs
            )
        else:
            search = RandomizedSearchCV(
                cfg['estimator'], grid, n_iter=n_iter,
                scoring=scorer, cv=CV, n_jobs=self.n_jobs
            )
        search.fit(
            X_use.values if isinstance(X_use, pd.DataFrame) else X_use,
            y_use.values if isinstance(y_use, (pd.Series, pd.DataFrame)) else y_use
        )
        best = search.best_params_
        logger.info(f"Best parameters: {best}")
        out = self.baseline(model_name, X_use, y_use, best)
        out['best_params'] = best
        return out

    def hp_search_fs(
        self,
        model_name: str,
        search_type: str = 'grid',
        param_grid: Optional[Dict[str, Any]] = None,
        n_features: Optional[int] = None,
        direction: str = 'forward',
        n_iter: int = 50,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Combined feature selection and hyperparameter search.
        """
        fs_out = self.feature_selection(
            model_name, n_features, direction, scoring
        )
        mask = np.array([fn in fs_out['selected_features']
                         for fn in self._get_feature_names(self.X)])
        Xs = self._select_columns(self.X, mask)
        hp_out = self.hp_search(
            model_name, search_type, param_grid,
            n_iter, scoring, Xs, self.y
        )
        hp_out['best_features'] = fs_out['selected_features']
        return hp_out

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Dispatch execution based on 'type' keyword in kwargs.

        Valid types: 'baseline', 'feature_selection', 'hp_search', 'hp_search_fs'.
        """
        method = kwargs.pop('type', 'baseline')
        if method not in ['baseline', 'feature_selection', 'hp_search', 'hp_search_fs', 'nested_feature_selection']:
            raise ValueError(f"Invalid execution type '{method}'")
        return getattr(self, method)(**kwargs)