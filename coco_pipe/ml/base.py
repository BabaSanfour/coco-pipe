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
        Extract feature importances or coefficients from a fitted estimator.

        Parameters
        ----------
        estimator : BaseEstimator
            A fitted sklearn estimator.

        Returns
        -------
        importances : ndarray or None
        """
        if hasattr(estimator, 'feature_importances_'):
            return estimator.feature_importances_
        if hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            if coef.ndim == 1:
                return coef
            if coef.ndim == 2:
                # average across outputs
                return np.mean(coef, axis=0)
        return None

    def cross_val(
        self,
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Perform cross-validation using scikit-learn's `cross_validate`.

        Wraps `metric_funcs` into `scoring`, collects per-fold predictions,
        scores, estimators, and feature importances, then refits final estimator.

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
            - 'final_estimator': fitted estimator on the full dataset.
            - 'final_importances': feature importances from the final estimator,
                or None if not supported.
        Raises
        ------
        ValueError
            If X and y have different numbers of samples or if no metrics are specified.
        Notes
        -----
        This method performs cross-validation on the provided estimator using
        the configured cross-validation strategy. It collects per-fold predictions,
        scores, and feature importances, then fits a final estimator on the full dataset.
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
        cv_fold_importances = {
            feature: np.array([imp[i] for imp in fold_importances if imp is not None])
            for i, feature in enumerate(self._get_feature_names(X))
        }

        # final estimator
        final_estimator = clone(estimator)
        if hasattr(final_estimator, 'random_state') and self.random_state is not None:
            setattr(final_estimator, 'random_state', self.random_state)
        final_estimator.fit(X_arr, y_arr)
        final_importances = self._extract_feature_importances(final_estimator)

        return {
            'cv_fold_scores': cv_fold_scores,
            'cv_fold_importances': cv_fold_importances,
            'cv_fold_predictions': fold_predictions,
            'cv_fold_estimators': cv_results['estimator'],
            'final_estimator': final_estimator,
            'final_importances': final_importances,
        }

    def baseline_evaluation(
        self,
        model_name: str,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        best_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single model via cross-validation and return metrics and feature importances.
        
        This method performs cross-validation on a specified model, gets scores on
        the validation folds, and returns a comprehensive evaluation summary including
        the final model fitted on all data and its feature importances (if applicable) 
        as well as the fitted estimators from each fold.
        
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
        else:
            estimator.set_params(self.model_configs[model_name].get('params', {}))

        X_use = X if X is not None else self.X
        y_use = y if y is not None else self.y
        
        results = self.cross_val(estimator, X_use, y_use)
        results.update({'model_name': model_name, 'params': best_params or {}})

        return results

    def feature_selection(
        self,
        model_name: str,
        n_features: Optional[int] = None,
        direction: str = 'forward',
        scoring: Optional[str] = None,
        test_size: float = 0.2,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Perform feature selection on a training set via cross-validation 
        and evaluate on a holdout test set.

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
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split.
        X : DataFrame or ndarray, optional
            Feature matrix. If None, uses self.X.
        y : Series or ndarray, optional
            Target array. If None, uses self.y.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'selected_features': List of selected feature names
        """

        # 1) Data + feature names
        X_use = X if X is not None else self.X
        y_use = y if y is not None else self.y
        feat_names = np.array(self._get_feature_names(X_use))

        # 2) How many to select & scoring
        n_sel   = n_features or (X_use.shape[1] // 2)
        scoring = scoring or self.metrics[0]
        scorer  = make_scorer(self.metric_funcs[scoring], needs_proba=False)

        # 3) Base estimator (fresh clone)
        base_est = clone(self.model_configs[model_name]['estimator'])
        if hasattr(base_est, 'random_state'):
            base_est.random_state = self.random_state

        # 4) Inner-CV splitter for SFS
        inner_cv = get_cv_splitter(random_state=self.random_state, **self.cv_kwargs)

        # 5) Build pipeline: SFS(inner_cv) -> estimator
        pipe = Pipeline([
            ('sfs', SequentialFeatureSelector(
                clone(base_est),
                n_features_to_select=n_sel,
                direction=direction,
                scoring=scorer,
                cv=inner_cv,
                n_jobs=self.n_jobs
            )),
            ('clf', clone(base_est))
        ])

        # 6) Outer-CV via cross_val
        cv_res = self.cross_val(pipe, X_use, y_use)

        # 7) Gather per-fold support masks and test-scores
        estimators = cv_res['cv_fold_estimators']
        scores     = cv_res['cv_fold_scores']
        n_folds    = len(estimators)

        outer_results = []
        all_selected = []

        for fold_idx, est in enumerate(estimators):
            sfs = est.named_steps['sfs']
            mask = sfs.get_support()
            selected = feat_names[mask].tolist()
            all_selected.extend(selected)

            # per-fold test-scores
            test_scores = {m: float(scores[m][fold_idx]) for m in self.metrics}

            outer_results.append({
                'fold': fold_idx,
                'selected_features': selected,
                'test_scores': test_scores
            })

        # 8) Aggregate: count & majority vote
        freq = Counter(all_selected)
        # features selected in majority of folds (> half)
        selected_features = [f for f, cnt in freq.items() if cnt > n_folds / 2]
        # sort descending by count
        selected_features.sort(key=lambda f: freq[f], reverse=True)

        # 9) Compute selection frequency map
        feature_frequency = {f: freq.get(f, 0) / n_folds for f in feat_names}

        return {
            'selected_features': selected_features,
            'outer_results': outer_results,
            'feature_frequency': feature_frequency
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