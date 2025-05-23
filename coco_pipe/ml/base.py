"""
coco_pipe/ml/base.py
----------------
Implements the core functionality and utilities shared by all ML pipelines.

Author: Hamza Abdelhedi <hamza.abdelhedii@gmail.com>
Date: 2025-05-18
Version: 0.0.1
License: TBD
"""

import logging
import warnings
from abc import ABC
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from coco_pipe.ml.config import DEFAULT_CV
from coco_pipe.ml.utils import get_cv_splitter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# TODO: Add descriptions and docstring

class BasePipeline(ABC):
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
        self.X = X; self.y = y
        self.metric_funcs = metric_funcs
        self.model_configs = model_configs
        self.metrics = list(metric_funcs.keys()) if default_metrics == "all" else list(default_metrics or [])
        self.cv_kwargs = cv_kwargs
        self.cv_strategy = cv_kwargs.get("cv_strategy", "stratified")
        self.n_jobs = n_jobs
        self.groups = groups
        self.random_state = random_state
        self._validate_input()
        self._validate_metrics()

    def _validate_input(self):
        if not isinstance(self.X, (pd.DataFrame, np.ndarray)):
            raise ValueError("X must be DataFrame or ndarray")
        if len(self.X) != len(self.y):
            raise ValueError("X and y length mismatch")

    def _validate_metrics(self):
        invalid = [m for m in self.metrics if m not in self.metric_funcs]
        if invalid:
            raise ValueError(f"Unknown metrics: {invalid}")
    
    @staticmethod
    def _get_splitter(**cv_kwargs):
        return get_cv_splitter(**cv_kwargs)

    @staticmethod
    def get_feature_importances(est: BaseEstimator) -> Optional[np.ndarray]:
        if hasattr(est, "feature_importances_"):
            return est.feature_importances_
        if hasattr(est, "coef_"):
            coef = est.coef_
            return coef[0] if coef.ndim > 1 else coef
        return None
    
    @staticmethod
    def _get_feature_names(X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        return [f"feature_{i}" for i in range(X.shape[1])]
    
    @staticmethod
    def compute_metrics(fold_preds, metrics, funcs, multioutput=False):
        import numpy as np
        scores = {m: [] for m in metrics}
        fold_sizes = [len(f["y_true"]) for f in fold_preds]
        all_true, all_pred, all_proba = [], [], []
        for f in fold_preds:
            all_true.append(f["y_true"])
            all_pred.append(f["y_pred"])
            if "y_proba" in f:
                all_proba.append(f["y_proba"])
            for m in metrics:
                if hasattr(funcs[m], "_sign"):
                    try:
                        scores[m].append(funcs[m]._sign * funcs[m]._score_func(y_true=f["y_true"], y_score=f["y_pred"]))
                    except:
                        scores[m].append(funcs[m]._sign * funcs[m]._score_func(y_true=f["y_true"], y_pred=f["y_pred"]))
                else:
                    scores[m].append(funcs[m](y_true=f["y_true"], y_pred=f["y_pred"]))
        arrs = {}
        weights = np.array(fold_sizes, dtype=float)
        for m, s in scores.items():
            scores[m] = np.array(s)
            weighted_mean = float((scores[m] * weights).sum() / weights.sum())
            std = float(np.sqrt((weights * (scores[m] - weighted_mean)**2).sum() / weights.sum()))  
            arrs[m] = {"mean": weighted_mean, "std": std, "scores": s}
            logging.info(f"{m}: {arrs[m]['mean']:.4f} (±{arrs[m]['std']:.4f})")
        return {"metrics": arrs,
                "predictions": {
                    "y_true": np.concatenate(all_true),
                    "y_pred": np.concatenate(all_pred),
                    # TODO: check if this is correct
                    "y_proba": np.concatenate(all_proba, axis=1) if all_proba and multioutput else np.concatenate(all_proba) if all_proba else None
                }}
    
    def update_model_params(self, model_name: str, params: Dict[str, Any]):
        self.model_configs[model_name]["estimator"].set_params(**params)

    def cross_validate(self, estimator: BaseEstimator, X, y):
        # --- adjust n_splits for stratified if too large ---
        if self.cv_strategy == "stratified":
            desired = self.cv_kwargs.get("n_splits", DEFAULT_CV["n_splits"])
            # compute minimum class count
            _, counts = np.unique(y, return_counts=True)
            min_count = counts.min()
            if desired > min_count:
                warnings.warn(
                    f"Requested n_splits={desired} > smallest class count ({min_count}); "
                    f"reducing to n_splits={min_count}.",
                    UserWarning
                )
                self.cv_kwargs["n_splits"] = int(min_count)
        self.cv_kwargs["random_state"] = self.random_state

        cv = self._get_splitter(**self.cv_kwargs)
        fold_preds, supports_proba = [], hasattr(estimator, "predict_proba")
        split_args: Tuple
        if self.cv_strategy in ["leave_p_out", "group_kfold"]:
            if self.groups is None:
                raise ValueError(f"'groups' required for strategy: {self.cv_strategy}")
            split_args = (X, y, self.groups)
        else:
            split_args = (X, y)
        for fold_idx, (tr, va) in enumerate(cv.split(*split_args), start=1):
            logging.info("Starting fold %d/%d", fold_idx, cv.get_n_splits(*split_args))
            X_train, X_val = X[tr], X[va]
            y_train, y_val = y[tr], y[va]
            est = clone(estimator)
            if hasattr(est, "random_state"):
                setattr(est, "random_state", self.random_state)
            est.fit(X_train, y_train)
            pred = dict(
                y_true=y_val,
                y_pred=est.predict(X_val),
                train_indices=tr, val_indices=va
            )
            if supports_proba:
                pred["y_proba"] = est.predict_proba(X_val)
            fold_preds.append(pred)
        # final fit
        final = clone(estimator)
        if hasattr(final, "random_state"):
            setattr(final, "random_state", self.random_state)
        final.fit(X, y)
        feature_importances = self.get_feature_importances(final)

        return {
            "fold_predictions": fold_preds,
            "estimator": final,
            "feature_importances": feature_importances,
        }

    def baseline(self, model_name: str, X: Optional[Union[pd.DataFrame, np.ndarray]] = None, 
                 y: Optional[Union[pd.Series, np.ndarray]] = None, 
                 best_params: Optional[Dict[str, Any]] = None):
        cfg = self.model_configs[model_name]
        if best_params is not None:
            cfg["estimator"].set_params(**best_params)
        metric_funcs = {m: self.metric_funcs[m] for m in self.metrics}
        X = X if X is not None else self.X
        y = y if y is not None else self.y
        cv_results = self.cross_validate(cfg["estimator"], X, y)
        results = self.compute_metrics(cv_results["fold_predictions"], self.metrics, metric_funcs)
        return {"model": cv_results["estimator"], 
                "feature_importances": cv_results["feature_importances"], 
                **results}

    def feature_selection(
        self, model_name: str, n_features: Optional[int] = None,
        direction: str = "forward", scoring: Optional[str] = None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ):
        X = X if X is not None else self.X
        y = y if y is not None else self.y
        from sklearn.feature_selection import SequentialFeatureSelector
        if scoring is None:
            scoring = self.metrics[0]
        sfs = SequentialFeatureSelector(
            self.model_configs[model_name]["estimator"],
            n_features_to_select=(n_features or self.X.shape[1] // 2),
            direction=direction,
            scoring=self.metric_funcs[scoring],
            n_jobs=self.n_jobs,
            cv=self._get_splitter(**self.cv_kwargs)
        )
        sfs.fit(X, y)
        mask = sfs.get_support()
        Xs = X[:, mask]
        out = self.baseline(model_name, Xs, y)
        orig_names = np.array(self._get_feature_names(X))
        selected = orig_names[mask].tolist()
        logging.info(f"Selected features: {selected}")
        out.update({"selected features": selected})
        return out

    def hp_search(
        self, model_name: str, search_type="grid",
        param_grid=None, n_iter=50, scoring: Optional[str] = None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ):
        X = X if X is not None else self.X
        y = y if y is not None else self.y
        if scoring is None:
            scoring = self.metrics[0]
        estimator = self.model_configs[model_name]["estimator"]
        grid = param_grid or self.model_configs[model_name].get("params", {})
        CV = self._get_splitter(**self.cv_kwargs)
        if search_type == "grid":
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(estimator, grid, scoring=self.metric_funcs[scoring],
                                  cv=CV, n_jobs=self.n_jobs)
        else:
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(
                estimator, grid, n_iter=n_iter,
                scoring=self.metric_funcs[scoring],
                cv=CV, n_jobs=self.n_jobs
            )
        search.fit(X, y)
        best_params = search.best_params_
        logging.info(f"Best parameters: {best_params}")
        # re-baseline with best parameters
        out = self.baseline(model_name, X, y, best_params)
        out.update({"best_params": best_params})
        return out
    
    def hp_search_fs(self, model_name: str, search_type="grid",
                     param_grid=None, n_features=None, direction="forward", 
                     n_iter=50, scoring: Optional[str] = None):
        out = self.feature_selection(model_name, n_features, direction, scoring)
        Xs = self.X[:, out["selected features"]]
        best_features = out["selected features"]
        out = self.hp_search(model_name, search_type, param_grid, n_iter, scoring, Xs, self.y)
        out.update({"best_features": best_features})
        return out

    def execute(self, type='baseline', **kwargs):
        if type == "hp_search_fs":
            return self.hp_search_fs(**kwargs)
        elif type == 'feature_selection':
            return self.feature_selection(**kwargs)
        elif type == 'hp_search':
            return self.hp_search(**kwargs)
        elif type == 'baseline':
            return self.baseline(**kwargs)