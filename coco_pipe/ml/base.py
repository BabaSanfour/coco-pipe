"""
base.py
----------------
Implements the core functionality and utilities shared by all ML pipelines.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    GroupKFold,
)
from sklearn.model_selection._split import BaseCrossValidator

from coco_pipe.ml.config import DEFAULT_CV

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class PipelineError(Exception):
    """Custom exception class for pipeline errors."""
    pass


class CrossValidationStrategy:
    """
    Manage different cross-validation strategies and metric computation.
    """

    @staticmethod
    def get_cv_splitter(
        strategy: str, **kwargs: Any
    ) -> BaseCrossValidator:
        """
        Return a scikit-learn CV splitter based on `strategy`.

        Args:
            strategy: One of 'stratified', 'leave_p_out', 'group_kfold'.
            **kwargs: n_splits, shuffle, random_state, n_groups, etc.

        Returns:
            A BaseCrossValidator instance.

        Raises:
            PipelineError: If strategy is not recognized.
        """
        if strategy == "stratified":
            n_splits = kwargs.get("n_splits", DEFAULT_CV["n_splits"])
            shuffle = kwargs.get("shuffle", DEFAULT_CV["shuffle"])
            random_state = kwargs.get("random_state", DEFAULT_CV["random_state"])

            # Warn if random_state set but shuffle=False
            if not shuffle and random_state is not None:
                warnings.warn(
                    "You set random_state=%r while shuffle=False. "
                    "random_state will have no effect unless shuffle=True."
                    % random_state,
                    UserWarning
                )
                # Create with shuffle=True so random_state attribute is stored,
                # then revert shuffle to False
                splitter = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=random_state
                )
                splitter.shuffle = False
                return splitter

            # Normal creation
            rs_arg = random_state if shuffle else None
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=rs_arg
            )

        if strategy == "leave_p_out":
            n_groups = kwargs.get("n_groups", DEFAULT_CV["n_groups"])
            return (
                LeavePGroupsOut(n_groups=n_groups)
                if n_groups > 1
                else LeaveOneGroupOut()
            )

        if strategy == "group_kfold":
            return GroupKFold(n_splits=kwargs.get("n_splits", DEFAULT_CV["n_splits"]))

        raise PipelineError(f"Unknown CV strategy: {strategy}")
    
    @staticmethod
    def get_feature_importances(
        est: BaseEstimator
    ) -> Optional[np.ndarray]:
        """
        Extract feature importances or coefficients from a fitted estimator.

        Args:
            est: fitted estimator

        Returns:
            Array of importances or None.
        """
        if hasattr(est, "feature_importances_"):
            return est.feature_importances_
        if hasattr(est, "coef_"):
            coef = est.coef_
            return coef[0] if coef.ndim > 1 else coef
        return None

    @staticmethod
    def cross_validate_with_predictions(
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv_strategy: str = DEFAULT_CV["strategy"],
        groups: Optional[Sequence[Any]] = None,
        random_state: int = DEFAULT_CV["random_state"],
        n_jobs: int = -1,
        **cv_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Perform CV and return fold predictions + final fitted estimator.

        Args:
            estimator: scikit-learn estimator
            X: feature matrix
            y: target vector
            cv_strategy: CV strategy name
            groups: group labels for group-based splits
            random_state: for reproducibility
            n_jobs: parallel jobs (if supported)
            **cv_kwargs: passed to get_cv_splitter

        Returns:
            Dict with keys:
              - 'fold_predictions': list of fold dicts
              - 'estimator': final estimator fitted on full data
              - 'feature_importances': array or None

        Notes
        -----
        The final estimator is trained on the full dataset after CV,
        making it ready for deployment or further analysis.

        """
        # --- adjust n_splits for stratified if too large ---
        if cv_strategy == "stratified":
            desired = cv_kwargs.get("n_splits", DEFAULT_CV["n_splits"])
            # compute minimum class count
            _, counts = np.unique(y, return_counts=True)
            min_count = counts.min()
            if desired > min_count:
                warnings.warn(
                    f"Requested n_splits={desired} > smallest class count ({min_count}); "
                    f"reducing to n_splits={min_count}.",
                    UserWarning
                )
                cv_kwargs["n_splits"] = int(min_count)

        cv_kwargs["random_state"] = random_state
        cv = CrossValidationStrategy.get_cv_splitter(cv_strategy, **cv_kwargs)

        fold_predictions: List[Dict[str, Any]] = []
        supports_proba = hasattr(estimator, "predict_proba")

        split_args: Tuple
        if cv_strategy in ["leave_p_out", "group_kfold"]:
            if groups is None:
                raise PipelineError(f"'groups' required for strategy: {cv_strategy}")
            split_args = (X, y, groups)
        else:
            split_args = (X, y)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(*split_args), start=1):
            logger.info("Starting fold %d/%d", fold_idx, cv.get_n_splits(*split_args))
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            est = clone(estimator)
            if hasattr(est, "random_state"):
                setattr(est, "random_state", random_state)
            est.fit(X_train, y_train)

            fold_pred = {
                "y_true": y_val,
                "y_pred": est.predict(X_val),
                "train_indices": train_idx,
                "val_indices": val_idx,
            }
            if supports_proba:
                fold_pred["y_proba"] = est.predict_proba(X_val)

            fold_predictions.append(fold_pred)

        # Final fit on full data
        final_estimator = clone(estimator)
        if hasattr(final_estimator, "random_state"):
            setattr(final_estimator, "random_state", random_state)
        final_estimator.fit(X, y)
        feature_importances = CrossValidationStrategy.get_feature_importances(
            final_estimator
        )

        return {
            "fold_predictions": fold_predictions,
            "estimator": final_estimator,
            "feature_importances": feature_importances,
        }



class BasePipeline(ABC):
    """
    Abstract base for ML pipelines.

    Subclasses must implement `run()`.
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv_strategy: str = DEFAULT_CV["strategy"],
        groups: Optional[Sequence[Any]] = None,
        random_state: int = DEFAULT_CV["random_state"],
        n_jobs: int = -1,
    ) -> None:
        self.X = X
        self.y = y
        self.cv_strategy = cv_strategy
        self.groups = groups
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._validate_input()

    def _validate_input(self) -> None:
        """Ensure X, y (and groups) have correct types and lengths."""
        if not isinstance(self.X, (pd.DataFrame, np.ndarray)):
            raise PipelineError("X must be a DataFrame or ndarray")
        if not isinstance(self.y, (pd.Series, np.ndarray)):
            raise PipelineError("y must be a Series or ndarray")
        if len(self.X) != len(self.y):
            raise PipelineError("X and y must have same number of samples")
        if self.groups is not None and len(self.groups) != len(self.y):
            raise PipelineError("groups length must match y length")

    def _validate_estimator(self, est: Any) -> bool:
        """Check estimator has fit() and predict()."""
        return all(hasattr(est, m) for m in ("fit", "predict"))

    def get_feature_names(self) -> np.ndarray:
        """Return DataFrame columns or numeric indices."""
        if hasattr(self.X, "columns"):
            return self.X.columns.values
        return np.arange(self.X.shape[1])

    @staticmethod
    def compute_metrics(
        fold_predictions: List[Dict[str, Any]],
        metrics: List[str],
        metric_funcs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute and log metrics across CV folds.

        Returns dict with:
         - 'metrics': {metric: {'mean', 'std', 'scores'}}
         - 'predictions': concatenated y_true, y_pred, (y_proba)
        """
        scores: Dict[str, List[float]] = {m: [] for m in metrics}
        all_true, all_pred, all_proba = [], [], []
        has_proba = "y_proba" in fold_predictions[0]

        for fold in fold_predictions:
            y_true = fold["y_true"]
            y_pred = fold["y_pred"]
            all_true.append(y_true)
            all_pred.append(y_pred)
            if has_proba:
                all_proba.append(fold["y_proba"])
            for m in metrics:
                if m not in metric_funcs:
                    raise PipelineError(f"No function for metric '{m}'")
                score = metric_funcs[m](y_true, y_pred)
                scores[m].append(score)

        results = {}
        for m, vals in scores.items():
            arr = np.array(vals)
            results[m] = {"mean": arr.mean(), "std": arr.std(), "scores": arr.tolist()}
            logger.info("%s: %.4f ± %.4f", m, results[m]["mean"], results[m]["std"])

        preds = {
            "y_true": np.concatenate(all_true),
            "y_pred": np.concatenate(all_pred),
        }
        if has_proba:
            preds["y_proba"] = np.concatenate(all_proba)

        return {"metrics": results, "predictions": preds}

    def cross_validate(
        self,
        estimator: BaseEstimator,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        **cv_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Wrapper around CrossValidationStrategy to get fold predictions.
        """
        if not self._validate_estimator(estimator):
            raise PipelineError("Estimator must implement fit() and predict()")
        X_used = self.X if X is None else X
        y_used = self.y if y is None else y
        return CrossValidationStrategy.cross_validate_with_predictions(
            estimator=estimator,
            X=X_used,
            y=y_used,
            cv_strategy=self.cv_strategy,
            groups=self.groups,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **cv_kwargs,
        )

    @abstractmethod
    def run(
        self,
        estimator: BaseEstimator,
        metrics: List[str],
        metric_funcs: Dict[str, Any],
        **cv_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute the pipeline: CV + metrics + any task-specific steps.
        """
        pass