"""
Base ML pipeline module with core functionality and utilities.

This module provides the base class for ML pipelines and common utilities
that are shared between different types of ML tasks (classification, regression, etc.).
"""

import logging
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    GroupKFold
)
from sklearn.metrics import make_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class CrossValidationStrategy:
    """
    Class to manage different cross-validation strategies.
    
    Supported strategies:
    - 'stratified': StratifiedKFold
    - 'leave_p_out': Leave-P-Subjects-Out
    - 'group_kfold': Group K-Fold
    """
    
    @staticmethod
    def get_cv_splitter(strategy: str, **kwargs):
        """
        Get the appropriate cross-validation splitter.
        
        Parameters
        ----------
        strategy : str
            The cross-validation strategy to use
        **kwargs : dict
            Additional arguments for the CV splitter:
            - n_splits: Number of folds (for stratified and group k-fold)
            - n_groups: Number of groups to leave out (for leave-p-out)
            - shuffle: Whether to shuffle (for stratified)
            - random_state: Random state (for shuffling)
            
        Returns
        -------
        splitter : object
            A scikit-learn cross-validation splitter
        """
        if strategy == "stratified":
            return StratifiedKFold(
                n_splits=kwargs.get("n_splits", 5),
                shuffle=kwargs.get("shuffle", True),
                random_state=kwargs.get("random_state", 42)
            )
        elif strategy == "leave_p_out":
            n_groups = kwargs.get("n_groups", 1)
            return LeavePGroupsOut(n_groups=n_groups) if n_groups > 1 else LeaveOneGroupOut()
        elif strategy == "group_kfold":
            return GroupKFold(n_splits=kwargs.get("n_splits", 5))
        else:
            raise ValueError(f"Unknown CV strategy: {strategy}")

class BasePipeline:
    """
    Base class for all ML pipelines.
    
    This class provides common functionality and utilities that are shared
    between different types of ML tasks (classification, regression, etc.).
    
    Attributes
    ----------
    X : {array-like, pandas.DataFrame}
        The feature set.
    y : array-like
        The target variable.
    groups : array-like, optional
        Group labels for the samples (e.g., subject IDs)
    cv_strategy : str
        Cross-validation strategy to use
    cv_params : dict
        Parameters for the cross-validation strategy
    random_state : int
        Seed for random number generation.
    n_jobs : int
        Number of jobs to run in parallel.
    """
    
    def __init__(self, X, y, groups=None, cv_strategy="stratified", cv_params=None,
                 random_state=42, n_jobs=-1):
        self.X = X
        self.y = y
        self.groups = groups
        self.cv_strategy = cv_strategy
        self.cv_params = cv_params or {}
        self.cv_params["random_state"] = random_state
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._validate_input()
        
    def _validate_input(self):
        """Validate input data."""
        if not isinstance(self.X, (pd.DataFrame, np.ndarray)):
            raise ValueError("X must be a pandas DataFrame or numpy array")
        if not isinstance(self.y, (pd.Series, np.ndarray)):
            raise ValueError("y must be a pandas Series or numpy array")
        if self.X.shape[0] != len(self.y):
            raise ValueError("X and y must have the same number of samples")
        if self.groups is not None and len(self.groups) != len(self.y):
            raise ValueError("groups must have the same length as y")
            
    def get_feature_names(self):
        """Get feature names from X."""
        if isinstance(self.X, pd.DataFrame):
            return list(self.X.columns)
        return [f"feature_{i}" for i in range(self.X.shape[1])]
        
    @staticmethod
    def _clone_estimator(estimator):
        """Clone an estimator while preserving random state."""
        if hasattr(estimator, 'random_state'):
            return clone(estimator)
        return clone(estimator)
        
    def _get_scorer(self, scoring):
        """Get a scorer object from a string or callable."""
        if callable(scoring):
            return make_scorer(scoring)
        return scoring
        
    def _get_cv_splitter(self):
        """Get the appropriate cross-validation splitter based on strategy."""
        return CrossValidationStrategy.get_cv_splitter(
            self.cv_strategy,
            **self.cv_params
        )
        
    def cross_validate(self, estimator, scoring=None):
        """
        Perform cross-validation for an estimator.
        
        Parameters
        ----------
        estimator : estimator object
            A scikit-learn estimator
        scoring : str or callable, optional
            Scoring metric
            
        Returns
        -------
        dict
            Cross-validation results
        """
        scorer = self._get_scorer(scoring)
        cv = self._get_cv_splitter()
        
        # For group-based CV, we need to pass the groups
        if self.cv_strategy in ["leave_p_out", "group_kfold"]:
            if self.groups is None:
                raise ValueError(f"groups must be provided for {self.cv_strategy}")
            scores = cross_val_score(
                estimator, self.X, self.y,
                groups=self.groups,
                cv=cv, scoring=scorer,
                n_jobs=self.n_jobs
            )
        else:
            scores = cross_val_score(
                estimator, self.X, self.y,
                cv=cv, scoring=scorer,
                n_jobs=self.n_jobs
            )
            
        return {
            'scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std()
        } 