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
from typing import Dict, List, Any, Optional, Union

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
    Base pipeline class with common functionality for all ML pipelines.
    
    This class provides core functionality such as:
    - Cross-validation with predictions
    - Model cloning and validation
    - Basic data handling
    """
    
    def __init__(self, X, y, random_state=42, n_jobs=-1):
        """
        Initialize the base pipeline.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        random_state : int, optional
            Random state for reproducibility
        n_jobs : int, optional
            Number of parallel jobs
        """
        self.X = X
        self.y = y
        self.random_state = random_state
        self.n_jobs = n_jobs
    
    def get_cv(self, n_splits=5):
        """Create a cross-validation splitter."""
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state
        )
    
    def _clone_estimator(self, estimator):
        """Clone an estimator with random state."""
        est = clone(estimator)
        if hasattr(est, 'random_state'):
            est.random_state = self.random_state
        return est

    def _validate_estimator(self, estimator):
        """
        Validate that an estimator has required methods.
        
        Parameters
        ----------
        estimator : object
            Estimator to validate
            
        Returns
        -------
        bool
            True if estimator is valid
        """
        required_methods = ['fit', 'predict']
        return all(hasattr(estimator, method) for method in required_methods)

    def get_feature_names(self):
        """Get feature names from X if available."""
        if hasattr(self.X, 'columns'):
            return self.X.columns
        return np.arange(self.X.shape[1])

    def _cross_validate_with_predictions(self, estimator, X=None, y=None, groups=None):
        """
        Perform cross-validation and return predictions for each fold.
        
        Parameters
        ----------
        estimator : estimator object
            The estimator to cross-validate
        X : array-like, optional
            Features to use (defaults to self.X)
        y : array-like, optional
            Target to use (defaults to self.y)
        groups : array-like, optional
            Group labels for CV splitting
            
        Returns
        -------
        dict
            Dictionary containing:
            - fold_predictions: List of dicts with predictions for each fold
            - estimator: Fitted estimator on full dataset
        """
        if not self._validate_estimator(estimator):
            raise ValueError("Estimator must implement fit() and predict() methods")
        
        X = self.X if X is None else X
        y = self.y if y is None else y
        
        cv = self.get_cv()
        fold_predictions = []
        
        # Determine if estimator supports probability predictions
        supports_proba = hasattr(estimator, 'predict_proba')
        
        split_args = (X, y) if groups is None else (X, y, groups)
        for train_idx, val_idx in cv.split(*split_args):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit and predict
            est = self._clone_estimator(estimator)
            est.fit(X_train, y_train)
            
            fold_pred = {
                'y_true': y_val,
                'y_pred': est.predict(X_val)
            }
            
            # Add probability predictions if supported
            if supports_proba:
                fold_pred['y_proba'] = est.predict_proba(X_val)
            
            fold_predictions.append(fold_pred)
        
        # Fit on full dataset
        estimator = self._clone_estimator(estimator)
        estimator.fit(X, y)
        
        return {
            'fold_predictions': fold_predictions,
            'estimator': estimator
        }

    def get_feature_importances(self, estimator):
        """
        Extract feature importances from an estimator if available.
        
        Parameters
        ----------
        estimator : estimator object
            Fitted estimator to extract importances from
            
        Returns
        -------
        array-like or None
            Feature importance scores if available
        """
        if hasattr(estimator, 'feature_importances_'):
            return estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            return coef[0] if len(coef.shape) > 1 else coef
        return None

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
            
    @staticmethod
    def _get_scorer(scoring):
        """Get a scorer object from a string or callable."""
        if callable(scoring):
            return make_scorer(scoring)
        return scoring
        
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
        cv = self.get_cv()
        
        # For group-based CV, we need to pass the groups
        if self.groups is None:
            scores = cross_val_score(
                estimator, self.X, self.y,
                cv=cv, scoring=scorer,
                n_jobs=self.n_jobs
            )
        else:
            scores = cross_val_score(
                estimator, self.X, self.y,
                groups=self.groups,
                cv=cv, scoring=scorer,
                n_jobs=self.n_jobs
            )
            
        return {
            'scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std()
        } 