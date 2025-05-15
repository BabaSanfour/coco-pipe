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
    StratifiedKFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    GroupKFold
)

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

    @staticmethod
    def cross_validate_with_predictions(estimator, X, y, cv_strategy="stratified", 
                                      groups=None, random_state=42, **cv_kwargs):
        """
        Perform cross-validation and return predictions for each fold.
        
        Parameters
        ----------
        estimator : estimator object
            The estimator to cross-validate
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        cv_strategy : str, optional
            Cross-validation strategy to use
        groups : array-like, optional
            Group labels for group-based CV
        random_state : int, optional
            Random state for reproducibility
        **cv_kwargs : dict
            Additional arguments for CV splitter
            
        Returns
        -------
        dict
            Dictionary containing:
            - fold_predictions: List of dicts with predictions for each fold
            - estimator: Fitted estimator on full dataset
        """
        # Set random state in cv_kwargs
        cv_kwargs['random_state'] = random_state
        
        # Get CV splitter
        cv = CrossValidationStrategy.get_cv_splitter(cv_strategy, **cv_kwargs)
        
        # Initialize results
        fold_predictions = []
        supports_proba = hasattr(estimator, 'predict_proba')
        
        # Determine split arguments based on CV strategy
        if cv_strategy in ["leave_p_out", "group_kfold"]:
            if groups is None:
                raise ValueError(f"groups must be provided for {cv_strategy}")
            split_args = (X, y, groups)
        else:
            split_args = (X, y)
        
        # Perform cross-validation
        for train_idx, val_idx in cv.split(*split_args):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone and fit estimator
            est = clone(estimator)
            if hasattr(est, 'random_state'):
                est.random_state = random_state
            est.fit(X_train, y_train)
            
            # Get predictions
            fold_pred = {
                'y_true': y_val,
                'y_pred': est.predict(X_val),
                'train_indices': train_idx,
                'val_indices': val_idx
            }
            
            if supports_proba:
                fold_pred['y_proba'] = est.predict_proba(X_val)
            
            fold_predictions.append(fold_pred)
        
        # Fit on full dataset
        final_estimator = clone(estimator)
        if hasattr(final_estimator, 'random_state'):
            final_estimator.random_state = random_state
        final_estimator.fit(X, y)
        
        return {
            'fold_predictions': fold_predictions,
            'estimator': final_estimator
        }

    @staticmethod
    def compute_cv_metrics(fold_predictions, metrics, metric_funcs):
        """
        Compute cross-validation metrics from fold predictions.
        
        This is the core implementation for computing CV metrics that can be used
        by different ML tasks (classification, regression, etc.). Each task should
        prepare its own metric functions and handle any special cases.
        
        Parameters
        ----------
        fold_predictions : list
            List of dictionaries containing predictions for each fold
        metrics : list
            List of metric names to compute
        metric_funcs : dict
            Dictionary mapping metric names to their scoring functions
            
        Returns
        -------
        dict
            Dictionary containing:
            - metrics: Dict with scores for each metric:
                - mean: Mean score across folds
                - std: Standard deviation across folds
                - scores: List of scores for each fold
            - predictions: Dict with:
                - y_true: Concatenated true values from all folds
                - y_pred: Concatenated predictions from all folds
                - y_proba: Concatenated probability predictions if available
        """
        metric_scores = {metric: [] for metric in metrics}
        
        # Initialize lists to store predictions
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        has_proba = 'y_proba' in fold_predictions[0]
        
        # Compute scores for each fold and collect predictions
        for fold in fold_predictions:
            y_true = fold['y_true']
            y_pred = fold['y_pred']
            
            # Store predictions
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            if has_proba:
                all_y_proba.append(fold['y_proba'])
            
            # Compute metrics
            for metric in metrics:
                score = metric_funcs[metric](y_true, y_pred)
                metric_scores[metric].append(score)
        
        # Compute statistics for metrics
        metric_results = {}
        for metric, scores in metric_scores.items():
            scores = np.array(scores)
            metric_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
        
        # Concatenate all predictions
        predictions = {
            'y_true': np.concatenate(all_y_true),
            'y_pred': np.concatenate(all_y_pred)
        }
        if has_proba:
            predictions['y_proba'] = np.concatenate(all_y_proba)
            
        return {
            'metrics': metric_results,
            'predictions': predictions
        }

class BasePipeline:
    """
    Base pipeline class with common functionality for all ML pipelines.
    
    This class provides core functionality such as:
    - Cross-validation with predictions
    - Model cloning and validation
    - Basic data handling
    """
    
    def __init__(self, X, y, cv_strategy="stratified", groups=None, random_state=42, n_jobs=-1):
        """
        Initialize the base pipeline.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        cv_strategy : str, optional
            Cross-validation strategy to use
        groups : array-like, optional
            Group labels for group-based CV
        random_state : int, optional
            Random state for reproducibility
        n_jobs : int, optional
            Number of parallel jobs
        """
        self.X = X
        self.y = y
        self.cv_strategy = cv_strategy
        self.groups = groups
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

    def _validate_estimator(self, estimator):
        """Validate that an estimator has required methods."""
        required_methods = ['fit', 'predict']
        return all(hasattr(estimator, method) for method in required_methods)

    def get_feature_names(self):
        """Get feature names from X if available."""
        if hasattr(self.X, 'columns'):
            return self.X.columns
        return np.arange(self.X.shape[1])

    def get_feature_importances(self, estimator):
        """Extract feature importances from an estimator if available."""
        if hasattr(estimator, 'feature_importances_'):
            return estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            return coef[0] if len(coef.shape) > 1 else coef
        return None

    def cross_validate(self, estimator, X=None, y=None, **cv_kwargs):
        """
        Perform cross-validation with predictions.
        
        This method uses CrossValidationStrategy to perform CV and return predictions.
        Subclasses should implement their own metric computation from these predictions.
        
        Parameters
        ----------
        estimator : estimator object
            The estimator to cross-validate
        X : array-like, optional
            Features to use (defaults to self.X)
        y : array-like, optional
            Target to use (defaults to self.y)
        **cv_kwargs : dict
            Additional arguments for CV splitter
            
        Returns
        -------
        dict
            Cross-validation results with predictions
        """
        if not self._validate_estimator(estimator):
            raise ValueError("Estimator must implement fit() and predict() methods")
        
        X = self.X if X is None else X
        y = self.y if y is None else y
        
        return CrossValidationStrategy.cross_validate_with_predictions(
            estimator=estimator,
            X=X,
            y=y,
            cv_strategy=self.cv_strategy,
            groups=self.groups,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **cv_kwargs
        )