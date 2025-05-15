"""
Base ML pipeline module with core functionality and utilities.

This module provides the foundational components for machine learning pipelines,
implementing core functionality that is shared across different types of ML tasks
(classification, regression, etc.).

Key Features:
- Cross-validation strategies (stratified, leave-p-out, group-based)
- Prediction collection and aggregation
- Metric computation across folds
- Input validation and error handling
- Model cloning and validation
- Feature importance extraction
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
    Class to manage different cross-validation strategies and metric computation.
    
    This class provides a unified interface for:
    - Selecting and configuring CV splitters
    - Performing cross-validation with prediction collection
    - Computing metrics across CV folds
    - Aggregating predictions and results
    
    Supported CV Strategies:
    - 'stratified': Stratified K-Fold CV (preserves class distribution)
    - 'leave_p_out': Leave-P-Groups-Out CV (for group-based validation)
    - 'group_kfold': Group K-Fold CV (no group overlap between folds)
    
    The class follows a consistent pattern:
    1. CV splitter selection and configuration
    2. Data splitting and model training
    3. Prediction collection for each fold
    4. Metric computation and aggregation
    """
    
    @staticmethod
    def get_cv_splitter(strategy: str, **kwargs):
        """
        Get the appropriate cross-validation splitter based on strategy.
        
        Parameters
        ----------
        strategy : str
            The cross-validation strategy to use:
            - 'stratified': Stratified K-Fold
            - 'leave_p_out': Leave-P-Groups-Out
            - 'group_kfold': Group K-Fold
        **kwargs : dict
            Additional arguments for CV splitter:
            - n_splits: Number of folds (for stratified and group k-fold)
            - n_groups: Number of groups to leave out (for leave-p-out)
            - shuffle: Whether to shuffle (for stratified)
            - random_state: Random state (for shuffling)
            
        Returns
        -------
        splitter : object
            A scikit-learn compatible cross-validation splitter
            
        Raises
        ------
        ValueError
            If strategy is not recognized
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
        Perform cross-validation and collect predictions for each fold.
        
        This method handles the complete CV process:
        1. Splits data according to CV strategy
        2. Trains model on each training fold
        3. Collects predictions for each validation fold
        4. Trains final model on full dataset
        
        Parameters
        ----------
        estimator : estimator object
            Scikit-learn compatible estimator to cross-validate
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target vector
        cv_strategy : str, optional (default="stratified")
            Cross-validation strategy to use
        groups : array-like of shape (n_samples,), optional
            Group labels for group-based CV strategies
        random_state : int, optional (default=42)
            Random state for reproducibility
        **cv_kwargs : dict
            Additional arguments for CV splitter
            
        Returns
        -------
        dict
            Dictionary containing:
            - fold_predictions: List of dicts with predictions for each fold
                - y_true: True values for the fold
                - y_pred: Predictions for the fold
                - y_proba: Probability predictions (if available)
                - train_indices: Training set indices
                - val_indices: Validation set indices
            - estimator: Final estimator trained on full dataset
            
        Notes
        -----
        The final estimator is trained on the full dataset after CV,
        making it ready for deployment or further analysis.
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
        
        This method handles metric computation across CV folds:
        1. Collects predictions from all folds
        2. Computes specified metrics for each fold
        3. Aggregates results with statistics
        
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
    Base pipeline class providing core ML pipeline functionality.
    
    This class serves as the foundation for specific ML pipelines (classification,
    regression, etc.) by providing common functionality:
    - Cross-validation with prediction collection
    - Input validation and error handling
    - Model cloning and validation
    - Feature importance extraction
    - Basic data handling
    
    The class is designed to be extended by task-specific pipelines that add
    their own metrics, models, and evaluation methods.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target vector
    cv_strategy : str, optional (default="stratified")
        Cross-validation strategy to use
    groups : array-like of shape (n_samples,), optional
        Group labels for group-based CV
    random_state : int, optional (default=42)
        Random state for reproducibility
    n_jobs : int, optional (default=-1)
        Number of parallel jobs
        
    Notes
    -----
    This class is not meant to be used directly, but rather serves as a base
    for specific pipeline implementations like ClassificationPipeline.
    """
    
    def __init__(self, X, y, cv_strategy="stratified", groups=None, random_state=42, n_jobs=-1):
        self.X = X
        self.y = y
        self.cv_strategy = cv_strategy
        self.groups = groups
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._validate_input()
    
    def _validate_input(self):
        """
        Validate input data format and dimensions.
        
        Raises
        ------
        ValueError
            If input validation fails
        """
        if not isinstance(self.X, (pd.DataFrame, np.ndarray)):
            raise ValueError("X must be a pandas DataFrame or numpy array")
        if not isinstance(self.y, (pd.Series, np.ndarray)):
            raise ValueError("y must be a pandas Series or numpy array")
        if self.X.shape[0] != len(self.y):
            raise ValueError("X and y must have the same number of samples")
        if self.groups is not None and len(self.groups) != len(self.y):
            raise ValueError("groups must have the same length as y")

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
            True if estimator has all required methods
        """
        required_methods = ['fit', 'predict']
        return all(hasattr(estimator, method) for method in required_methods)

    def get_feature_names(self):
        """
        Get feature names from X if available.
        
        Returns
        -------
        array-like
            Feature names if X is a DataFrame, otherwise feature indices
        """
        if hasattr(self.X, 'columns'):
            return self.X.columns
        return np.arange(self.X.shape[1])

    def get_feature_importances(self, estimator):
        """
        Extract feature importances from an estimator if available.
        
        Parameters
        ----------
        estimator : estimator object
            Fitted estimator to extract feature importances from
            
        Returns
        -------
        array-like or None
            Feature importances if available, None otherwise
        """
        if hasattr(estimator, 'feature_importances_'):
            return estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            return coef[0] if len(coef.shape) > 1 else coef
        return None

    def cross_validate(self, estimator, X=None, y=None, **cv_kwargs):
        """
        Perform cross-validation with predictions.
        
        This method uses CrossValidationStrategy to perform CV and return
        predictions. Specific pipeline implementations should handle metric
        computation from these predictions.
        
        Parameters
        ----------
        estimator : estimator object
            Estimator to cross-validate
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
            
        Raises
        ------
        ValueError
            If estimator validation fails
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