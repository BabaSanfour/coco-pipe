"""
Base classification pipeline with common functionality for all classification tasks.

This module provides the foundational components for classification pipelines,
implementing core functionality that is shared across different types of
classification tasks (binary, multiclass, multivariate).
"""

import logging
import numpy as np
from sklearn.metrics import (
    make_scorer, recall_score, accuracy_score, 
    f1_score, precision_score,
    matthews_corrcoef, balanced_accuracy_score
)
from .base import BasePipeline

# Common classification metrics
CLASSIFICATION_METRICS = {
    "accuracy": make_scorer(accuracy_score),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "f1": make_scorer(f1_score, average='weighted'),
    "precision": make_scorer(precision_score, average='weighted'),
    "recall": make_scorer(recall_score, average='weighted'),
    "mcc": make_scorer(matthews_corrcoef),
}

class BaseClassificationPipeline(BasePipeline):
    """
    Base class for all classification pipelines with common functionality.
    
    This class extends BasePipeline with classification-specific functionality:
    - Common classification metrics
    - Basic model configurations
    - Shared evaluation methods
    - Feature selection support
    - Hyperparameter optimization
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target vector(s)
    models : str or list, optional (default="all")
        Models to include in the pipeline
    metrics : str or list, optional (default=None)
        Metrics to evaluate
    random_state : int, optional (default=42)
        Random state for reproducibility
    n_jobs : int, optional (default=-1)
        Number of parallel jobs
    """
    
    def __init__(self, X, y, models="all", metrics=None, random_state=42, n_jobs=-1):
        super().__init__(X, y, random_state=random_state, n_jobs=n_jobs)
        self.metrics = ['accuracy'] if metrics is None else (
            [metrics] if isinstance(metrics, str) else metrics
        )
        self.models = models
        self._validate_metrics()
        self._setup_base_models()
        
    def _validate_metrics(self):
        """Validate that all requested metrics are available."""
        invalid_metrics = [m for m in self.metrics if m not in CLASSIFICATION_METRICS]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. "
                           f"Available metrics: {list(CLASSIFICATION_METRICS.keys())}")
    
    def _setup_base_models(self):
        """Setup basic model configurations common to all classification types."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        
        self.all_models = {
            "Logistic Regression": {
                "estimator": LogisticRegression(random_state=self.random_state),
                "params": {
                    "C": [0.1, 1, 10],
                    "penalty": ["l2"],
                }
            },
            "Decision Tree": {
                "estimator": DecisionTreeClassifier(random_state=self.random_state),
                "params": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "criterion": ["gini", "entropy"]
                }
            },
            "Random Forest": {
                "estimator": RandomForestClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5],
                    "criterion": ["gini", "entropy"]
                }
            },
            "SVC": {
                "estimator": SVC(random_state=self.random_state, probability=True),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"]
                }
            }
        }
        
        if self.models == "all":
            self.models = self.all_models
        elif isinstance(self.models, str):
            if self.models in self.all_models:
                self.models = {self.models: self.all_models[self.models]}
            else:
                raise ValueError(f"Model '{self.models}' not available. Available models: {list(self.all_models.keys())}")
        elif isinstance(self.models, list):
            invalid_models = [m for m in self.models if m not in self.all_models]
            if invalid_models:
                raise ValueError(f"Models {invalid_models} not available. Available models: {list(self.all_models.keys())}")
            self.models = {m: self.all_models[m] for m in self.models}
        else:
            raise ValueError("models must be 'all', a string, or a list of strings")
            
    
    def baseline(self, estimator: None, X: None, y: None):
        """
        Run baseline evaluation for estimator.
        
        Returns
        -------
        dict
            Dictionary with results:
            - cv_results: Cross-validation results
            - estimator: Fitted estimator
            - predictions: Predictions
            - feature_importances: Feature importances
        """
        logging.info("Starting baseline evaluation")

        # Get estimator
        if isinstance(estimator, str):
            estimator = self.models[estimator]["estimator"]

        # Perform cross-validation
        return self.cross_validate(estimator, X=X, y=y)

    def feature_selection(self, model_name=None, n_features=None, direction="forward", scoring=None):
        """
        Perform feature selection using sequential feature selection.
        
        Parameters
        ----------
        model_name : str, optional
            Name of the model to use for selection. If None, uses the first model in self.models.
        n_features : int, optional
            Number of features to select. If None, uses the N//2
        direction : {'forward', 'backward'}, optional (default='forward')
            Direction of feature selection:
            - 'forward': Start with no features and add one at a time
            - 'backward': Start with all features and remove one at a time
        scoring : str, optional
            Metric to use for selection. If None, uses first metric in self.metrics.
            
        Returns
        -------
        dict
            Dictionary containing:
            - cv_results: Cross-validation results
            - estimator: Fitted estimator
            - predictions: Predictions
            - feature_importances: Feature importances
            - selected_features: Selected features
            - selected_feature_names: Selected feature names
        """
        from sklearn.feature_selection import SequentialFeatureSelector
        # Get estimator
        estimator = self.models[model_name]["estimator"]

        # Set n_features values to try
        if n_features is None:
            n_features = self.X.shape[1]//2
        
        # Get scoring metric
        if scoring is None:
            scoring = self.metrics[0]
        scorer = CLASSIFICATION_METRICS[scoring]
        
        logging.info(f"Selecting {n_features} features using {direction} selection")
            
        # Create sfs
        sfs = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=n_features,
            direction=direction,
            scoring=scorer,
            n_jobs=self.n_jobs,
            cv=self.cv_strategy
        )
            
        # Fit selector
        sfs.fit(self.X, self.y)
            
        # Get selected features
        support = sfs.get_support()
        selected_features = np.where(support)[0]
        selected_names = self.get_feature_names()[selected_features]
        logging.info(f"Selected features: {selected_names}")
        # Evaluate with selected features
        X_selected = self.X[:, selected_features]
        # use baseline to evaluate
        results = self.baseline(estimator, X_selected, self.y)
        results.update({
            'selected_features': selected_features,
            'selected_feature_names': selected_names,
        })
        return results


    def hp_search(self, model_name, param_grid=None, search_type='grid', 
                 n_iter=100, scoring=None):
        """
        Perform hyperparameter optimization.
        
        Parameters
        ----------
        model_name : str, optional
            Name of the model to use for selection. If None, uses the first model in self.models.
        param_grid : dict, optional
            Custom parameter grid. If None, uses default from model config.
        search_type : {'grid', 'random'}, optional (default='grid')
            Type of search to perform:
            - 'grid': Exhaustive search over parameter combinations
            - 'random': Random search over parameter space
        n_iter : int, optional (default=100)
            Number of iterations for random search
        scoring : str, optional
            Metric to optimize. If None, uses first metric in self.metrics.
            
        Returns
        -------
        dict
            Dictionary containing:
            - cv_results: Cross-validation results
            - estimator: Fitted estimator
            - predictions: Predictions
            - feature_importances: Feature importances
            - best_params: Best parameter combination found
        """
        # Get estimator
        estimator = self.models[model_name]["estimator"]

        # Get parameter grid
        if param_grid is None:
            param_grid = self.models[model_name]["params"]

        # Get scoring metric
        if scoring is None:
            scoring = self.metrics[0]
        scorer = CLASSIFICATION_METRICS[scoring]
        
        logging.info(f"Starting {search_type} search for {model_name}")
        
        # Create search object
        if search_type == 'grid':
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring=scorer,
                cv=self.cv_strategy,
                n_jobs=self.n_jobs,
                return_train_score=True
            )
        elif search_type == 'random':
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=scorer,
                cv=self.cv_strategy,
                n_jobs=self.n_jobs,
                return_train_score=True
            )
        else:
            raise ValueError("search_type must be 'grid' or 'random'")
            
        # Fit search
        search.fit(self.X, self.y)
        logging.info(f"Best parameters: {search.best_params_}")
        # Set best parameters
        estimator = estimator.set_params(**search.best_params_)
        # Evaluate with best parameters
        results = self.baseline(estimator, self.X, self.y)
        results.update({
            'best_params': search.best_params_,
        })
                    
        return results
    
    def execute(self, type='baseline', **kwargs):
        """
        Run the pipeline.
        """
        self.results = []
        for model_name in self.models.keys():
            if type == 'feature_selection':
                results = self.feature_selection(model_name, **kwargs)
            elif type == 'hp_search':
                results = self.hp_search(model_name, **kwargs)
            elif type == 'baseline':
                results = self.baseline(model_name, **kwargs)
            self.results.append(results)
        return self.results