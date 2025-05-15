"""
Classification module with extended model selection and evaluation.

This module provides classification-specific functionality, including:
- Binary and multiclass classification
- Feature selection
- Hyperparameter optimization
- Model evaluation with various metrics
- Model management utilities
"""

import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import (
    make_scorer, recall_score, accuracy_score, 
    f1_score, roc_auc_score, precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from .base import BasePipeline

def sensitivity_score(y_true, y_pred):
    """Calculate sensitivity (recall for positive class)."""
    return recall_score(y_true, y_pred, pos_label=1)

def specificity_score(y_true, y_pred):
    """Calculate specificity (recall for negative class)."""
    return recall_score(y_true, y_pred, pos_label=0)

CLASSIFICATION_METRICS = {
    "accuracy": make_scorer(accuracy_score),
    "sensitivity": make_scorer(sensitivity_score),
    "specificity": make_scorer(specificity_score),
    "f1": make_scorer(f1_score),
    "precision": make_scorer(precision_score),
    "auc": make_scorer(roc_auc_score, needs_proba=True)
}

class ClassificationPipeline(BasePipeline):
    """
    A pipeline for classification tasks with model selection and evaluation.
    
    This class extends BasePipeline with classification-specific functionality:
    - Model selection from common classifiers
    - Feature selection
    - Hyperparameter optimization
    - Classification-specific metrics
    - Model management utilities
    """
    
    def __init__(self, X, y, models="all", metrics=None, random_state=42, n_jobs=-1):
        """
        Initialize the classification pipeline.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        models : str or list, optional
            Models to include
        metrics : str or list, optional
            Metrics to evaluate. Can be a single metric name or a list.
            Default metrics: ['accuracy', 'f1']
        random_state : int, optional
            Random state for reproducibility
        n_jobs : int, optional
            Number of parallel jobs
        """
        super().__init__(X, y, random_state, n_jobs)
        self.metrics = ['accuracy'] if metrics is None else (
            [metrics] if isinstance(metrics, str) else metrics
        )
        self._validate_metrics()
        self._setup_models(models)
        
    def _validate_metrics(self):
        """Validate that all requested metrics are available."""
        invalid_metrics = [m for m in self.metrics if m not in CLASSIFICATION_METRICS]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. "
                           f"Available metrics: {list(CLASSIFICATION_METRICS.keys())}")
    
    def _evaluate_predictions(self, y_true, y_pred, y_proba=None):
        """
        Evaluate predictions using all requested metrics.
        
        Returns
        -------
        dict
            Dictionary of metric scores
        """
        scores = {}
        for metric in self.metrics:
            if metric == 'auc' and y_proba is not None:
                scores[metric] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                scorer = CLASSIFICATION_METRICS[metric]._score_func
                scores[metric] = scorer(y_true, y_pred)
        return scores

    def _setup_models(self, models):
        """Setup available models and their parameter grids."""
        self.all_models = {
            "Logistic Regression": {
                "estimator": LogisticRegression(random_state=self.random_state, max_iter=1000),
                "params": {
                    "C": [0.1, 1, 10],
                    "penalty": ["l2"],
                    "solver": ["lbfgs", "saga"]
                },
            },
            "Decision Tree": {
                "estimator": DecisionTreeClassifier(random_state=self.random_state),
                "params": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
            },
            "Random Forest": {
                "estimator": RandomForestClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["auto", "sqrt", "log2"]
                },
            },
            "Gradient Boosting": {
                "estimator": GradientBoostingClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 1],
                    "max_depth": [3, 5, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["auto", "sqrt", "log2"]
                },
            },
            "AdaBoost": {
                "estimator": AdaBoostClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1]
                },
            },
            "Extra Trees": {
                "estimator": ExtraTreesClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["auto", "sqrt", "log2"]
                },
            },
            "SVC": {
                "estimator": SVC(random_state=self.random_state, probability=True),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto"],
                    "degree": [2, 3, 4]  # for poly kernel
                },
            },
            "K-Nearest Neighbors": {
                "estimator": KNeighborsClassifier(),
                "params": {
                    "n_neighbors": [3, 5, 7, 10],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],  # Manhattan or Euclidean
                    "metric": ["minkowski", "cosine"]
                },
            },
            "Linear Discriminant Analysis": {
                "estimator": LinearDiscriminantAnalysis(),
                "params": {
                    "solver": ["svd", "lsqr", "eigen"],
                    "shrinkage": [None, "auto", 0.1, 0.5, 0.9]
                },
            },
            "Quadratic Discriminant Analysis": {
                "estimator": QuadraticDiscriminantAnalysis(),
                "params": {
                    "reg_param": [0.0, 0.1, 0.5],
                    "store_covariance": [True, False]
                },
            },
            "Gaussian Naive Bayes": {
                "estimator": GaussianNB(),
                "params": {
                    "var_smoothing": [1e-9, 1e-8, 1e-7]
                },
            },
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.all_models["XGBoost"] = {
                "estimator": xgb.XGBClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7, 9],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0],
                    "min_child_weight": [1, 3, 5],
                    "gamma": [0, 0.1, 0.2]
                },
            }
            
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.all_models["LightGBM"] = {
                "estimator": lgb.LGBMClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7, -1],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "num_leaves": [31, 63, 127],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0],
                    "min_child_samples": [20, 30, 50]
                },
            }

        if models == "all":
            self.models = self.all_models
        elif isinstance(models, str):
            if models in self.all_models:
                self.models = {models: self.all_models[models]}
            else:
                raise ValueError(f"Model '{models}' not available")
        elif isinstance(models, list):
            self.models = {m: self.all_models[m] for m in models if m in self.all_models}
        else:
            raise ValueError("models must be 'all', a string, or a list of strings")

    def list_available_models(self, verbose: bool = False) -> Dict[str, Any]:
        """
        List all available models and their configurations.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, include hyperparameter grids in output
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'models': List of available model names
            - 'details': Dict of model configurations (if verbose=True)
            - 'total_models': Number of available models
        """
        result = {
            'models': list(self.all_models.keys()),
            'total_models': len(self.all_models)
        }
        
        if verbose:
            result['details'] = {
                name: {
                    'type': type(model['estimator']).__name__,
                    'parameters': model['params']
                }
                for name, model in self.all_models.items()
            }
        
        return result

    def add_model(self, name: str, estimator: Any, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Add a new model to the available models list.
        
        Parameters
        ----------
        name : str
            Name of the model to add
        estimator : Any
            Scikit-learn compatible estimator object
        param_grid : Dict[str, List[Any]]
            Dictionary of parameters to search during optimization
            
        """
        try:
            # Validate input
            if name in self.all_models:
                logging.warning(f"Model '{name}' already exists. Use update_model_params to modify it.")
                return
            # Validate estimator has fit and predict methods
            if not (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')):
                raise ValueError("Estimator must implement fit() and predict() methods")
            
            # Add the model
            self.all_models[name] = {
                "estimator": estimator,
                "params": param_grid
            }
            
            # If using 'all' models, update self.models
            if hasattr(self, 'models') and len(self.models) == len(self.all_models) - 1:
                self.models = self.all_models
            logging.info(f"Successfully added model '{name}'")
            
        except Exception as e:
            logging.error(f"Failed to add model: {str(e)}")
            raise

    def update_model_params(self, name: str, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Update the parameter grid for an existing model.
        
        Parameters
        ----------
        name : str
            Name of the model to update
        param_grid : Dict[str, List[Any]]
            New parameter grid for hyperparameter optimization
            
        """
        try:
            if name not in self.all_models:
                raise ValueError(f"Model '{name}' not found")
            
            # Validate parameters against estimator
            estimator = self.all_models[name]['estimator']
            invalid_params = [
                param for param in param_grid.keys()
                if not hasattr(estimator, param)
            ]
            
            if invalid_params:
                raise ValueError(f"Invalid parameters for {name}: {', '.join(invalid_params)}")
            
            # Update parameters
            self.all_models[name]['params'] = param_grid
            
            # If the model is in current selection, update it there too
            if name in self.models:
                self.models[name]['params'] = param_grid
            
            logging.info(f"Successfully updated parameters for '{name}'")
            
        except Exception as e:
            logging.error(f"Failed to update model parameters: {str(e)}")
            raise

    def remove_model(self, name: str) -> Dict[str, Any]:
        """
        Remove a model from the available models list.
        
        Parameters
        ----------
        name : str
            Name of the model to remove
            
        """
        try:
            if name not in self.all_models:
                raise ValueError(f"Model '{name}' not found")
            
            # Remove from all_models and models if present
            del self.all_models[name]
            if name in self.models:
                del self.models[name]
            
            logging.info(f"Successfully removed model '{name}'")
            
            
        except Exception as e:
            logging.error(f"Failed to remove model: {str(e)}")
            raise

    def baseline(self):
        """
        Run baseline evaluation for all selected models.
        
        Returns
        -------
        dict
            Dictionary with model results containing:
            - cv_results: Cross-validation scores for each metric
            - estimator: Fitted estimator
            - feature_importances: Feature importance scores if available
            - predictions: Dict with true and predicted values
        """
        results = {}
        logging.info("Starting baseline evaluation for all selected models")
        
        for name, model_dict in self.models.items():
            logging.info(f"Evaluating {name}")
            estimator = self._clone_estimator(model_dict["estimator"])
            
            # Perform cross-validation for each metric
            cv_results = {}
            for metric in self.metrics:
                cv_scores = self.cross_validate(
                    estimator, 
                    scoring=CLASSIFICATION_METRICS[metric],
                    return_estimator=True,
                    return_predictions=True
                )
                cv_results[metric] = {
                    'scores': cv_scores['cv_scores'],
                    'mean': cv_scores['mean_score'],
                    'std': cv_scores['std_score']
                }
                logging.info(f"{name} - {metric}: {cv_scores['mean_score']:.4f} (±{cv_scores['std_score']:.4f})")
            
            # Get feature importances if available
            feature_importances = None
            if hasattr(estimator, 'feature_importances_'):
                feature_importances = estimator.feature_importances_
            elif hasattr(estimator, 'coef_'):
                feature_importances = estimator.coef_[0] if len(estimator.coef_.shape) > 1 else estimator.coef_
            
            # Fit on full dataset and get predictions
            estimator.fit(self.X, self.y)
            y_pred = estimator.predict(self.X)
            y_proba = estimator.predict_proba(self.X) if hasattr(estimator, 'predict_proba') else None
            
            # Evaluate on full dataset
            full_scores = self._evaluate_predictions(self.y, y_pred, y_proba)
            
            results[name] = {
                'cv_results': cv_results,
                'estimator': estimator,
                'feature_importances': feature_importances,
                'predictions': {
                    'y_true': self.y,
                    'y_pred': y_pred,
                    'y_proba': y_proba
                },
                'scores': full_scores
            }
        
        logging.info("Baseline evaluation completed")
        return results

    def feature_selection(self, estimator, n_features, direction="forward"):
        """
        Perform sequential feature selection.
        
        Returns
        -------
        dict
            Dictionary containing:
            - selected_features: Names of selected features
            - support: Boolean mask of selected features
            - estimator: Fitted estimator
            - feature_importances: Feature importance scores if available
            - cv_results: Cross-validation scores for each metric
            - predictions: Dict with true and predicted values
        """
        logging.info(f"Starting {direction} feature selection for {n_features} features")
        
        sfs = SequentialFeatureSelector(
            estimator=self._clone_estimator(estimator),
            n_features_to_select=n_features,
            direction=direction,
            n_jobs=self.n_jobs
        )
        
        sfs.fit(self.X, self.y)
        selected_features = self.get_feature_names()[sfs.get_support()]
        logging.info(f"Selected features: {', '.join(selected_features)}")
        
        # Evaluate with selected features
        X_selected = self.X[:, sfs.get_support()]
        cv_results = {}
        
        for metric in self.metrics:
            cv_scores = self.cross_validate(
                sfs.estimator_,
                X=X_selected,
                scoring=CLASSIFICATION_METRICS[metric],
                return_estimator=True,
                return_predictions=True
            )
            cv_results[metric] = {
                'scores': cv_scores['cv_scores'],
                'mean': cv_scores['mean_score'],
                'std': cv_scores['std_score']
            }
            logging.info(f"CV {metric} with selected features: {cv_scores['mean_score']:.4f} (±{cv_scores['std_score']:.4f})")
        
        # Get predictions on full dataset
        sfs.estimator_.fit(X_selected, self.y)
        y_pred = sfs.estimator_.predict(X_selected)
        y_proba = sfs.estimator_.predict_proba(X_selected) if hasattr(sfs.estimator_, 'predict_proba') else None
        
        # Get feature importances if available
        feature_importances = None
        if hasattr(sfs.estimator_, 'feature_importances_'):
            feature_importances = sfs.estimator_.feature_importances_
        elif hasattr(sfs.estimator_, 'coef_'):
            feature_importances = sfs.estimator_.coef_[0] if len(sfs.estimator_.coef_.shape) > 1 else sfs.estimator_.coef_
        
        return {
            'selected_features': selected_features,
            'support': sfs.get_support(),
            'estimator': sfs.estimator_,
            'feature_importances': feature_importances,
            'cv_results': cv_results,
            'predictions': {
                'y_true': self.y,
                'y_pred': y_pred,
                'y_proba': y_proba
            },
            'scores': self._evaluate_predictions(self.y, y_pred, y_proba)
        }

    def hp_search(self, model_name):
        """
        Perform hyperparameter search for a specific model.
        
        Returns
        -------
        dict
            Dictionary containing:
            - best_params: Best hyperparameters found
            - best_estimator: Best fitted estimator
            - feature_importances: Feature importance scores if available
            - cv_results: Cross-validation scores for each metric
            - predictions: Dict with true and predicted values
        """
        if model_name not in self.models:
            msg = f"Model '{model_name}' not found"
            logging.error(msg)
            raise ValueError(msg)
        
        logging.info(f"Starting hyperparameter search for {model_name}")
        model_dict = self.models[model_name]
        
        # Create a scorer that optimizes the first metric
        primary_metric = self.metrics[0]
        grid_search = GridSearchCV(
            estimator=self._clone_estimator(model_dict["estimator"]),
            param_grid=model_dict["params"],
            scoring=CLASSIFICATION_METRICS[primary_metric],
            n_jobs=self.n_jobs,
            cv=5,
            return_train_score=True
        )
        
        grid_search.fit(self.X, self.y)
        logging.info(f"Best parameters: {grid_search.best_params_}")
        
        # Get predictions
        y_pred = grid_search.predict(self.X)
        y_proba = grid_search.predict_proba(self.X) if hasattr(grid_search, 'predict_proba') else None
        
        # Evaluate all metrics
        scores = self._evaluate_predictions(self.y, y_pred, y_proba)
        for metric, score in scores.items():
            logging.info(f"Best {metric}: {score:.4f}")
        
        # Get feature importances
        feature_importances = None
        if hasattr(grid_search.best_estimator_, 'feature_importances_'):
            feature_importances = grid_search.best_estimator_.feature_importances_
        elif hasattr(grid_search.best_estimator_, 'coef_'):
            feature_importances = grid_search.best_estimator_.coef_[0] if len(grid_search.best_estimator_.coef_.shape) > 1 else grid_search.best_estimator_.coef_
        
        return {
            'best_params': grid_search.best_params_,
            'best_estimator': grid_search.best_estimator_,
            'feature_importances': feature_importances,
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'],
                'std_test_score': grid_search.cv_results_['std_test_score'],
                'mean_train_score': grid_search.cv_results_['mean_train_score'],
                'std_train_score': grid_search.cv_results_['std_train_score'],
                'params': grid_search.cv_results_['params']
            },
            'predictions': {
                'y_true': self.y,
                'y_pred': y_pred,
                'y_proba': y_proba
            },
            'scores': scores
        } 