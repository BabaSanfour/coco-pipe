"""
Classification module with extended model selection and evaluation.

This module provides classification-specific functionality for machine learning pipelines,
including model selection, evaluation, and optimization. It supports both binary and 
multiclass classification tasks with various metrics and cross-validation strategies.

Key Features:
- Multiple classification models with pre-configured parameter grids
- Cross-validation with various strategies (stratified, leave-p-out, group-based)
- Feature selection with forward/backward selection
- Hyperparameter optimization with grid search
- Comprehensive model evaluation with multiple metrics
- Support for probability-based metrics (e.g., AUC)
- Support for multiclass classification
"""

import logging
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import (
    make_scorer, recall_score, accuracy_score, 
    f1_score, roc_auc_score, precision_score,
    matthews_corrcoef, balanced_accuracy_score,
    cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
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
    """Calculate sensitivity (recall) with support for multiclass."""
    return recall_score(y_true, y_pred, average='weighted')

def specificity_score(y_true, y_pred):
    """Calculate specificity with support for multiclass."""
    return recall_score(y_true, y_pred, average='weighted')

def multiclass_roc_auc_score(y_true, y_proba):
    """Calculate ROC AUC score for multiclass problems using one-vs-rest."""
    if y_proba.shape[1] == 2:  # Binary classification
        return roc_auc_score(y_true, y_proba[:, 1])
    else:  # Multiclass
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        return roc_auc_score(y_true_bin, y_proba, average='weighted', multi_class='ovr')

# Dictionary of classification metrics with their scorer functions
CLASSIFICATION_METRICS = {
    "accuracy": make_scorer(accuracy_score),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "sensitivity": make_scorer(sensitivity_score),
    "specificity": make_scorer(specificity_score),
    "f1": make_scorer(f1_score, average='weighted'),
    "precision": make_scorer(precision_score, average='weighted'),
    "auc": make_scorer(multiclass_roc_auc_score, needs_proba=True),
    "mcc": make_scorer(matthews_corrcoef),
    "kappa": make_scorer(cohen_kappa_score)
}

class ClassificationPipeline(BasePipeline):
    """
    A comprehensive pipeline for classification tasks with model selection and evaluation.
    
    This pipeline extends BasePipeline with classification-specific functionality:
    - Model selection from common classifiers (logistic regression, random forest, etc.)
    - Feature selection using sequential methods
    - Hyperparameter optimization via grid search
    - Classification-specific metrics (accuracy, F1, AUC, etc.)
    - Support for probability-based metrics
    - Cross-validation with various strategies
    - Support for multiclass classification
    
    The pipeline follows a consistent pattern for all operations:
    1. Model initialization/selection
    2. Cross-validation with prediction collection
    3. Metric computation across folds
    4. Final model training on full dataset
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target vector
    models : str or list, optional (default="all")
        Models to include in the pipeline. Can be:
        - "all": Include all available models
        - str: Name of a specific model
        - list: List of model names
    metrics : str or list, optional (default=None)
        Metrics to evaluate. Can be:
        - None: Use default metrics ['accuracy', 'balanced_accuracy']
        - str: Single metric name
        - list: List of metric names
        Available metrics: accuracy, balanced_accuracy, sensitivity, specificity,
                         f1, precision, auc, mcc, kappa
    random_state : int, optional (default=42)
        Random state for reproducibility
    n_jobs : int, optional (default=-1)
        Number of parallel jobs. -1 means use all processors
    
    Attributes
    ----------
    models : dict
        Dictionary of selected models with their configurations
    metrics : list
        List of metric names to evaluate
    all_models : dict
        Dictionary of all available models with their default parameters
    n_classes : int
        Number of unique classes in the target variable
    
    Examples
    --------
    >>> from coco_pipe.ml import ClassificationPipeline
    >>> pipeline = ClassificationPipeline(X, y, models=['Logistic Regression', 'Random Forest'])
    >>> results = pipeline.baseline()
    >>> feature_results = pipeline.feature_selection(estimator, n_features=10)
    >>> hp_results = pipeline.hp_search('Random Forest')
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
            Default metrics: ['accuracy', 'balanced_accuracy']
        random_state : int, optional
            Random state for reproducibility
        n_jobs : int, optional
            Number of parallel jobs
        """
        super().__init__(X, y, random_state, n_jobs)
        self.metrics = ['accuracy', 'balanced_accuracy'] if metrics is None else (
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
        
        This method computes both overall and per-class metrics when applicable.
        For multiclass problems, it provides:
        - Weighted average metrics across all classes
        - Per-class metrics for detailed analysis
        - Confusion matrix for detailed error analysis
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_proba : array-like, optional
            Predicted probabilities for each class
            
        Returns
        -------
        dict
            Dictionary containing:
            - overall_metrics: Dict of overall metric scores
            - per_class_metrics: Dict of per-class metric scores (if applicable)
            - confusion_matrix: Confusion matrix as a nested list
        """
        from sklearn.metrics import confusion_matrix
        
        # Compute overall metrics
        overall_scores = {}
        for metric in self.metrics:
            if metric == 'auc' and y_proba is not None:
                overall_scores[metric] = multiclass_roc_auc_score(y_true, y_proba)
            else:
                scorer = CLASSIFICATION_METRICS[metric]._score_func
                if metric in ['sensitivity', 'specificity', 'f1', 'precision']:
                    # These metrics support averaging
                    overall_scores[metric] = scorer(y_true, y_pred, average='weighted')
                else:
                    overall_scores[metric] = scorer(y_true, y_pred)
        
        # For binary/multiclass, compute per-class metrics
        classes = np.unique(y_true)
        per_class_scores = {}
        
        if len(classes) > 2:  # Multiclass
            for cls in classes:
                cls_metrics = {}
                # Convert to binary problem for this class
                y_true_bin = (y_true == cls).astype(int)
                y_pred_bin = (y_pred == cls).astype(int)
                
                # Compute binary metrics for this class
                cls_metrics['precision'] = precision_score(y_true_bin, y_pred_bin)
                cls_metrics['recall'] = recall_score(y_true_bin, y_pred_bin)
                cls_metrics['f1'] = f1_score(y_true_bin, y_pred_bin)
                
                if y_proba is not None:
                    # ROC AUC for this class vs rest
                    try:
                        cls_metrics['auc'] = roc_auc_score(y_true_bin, y_proba[:, int(cls)])
                    except:
                        cls_metrics['auc'] = None
                
                per_class_scores[f'class_{cls}'] = cls_metrics
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred).tolist()
        
        return {
            'overall_metrics': overall_scores,
            'per_class_metrics': per_class_scores,
            'confusion_matrix': conf_matrix
        }

    def _setup_models(self, models):
        """Setup available models and their parameter grids."""
        # Determine number of classes
        self.n_classes = len(np.unique(self.y))
        
        self.all_models = {
            "Logistic Regression": {
                "estimator": LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    multi_class='multinomial' if self.n_classes > 2 else 'auto',
                    penalty=self.penalty
                ),
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
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy", "log_loss"]
                },
            },
            "Random Forest": {
                "estimator": RandomForestClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                    "criterion": ["gini", "entropy", "log_loss"],
                    "class_weight": ["balanced", "balanced_subsample", None]
                },
            },
            "Gradient Boosting": {
                "estimator": GradientBoostingClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "subsample": [0.8, 0.9, 1.0],
                    "max_features": ["sqrt", "log2"]
                },
            },
            "AdaBoost": {
                "estimator": AdaBoostClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1],
                    "algorithm": ["SAMME", "SAMME.R"]
                },
            },
            "Extra Trees": {
                "estimator": ExtraTreesClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                    "criterion": ["gini", "entropy", "log_loss"],
                    "class_weight": ["balanced", "balanced_subsample", None]
                },
            },
            "SVC": {
                "estimator": SVC(random_state=self.random_state, probability=True),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto"],
                    "degree": [2, 3] if self.n_classes > 2 else [2, 3, 4],
                    "decision_function_shape": ["ovr", "ovo"] if self.n_classes > 2 else ["ovr"],
                    "class_weight": ["balanced", None]
                },
            },
            "K-Nearest Neighbors": {
                "estimator": KNeighborsClassifier(),
                "params": {
                    "n_neighbors": [3, 5, 7, 11],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                    "metric": ["minkowski", "cosine", "manhattan"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
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
                    "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
                },
            },
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.all_models["XGBoost"] = {
                "estimator": xgb.XGBClassifier(
                    random_state=self.random_state,
                    objective='multiclass' if self.n_classes > 2 else 'binary:logistic',
                    num_class=self.n_classes if self.n_classes > 2 else None
                ),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7, 9],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0],
                    "min_child_weight": [1, 3, 5],
                    "gamma": [0, 0.1, 0.2],
                    "scale_pos_weight": [1] if self.n_classes > 2 else [1, sum(self.y == 0) / sum(self.y == 1)]
                },
            }
            
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.all_models["LightGBM"] = {
                "estimator": lgb.LGBMClassifier(
                    random_state=self.random_state,
                    objective='multiclass' if self.n_classes > 2 else 'binary',
                    num_class=self.n_classes if self.n_classes > 2 else None
                ),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7, -1],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "num_leaves": [31, 63, 127],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0],
                    "min_child_samples": [20, 30, 50],
                    "class_weight": ["balanced", None]
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
        
        This method provides information about all models that can be used in the pipeline,
        including their current parameter grids and multiclass support status.
        
        Parameters
        ----------
        verbose : bool, optional (default=False)
            If True, include detailed information about each model:
            - Parameter grids
            - Multiclass support
            - Current configuration
            - Available metrics
            
        Returns
        -------
        dict
            Dictionary containing:
            - models: List of available model names
            - total_models: Number of available models
            - details: Dict of model configurations (if verbose=True)
            - metrics: List of available metrics
            - current_task: 'binary' or 'multiclass'
        
        Examples
        --------
        >>> pipeline = ClassificationPipeline(X, y)
        >>> # Get basic model list
        >>> models = pipeline.list_available_models()
        >>> print(f"Available models: {models['models']}")
        >>> # Get detailed information
        >>> details = pipeline.list_available_models(verbose=True)
        >>> print(f"Model details: {details['details']}")
        """
        result = {
            'models': list(self.all_models.keys()),
            'total_models': len(self.all_models),
            'metrics': list(CLASSIFICATION_METRICS.keys()),
            'current_task': 'multiclass' if self.n_classes > 2 else 'binary'
        }
        
        if verbose:
            result['details'] = {
                name: {
                    'type': type(model['estimator']).__name__,
                    'parameters': model['params'],
                    'multiclass_support': True,  # All our models support multiclass
                    'current_config': {
                        param: getattr(model['estimator'], param)
                        for param in model['params'].keys()
                        if hasattr(model['estimator'], param)
                    }
                }
                for name, model in self.all_models.items()
            }
        
        return result

    def add_model(self, name: str, estimator: Any, param_grid: Dict[str, List[Any]]) -> None:
        """
        Add a new model to the available models list.
        
        This method allows adding custom models to the pipeline. The model must be
        scikit-learn compatible (implement fit, predict, and predict_proba methods).
        
        Parameters
        ----------
        name : str
            Name of the model to add
        estimator : estimator object
            Scikit-learn compatible estimator that implements:
            - fit(X, y)
            - predict(X)
            - predict_proba(X) for probability-based metrics
        param_grid : Dict[str, List[Any]]
            Dictionary of parameters to search during optimization
            
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the model name already exists or if the estimator is not compatible
        
        Examples
        --------
        >>> from sklearn.ensemble import BaggingClassifier
        >>> pipeline = ClassificationPipeline(X, y)
        >>> # Add a new model
        >>> pipeline.add_model(
        ...     name="Bagging",
        ...     estimator=BaggingClassifier(random_state=42),
        ...     param_grid={
        ...         "n_estimators": [10, 20, 30],
        ...         "max_samples": [0.5, 0.7, 1.0]
        ...     }
        ... )
        """
        try:
            # Validate input
            if name in self.all_models:
                raise ValueError(f"Model '{name}' already exists. Use update_model_params to modify it.")
            
            # Validate estimator has required methods
            required_methods = ['fit', 'predict']
            missing_methods = [method for method in required_methods 
                             if not hasattr(estimator, method)]
            
            if missing_methods:
                raise ValueError(
                    f"Estimator must implement: {', '.join(required_methods)}. "
                    f"Missing methods: {', '.join(missing_methods)}"
                )
            
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

    def update_model_params(self, name: str, param_grid: Dict[str, List[Any]]) -> None:
        """
        Update the parameter grid for an existing model.
        
        This method allows modifying the hyperparameter search space for a model.
        It validates that all parameters exist in the estimator before updating.
        
        Parameters
        ----------
        name : str
            Name of the model to update
        param_grid : Dict[str, List[Any]]
            New parameter grid for hyperparameter search
            
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the model doesn't exist or if invalid parameters are provided
            
        Examples
        --------
        >>> pipeline = ClassificationPipeline(X, y)
        >>> # Update Random Forest parameters
        >>> pipeline.update_model_params(
        ...     "Random Forest",
        ...     {
        ...         "n_estimators": [50, 100],
        ...         "max_depth": [5, 10]
        ...     }
        ... )
        """
        try:
            if name not in self.all_models:
                raise ValueError(f"Model '{name}' not found")
            
            # Validate all parameters exist in the estimator
            estimator = self.all_models[name]["estimator"]
            invalid_params = [
                param for param in param_grid.keys()
                if not hasattr(estimator, param)
            ]
            
            if invalid_params:
                raise ValueError(
                    f"Invalid parameters for {name}: {invalid_params}. "
                    f"Available parameters: {[p for p in dir(estimator) if not p.startswith('_')]}"
                )
            
            # Update the parameter grid
            self.all_models[name]["params"] = param_grid
            logging.info(f"Successfully updated parameters for model '{name}'")
            
        except Exception as e:
            logging.error(f"Failed to update model parameters: {str(e)}")
            raise

    def remove_model(self, name: str) -> None:
        """
        Remove a model from the available models list.
        
        Parameters
        ----------
        name : str
            Name of the model to remove
            
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the model doesn't exist
            
        Examples
        --------
        >>> pipeline = ClassificationPipeline(X, y)
        >>> pipeline.remove_model("Decision Tree")
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

    def _compute_cv_metrics(self, fold_predictions):
        """
        Compute classification metrics from cross-validation predictions.
        
        Parameters
        ----------
        fold_predictions : list
            List of dictionaries containing predictions for each fold
            
        Returns
        -------
        dict
            Dictionary containing:
            - metrics: Dict with scores for each metric
            - per_class_metrics: Dict with per-class scores
            - predictions: Dict with aggregated predictions
            - confusion_matrices: List of confusion matrices from each fold
        """
        # Initialize storage for metrics
        all_metrics = {metric: [] for metric in self.metrics}
        all_per_class_metrics = {}
        all_confusion_matrices = []
        
        # Compute metrics for each fold
        for fold in fold_predictions:
            # Get predictions for this fold
            y_true = fold['y_true']
            y_pred = fold['y_pred']
            y_proba = fold.get('y_proba')
            
            # Evaluate predictions
            eval_results = self._evaluate_predictions(y_true, y_pred, y_proba)
            
            # Store overall metrics
            for metric, score in eval_results['overall_metrics'].items():
                all_metrics[metric].append(score)
            
            # Store per-class metrics
            if eval_results['per_class_metrics']:
                for cls, cls_metrics in eval_results['per_class_metrics'].items():
                    if cls not in all_per_class_metrics:
                        all_per_class_metrics[cls] = {
                            metric: [] for metric in cls_metrics.keys()
                        }
                    for metric, score in cls_metrics.items():
                        if score is not None:  # Only append valid scores
                            all_per_class_metrics[cls][metric].append(score)
            
            # Store confusion matrix
            all_confusion_matrices.append(eval_results['confusion_matrix'])
        
        # Compute mean and std for overall metrics
        metrics_summary = {}
        for metric in self.metrics:
            scores = np.array(all_metrics[metric])
            metrics_summary[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
        
        # Compute mean and std for per-class metrics
        per_class_summary = {}
        for cls, cls_metrics in all_per_class_metrics.items():
            per_class_summary[cls] = {}
            for metric, scores in cls_metrics.items():
                if scores:  # Only compute if we have valid scores
                    scores_array = np.array(scores)
                    per_class_summary[cls][metric] = {
                        'mean': scores_array.mean(),
                        'std': scores_array.std(),
                        'scores': scores_array.tolist()
                    }
        
        return {
            'metrics': metrics_summary,
            'per_class_metrics': per_class_summary,
            'confusion_matrices': all_confusion_matrices,
            'predictions': {
                'y_true': np.concatenate([fold['y_true'] for fold in fold_predictions]),
                'y_pred': np.concatenate([fold['y_pred'] for fold in fold_predictions]),
                'y_proba': np.concatenate([fold['y_proba'] for fold in fold_predictions]) if 'y_proba' in fold_predictions[0] else None
            }
        }
    
    def _baseline(self, name, model_dict, X=None, y=None):
        """
        Run baseline evaluation for a single model.
        
        This is the core evaluation method used by baseline, feature_selection,
        and hp_search. It performs cross-validation, computes metrics, and returns
        a consistent set of results.
        
        Parameters
        ----------
        name : str
            Name of the model for logging purposes
        model_dict : dict
            Dictionary containing:
            - estimator: The scikit-learn compatible estimator
        X : array-like, optional (default=None)
            Feature matrix. If None, uses self.X
        y : array-like, optional (default=None)
            Target vector. If None, uses self.y
            
        Returns
        -------
        dict
            Dictionary containing:
            - cv_results: Cross-validation metrics for each metric
            - estimator: Final model trained on full dataset
            - feature_importances: Feature importance scores if available
            - predictions: Dict with predictions from CV
        """
        X = self.X if X is None else X
        y = self.y if y is None else y
        
        estimator = self._clone_estimator(model_dict["estimator"])
            
        # Perform cross-validation
        cv_output = self.cross_validate(estimator)
        cv_metrics = self._compute_cv_metrics(cv_output['fold_predictions'])
            
        # Log results for each metric
        for metric, scores in cv_metrics['metrics'].items():
            logging.info(f"{name} - {metric}: {scores['mean']:.4f} (±{scores['std']:.4f})")
            
        # Get feature importances from the final model
        feature_importances = self.get_feature_importances(cv_output['estimator'])

        return {
            'cv_results': cv_metrics['metrics'],
            'estimator': cv_output['estimator'],  # final model trained on full dataset
            'feature_importances': feature_importances,
            'predictions': cv_metrics['predictions'],
        }

    def baseline(self):
        """
        Run baseline evaluation for all selected models.
        
        This method evaluates all selected models using cross-validation,
        computes specified metrics, and returns comprehensive results including
        predictions and feature importances.
        
        Returns
        -------
        dict
            Dictionary with results for each model:
            - cv_results: Cross-validation scores for each metric
            - estimator: Final model trained on full dataset
            - feature_importances: Feature importance scores if available
            - predictions: Dict with predictions from CV
        
        Examples
        --------
        >>> pipeline = ClassificationPipeline(X, y)
        >>> results = pipeline.baseline()
        >>> print(f"Best accuracy: {results['Random Forest']['cv_results']['accuracy']['mean']:.4f}")
        """
        results = {}
        logging.info("Starting baseline evaluation for all selected models")
        
        for name, model_dict in self.models.items():
            logging.info(f"Evaluating {name}")
            results[name] = self._baseline(name, model_dict)
                    
        logging.info("Baseline evaluation completed")
        return results

    def _feature_selection(self, estimator, n_features, direction="forward", name=None):
        """
        Perform feature selection for a single estimator.
        
        Parameters
        ----------
        estimator : estimator object
            Scikit-learn compatible estimator to use for selection
        n_features : int
            Number of features to select
        direction : {'forward', 'backward'}, optional (default='forward')
            Direction of feature selection
        name : str, optional
            Name for logging purposes
            
        Returns
        -------
        dict
            Dictionary containing feature selection results
        """
        from sklearn.feature_selection import SequentialFeatureSelector
        name = name or f"{type(estimator).__name__} feature selection"
        logging.info(f"Starting {direction} feature selection for {name} with {n_features} features")
        
        # Create a scorer that optimizes the first metric
        primary_metric = self.metrics[0]
        scorer = CLASSIFICATION_METRICS[primary_metric]
        sfs = SequentialFeatureSelector(
            estimator=self._clone_estimator(estimator),
            n_features_to_select=n_features,
            direction=direction,
            n_jobs=self.n_jobs,
            scoring=scorer
        )
        
        sfs.fit(self.X, self.y)
        selected_features = self.get_feature_names()[sfs.get_support()]
        logging.info(f"Selected features for {name}: {', '.join(selected_features)}")
        
        # Evaluate selected features
        X_selected = self.X[:, sfs.get_support()]
        
        # Evaluate selected features using _baseline
        cross_val_results = self._baseline(
            name=f'CV with selected features for {name}',
            model_dict={'estimator': sfs.estimator_},
            X=X_selected
        )
        cross_val_results.update(
            {
                "selected_features": selected_features,
                "support": sfs.get_support()
            }
        )
        
        return cross_val_results

    def feature_selection(self, estimators, n_features, direction="forward"):
        """
        Perform feature selection for multiple estimators.
        
        Parameters
        ----------
        estimators : estimator object or dict
            Single estimator or dictionary of estimators to evaluate.
            If dict, keys are used as names.
        n_features : int
            Number of features to select
        direction : {'forward', 'backward'}, optional (default='forward')
            Direction of feature selection:
            - 'forward': Start with no features and add one at a time
            - 'backward': Start with all features and remove one at a time
            
        Returns
        -------
        dict
            Dictionary mapping estimator names to their feature selection results
            
        Examples
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> rf = RandomForestClassifier()
        >>> # Single estimator
        >>> results = pipeline.feature_selection(rf, n_features=10)
        >>> # Multiple estimators
        >>> estimators = {
        ...     'RF': RandomForestClassifier(),
        ...     'LR': LogisticRegression()
        ... }
        >>> results = pipeline.feature_selection(estimators, n_features=10)
        
        Notes
        -----
        Feature selection uses the first metric in self.metrics as the scoring metric
        for evaluating feature importance. This is important to consider when
        initializing the pipeline. For example:
        >>> pipeline = ClassificationPipeline(X, y, metrics=['accuracy', 'f1'])
        Here, 'accuracy' will be used as the feature selection metric.
        """
        # Handle single estimator case
        if not isinstance(estimators, dict):
            return self._feature_selection(estimators, n_features, direction)
        
        # Handle multiple estimators
        results = {}
        for name, estimator in estimators.items():
            logging.info(f"Performing feature selection for {name}")
            results[name] = self._feature_selection(
                estimator, n_features, direction, name=name
            )
        
        return results

    def _hp_search(self, model_name, param_grid=None, search_type='grid', n_iter=100):
        """
        Perform hyperparameter search for a single model.
        
        Parameters
        ----------
        model_name : str
            Name of the model to optimize
        param_grid : dict, optional
            Custom parameter grid. If None, uses default from self.models
        search_type : {'grid', 'random'}, optional (default='grid')
            Type of search to perform:
            - 'grid': Exhaustive search over all parameter combinations
            - 'random': Random search over parameter space
        n_iter : int, optional (default=100)
            Number of parameter settings sampled if using random search.
            Not used if search_type='grid'.
            
        Returns
        -------
        dict
            Dictionary containing hyperparameter optimization results
            
        Notes
        -----
        The search uses the first metric in self.metrics as the scoring metric.
        This is important to consider when ordering your metrics during pipeline
        initialization.
        """
        if model_name not in self.models and param_grid is None:
            raise ValueError(f"Model '{model_name}' not found and no param_grid provided")
        
        logging.info(f"Starting {search_type} search for {model_name}")
        
        # Get model and param grid
        if param_grid is None:
            model_dict = self.models[model_name]
            estimator = model_dict["estimator"]
            param_grid = model_dict["params"]
        else:
            estimator = self.models[model_name]["estimator"]
        
        # Create a scorer that optimizes the first metric
        primary_metric = self.metrics[0]
        logging.info(f"Using {primary_metric} as the optimization metric")
        
        # Choose search strategy
        if search_type == 'grid':
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(
                estimator=self._clone_estimator(estimator),
                param_grid=param_grid,
                scoring=CLASSIFICATION_METRICS[primary_metric],
                n_jobs=self.n_jobs,
                cv=5,
                return_train_score=True
            )
        elif search_type == 'random':
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(
                estimator=self._clone_estimator(estimator),
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=CLASSIFICATION_METRICS[primary_metric],
                n_jobs=self.n_jobs,
                cv=5,
                return_train_score=True
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}. Use 'grid' or 'random'.")
        
        search.fit(self.X, self.y)
        logging.info(f"Best parameters for {model_name}: {search.best_params_}")
        
        # Evaluate best model using _baseline
        cross_val_results = self._baseline(
            name=f'Best model from {model_name} ({search_type} search)',
            model_dict={'estimator': search.best_estimator_}
        )
        
        # Add search specific results
        cross_val_results.update({
            'best_params': search.best_params_,
        })
        
        return cross_val_results

    def hp_search(self, models, param_grid=None, search_type='grid', n_iter=100):
        """
        Perform hyperparameter search for multiple models.
        
        Parameters
        ----------
        models : str or list
            Single model name or list of model names to optimize.
            Can also be a dict mapping model names to custom param_grides.
        param_grid : dict, optional
            Custom parameter searches for each model.
            If provided, must be a dict mapping model names to param_grides.
        search_type : {'grid', 'random'}, optional (default='grid')
            Type of search to perform:
            - 'grid': Exhaustive search over all parameter combinations
            - 'random': Random search over parameter space
        n_iter : int, optional (default=100)
            Number of parameter settings sampled if using random search.
            Not used if search_type='grid'.
            
        Returns
        -------
        dict
            Dictionary mapping model names to their optimization results
            
        Examples
        --------
        >>> # Grid search with single model
        >>> results = pipeline.hp_search('Random Forest')
        >>> # Random search with multiple models
        >>> results = pipeline.hp_search(['Random Forest', 'SVC'], search_type='random')
        >>> # Custom param grids with grid search
        >>> param_grids = {
        ...     'Random Forest': {
        ...         'n_estimators': [100, 200],
        ...         'max_depth': [5, 10]
        ...     }
        ... }
        >>> results = pipeline.hp_search('Random Forest', param_grids)
        
        Notes
        -----
        Both hyperparameter search and feature selection use the first metric in
        self.metrics as their scoring metric. This is important to consider when
        initializing the pipeline. For example:
        >>> pipeline = ClassificationPipeline(X, y, metrics=['accuracy', 'f1'])
        Here, 'accuracy' will be used as the optimization metric.
        """
        # Handle string input (single model)
        if isinstance(models, str):
            models = [models]
        
        # Handle dict input (models with custom param_grids)
        if isinstance(models, dict):
            param_grids = models
            models = list(models.keys())
        
        results = {}
        for model_name in models:
            param_grid = param_grid.get(model_name) if param_grid else None
            results[model_name] = self._hp_search(
                model_name, 
                param_grid, 
                search_type=search_type,
                n_iter=n_iter
            )
        
        return results 