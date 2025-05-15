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
"""

import logging
from typing import Dict, List, Any
import numpy as np
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
from .base import CrossValidationStrategy

def sensitivity_score(y_true, y_pred):
    """Calculate sensitivity (recall for positive class)."""
    return recall_score(y_true, y_pred, pos_label=1)

def specificity_score(y_true, y_pred):
    """Calculate specificity (recall for negative class)."""
    return recall_score(y_true, y_pred, pos_label=0)

# Dictionary of classification metrics with their scorer functions
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
    A comprehensive pipeline for classification tasks with model selection and evaluation.
    
    This pipeline extends BasePipeline with classification-specific functionality:
    - Model selection from common classifiers (logistic regression, random forest, etc.)
    - Feature selection using sequential methods
    - Hyperparameter optimization via grid search
    - Classification-specific metrics (accuracy, F1, AUC, etc.)
    - Support for probability-based metrics
    - Cross-validation with various strategies
    
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
        - None: Use default metrics ['accuracy']
        - str: Single metric name
        - list: List of metric names
        Available metrics: accuracy, sensitivity, specificity, f1, precision, auc
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
            - predictions: Dict with aggregated predictions
        """
        # Handle metrics that don't need probabilities first
        standard_metrics = [m for m in self.metrics if m != 'auc']
        if standard_metrics:
            standard_funcs = {m: CLASSIFICATION_METRICS[m]._score_func for m in standard_metrics}
            results = CrossValidationStrategy.compute_cv_metrics(
                fold_predictions, 
                standard_metrics, 
                standard_funcs
            )
        else:
            results = {'metrics': {}, 'predictions': {}}
        
        # Handle AUC separately if needed
        if 'auc' in self.metrics:
            auc_scores = []
            for fold in fold_predictions:
                if 'y_proba' not in fold:
                    raise ValueError("AUC metric requires probability predictions")
                auc_scores.append(roc_auc_score(fold['y_true'], fold['y_proba'][:, 1]))
            
            scores = np.array(auc_scores)
            results['metrics']['auc'] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            
        return results
    
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

    def _hp_search(self, model_name, param_search=None, search_type='grid', n_iter=100):
        """
        Perform hyperparameter search for a single model.
        
        Parameters
        ----------
        model_name : str
            Name of the model to optimize
        param_search : dict, optional
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
        if param_search is None:
            model_dict = self.models[model_name]
            estimator = model_dict["estimator"]
            param_search = model_dict["params"]
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
                param_grid=param_search,
                scoring=CLASSIFICATION_METRICS[primary_metric],
                n_jobs=self.n_jobs,
                cv=5,
                return_train_score=True
            )
        elif search_type == 'random':
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(
                estimator=self._clone_estimator(estimator),
                param_distributions=param_search,
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

    def hp_search(self, models, param_search=None, search_type='grid', n_iter=100):
        """
        Perform hyperparameter search for multiple models.
        
        Parameters
        ----------
        models : str or list
            Single model name or list of model names to optimize.
            Can also be a dict mapping model names to custom param_searches.
        param_search : dict, optional
            Custom parameter searches for each model.
            If provided, must be a dict mapping model names to param_searches.
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
            param_search = param_search.get(model_name) if param_search else None
            results[model_name] = self._hp_search(
                model_name, 
                param_search, 
                search_type=search_type,
                n_iter=n_iter
            )
        
        return results 