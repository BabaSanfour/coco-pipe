#!/usr/bin/env python3
"""
Core ML pipeline module with extended model selection, utilities, and customizable scoring.

This module defines the :class:`MLPipeline` class which provides methods for:
  
  - Baseline classification,
  - Feature selection using :class:`sklearn.feature_selection.SequentialFeatureSelector`,
  - Hyperparameter search using :class:`sklearn.model_selection.GridSearchCV`,
  - Combined feature selection and hyperparameter search,
  - Unsupervised clustering using :class:`sklearn.cluster.KMeans` and silhouette score.

Additionally, utility functions are provided to:
  
  - List (and print) available models,
  - Add new models with their hyperparameter grids,
  - Update the hyperparameter grid of already existing models.

The user specifies the input data (X and y) and the default scoring metric at initialization.
The supported metrics for reporting and optimization are:
  
  - ``"accuracy"`` (default),
  - ``"sensitivity"``: recall for the positive class,
  - ``"specificity"``: recall for the negative class,
  - ``"f1-score"``,
  - ``"auc"``.
"""

import logging
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone

# Import common classifiers from scikit-learn.
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

# Import make_scorer for custom metrics.
from sklearn.metrics import make_scorer, recall_score

def sensitivity_score(y_true, y_pred):
    """
    Calculate sensitivity (recall for the positive class).

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Sensitivity score.
    """
    return recall_score(y_true, y_pred, pos_label=1)

def specificity_score(y_true, y_pred):
    """
    Calculate specificity (recall for the negative class).

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Specificity score.
    """
    return recall_score(y_true, y_pred, pos_label=0)

CUSTOM_SCORERS = {
    "accuracy": "accuracy",
    "sensitivity": make_scorer(sensitivity_score),
    "specificity": make_scorer(specificity_score),
    "f1-score": "f1",
    "auc": "roc_auc"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class MLPipeline:
    """
    A machine learning pipeline for classification and clustering tasks.

    This class encapsulates various methods to run baseline models, perform feature selection,
    conduct hyperparameter search, and execute unsupervised clustering. The data (X and y)
    as well as a default scoring metric are specified at initialization and then used by default
    throughout all methods.

    Attributes
    ----------
    X : {array-like, pandas.DataFrame}
        The feature set.
    y : array-like
        The target labels.
    scoring : str
        Default scoring metric to use (e.g., "accuracy", "sensitivity", "specificity", "f1-score", "auc").
    random_state : int
        Seed for random number generation to ensure reproducibility.
    n_jobs : int
        Number of jobs to run in parallel (default is -1 for all processors).
    all_models : dict
        A dictionary of all available models with their default estimator instances and hyperparameter grids.
    models : dict
        A dictionary of selected models (a subset or all of :attr:`all_models`) according to
        the input parameter ``models``.
    """

    def __init__(self, X, y, models="all", scoring="accuracy", random_state=42, n_jobs=-1):
        self.X = X
        self.y = y
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.all_models = {
            "Logistic Regression": {
                "estimator": LogisticRegression(random_state=random_state, max_iter=1000),
                "params": {"C": [0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs", "saga"]},
            },
            "Decision Tree": {
                "estimator": DecisionTreeClassifier(random_state=random_state),
                "params": {"max_depth": [3, 5, 10, None],
                           "min_samples_split": [2, 5, 10],
                           "min_samples_leaf": [1, 2, 4]},
            },
            "Random Forest": {
                "estimator": RandomForestClassifier(random_state=random_state),
                "params": {"n_estimators": [100, 200, 300],
                           "max_depth": [3, 5, 10, None],
                           "min_samples_split": [2, 5, 10],
                           "min_samples_leaf": [1, 2, 4],
                           "max_features": ["auto", "sqrt", "log2"]},
            },
            "Extra Trees": {
                "estimator": ExtraTreesClassifier(random_state=random_state),
                "params": {"n_estimators": [100, 200, 300],
                           "max_depth": [3, 5, 10, None],
                           "min_samples_split": [2, 5, 10],
                           "min_samples_leaf": [1, 2, 4],
                           "max_features": ["auto", "sqrt", "log2"]},
            },
            "Gradient Boosting": {
                "estimator": GradientBoostingClassifier(random_state=random_state),
                "params": {"n_estimators": [100, 200, 300],
                           "learning_rate": [0.01, 0.1, 1],
                           "max_depth": [3, 5, 10],
                           "min_samples_split": [2, 5, 10],
                           "min_samples_leaf": [1, 2, 4],
                           "max_features": ["auto", "sqrt", "log2"]},
            },
            "AdaBoost": {
                "estimator": AdaBoostClassifier(random_state=random_state),
                "params": {"n_estimators": [50, 100, 200],
                           "learning_rate": [0.01, 0.1, 1]},
            },
            "SVC": {
                "estimator": SVC(random_state=random_state, probability=True),
                "params": {"C": [0.1, 1, 10],
                           "kernel": ["linear", "rbf", "poly"],
                           "gamma": ["scale", "auto"]},
            },
            "K-Nearest Neighbors": {
                "estimator": KNeighborsClassifier(),
                "params": {"n_neighbors": [3, 5, 10],
                           "weights": ["uniform", "distance"],
                           "p": [1, 2]},
            },
            "Linear Discriminant Analysis": {
                "estimator": LinearDiscriminantAnalysis(),
                "params": {},
            },
            "Quadratic Discriminant Analysis": {
                "estimator": QuadraticDiscriminantAnalysis(),
                "params": {},
            },
            "Gaussian Naive Bayes": {
                "estimator": GaussianNB(),
                "params": {},
            },
        }
        self._models = models

        if models == "all":
            self.models = self.all_models.copy()
        elif isinstance(models, str):
            if models in self.all_models:
                self.models = {models: self.all_models[models]}
            else:
                raise ValueError(f"Model '{models}' is not available.")
        elif isinstance(models, list):
            if all(isinstance(x, int) for x in models):
                all_keys = list(self.all_models.keys())
                self.models = {all_keys[i]: self.all_models[all_keys[i]]
                               for i in models if i < len(all_keys)}
            elif all(isinstance(x, str) for x in models):
                self.models = {}
                for model in models:
                    if model in self.all_models:
                        self.models[model] = self.all_models[model]
                    else:
                        raise ValueError(f"Model '{model}' is not available.")
            else:
                raise ValueError("models list must contain all integers or all strings.")
        else:
            raise ValueError("models must be 'all', a string, or a list of strings/integers.")

    # === Utility Functions ===

    def list_available_models(self):
        """
        List all available model names.

        Returns
        -------
        list
            A list of strings representing the names of the available models.
        """
        return list(self.all_models.keys())

    def print_available_models(self, verbose=False):
        """
        Print all available models. If verbose is True, also print their hyperparameter grids.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints detailed information for each model (default is False).
        """
        print("Available models:")
        for model_name, info in self.all_models.items():
            print(f" - {model_name}")
            if verbose:
                print(f"    Parameters: {info['params']}")

    def add_model(self, model_name, estimator, param_grid):
        """
        Add a new model to the available models.

        Parameters
        ----------
        model_name : str
            The name for the new model.
        estimator : object
            An instance of a scikit-learn estimator.
        param_grid : dict
            A dictionary specifying the hyperparameter grid for the model.

        Notes
        -----
        If this pipeline was initialized with "all", the new model will automatically be
        added to :attr:`models`.
        """
        if model_name in self.all_models:
            logging.info(f"Model '{model_name}' already exists. Use update_model_params to modify it.")
        else:
            self.all_models[model_name] = {"estimator": estimator, "params": param_grid}
            if self._models == "all":
                self.models[model_name] = self.all_models[model_name]
            logging.info(f"Model '{model_name}' added successfully.")

    def update_model_params(self, model_name, new_param_grid):
        """
        Update the hyperparameter grid for an existing model.

        Parameters
        ----------
        model_name : str
            The name of the model to update.
        new_param_grid : dict
            The new hyperparameter grid dictionary.
        """
        if model_name not in self.all_models:
            raise ValueError(f"Model '{model_name}' not found. Use add_model to add a new model.")
        self.all_models[model_name]["params"] = new_param_grid
        if model_name in self.models:
            self.models[model_name]["params"] = new_param_grid
        logging.info(f"Model '{model_name}' parameters updated successfully.")

    # === End Utility Functions ===

    def get_cv(self, n_splits=5):
        """
        Create a StratifiedKFold cross-validation object.

        Parameters
        ----------
        n_splits : int, optional
            Number of folds for cross-validation (default is 5).

        Returns
        -------
        StratifiedKFold
            A StratifiedKFold object with shuffling enabled.
        """
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    def baseline(self, scoring=None):
        """
        Run a baseline pipeline for each selected model using stored X and y.

        For each model, this method performs cross-validation using the specified scoring metric
        (or the default defined during initialization) and then fits the model on the full dataset.

        Parameters
        ----------
        scoring : str, optional
            Scoring metric to use. If not provided, the default scoring from the instance is used.

        Returns
        -------
        dict
            A dictionary containing:
            - "results": cross-validation scores for each model,
            - "saved_models": the fitted model instances,
            - "best_score": the score from the last evaluated model,
            - other keys set to None.
        """
        if scoring is None:
            scoring = self.scoring
        cv = self.get_cv()
        results = {}
        saved_models = {}
        last_score_mean = None

        scorer = CUSTOM_SCORERS.get(scoring, scoring)
        for model_name, model_dict in self.models.items():
            estimator = clone(model_dict["estimator"])
            scores = cross_val_score(estimator, self.X, self.y, cv=cv, scoring=scorer, n_jobs=self.n_jobs)
            score_mean = scores.mean()
            logging.info(f"{model_name} {scoring}: {score_mean:.4f}")
            estimator.fit(self.X, self.y)
            results[model_name] = score_mean
            saved_models[model_name] = estimator
            last_score_mean = score_mean

        return {
            "results": results,
            "saved_models": saved_models,
            "selected_features": None,
            "best_score": last_score_mean,
            "best_params": None,
            "fitted_model": None,
            "feature_importances": None,
            "cluster_labels": None,
            "silhouette_score": None,
        }

    def _feature_selection(self, num_features, model_name, scoring=None):
        """
        Internal method to perform feature selection using SequentialFeatureSelector
        for a specified model.

        Parameters
        ----------
        num_features : int
            The maximum number of features to select.
        model_name : str
            The model for which feature selection is performed.
        scoring : str, optional
            The scoring metric to use. If not provided, the default from the instance is used.

        Returns
        -------
        dict
            A dictionary containing the selected features (keyed by the number of features)
            and associated scores.
        """
        from sklearn.feature_selection import SequentialFeatureSelector
        if scoring is None:
            scoring = self.scoring
        cv = self.get_cv()
        selected_features_dict = {}

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not available for feature selection.")

        base_model = self.models[model_name]["estimator"]
        scorer = CUSTOM_SCORERS.get(scoring, scoring)
        for k in range(1, num_features + 1):
            logging.info(f"Processing {k} features for {model_name}")
            model_instance = clone(base_model)
            sfs = SequentialFeatureSelector(
                model_instance,
                n_features_to_select=k,
                direction="forward",
                cv=cv,
                n_jobs=self.n_jobs,
                scoring=scorer
            )
            sfs.fit(self.X, self.y)
            selected_features = sfs.get_support(indices=True)
            X_selected = self.X.iloc[:, selected_features] if isinstance(self.X, pd.DataFrame) else self.X[:, selected_features]
            scores = cross_val_score(clone(base_model), X_selected, self.y, cv=cv, scoring=scorer, n_jobs=self.n_jobs)
            score_mean = scores.mean()
            logging.info(f"{scoring} with {k} features for {model_name}: {score_mean:.4f}")
            result_dict = {
                "selected_features": selected_features,
                scoring: score_mean,
                "fitted_model": sfs.estimator,
                "feature_importances": sfs.estimator.feature_importances_ if hasattr(sfs.estimator, "feature_importances_") else None
            }
            selected_features_dict[k] = result_dict

        return {"selected_features": selected_features_dict}

    def feature_selection(self, num_features, scoring=None):
        """
        Run feature selection for each selected model using stored X and y.

        Parameters
        ----------
        num_features : int
            The maximum number of features to select.
        scoring : str, optional
            The scoring metric to use. If not provided, the default from the instance is used.

        Returns
        -------
        dict
            A dictionary containing, for each model, the selected features (keyed by the number of features)
            and associated scores.
        """
        if scoring is None:
            scoring = self.scoring

        if num_features < 1:
            raise ValueError("num_features must be at least 1.")
        if not isinstance(num_features, int):
            raise ValueError("num_features must be an integer.")

        all_fs_results = {}
        for model_name in self.models:
            logging.info(f"Performing feature selection for {model_name}")
            fs_result = self._feature_selection(num_features, model_name, scoring)
            all_fs_results[model_name] = fs_result["selected_features"]

        return {"selected_features": all_fs_results}

    def _hp_search(self, model_name, scoring=None):
        """
        Perform hyperparameter search using GridSearchCV for a given model on stored X and y.

        Parameters
        ----------
        model_name : str
            The model to optimize.
        scoring : str, optional
            The scoring metric to use. If not provided, the default from the instance is used.

        Returns
        -------
        dict
            A dictionary containing the best score, best hyperparameters, the best estimator,
            and feature importances if available.
        """
        from sklearn.model_selection import GridSearchCV

        if scoring is None:
            scoring = self.scoring

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not available for HP search.")

        base_model = clone(self.models[model_name]["estimator"])
        param_grid = self.models[model_name]["params"]
        cv = self.get_cv()
        scorer = CUSTOM_SCORERS.get(scoring, scoring)
        search = GridSearchCV(
            base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scorer,
            n_jobs=self.n_jobs
        )
        search.fit(self.X, self.y)
        best_score = search.best_score_
        best_params = search.best_params_
        best_estimator = search.best_estimator_
        logging.info(f"{model_name} - Best parameters: {best_params}")
        logging.info(f"{model_name} - Best {scoring}: {best_score:.4f}")

        feature_importances = best_estimator.feature_importances_ if hasattr(best_estimator, "feature_importances_") else None

        return {
            "best_score": best_score,
            "best_params": best_params,
            "fitted_model": best_estimator,
            "feature_importances": feature_importances,
        }
    
    def hp_search(self, scoring=None):
        """
        Run hyperparameter search for each selected model using stored X and y.

        Parameters
        ----------
        scoring : str, optional
            The scoring metric to use. If not provided, the default from the instance is used.

        Returns
        -------
        dict
            A dictionary containing, for each model, the best score, best parameters,
            fitted model, and feature importances if available.
        """
        if scoring is None:
            scoring = self.scoring

        all_hp_results = {}
        for model_name in self.models:
            logging.info(f"Performing HP search for {model_name}")
            hp_result = self._hp_search(model_name, scoring)
            all_hp_results[model_name] = hp_result

        return {"hp_results": all_hp_results}

    def _fs_hp_search(self, num_features, model_name, scoring=None):
        """
        Combine feature selection and hyperparameter search for a specified model using stored X and y.

        Parameters
        ----------
        num_features : int
            Maximum number of features to select.
        model_name : str
            The model for which to perform the combined search.
        scoring : str, optional
            The scoring metric to use. If not provided, the default from the instance is used.

        Returns
        -------
        dict
            A dictionary with keys representing the number of features selected and values
            containing the results of the hyperparameter search on that subset.
        """
        if scoring is None:
            scoring = self.scoring

        fs_results = self._feature_selection(num_features, model_name, scoring)["selected_features"]
        combined_results = {}
        for k, fs_result in fs_results.items():
            logging.info(f"Performing HP search on {k} selected features for {model_name}")
            selected_features = fs_result["selected_features"]
            X_selected = self.X.iloc[:, selected_features] if isinstance(self.X, pd.DataFrame) else self.X[:, selected_features]
            hp_result = self.__class__(X_selected, self.y, models=model_name,
                                       scoring=scoring, random_state=self.random_state, n_jobs=self.n_jobs)._hp_search(model_name, scoring)
            logging.info(f"{model_name} - Best parameters with {k} features: {hp_result['best_params']}")
            logging.info(f"{model_name} - Best {scoring} with {k} features: {hp_result['best_score']:.4f}")
            combined_results[k] = {
                "selected_features": selected_features,
                scoring: hp_result["best_score"],
                "best_params": hp_result["best_params"],
                "fitted_model": hp_result["fitted_model"],
                "feature_importances": hp_result["feature_importances"],
            }
        return {"selected_features": combined_results}
    
    def feature_selection_hp_search(self, num_features, scoring=None):
        """
        Run combined feature selection and hyperparameter search for each selected model using stored X and y.

        Parameters
        ----------
        num_features : int
            Maximum number of features to select.
        scoring : str, optional
            The scoring metric to use. If not provided, the default from the instance is used.

        Returns
        -------
        dict
            A dictionary containing the results of the combined feature selection and HP search.
        """
        if scoring is None:
            scoring = self.scoring

        all_combined_results = {}
        for model_name in self.models:
            logging.info(f"Performing combined feature selection and HP search for {model_name}")
            combined_result = self._fs_hp_search(num_features, model_name, scoring)
            all_combined_results[model_name] = combined_result["selected_features"]

        return {"combined_results": all_combined_results}
    
    # === Unsupervised Clustering ===

    def unsupervised(self, n_clusters=2):
        """
        Run an unsupervised clustering pipeline using KMeans on the stored X.

        Parameters
        ----------
        n_clusters : int, optional
            The number of clusters to form (default is 2).

        Returns
        -------
        dict
            A dictionary containing the cluster labels and the silhouette score.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(self.X)
        silhouette_avg = silhouette_score(self.X, cluster_labels)
        logging.info(f"KMeans with {n_clusters} clusters - Silhouette score: {silhouette_avg:.4f}")
        return {"cluster_labels": cluster_labels, "silhouette_score": silhouette_avg}

# === Wrapper functions for easy access ===

def pipeline_baseline(X, y, scoring="accuracy", models="all", random_state=42, n_jobs=-1):
    """
    Wrapper for running the baseline pipeline.

    Parameters
    ----------
    X : {array-like, pandas.DataFrame}
        Feature set.
    y : array-like
        Labels.
    scoring : str, optional
        Scoring metric (default is "accuracy").
    models : {"all", str, list}, optional
        Models to include in the baseline (default is "all").
    random_state : int, optional
        Random state for reproducibility.
    n_jobs : int, optional
        Number of parallel jobs (default is -1).

    Returns
    -------
    dict
        The results of the baseline pipeline.
    """
    pipeline = MLPipeline(X, y, models=models, scoring=scoring, random_state=random_state, n_jobs=n_jobs)
    return pipeline.baseline()

def pipeline_feature_selection(X, y, num_features, models="all", scoring="accuracy", random_state=42, n_jobs=-1):
    """
    Wrapper for running feature selection for a given model.

    Parameters
    ----------
    X : {array-like, pandas.DataFrame}
        Feature set.
    y : array-like
        Labels.
    num_features : int
        Maximum number of features to select.
    models : {"all", str, list}, optional
        Models to include in the baseline (default is "all").
    scoring : str, optional
        Scoring metric (default is "accuracy").
    random_state : int, optional
        Random state.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    dict
        The feature selection results.
    """
    pipeline = MLPipeline(X, y, models=models, scoring=scoring, random_state=random_state, n_jobs=n_jobs)
    return pipeline.feature_selection(num_features)

def pipeline_HP_search(X, y, models, scoring="accuracy", random_state=42, n_jobs=-1):
    """
    Wrapper for performing hyperparameter search for a given model.

    Parameters
    ----------
    X : {array-like, pandas.DataFrame}
        Feature set.
    y : array-like
        Labels.
    models : {"all", str, list}, optional
        Models to include in the baseline (default is "all").
    scoring : str, optional
        Scoring metric (default is "accuracy").
    random_state : int, optional
        Random state.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    dict
        The hyperparameter search results.
    """
    pipeline = MLPipeline(X, y, models=models, scoring=scoring, random_state=random_state, n_jobs=n_jobs)
    return pipeline.hp_search()

def pipeline_feature_selection_HP_search(X, y, num_features, models, scoring="accuracy", random_state=42, n_jobs=-1):
    """
    Wrapper for running combined feature selection and hyperparameter search for a given model.

    Parameters
    ----------
    X : {array-like, pandas.DataFrame}
        Feature set.
    y : array-like
        Labels.
    num_features : int
        Maximum number of features to select.
    models : {"all", str, list}, optional
        Models to include in the baseline (default is "all").
    scoring : str, optional
        Scoring metric (default is "accuracy").
    random_state : int, optional
        Random state.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    dict
        The combined feature selection and HP search results.
    """
    pipeline = MLPipeline(X, y, models=models, scoring=scoring, random_state=random_state, n_jobs=n_jobs)
    return pipeline.feature_selection_hp_search(num_features)

def pipeline_unsupervised(X, y, n_clusters=2, random_state=42, n_jobs=-1):
    """
    Wrapper for running the unsupervised clustering pipeline.

    Parameters
    ----------
    X : {array-like, pandas.DataFrame}
        Feature set.
    y : array-like
        Labels (not used in clustering but provided for consistency).
    n_clusters : int, optional
        Number of clusters (default is 2).
    random_state : int, optional
        Random state.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    tuple
        A tuple containing the silhouette score and cluster labels.
    """
    pipeline = MLPipeline(X, y, scoring="accuracy", random_state=random_state, n_jobs=n_jobs)
    result = pipeline.unsupervised(n_clusters)
    return result["silhouette_score"], result["cluster_labels"]