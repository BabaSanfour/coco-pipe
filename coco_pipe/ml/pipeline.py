#!/usr/bin/env python3
"""
coco_pipe/ml/pipeline.py
----------------
Wrapper for ML pipelines.

Author: Hamza Abdelhedi <hamza.abdelhedii@gmail.com>
Date: 2025-05-18
Version: 0.0.1
License: TBD
"""

from typing import Any, Dict
from .classification import ClassificationPipeline
from .regression import RegressionPipeline


class MLPipeline:
    """
    Unified front‐end for both classification and regression pipelines.

    Parameters
    ----------
    X : array‐like of shape (n_samples, n_features)
        Feature matrix.
    y : array‐like of shape (n_samples,) or (n_samples, n_targets)
        Target vector or matrix.
    config : dict
        Configuration dictionary. Must include:
          - task: "classification" or "regression"
          - analysis_type, models, metrics, random_state, cv_strategy, n_splits,
            n_features, direction, search_type, n_iter, scoring, n_jobs,
            save_intermediate, results_dir, results_file, cv_kwargs

    Raises
    ------
    ValueError
        If the `task` in the configuration is not "classification" or "regression".
    KeyError
        If any required configuration parameters are missing.

    Notes
    -----
    This class serves as a high-level interface to either a classification or regression
    pipeline based on the `task` specified in the configuration. It abstracts away
    the details of the underlying pipeline classes, allowing users to focus on
    configuring the task and running the analysis without worrying about the specifics
    of the implementation.  

    Examples
    --------
    >>> cfg = {
    ...   "task": "classification",
    ...   "analysis_type": "baseline",
    ...   "models": ["Logistic Regression"],
    ...   "metrics": ["accuracy"],
    ...   "cv_strategy": "stratified",
    ...   "n_splits": 5,
    ...   "random_state": 42
    ... }
    >>> ml = MLPipeline(X, y, cfg)
    >>> results = ml.run()
    """

    def __init__(self, X, y, config):
        self.X = X
        self.y = y
        self.config = config
        # Task: classification or regression
        self.task = config.get("task")
        if self.task == "regression":
            self.pipeline_cls = RegressionPipeline
        elif self.task == "classification":
            self.pipeline_cls = ClassificationPipeline
        else:
            raise ValueError(f"Invalid task: {self.task}")

        # Mode: univariate (per-target) or multivariate (all targets)
        self.mode = config.get("mode", "multivariate")
        if self.mode not in ("univariate", "multivariate"):
            raise ValueError(f"Invalid mode: {self.mode!r}; must be 'univariate' or 'multivariate'")

        # Extract cv_kwargs without duplicates
        cv_kwargs = config.get("cv_kwargs", {})
        cv_kwargs.pop("cv_strategy", None)
        cv_kwargs.pop("n_splits", None)
        cv_kwargs.pop("random_state", None)
        self.cv_kwargs = cv_kwargs

    def run(self):
        """
        Run the ML pipeline.
        
        Return
        ------
        dict
            Mapping of model names (or output-column indices in univariate mode)
            to their result dicts. Each result dict now has at least:
                - 'model_name': Name of the evaluated model.
                - 'metric_scores': Scoring metrics aggregated across folds with keys 'mean', 'std', and 'fold_scores'.
                - 'feature_importances': Dictionary of feature importance statistics (mean, std, weighted values) or None.
                - 'predictions': Dictionary containing concatenated y_true, y_pred, and optionally y_proba from cross-validation.
                - 'params': The initial model parameters used during evaluation.
                - 'folds_estimators': List of fitted estimator instances from each fold of cross-validation.
            if the analysis_type is 'feature_selection' results will also include:
                - 'selected_features': The combined set of features selected across all CV folds.
                - 'feature_frequency': A dictionary mapping each feature name to its selection frequency.
                - 'feature_importances': A dictionary with weighted mean and std of importances across folds.
                - 'selected_per_fold': Dictionary mapping fold indices to lists of feature names selected in each fold.
                - 'best_fold': Information about the best-performing fold, including:
                    - 'fold': Index of the best fold.
                    - 'features': Features selected in that fold.
                    - <metric>: Metric score for that fold.
                    - 'estimator': The fitted estimator for the best fold.
                - 'fs_parameters': dict
                    Parameters used for feature selection, including:
                    - 'n_features': Number of features selected.
                    - 'direction': Direction of feature selection ('forward' or 'backward').
                    - 'scoring': Metric used for feature selection.
            if the analysis_type is 'hp_search' results will also include:
                - 'best_params': Aggregated best parameter settings determined by majority voting across folds.
                - 'param_frequency': Dictionary mapping each hyperparameter value to its frequency across folds.
                - 'best_params_per_fold': Dictionary mapping fold indices to their best parameter settings.
                - 'best_fold': Information about the best-performing fold, including:
                - 'hp_search_parameters': dict
                    Meta-information on the hyperparameter search containing:
                        - search type: The type of search performed ('grid' or 'random').
                        - param grid: The parameter grid provided (or from model_configs if None).
                        - scoring: The metric used for evaluation.
                        - n_iter: The number of parameter settings sampled (for randomized search).
            if the analysis_type is 'hp_search_fs', results will include a combination of
            feature selection and hyperparameter search results, including:
                - 'selected_features': The combined set of features selected across all CV folds.
                - 'feature_frequency': A dictionary mapping each feature name to its selection frequency.
                - 'feature_importances': A dictionary with weighted mean and std of importances across folds.
                - 'selected_per_fold': Dictionary mapping fold indices to lists of feature names selected in each fold.
                - 'best_fold': Information about the best-performing fold, including:
                    - 'fold': Index of the best fold.
                    - 'features': Features selected in that fold.
                    - <metric>: Metric score for that fold.
                    - 'estimator': The fitted estimator for the best fold.
                - 'fs_parameters': dict
                    Parameters used for feature selection, including:
                        - 'n_features': Number of features selected.
                        - 'direction': Direction of feature selection ('forward' or 'backward').
                        - 'scoring': Metric used for feature selection.

        Raises
        ------
        ValueError
            If the analysis_type is 'feature_selection' or 'hp_search_fs' in univariate mode.
        """
        # Common kwargs for pipeline instantiation
        common_kwargs = dict(
            X=self.X,
            y=None,  # to be set per run
            analysis_type=self.config.get("analysis_type", "baseline"),
            models=self.config.get("models", "all"),
            metrics=self.config.get("metrics", None),
            random_state=self.config.get("random_state", 42),
            cv_strategy=self.config.get("cv_strategy", "stratified"),
            n_splits=self.config.get("n_splits", 5),
            n_features=self.config.get("n_features", None),
            direction=self.config.get("direction", "forward"),
            search_type=self.config.get("search_type", "grid"),
            n_iter=self.config.get("n_iter", 100),
            scoring=self.config.get("scoring", None),
            n_jobs=self.config.get("n_jobs", -1),
            save_intermediate=self.config.get("save_intermediate", False),
            results_dir=self.config.get("results_dir", "results"),
            results_file=self.config.get("results_file", "results"),
            cv_kwargs=self.cv_kwargs
        )

        # Multivariate mode or single-output always treated as one run
        if self.mode == "multivariate" or getattr(self.y, 'ndim', 1) == 1:
            common_kwargs["y"] = self.y
            pipeline = self.pipeline_cls(**common_kwargs)
            return pipeline.run()

        # Univariate mode on each column of a 2D target
        # Feature selection-type analyses not supported per-target
        if common_kwargs["analysis_type"] in ("feature_selection", "hp_search_fs"):
            raise ValueError(f"Cannot perform {common_kwargs['analysis_type']} in univariate mode")

        results = {}
        # Iterate over each output column
        for idx in range(self.y.shape[1]):
            yi = self.y[:, idx]
            common_kwargs["y"] = yi
            pipeline = self.pipeline_cls(**common_kwargs)
            results[idx] = pipeline.run()
        return results
