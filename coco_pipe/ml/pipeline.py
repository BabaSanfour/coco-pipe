"""
High-level pipeline operations for machine learning tasks.

This module provides a unified interface for running ML analyses with flexible
feature grouping, supporting both classification and regression tasks. It handles:
- Task type detection and pipeline selection
- Feature group management
- Analysis execution with different subset options
- Results management and persistence
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple

from .classification import classification
from .regression import regression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class FeatureGroupManager:
    """
    Manage feature groups and subsets for ML analysis.
    
    This class handles:
    - Feature group organization
    - Feature subset creation
    - Global feature management
    """
    
    def __init__(self, X: pd.DataFrame, groups: Optional[Dict[str, List[str]]] = None,
                 global_features: Optional[List[str]] = None):
        self.X = X
        self.groups = groups or {}
        self.global_features = global_features or []
        self._validate_features()
        
    def _validate_features(self):
        """Validate feature groups and global features."""
        all_features = set(self.X.columns)
        
        # Validate global features
        invalid_global = set(self.global_features) - all_features
        if invalid_global:
            raise ValueError(f"Invalid global features: {invalid_global}")
            
        # Validate group features
        for group, features in self.groups.items():
            invalid_group = set(features) - all_features
            if invalid_group:
                raise ValueError(f"Invalid features in group {group}: {invalid_group}")
                
    def get_feature_subset(self, subset_type: str, group: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Get a subset of features based on specified criteria.
        
        Parameters
        ----------
        subset_type : str
            Type of subset to create:
            - "all": All features
            - "group": Features from specific group
            - "group_with_global": Group features plus global features
        group : str, optional
            Group name when using group-based subsets
            
        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            Feature subset and list of feature names
        """
        if subset_type == "all":
            return self.X, list(self.X.columns)
        elif subset_type == "group":
            if not group or group not in self.groups:
                raise ValueError(f"Invalid group: {group}")
            features = self.groups[group]
            return self.X[features], features
        elif subset_type == "group_with_global":
            if not group or group not in self.groups:
                raise ValueError(f"Invalid group: {group}")
            features = self.groups[group] + self.global_features
            return self.X[features], features
        else:
            raise ValueError(f"Invalid subset type: {subset_type}")

class MLPipeline:
    """
    Unified pipeline for ML tasks with automatic task type detection.
    
    This class provides:
    - Automatic task type detection (classification/regression)
    - Feature group management
    - Consistent interface for all ML tasks
    - Result collection and organization
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : Union[pd.Series, pd.DataFrame]
        Target variable(s)
    task_type : str, optional (default=None)
        Force specific task type ("classification" or "regression")
        If None, automatically detected
    feature_groups : Dict[str, List[str]], optional
        Feature group definitions
    global_features : List[str], optional
        Features to include in all groups
    cv_strategy : str, optional (default="stratified")
        Cross-validation strategy
    random_state : int, optional (default=42)
        Random state for reproducibility
    n_jobs : int, optional (default=-1)
        Number of parallel jobs
    """
    
    def __init__(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame],
                 task_type: Optional[str] = None, feature_groups: Optional[Dict[str, List[str]]] = None,
                 global_features: Optional[List[str]] = None, cv_strategy: str = "stratified",
                 random_state: int = 42, n_jobs: int = -1):
        self.X = X
        self.y = y
        self.cv_strategy = cv_strategy
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Setup feature management
        self.feature_manager = FeatureGroupManager(X, feature_groups, global_features)
        
        # Detect or set task type
        self.task_type = self._detect_task_type() if task_type is None else task_type
        if self.task_type not in ["classification", "regression"]:
            raise ValueError(f"Invalid task type: {self.task_type}")
            
        logging.info(f"Initialized {self.task_type} pipeline")
        
    def _detect_task_type(self) -> str:
        """
        Automatically detect whether this is a classification or regression task.
        
        Returns
        -------
        str
            "classification" or "regression"
        """
        y = np.array(self.y)
        
        # Check if multivariate
        if len(y.shape) == 2:
            # Check first column type
            y = y[:, 0]
            
        # Check if categorical/discrete
        unique_values = np.unique(y)
        if len(unique_values) < 10 or all(isinstance(val, (bool, np.bool_)) or float(val).is_integer() for val in unique_values):
            return "classification"
        return "regression"
        
    def run(self, analysis_type: str = "baseline", subset_type: str = "all",
            group: Optional[str] = None, models: Union[str, List[str]] = "all",
            metrics: Optional[Union[str, List[str]]] = None, n_features: Optional[int] = None,
            direction: str = "forward", search_type: str = "grid",
            n_iter: int = 100, scoring: Optional[str] = None) -> Dict:
        """
        Run ML analysis with specified configuration.
        
        Parameters
        ----------
        analysis_type : str, optional (default="baseline")
            Type of analysis to run:
            - "baseline": Basic model evaluation
            - "feature_selection": Feature selection
            - "hp_search": Hyperparameter optimization
        subset_type : str, optional (default="all")
            Type of feature subset to use:
            - "all": All features
            - "group": Features from specific group
            - "group_with_global": Group features plus global features
        group : str, optional
            Group name when using group-based subsets
        models : Union[str, List[str]], optional (default="all")
            Models to include in analysis
        metrics : Union[str, List[str]], optional
            Metrics to evaluate
        n_features : int, optional
            Number of features for feature selection
        direction : str, optional (default="forward")
            Direction for feature selection
        search_type : str, optional (default="grid")
            Type of hyperparameter search
        n_iter : int, optional (default=100)
            Number of iterations for random search
        scoring : str, optional
            Scoring metric for optimization
            
        Returns
        -------
        Dict
            Analysis results including:
            - task_type: Detected/specified task type
            - subset_info: Information about feature subset used
            - pipeline: The fitted pipeline object
            - results: Analysis results
        """
        # Get feature subset
        X_subset, features = self.feature_manager.get_feature_subset(subset_type, group)
        
        # Prepare common arguments
        common_args = {
            "X": X_subset,
            "y": self.y,
            "type": analysis_type,
            "models": models,
            "metrics": metrics,
            "random_state": self.random_state,
            "cv_strategy": self.cv_strategy,
            "n_jobs": self.n_jobs
        }
        
        # Add analysis-specific arguments
        if analysis_type == "feature_selection":
            common_args.update({
                "n_features": n_features,
                "direction": direction,
                "scoring": scoring
            })
        elif analysis_type == "hp_search":
            common_args.update({
                "search_type": search_type,
                "n_iter": n_iter,
                "scoring": scoring
            })
            
        # Run appropriate pipeline
        if self.task_type == "classification":
            results = classification(**common_args)
        else:
            results = regression(**common_args)
            
        # Add metadata
        results.update({
            'task_type': self.task_type,
            'subset_info': {
                'type': subset_type,
                'group': group,
                'n_features': len(features),
                'features': features
            }
        })
        
        return results
    
    def run_group_analysis(self, analysis_config: List[Dict]) -> Dict[str, Dict]:
        """
        Run multiple analyses across feature groups.
        
        Parameters
        ----------
        analysis_config : List[Dict]
            List of analysis configurations, each containing:
            - name: Analysis name
            - type: Analysis type
            - subset_type: Feature subset type
            - group: Group name (optional)
            - models: Models to use
            - metrics: Metrics to evaluate
            - Other analysis-specific parameters
            
        Returns
        -------
        Dict[str, Dict]
            Results for each analysis
        """
        results = {}
        for config in analysis_config:
            name = config.pop('name', 'unnamed_analysis')
            logging.info(f"Running analysis: {name}")
            try:
                results[name] = self.run(**config)
            except Exception as e:
                logging.error(f"Error in analysis {name}: {str(e)}")
                results[name] = {'error': str(e)}
        return results

def run_ml_analysis(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    task_type: Optional[str] = None,
    analysis_config: Optional[List[Dict]] = None,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    global_features: Optional[List[str]] = None,
    cv_strategy: str = "stratified",
    random_state: int = 42,
    n_jobs: int = -1
) -> Dict:
    """
    High-level function to run ML analyses.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : Union[pd.Series, pd.DataFrame]
        Target variable(s)
    task_type : str, optional
        Force specific task type ("classification" or "regression")
    analysis_config : List[Dict], optional
        List of analysis configurations
    feature_groups : Dict[str, List[str]], optional
        Feature group definitions
    global_features : List[str], optional
        Features to include in all groups
    cv_strategy : str, optional (default="stratified")
        Cross-validation strategy
    random_state : int, optional (default=42)
        Random state for reproducibility
    n_jobs : int, optional (default=-1)
        Number of parallel jobs
        
    Returns
    -------
    Dict
        Analysis results
    """
    pipeline = MLPipeline(
        X=X,
        y=y,
        task_type=task_type,
        feature_groups=feature_groups,
        global_features=global_features,
        cv_strategy=cv_strategy,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    if analysis_config:
        return pipeline.run_group_analysis(analysis_config)
    else:
        # Run default baseline analysis
        return pipeline.run() 