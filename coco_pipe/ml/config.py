"""
coco_pipe/ml/config.py
-----------------
Default cross-validation parameters, metric functions, and model configurations
for binary, multiclass, multi-output, and regression tasks.
"""
import numpy as np
from typing import Callable, Dict, Any
from sklearn.metrics import (
    recall_score, accuracy_score, f1_score,
    precision_score, matthews_corrcoef, balanced_accuracy_score,
    r2_score, mean_squared_error, mean_absolute_error, explained_variance_score,
    max_error, roc_auc_score, average_precision_score,
    hamming_loss, jaccard_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso,
    ElasticNet
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

# Default cross-validation kwargs
DEFAULT_CV: Dict[str, Any] = {
    "cv_strategy": "stratified",
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Classification metrics (raw functions)
CLASSIFICATION_METRICS: Dict[str, Callable] = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted"),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted"),
    "mcc": matthews_corrcoef,
}

# Binary-specific metrics
BINARY_METRICS: Dict[str, Callable] = {
    **CLASSIFICATION_METRICS,
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
}

def multiclass_roc_auc_score(y_true, y_proba):
    """
    Compute one-vs-rest multiclass ROC AUC.
    """
    # Ensure y_proba is 2D
    if y_proba.ndim == 1:
        y_proba = y_proba.reshape(-1, 1)
    
    # Ensure we have the right shape
    if y_proba.shape[0] != len(y_true):
        raise ValueError(f"Shape mismatch: y_true has {len(y_true)} samples, y_proba has {y_proba.shape[0]}")
    
    classes = np.unique(y_true)
    
    # Handle binary case
    if len(classes) == 2:
        if y_proba.shape[1] == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        elif y_proba.shape[1] == 1:
            return roc_auc_score(y_true, y_proba[:, 0])
        else:
            return roc_auc_score(y_true, y_proba)
    
    # Handle multiclass case
    if y_proba.shape[1] == len(classes):
        return roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
    else:
        # Shape mismatch - fallback to simpler approach
        try:
            y_true_bin = label_binarize(y_true, classes=classes)
            if y_true_bin.shape[1] == 1:
                # Binary case disguised as multiclass
                return roc_auc_score(y_true_bin[:, 0], y_proba[:, 0])
            return roc_auc_score(y_true_bin, y_proba, average='weighted', multi_class='ovr')
        except Exception:
            return 0.5
            
            
            # Multiclass metrics
MULTICLASS_METRICS: Dict[str, Callable] = {
    **CLASSIFICATION_METRICS,
    "roc_auc": multiclass_roc_auc_score,
}

# Multi-output classification metrics
MULTIOUTPUT_CLASS_METRICS: Dict[str, Callable] = {
    "subset_accuracy": accuracy_score,
    "hamming_loss": lambda y_true, y_pred: 1.0 - hamming_loss(y_true, y_pred),
    "jaccard_samples": lambda y_true, y_pred: jaccard_score(
        y_true, y_pred, average="samples", zero_division=0
    ),
    "precision_samples": lambda y_true, y_pred: precision_score(
        y_true, y_pred, average="samples", zero_division=0
    ),
    "recall_samples": lambda y_true, y_pred: recall_score(
        y_true, y_pred, average="samples", zero_division=0
    ),
    "f1_samples": lambda y_true, y_pred: f1_score(
        y_true, y_pred, average="samples", zero_division=0
    ),
}

# Regression metrics
REGRESSION_METRICS: Dict[str, Callable] = {
    "r2": r2_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "explained_variance": explained_variance_score,
    "max_error": max_error,
}

# Multi-output regression metrics
def mean_r2_score(y_true, y_pred):
    """Mean R2 over multiple outputs."""
    return np.mean([r2_score(y_true[:, i], y_pred[:, i])
                    for i in range(y_true.shape[1])])

def neg_mean_mse(y_true, y_pred):
    """Negative mean MSE over multiple outputs."""
    return -np.mean([mean_squared_error(y_true[:, i], y_pred[:, i])
                     for i in range(y_true.shape[1])])

def neg_mean_mae(y_true, y_pred):
    """Negative mean MAE over multiple outputs."""
    return -np.mean([mean_absolute_error(y_true[:, i], y_pred[:, i])
                     for i in range(y_true.shape[1])])

MULTIOUTPUT_REG_METRICS: Dict[str, Callable] = {
    **REGRESSION_METRICS,
    "mean_r2": mean_r2_score,
    "neg_mean_mse": neg_mean_mse,
    "neg_mean_mae": neg_mean_mae,
}

# Binary classification models
BINARY_MODELS: Dict[str, Dict[str, Any]] = {
    "Logistic Regression": {
        "estimator": LogisticRegression,
        "default_params": {"random_state": 42},
        "hp_search_params": {"C": [0.1, 1, 10], "penalty": ["l2"]},
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier,
        "default_params": {"random_state": 42},
        "hp_search_params": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    },
    "Random Forest": {
        "estimator": RandomForestClassifier,
        "default_params": {"random_state": 42},
        "hp_search_params": {"n_estimators": [100, 200], "max_depth": [3, 5, 10, None]},
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier,
        "default_params": {"random_state": 42},
        "hp_search_params": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},
    },
    "SVC": {
        "estimator": SVC,
        "default_params": {"probability": True, "random_state": 42},
        "hp_search_params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    },
    "KNN": {
        "estimator": KNeighborsClassifier,
        "default_params": {},
        "hp_search_params": {"n_neighbors": [3, 5, 7]},
    },
    "Extra Trees": {
        "estimator": ExtraTreesClassifier,
        "default_params": {"random_state": 42},
        "hp_search_params": {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
    },
    "AdaBoost": {
        "estimator": AdaBoostClassifier,
        "default_params": {"random_state": 42},
        "hp_search_params": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
    },
    "HistGradientBoosting": {
        "estimator": HistGradientBoostingClassifier,
        "default_params": {"random_state": 42},
        "hp_search_params": {"max_iter": [100, 200], "learning_rate": [0.01, 0.1]},
    },
}

# Multiclass classification models (same as binary for most)
MULTICLASS_MODELS: Dict[str, Dict[str, Any]] = {
    name: cfg for name, cfg in BINARY_MODELS.items() if name in [
        "Logistic Regression", "Decision Tree", "Random Forest",
        "Gradient Boosting", "SVC", "KNN", "Extra Trees", "AdaBoost", "HistGradientBoosting"
    ]
}

# Multi-output classification models
MULTIOUTPUT_CLASS_MODELS: Dict[str, Dict[str, Any]] = {
    name: {
        "estimator": MultiOutputClassifier,
        "default_params": {"estimator": cfg["estimator"](**cfg["default_params"])},
        "hp_search_params": {f"estimator__{k}": v for k, v in cfg["hp_search_params"].items()}
    }
    for name, cfg in MULTICLASS_MODELS.items()
}

# Regression models
REGRESSION_MODELS: Dict[str, Dict[str, Any]] = {
    "Linear Regression": {
        "estimator": LinearRegression,
        "default_params": {},
        "hp_search_params": {}
    },
    "Ridge": {
        "estimator": Ridge,
        "default_params": {"random_state": 42},
        "hp_search_params": {"alpha": [0.1, 1.0, 10.0]}
    },
    "Lasso": {
        "estimator": Lasso,
        "default_params": {"random_state": 42},
        "hp_search_params": {"alpha": [0.1, 1.0, 10.0]}
    },
    "ElasticNet": {
        "estimator": ElasticNet,
        "default_params": {"random_state": 42},
        "hp_search_params": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]}
    },
    "Decision Tree Regr": {
        "estimator": DecisionTreeRegressor,
        "default_params": {"random_state": 42},
        "hp_search_params": {"max_depth": [None, 3, 5], "min_samples_split": [2, 5]}
    },
    "Random Forest Regr": {
        "estimator": RandomForestRegressor,
        "default_params": {"random_state": 42},
        "hp_search_params": {"n_estimators": [100, 200], "max_depth": [3, 5, None]}
    },
    "Gradient Boosting Regr": {
        "estimator": GradientBoostingRegressor,
        "default_params": {"random_state": 42},
        "hp_search_params": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}
    },
    "SVR": {
        "estimator": SVR,
        "default_params": {},
        "hp_search_params": {"kernel": ["linear", "rbf"], "C": [0.1, 1.0]}
    },
    "KNN Regr": {
        "estimator": KNeighborsRegressor,
        "default_params": {},
        "hp_search_params": {"n_neighbors": [3, 5, 7]}
    },
    "Extra Trees Regr": {
        "estimator": ExtraTreesRegressor,
        "default_params": {"random_state": 42},
        "hp_search_params": {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
    },
    "AdaBoost Regr": {
        "estimator": AdaBoostRegressor,
        "default_params": {"random_state": 42},
        "hp_search_params": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]}
    },
    "HistGradientBoosting Regr": {
        "estimator": HistGradientBoostingRegressor,
        "default_params": {"random_state": 42},
        "hp_search_params": {"max_iter": [100, 200], "learning_rate": [0.01, 0.1]}
    }
}

# Multi-output regression models
MULTIOUTPUT_REG_MODELS: Dict[str, Dict[str, Any]] = {
    name: {
        "estimator": MultiOutputRegressor,
        "default_params": {"estimator": cfg["estimator"](**cfg["default_params"])},
        "hp_search_params": {f"estimator__{k}": v for k, v in cfg["hp_search_params"].items()}
    }
    for name, cfg in REGRESSION_MODELS.items()
}
