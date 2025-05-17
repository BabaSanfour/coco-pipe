import numpy as np
from sklearn.metrics import (
    make_scorer,
    recall_score, accuracy_score, f1_score,
    precision_score, matthews_corrcoef, balanced_accuracy_score,
    r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, 
    max_error, roc_auc_score, average_precision_score,
    hamming_loss, jaccard_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR

DEFAULT_CV = {
    "strategy": "stratified",
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
    "n_groups": 1,
}

CLASSIFICATION_METRICS = {
    "accuracy": make_scorer(accuracy_score),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "f1": make_scorer(f1_score, average="weighted"),
    "precision": make_scorer(precision_score, average="weighted"),
    "recall": make_scorer(recall_score, average="weighted"),
    "mcc": make_scorer(matthews_corrcoef),
}

BINARY_METRICS = {
    **CLASSIFICATION_METRICS,
    "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
    "average_precision": make_scorer(average_precision_score, needs_proba=True),
}

MULTIOUTPUT_METRICS = {
    # Strict subset accuracy: 1 iff all labels correct
    "subset_accuracy": make_scorer(accuracy_score),
    # fraction of incorrect labels (we invert so higher is better)
    "hamming_loss": lambda y_true, y_pred: 1.0 - hamming_loss(y_true, y_pred),

    # intersection over union per sample
    "jaccard_samples": lambda y_true, y_pred: jaccard_score(
        y_true, y_pred, average="samples", zero_division=0
    ),

    # per-sample precision/recall/f1
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

# Default model configurations for binary classification
BINARY_MODELS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(random_state=0),
        "params": {"C": [0.1, 1, 10], "penalty": ["l2"]},
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=0),
        "params": {"max_depth": [3,5,10,None], "min_samples_split": [2,5,10]},
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=0),
        "params": {"n_estimators": [100,200], "max_depth": [3,5,10,None]},
    },
    "SVC": {
        "estimator": SVC(random_state=0, probability=True),
        "params": {"C": [0.1,1,10], "kernel": ["linear","rbf"]},
    },
}

MULTICLASS_MODELS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(
            multi_class="multinomial", solver="lbfgs", random_state=42, max_iter=1000
        ),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs", "saga"],
            "multi_class": ["multinomial"],
        },
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 3, 5],
            "min_samples_split": [2, 5],
            "criterion": ["gini", "entropy"],
        },
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
            "min_samples_split": [2, 5],
        },
    },
    "SVC": {
        "estimator": SVC(probability=True, random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],
            "decision_function_shape": ["ovr", "ovo"],
        },
    },
}

REGRESSION_METRICS = {
    "r2": r2_score,
    "neg_mse": lambda y_true, y_pred: -mean_squared_error(y_true, y_pred),
    "neg_mae": lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred),
    "neg_rmse": lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)), 
    "explained_variance": explained_variance_score,
    "neg_max_error": lambda y_true, y_pred: -max_error(y_true, y_pred),
}

REGRESSION_MODELS = {
    "Linear Regression": {
        "estimator": LinearRegression(),
        "params": {}
    },
    "Ridge": {
        "estimator": Ridge(random_state=42),
        "params": {"alpha": [0.1, 1.0, 10.0]}
    },
    "Lasso": {
        "estimator": Lasso(random_state=42),
        "params": {"alpha": [0.1, 1.0, 10.0]}
    },
    "ElasticNet": {
        "estimator": ElasticNet(random_state=42),
        "params": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]}
    },
    "Random Forest": {
        "estimator": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 3, 5],
            "min_samples_split": [2, 5]
        }
    },
    "SVR": {
        "estimator": SVR(),
        "params": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1.0, 10.0],
            "epsilon": [0.1, 0.2]
        }
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    },
}