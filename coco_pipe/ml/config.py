import numpy as np
from sklearn.metrics import (
    make_scorer,
    recall_score, accuracy_score, f1_score,
    precision_score, matthews_corrcoef, balanced_accuracy_score,
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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


REGRESSION_METRICS = {
    "r2": make_scorer(r2_score),
    "neg_mse": make_scorer(mean_squared_error, greater_is_better=False),
    "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "neg_rmse": make_scorer(
        lambda y_t, y_p: np.sqrt(mean_squared_error(y_t, y_p)),
        greater_is_better=False
    ),
}