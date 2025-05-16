import numpy as np
from sklearn.metrics import (
    make_scorer,
    recall_score, accuracy_score, f1_score,
    precision_score, matthews_corrcoef, balanced_accuracy_score,
    r2_score, mean_squared_error, mean_absolute_error
)

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

REGRESSION_METRICS = {
    "r2": make_scorer(r2_score),
    "neg_mse": make_scorer(mean_squared_error, greater_is_better=False),
    "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "neg_rmse": make_scorer(
        lambda y_t, y_p: np.sqrt(mean_squared_error(y_t, y_p)),
        greater_is_better=False
    ),
}