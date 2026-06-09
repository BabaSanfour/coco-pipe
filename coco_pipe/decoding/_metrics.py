"""
Internal Metric Registry for Decoding.
=====================================

This module defines the registry of available metrics for classification
and regression tasks. It provides metadata about each metric, such as
the required estimator response (predict vs predict_proba) and whether
higher values are better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    zero_one_loss,
)

from ._constants import MetricFamily, MetricTask, ResponseMethod


@dataclass(frozen=True)
class MetricSpec:
    """
    Metadata specification for a decoding metric.

    This container stores all necessary information to resolve and compute
    a metric, including its task type, family for reporting, and the
    required estimator response method.

    Parameters
    ----------
    name : str
        The unique identifier for the metric.
    task : MetricTask
        The type of task ("classification" or "regression").
    scorer : Callable
        The callable with signature ``(y_true, y_pred) -> float``.
    response_method : ResponseMethod, default="predict"
        The estimator method required to produce ``y_pred`` (e.g., "proba").
    family : MetricFamily, default="label"
        The category of the metric for reporting and visualization.
    greater_is_better : bool, default=True
        Whether a higher score indicates better performance.

    Examples
    --------
    >>> from sklearn.metrics import accuracy_score
    >>> spec = MetricSpec("accuracy", "classification", accuracy_score)
    >>> spec.name
    'accuracy'
    """

    name: str
    task: MetricTask
    scorer: Callable
    response_method: ResponseMethod = "predict"
    family: MetricFamily = "label"
    greater_is_better: bool = True


def _specificity_score(y_true, y_pred) -> float:
    """
    Compute specificity (recall of the negative class).

    In many clinical or brain-decoding tasks, the ability to correctly
    identify non-targets (negatives) is as important as identifying
    targets. Specificity measures the true negative rate ($TN / (TN + FP)$).

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    score : float
        The calculated specificity.

    Examples
    --------
    >>> _specificity_score([0, 1], [0, 1])
    1.0

    See Also
    --------
    _sensitivity_score : True positive rate.
    """
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


def _sensitivity_score(y_true, y_pred) -> float:
    """
    Compute sensitivity (recall of the positive class).

    This implementation ensures binary-only enforcement and robust
    zero-division handling for unbalanced neuroimaging datasets. It
    measures the true positive rate ($TP / (TP + FN)$).

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    score : float
        The calculated sensitivity.

    Raises
    ------
    ValueError
        If the input labels contain more than 2 classes.

    Examples
    --------
    >>> _sensitivity_score([0, 1], [0, 1])
    1.0

    See Also
    --------
    _specificity_score : True negative rate.
    """
    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError("Sensitivity is only defined for binary classification.")
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def _pr_auc_score(y_true, probas_pred) -> float:
    """
    Compute Area Under the Precision-Recall Curve via trapezoidal integration.

    PR-AUC computed via trapezoidal integration can differ from Average
    Precision (AP) on imbalanced datasets. This method directly
    approximates the integral of the PR curve, providing a more direct
    estimate of the curve's area for scientific reporting.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    probas_pred : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    score : float
        The calculated PR-AUC.

    Examples
    --------
    >>> _pr_auc_score([0, 1], [0.1, 0.9])
    1.0

    See Also
    --------
    sklearn.metrics.average_precision_score : Standard AP implementation.
    """
    # Note: precision_recall_curve returns precision, recall, thresholds
    # where recall is in descending order; auc() handles this.
    precision, recall, _ = precision_recall_curve(y_true, probas_pred)
    return float(auc(recall, precision))


METRIC_REGISTRY: dict[str, MetricSpec] = {
    # Classification from hard predictions (family="label" or "confusion")
    "accuracy": MetricSpec("accuracy", "classification", accuracy_score),
    "balanced_accuracy": MetricSpec(
        "balanced_accuracy",
        "classification",
        balanced_accuracy_score,
        family="confusion",
    ),
    "f1": MetricSpec(
        "f1",
        "classification",
        lambda y, p: f1_score(y, p, average="weighted"),
        family="confusion",
    ),
    "f1_macro": MetricSpec(
        "f1_macro",
        "classification",
        lambda y, p: f1_score(y, p, average="macro"),
        family="confusion",
    ),
    "f1_micro": MetricSpec(
        "f1_micro",
        "classification",
        lambda y, p: f1_score(y, p, average="micro"),
        family="confusion",
    ),
    "precision": MetricSpec(
        "precision",
        "classification",
        lambda y, p: precision_score(y, p, average="weighted", zero_division=0),
        family="confusion",
    ),
    "recall": MetricSpec(
        "recall",
        "classification",
        lambda y, p: recall_score(y, p, average="weighted", zero_division=0),
        family="confusion",
    ),
    "sensitivity": MetricSpec(
        "sensitivity",
        "classification",
        _sensitivity_score,
        family="confusion",
    ),
    "specificity": MetricSpec(
        "specificity",
        "classification",
        _specificity_score,
        family="confusion",
    ),
    "zero_one_loss": MetricSpec(
        "zero_one_loss",
        "classification",
        zero_one_loss,
        family="label",
        greater_is_better=False,
    ),
    "hamming_loss": MetricSpec(
        "hamming_loss",
        "classification",
        hamming_loss,
        family="label",
        greater_is_better=False,
    ),
    "jaccard": MetricSpec(
        "jaccard",
        "classification",
        lambda y, p: jaccard_score(y, p, average="weighted"),
        family="confusion",
    ),
    "matthews_corrcoef": MetricSpec(
        "matthews_corrcoef",
        "classification",
        matthews_corrcoef,
        family="confusion",
    ),
    "cohen_kappa": MetricSpec(
        "cohen_kappa",
        "classification",
        cohen_kappa_score,
        family="confusion",
    ),
    # Classification from probabilities (family="threshold_sweep")
    "roc_auc": MetricSpec(
        "roc_auc",
        "classification",
        roc_auc_score,
        "proba_or_score",
        family="threshold_sweep",
    ),
    "roc_auc_ovr_weighted": MetricSpec(
        "roc_auc_ovr_weighted",
        "classification",
        lambda y, p: roc_auc_score(y, p, multi_class="ovr", average="weighted"),
        "proba",
        family="threshold_sweep",
    ),
    "average_precision": MetricSpec(
        "average_precision",
        "classification",
        average_precision_score,
        "proba_or_score",
        family="threshold_sweep",
    ),
    "pr_auc": MetricSpec(
        "pr_auc",
        "classification",
        _pr_auc_score,
        "proba_or_score",
        family="threshold_sweep",
    ),
    "log_loss": MetricSpec(
        "log_loss",
        "classification",
        log_loss,
        "proba",
        family="score_probability",
        greater_is_better=False,
    ),
    "brier_score": MetricSpec(
        "brier_score",
        "classification",
        brier_score_loss,
        "proba",
        family="calibration",
        greater_is_better=False,
    ),
    # Regression (family="regression")
    "r2": MetricSpec("r2", "regression", r2_score, family="regression"),
    "neg_mean_squared_error": MetricSpec(
        "neg_mean_squared_error",
        "regression",
        lambda y, p: -mean_squared_error(y, p),
        family="regression",
        greater_is_better=True,
    ),
    "neg_mean_absolute_error": MetricSpec(
        "neg_mean_absolute_error",
        "regression",
        lambda y, p: -mean_absolute_error(y, p),
        family="regression",
        greater_is_better=True,
    ),
    "explained_variance": MetricSpec(
        "explained_variance",
        "regression",
        explained_variance_score,
        family="regression",
    ),
    "neg_root_mean_squared_error": MetricSpec(
        "neg_root_mean_squared_error",
        "regression",
        lambda y, p: -float(np.sqrt(mean_squared_error(y, p))),
        family="regression",
        greater_is_better=True,
    ),
}


def get_scorer(name: str) -> Callable:
    """
    Retrieve the callable scoring function for a given metric.

    Returns the bare function required by scikit-learn's `fit` and
    `score` APIs.

    Parameters
    ----------
    name : str
        Metric name, for example ``accuracy`` or ``neg_mean_squared_error``.

    Returns
    -------
    scorer : Callable
        Metric function with signature ``(y_true, y_pred) -> float``.

    Raises
    ------
    ValueError
        If the metric name is not found in the registry.

    Examples
    --------
    >>> scorer = get_scorer("accuracy")
    >>> scorer([0, 1], [0, 1])
    1.0

    See Also
    --------
    get_metric_spec : Retrieve the full metadata object.
    """
    return get_metric_spec(name).scorer


def get_metric_spec(name: str) -> MetricSpec:
    """
    Return the full metadata specification for a given metric.

    The specification includes the task type, response method, and
    reporting family for the metric.

    Parameters
    ----------
    name : str
        The unique name of the metric to look up.

    Returns
    -------
    spec : MetricSpec
        The metadata object for the requested metric.

    Raises
    ------
    ValueError
        If the metric name is not found in the registry.

    Examples
    --------
    >>> spec = get_metric_spec("roc_auc")
    >>> spec.response_method
    'proba_or_score'

    See Also
    --------
    get_scorer : Retrieve the bare scoring function.
    """
    if name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{name}'. Available: "
            f"{sorted(list(METRIC_REGISTRY.keys()))}"
        )
    return METRIC_REGISTRY[name]


def get_metric_names(
    task: MetricTask | None = None,
    family: MetricFamily | None = None,
) -> list[str]:
    """
    Return a list of known metric names, optionally filtered.

    Enables dynamic discovery of metrics based on the current decoding
    task or the desired reporting family.

    Parameters
    ----------
    task : MetricTask, optional
        Filter by task type ("classification" or "regression").
    family : MetricFamily, optional
        Filter by metric family (e.g., "confusion").

    Returns
    -------
    names : list of str
        The sorted list of matching metric names.

    Examples
    --------
    >>> get_metric_names(task="regression")
    ['explained_variance', ..., 'r2']

    See Also
    --------
    get_metric_families : Discover available reporting families.
    """
    return sorted(
        name
        for name, spec in METRIC_REGISTRY.items()
        if (task is None or spec.task == task)
        and (family is None or spec.family == family)
    )


def get_metric_families(task: MetricTask | None = None) -> list[MetricFamily]:
    """
    Return a list of known metric families, optionally filtered.

    Returns the categories of metrics available, useful for generating
    comprehensive reporting dashboards.

    Parameters
    ----------
    task : MetricTask, optional
        Filter families by the task type they support.

    Returns
    -------
    families : list of MetricFamily
        The sorted list of matching metric families.

    Examples
    --------
    >>> get_metric_families(task="classification")
    ['calibration', 'confusion', 'label', 'score_probability', 'threshold_sweep']

    See Also
    --------
    get_metric_names : Retrieve the individual metric identifiers.
    """
    return sorted(
        {
            spec.family
            for spec in METRIC_REGISTRY.values()
            if task is None or spec.task == task
        }
    )
