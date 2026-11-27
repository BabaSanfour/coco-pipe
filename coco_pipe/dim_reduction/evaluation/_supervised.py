"""
Private supervised scoring helpers for dim-reduction evaluation.

These helpers are intentionally local to ``coco_pipe.dim_reduction``. They are
used to score embedding separability and are not part of the decoding API.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ...utils import balanced_sample_weights


def _cv_random_state(shuffle: bool, random_state: int | None) -> int | None:
    return random_state if shuffle else None


def _make_splitter(
    strategy: str,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int | None,
    groups: np.ndarray | None,
):
    if strategy == "stratified_group_kfold":
        if groups is None:
            raise ValueError("groups are required for stratified_group_kfold.")
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=_cv_random_state(shuffle, random_state),
        )
    if strategy == "group_kfold":
        if groups is None:
            raise ValueError("groups are required for group_kfold.")
        return GroupKFold(n_splits=n_splits)
    if strategy == "stratified":
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=_cv_random_state(shuffle, random_state),
        )
    raise ValueError(f"Unsupported supervised scoring CV strategy: {strategy}.")


def _cross_validate_score(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: Sequence,
    *,
    groups: Sequence | None = None,
    cv_strategy: str = "stratified",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = 42,
    metric: str = "balanced_accuracy",
    use_scaler: bool = False,
    sample_weight: str | None = None,
) -> float:
    """
    Compute a mean supervised CV score for dim-reduction separation metrics.

    This private helper is deliberately narrow. It currently exists for the
    separation metrics and supports only the CV strategies used by
    dim-reduction evaluation. With ``sample_weight="group"``, the scaling, the
    fit and the score weight each group once within each class
    (:func:`~coco_pipe.utils.balanced_sample_weights`); the estimator must then
    have no ``class_weight``.
    """
    if sample_weight not in (None, "group"):
        raise ValueError(
            f"sample_weight must be None or 'group', got {sample_weight!r}."
        )
    if sample_weight == "group" and groups is None:
        raise ValueError("sample_weight='group' needs groups.")
    if metric != "balanced_accuracy":
        raise ValueError(
            "Dim-reduction supervised scoring currently supports only "
            "'balanced_accuracy'."
        )

    X_values = np.asarray(X)
    y_values = np.asarray(y).reshape(-1)
    if len(X_values) != len(y_values):
        raise ValueError("X and y must have matching sample counts.")
    # integer classes: a random forest with class_weight="balanced" fails on
    # text labels in scikit-learn 1.9
    y_values = LabelEncoder().fit_transform(y_values)

    group_values = None
    if groups is not None:
        group_values = np.asarray(groups).reshape(-1)
        if len(group_values) != len(y_values):
            raise ValueError("groups must align with X and y.")

    splitter = _make_splitter(
        cv_strategy,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        groups=group_values,
    )
    base_estimator = estimator
    if use_scaler:
        base_estimator = Pipeline(
            [("scaler", StandardScaler()), ("clf", clone(estimator))]
        )

    scores = []
    for train_idx, test_idx in splitter.split(X_values, y_values, group_values):
        y_train, y_test = y_values[train_idx], y_values[test_idx]
        fit_params, test_weights = {}, None
        if sample_weight == "group":
            weights = balanced_sample_weights(y_train, group_values[train_idx])
            fit_params = (
                {"scaler__sample_weight": weights, "clf__sample_weight": weights}
                if use_scaler
                else {"sample_weight": weights}
            )
            test_weights = balanced_sample_weights(y_test, group_values[test_idx])
        model = clone(base_estimator)
        model.fit(X_values[train_idx], y_train, **fit_params)
        score = balanced_accuracy_score(
            y_test, model.predict(X_values[test_idx]), sample_weight=test_weights
        )
        scores.append(float(score))

    return float(np.nanmean(scores)) if scores else float("nan")
