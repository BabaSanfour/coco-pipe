"""
Private supervised scoring helpers for dim-reduction evaluation.

These helpers are intentionally local to ``coco_pipe.dim_reduction``. They are
used to score embedding separability and are not part of the decoding API.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _cv_random_state(shuffle: bool, random_state: Optional[int]) -> Optional[int]:
    return random_state if shuffle else None


def _make_splitter(
    strategy: str,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: Optional[int],
    groups: Optional[np.ndarray],
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
    groups: Optional[Sequence] = None,
    cv_strategy: str = "stratified",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = 42,
    metric: str = "balanced_accuracy",
    use_scaler: bool = False,
) -> float:
    """
    Compute a mean supervised CV score for dim-reduction separation metrics.

    This private helper is deliberately narrow. It currently exists for
    ``separation_logreg_balanced_accuracy`` and supports only the CV strategies
    used by dim-reduction evaluation.
    """
    if metric != "balanced_accuracy":
        raise ValueError(
            "Dim-reduction supervised scoring currently supports only "
            "'balanced_accuracy'."
        )

    X_values = np.asarray(X)
    y_values = np.asarray(y).reshape(-1)
    if len(X_values) != len(y_values):
        raise ValueError("X and y must have matching sample counts.")

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
        model = clone(base_estimator)
        model.fit(X_values[train_idx], y_values[train_idx])
        y_pred = model.predict(X_values[test_idx])
        scores.append(float(balanced_accuracy_score(y_values[test_idx], y_pred)))

    return float(np.nanmean(scores)) if scores else float("nan")
