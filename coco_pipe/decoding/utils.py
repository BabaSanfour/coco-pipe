"""
Decoding Utilities
==================

Helper functions and classes for the decoding module, primarily focused on 
Cross-Validation (CV) strategy management.

This module provides:
- `get_cv_splitter`: A factory function to instantiate Scikit-Learn cross-validators 
  from a Pydantic `CVConfig`.
- `SimpleSplit`: A custom validator for a single train/test split.
- `_CVWithGroups`: A wrapper to ensure group constraints are respected even when 
  Scikit-Learn's `cross_val_score` internals might obscure them.
"""
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from typing import Callable

from .configs import CVConfig


class _CVWithGroups(BaseCrossValidator):
    """
    Internal wrapper to bind specific groups to a CV splitter.

    This ensures that `.split(X, y)` always uses the strict `groups` provided 
    at initialization, ignoring any groups passed at runtime. This is critical 
    for preventing data leakage when complex grouping logic is defined upstream.

    Parameters
    ----------
    cv : BaseCrossValidator
        The underlying Scikit-Learn cross-validator (e.g., GroupKFold).
    groups : array-like
        The group labels to enforce for all splits.
    """

    def __init__(self, cv, groups):
        self.cv = cv
        self.groups = groups

    def split(self, X, y=None, groups=None):
        # ignore incoming groups, always use our stored one
        return self.cv.split(X, y, self.groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.cv.get_n_splits(X, y, self.groups)


class SimpleSplit(BaseCrossValidator):
    """
    A unified 1-fold CV strategy wrapping `train_test_split`.

    This allows "hold-out" validation to be treated as a Cross-Validation 
    strategy with `n_splits=1`, integrating seamlessly into loops that 
    expect a generator of indices.

    Parameters
    ----------
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split.
    stratify : array-like, optional
        If not None, data is split in a stratified fashion, using this array 
        as the class labels.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        stratify: Optional[Union[pd.Series, np.ndarray]] = None,
    ):
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1.")
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[Sequence] = None,
    ):
        """
        Yield a single (train_index, test_index) tuple.
        """
        idx = np.arange(len(X))
        strat = self.stratify if self.stratify is not None else None
        train_idx, test_idx = train_test_split(
            idx,
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
            stratify=strat,
        )
        yield train_idx, test_idx

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Always returns 1 split."""
        return 1


def get_cv_splitter(config: CVConfig, groups: Optional[Sequence] = None) -> BaseCrossValidator:
    """
    Factory function to create a Scikit-Learn compliant cross-validator.

    Constructs the appropriate splitter based on the provided `CVConfig` strategy.
    If `groups` are provided, they are bound to the splitter using `_CVWithGroups`
    to guarantee consistent grouping across pipeline steps.

    Parameters
    ----------
    config : CVConfig
        Validated configuration object specifying:
        - strategy: 'stratified', 'kfold', 'group_kfold', 'leave_p_out', etc.
        - n_splits: Number of folds (where applicable).
        - shuffle: Whether to shuffle data (where applicable).
        - random_state: Seed for reproducibility.
    groups : sequence, optional
        Group labels for the samples. Required for 'group_kfold', 'leave_p_out', 
        and 'stratified_group_kfold'.
        If provided, the returned validator will ignore any groups passed to its 
        `.split()` method and use these instead.

    Returns
    -------
    BaseCrossValidator
        An initialized cross-validator instance.

    Raises
    ------
    ValueError
        If an unknown CV strategy is specified or if required parameters (like 
        n_groups for leave_p_out) are missing from the configuration.
    """
    strat = config.strategy.lower()
    
    # Common arguments
    common_kwargs = {}
    if strat not in ["leave_one_out", "leave_p_out", "split"]:
        common_kwargs["n_splits"] = config.n_splits
    
    if strat in ["stratified", "kfold", "stratified_group_kfold", "split"]:
        common_kwargs["shuffle"] = config.shuffle
        common_kwargs["random_state"] = config.random_state if config.shuffle else None

    # Strategy Selection
    if strat == "stratified":
        splitter = StratifiedKFold(**common_kwargs)

    elif strat == "kfold":
        splitter = KFold(**common_kwargs)

    elif strat == "group_kfold":
        # GroupKFold doesn't take shuffle/random_state
        splitter = GroupKFold(n_splits=config.n_splits)

    elif strat == "stratified_group_kfold":
        splitter = StratifiedGroupKFold(**common_kwargs)

    elif strat == "leave_p_out":
        splitter = LeavePGroupsOut(n_groups=config.n_splits)

    elif strat == "leave_one_out":
        splitter = LeaveOneGroupOut()

    elif strat == "split":
        splitter = SimpleSplit(
            test_size=0.2,
            shuffle=config.shuffle,
            random_state=config.random_state,
        )

    else:
        raise ValueError(f"Unknown CV strategy: {config.strategy}")

    # if the user provided groups, wrap the splitter so .split always sees them
    if groups is not None:
        splitter = _CVWithGroups(splitter, groups)

    return splitter


def get_scorer(name: str) -> Callable:
    """
    Retrieve or construct a Scikit-Learn compliant scorer by name.

    Parameters
    ----------
    name : str
        The name of the metric (e.g., 'accuracy', 'f1_macro', 'neg_mean_squared_error').

    Returns
    -------
    Callable
        A scoring function with signature `(y_true, y_pred) -> float`.

    Raises
    ------
    ValueError
        If the metric name is unknown.
    """
    metrics = {
        # Classification
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "roc_auc": roc_auc_score,
        "f1": lambda y, p: f1_score(y, p, average="weighted"),
        "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
        "f1_micro": lambda y, p: f1_score(y, p, average="micro"),
        "precision": lambda y, p: precision_score(y, p, average="weighted", zero_division=0),
        "recall": lambda y, p: recall_score(y, p, average="weighted", zero_division=0),
        # Regression
        "r2": r2_score,
        "neg_mean_squared_error": lambda y, p: -mean_squared_error(y, p),
        "neg_mean_absolute_error": lambda y, p: -mean_absolute_error(y, p),
        "explained_variance": explained_variance_score,
    }

    if name not in metrics:
        raise ValueError(
            f"Unknown metric '{name}'. Available: {sorted(list(metrics.keys()))}"
        )
    return metrics[name]
