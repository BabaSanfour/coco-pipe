#!/usr/bin/env python3
"""
coco_pipe/ml/utils.py
----------------
Utility functions for ML pipelines: CV splitter factory, simple train/test split,
listing supported strategies, and parameter validation.
"""
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    GroupKFold,
    StratifiedGroupKFold,
    train_test_split,
)

from coco_pipe.ml.config import DEFAULT_CV


class SimpleSplit(BaseCrossValidator):
    """
    Single train/test split using scikit-learn's train_test_split.

    Parameters
    ----------
    test_size : float
        Fraction of data to reserve for the test fold (0 < test_size < 1).
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    random_state : int or None
        Seed for reproducible shuffling.
    stratify : array-like or None, optional
        If not None, data is split in a stratified fashion, using y for stratification.
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

def get_cv_splitter(
    cv_strategy: str,
    **kwargs: Any
) -> BaseCrossValidator:
    """
    Return a scikit-learn CV splitter based on strategy name.

    Parameters
    ----------
    cv_strategy : {'stratified', 'kfold', 'group_kfold', 'leave_p_out', 'leave_one_out', 'split'}
        Name of the CV strategy.
    kwargs :
        Additional parameters for the splitter:
        - n_splits, shuffle, random_state for KFold types.
        - n_repeats for repeated KFold.
        - test_size, stratify for 'split'.
        - n_groups for LeavePGroupsOut.
        - groups array for group-based splitters.

    Returns
    -------
    BaseCrossValidator
        Configured CV splitter.

    Raises
    ------
    ValueError
        For unknown strategy or missing required kwargs.
    """
    strat = cv_strategy.lower()
    n_splits = kwargs.get('n_splits', DEFAULT_CV['n_splits'])
    shuffle = kwargs.get('shuffle', DEFAULT_CV['shuffle'])
    random_state = kwargs.get('random_state', DEFAULT_CV['random_state'])

    if strat == 'stratified':
        if not shuffle and random_state is not None:
            warnings.warn(
                "random_state has no effect when shuffle=False.",
                UserWarning
            )
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

    elif strat == 'kfold':
        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

    elif strat == 'group_kfold':
        return GroupKFold(n_splits=n_splits)

    elif strat == 'stratified_group_kfold':
        if not shuffle and random_state is not None:
            warnings.warn(
                "random_state has no effect when shuffle=False.",
                UserWarning
            )
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

    elif strat == 'leave_p_out':
        n_groups = kwargs.get('n_groups')
        if not n_groups:
            raise ValueError("`n_groups` required for leave_p_out strategy.")
        return LeavePGroupsOut(n_groups=n_groups)

    elif strat == 'leave_one_out':
        return LeaveOneGroupOut()

    elif strat == 'split':
        test_size = kwargs.get('test_size', 0.2)
        stratify = kwargs.get('stratify')
        return SimpleSplit(
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
            stratify=stratify,
        )

    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}")