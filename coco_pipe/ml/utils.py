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

class _CVWithGroups(BaseCrossValidator):
    """
    Wrap any existing CV splitter so that its .split always uses
    the provided groups array.
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
    Return a CV splitter based on strategy name, optionally carrying a groups array.
    """
    # pull out groups if the user passed them
    groups = kwargs.pop('groups', None)

    strat = cv_strategy.lower()
    n_splits    = kwargs.get('n_splits', DEFAULT_CV['n_splits'])
    shuffle     = kwargs.get('shuffle', DEFAULT_CV['shuffle'])
    random_state= kwargs.get('random_state', DEFAULT_CV['random_state'])

    if strat == 'stratified':
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

    elif strat == 'kfold':
        splitter = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

    elif strat == 'group_kfold':
        splitter = GroupKFold(n_splits=n_splits)

    elif strat == 'stratified_group_kfold':
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

    elif strat == 'leave_p_out':
        n_groups = kwargs.get('n_groups')
        if not n_groups:
            raise ValueError("`n_groups` required for leave_p_out strategy.")
        splitter = LeavePGroupsOut(n_groups=n_groups)

    elif strat == 'leave_one_out':
        splitter = LeaveOneGroupOut()

    elif strat == 'split':
        test_size = kwargs.get('test_size', DEFAULT_CV.get('test_size', 0.2))
        stratify  = kwargs.get('stratify')
        splitter = SimpleSplit(
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
            stratify=stratify,
        )

    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}")

    # if the user provided groups, wrap the splitter so .split always sees them
    if groups is not None:
        splitter = _CVWithGroups(splitter, groups)

    return splitter

