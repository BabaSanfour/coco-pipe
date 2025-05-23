"""
coco_pipe/ml/utils.py
----------------
Utility functions for ML pipelines.

Author: Hamza Abdelhedi <hamza.abdelhedii@gmail.com>
Date: 2025-05-18
Version: 0.0.1
License: TBD
"""
from typing import Any
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold, \
    LeaveOneGroupOut, LeavePGroupsOut, GroupKFold, train_test_split
import warnings
from coco_pipe.ml.config import DEFAULT_CV

def get_cv_splitter(cv_strategy: str, **kwargs: Any) -> BaseCrossValidator:
    """
    Return a scikit-learn CV splitter based on `strategy`.

    Supports:
      - 'stratified' : StratifiedKFold
      - 'kfold'      : plain KFold
      - 'leave_p_out': LeavePGroupsOut / LeaveOneGroupOut
      - 'group_kfold': GroupKFold
      - 'split'      : Simple train/test split

    Parameters:
        :cv_strategy: str, CV strategy to use
        :kwargs: Any, Additional arguments for the CV splitter

    Returns:
        :BaseCrossValidator, CV splitter
    """
    if cv_strategy == "stratified":
        n_splits = kwargs.get("n_splits", DEFAULT_CV["n_splits"])
        shuffle = kwargs.get("shuffle", DEFAULT_CV["shuffle"])
        random_state = kwargs.get("random_state", DEFAULT_CV["random_state"])
        # warn if random_state without shuffle
        if not shuffle and random_state is not None:
            warnings.warn(
                f"You set random_state={random_state} while shuffle=False; "
                "random_state will have no effect.",
                UserWarning
            )
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None
        )

    if cv_strategy == "kfold":
        return KFold(
            n_splits=kwargs.get("n_splits", DEFAULT_CV["n_splits"]),
            shuffle=kwargs.get("shuffle", DEFAULT_CV["shuffle"]),
            random_state=kwargs.get("random_state", DEFAULT_CV["random_state"])
        )

    if cv_strategy == "leave_p_out":
        n_groups = kwargs.get("n_groups", DEFAULT_CV["n_groups"])
        return (
            LeavePGroupsOut(n_groups=n_groups)
            if n_groups > 1 else LeaveOneGroupOut()
        )

    if cv_strategy == "group_kfold":
        return GroupKFold(n_splits=kwargs.get("n_splits", DEFAULT_CV["n_splits"]))

    if cv_strategy == "split":
        test_size = kwargs.get("test_size", 0.2)
        shuffle = kwargs.get("shuffle", DEFAULT_CV["shuffle"])
        random_state = kwargs.get("random_state", DEFAULT_CV["random_state"])
        return lambda X, y=None, groups=None: [train_test_split(
            range(len(X)), 
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state if shuffle else None
        )]

    raise ValueError(f"Unknown CV strategy: {cv_strategy}")
