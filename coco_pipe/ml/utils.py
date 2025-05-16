# coco_pipe/ml/utils.py
from typing import Any
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold, \
    LeaveOneGroupOut, LeavePGroupsOut, GroupKFold
import warnings
from coco_pipe.ml.config import DEFAULT_CV

def get_cv_splitter(strategy: str, **kwargs: Any) -> BaseCrossValidator:
    """
    Return a scikit-learn CV splitter based on `strategy`.

    Supports:
      - 'stratified' : StratifiedKFold
      - 'kfold'      : plain KFold
      - 'leave_p_out': LeavePGroupsOut / LeaveOneGroupOut
      - 'group_kfold': GroupKFold
    """
    if strategy == "stratified":
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
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            splitter.shuffle = False
            return splitter
        rs_arg = random_state if shuffle else None
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=rs_arg)

    if strategy == "kfold":
        return KFold(
            n_splits=kwargs.get("n_splits", DEFAULT_CV["n_splits"]),
            shuffle=kwargs.get("shuffle", DEFAULT_CV["shuffle"]),
            random_state=kwargs.get("random_state", DEFAULT_CV["random_state"])
        )

    if strategy == "leave_p_out":
        n_groups = kwargs.get("n_groups", DEFAULT_CV["n_groups"])
        return (
            LeavePGroupsOut(n_groups=n_groups)
            if n_groups > 1 else LeaveOneGroupOut()
        )

    if strategy == "group_kfold":
        return GroupKFold(n_splits=kwargs.get("n_splits", DEFAULT_CV["n_splits"]))

    raise ValueError(f"Unknown CV strategy: {strategy}")
