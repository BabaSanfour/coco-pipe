"""
Decoding Splitters
==================

Internal cross-validation splitters for the decoding module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from ._constants import GROUP_CV_STRATEGIES, MetricTask
from .configs import CVConfig


class _CVWithGroups(BaseCrossValidator):
    """
    Bind fixed groups to a cross-validator.

    This wrapper ensures that the same group array is supplied whenever
    ``split`` or ``get_n_splits`` is called. It is particularly useful
    for scientific validation where group identities (e.g., Subject IDs)
    must be strictly preserved across nested CV folds to prevent leakage.

    Parameters
    ----------
    cv : BaseCrossValidator
        The underlying scikit-learn splitter to wrap.
    groups : Any
        The group labels (e.g., Subject IDs) to bind to the splitter.
    """

    def __init__(self, cv: BaseCrossValidator, groups: Any):
        """
        Initialize the group wrapper.

        Parameters
        ----------
        cv : BaseCrossValidator
            The underlying scikit-learn splitter to wrap.
        groups : Any
            The group labels (e.g., Subject IDs) to bind to the splitter.
        """
        self.cv = cv
        self.groups = (
            groups
            if isinstance(groups, (np.ndarray, pd.Series))
            else np.asarray(groups)
        )

    def _get_effective_groups(self, X: Any, groups: Any = None) -> Any:
        """
        Return groups aligned to ``X`` or fail before leakage can occur.

        Scientific rationale: In nested cross-validation, `X` is often a
        subset (fold) of the original data. This method ensures that the
        groups being passed to the inner splitter match the rows of `X`.

        Parameters
        ----------
        X : array-like
            The data being split.
        groups : array-like, optional
            Explicit groups to use. If provided, these take precedence.

        Returns
        -------
        aligned_groups : array-like
            The groups array aligned to `X`.

        Raises
        ------
        ValueError
            If group lengths do not match `X`.
        """
        if groups is not None:
            groups_arr = (
                groups
                if isinstance(groups, (np.ndarray, pd.Series))
                else np.asarray(groups)
            )
            if X is not None and len(groups_arr) != len(X):
                raise ValueError(
                    "Explicit groups length does not match X for CV splitting: "
                    f"{len(groups_arr)} != {len(X)}."
                )
            return groups_arr

        if X is None:
            return self.groups

        if len(X) == len(self.groups):
            return self.groups

        if hasattr(X, "index") and hasattr(self.groups, "loc"):
            try:
                aligned = self.groups.loc[X.index]
                if len(aligned) == len(X):
                    return aligned
            except Exception:
                pass

        raise ValueError(
            "Bound groups length does not match X for CV splitting: "
            f"{len(self.groups)} != {len(X)}. Pass a groups array aligned to "
            "the X being split instead of reusing a full-array group binding "
            "inside nested CV."
        )

    def split(self, X: Any, y: Any = None, groups: Any = None):
        """Generate indices to split data into training and test set."""
        return self.cv.split(X, y, self._get_effective_groups(X, groups))

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of splitting iterations in the cross-validator."""
        if X is not None:
            effective_groups = self._get_effective_groups(X, groups)
        else:
            effective_groups = groups if groups is not None else self.groups
        return self.cv.get_n_splits(X, y, effective_groups)

    def __sklearn_tags__(self) -> Dict[str, Any]:
        """Sklearn 1.6+ compatibility for estimator tags."""
        tags = getattr(self.cv, "__sklearn_tags__", lambda: {})()
        if not tags:
            tags = getattr(self.cv, "_get_tags", lambda: {})()
        return {**tags, "non_deterministic": tags.get("non_deterministic", False)}

    def _get_tags(self) -> Dict[str, Any]:
        """Legacy sklearn tag support."""
        return self.__sklearn_tags__()

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {"cv": self.cv, "groups": self.groups}

    def __repr__(self) -> str:
        return f"_CVWithGroups(cv={self.cv!r})"


def cv_uses_groups(cv: Any) -> bool:
    """Return True for runtime CV splitter objects that consume groups."""
    return isinstance(
        cv,
        (
            _CVWithGroups,
            GroupKFold,
            StratifiedGroupKFold,
            LeaveOneGroupOut,
            LeavePGroupsOut,
            GroupShuffleSplit,
        ),
    )


class SimpleSplit(BaseCrossValidator):
    """
    One-shot train/test split using ``train_test_split``.

    This provides a scikit-learn compatible interface for a single hold-out
    validation set. It is often used for final model evaluation or when
    the dataset is large enough that K-Fold CV is computationally
    prohibitive.

    Parameters
    ----------
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    random_state : int, optional
        Random seed for reproducibility.
    stratify : bool or array-like, optional
        If True, use 'y' for stratification. If an array, use it directly.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        stratify: Optional[Union[bool, pd.Series, np.ndarray]] = None,
    ):
        """
        Initialize the hold-out splitter.

        Parameters
        ----------
        test_size : float, default=0.2
            The proportion of the dataset to include in the test split.
        shuffle : bool, default=True
            Whether to shuffle the data before splitting.
        random_state : int, optional
            Random seed for reproducibility.
        stratify : bool or array-like, optional
            If True, use 'y' for stratification. If an array, use it directly.
        """
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
        """Generate indices for a single hold-out split."""
        idx = np.arange(len(X))
        if self.stratify is True:
            strat = y
        elif self.stratify is False:
            strat = None
        else:
            strat = self.stratify

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
        """Return the number of splits (always 1)."""
        return 1

    def __sklearn_tags__(self) -> Dict[str, Any]:
        """Sklearn 1.6+ compatibility for estimator tags."""
        return {"non_deterministic": self.shuffle}

    def _get_tags(self) -> Dict[str, Any]:
        """Legacy sklearn tag support."""
        return self.__sklearn_tags__()

    def __repr__(self) -> str:
        return f"SimpleSplit(test_size={self.test_size}, shuffle={self.shuffle})"


def get_cv_splitter(
    config: CVConfig,
    groups: Optional[Sequence[Any]] = None,
    y: Optional[Sequence[Any]] = None,
    task: Optional[MetricTask] = None,
    require_groups: bool = True,
) -> BaseCrossValidator:
    """
    Create a scikit-learn cross-validator from ``CVConfig``.

    This factory handles the mapping from strategy names to scikit-learn
    splitter objects and performs validation to ensure the strategy is
    scientifically sound for the given task (e.g., preventing
    stratification for regression).

    Parameters
    ----------
    config : CVConfig
        The cross-validation configuration.
    groups : Sequence, optional
        Grouping labels for the samples. Required for group-based strategies.
    y : Sequence, optional
        Target labels. Used for stratified strategies.
    task : MetricTask, optional
        The task type ('classification' or 'regression'). Used to validate
        stratification requests.
    require_groups : bool, default=True
        Whether to raise an error if groups are missing for group-based strategies.

    Returns
    -------
    splitter : BaseCrossValidator
        A configured scikit-learn compatible splitter.

    Raises
    ------
    ValueError
        If the strategy is unknown, if groups are missing for a group strategy,
        or if stratification is requested for a regression task.

    Examples
    --------
    >>> from coco_pipe.decoding.configs import CVConfig
    >>> cfg = CVConfig(strategy="stratified", n_splits=5)
    >>> splitter = get_cv_splitter(cfg)

    See Also
    --------
    SimpleSplit : One-shot holdout splitter.
    _CVWithGroups : Wrapper for binding groups to splitters.
    """
    strat = config.strategy.lower()

    # 1. Scientific Validation
    if strat in GROUP_CV_STRATEGIES and require_groups and groups is None:
        raise ValueError(f"CV strategy '{config.strategy}' requires groups.")

    if task == "regression" and "stratified" in strat:
        raise ValueError(
            f"Stratified CV strategy '{config.strategy}' is not supported for "
            "regression tasks. Use 'kfold' or 'group_kfold' instead."
        )

    common_kwargs = {}
    if strat not in ["leave_one_group_out", "leave_p_out", "split", "timeseries"]:
        common_kwargs["n_splits"] = config.n_splits

    if strat in ["stratified", "kfold", "stratified_group_kfold", "split"]:
        common_kwargs["shuffle"] = config.shuffle
        common_kwargs["random_state"] = config.random_state if config.shuffle else None

    # 2. Factory Logic
    if strat == "stratified":
        splitter = StratifiedKFold(**common_kwargs)
    elif strat == "kfold":
        splitter = KFold(**common_kwargs)
    elif strat == "group_kfold":
        splitter = GroupKFold(n_splits=config.n_splits)
    elif strat == "stratified_group_kfold":
        splitter = StratifiedGroupKFold(**common_kwargs)
    elif strat == "leave_p_out":
        splitter = LeavePGroupsOut(n_groups=config.n_splits)
    elif strat == "leave_one_group_out":
        splitter = LeaveOneGroupOut()
    elif strat == "group_shuffle_split":
        splitter = GroupShuffleSplit(
            n_splits=config.n_splits,
            test_size=config.test_size,
            random_state=config.random_state,
        )
    elif strat == "timeseries":
        splitter = TimeSeriesSplit(n_splits=config.n_splits)
    elif strat == "split":
        splitter = SimpleSplit(
            test_size=config.test_size,
            shuffle=config.shuffle,
            random_state=config.random_state,
            stratify=y if config.stratify and y is not None else config.stratify,
        )
    else:
        raise ValueError(f"Unknown CV strategy: {config.strategy}")

    # 3. Contextual Wrapping
    if groups is not None:
        if strat not in GROUP_CV_STRATEGIES:
            import warnings

            warnings.warn(
                f"CV groups were provided but the strategy '{config.strategy}' is not "
                "group-aware. Groups will be bound for technical compatibility but "
                "ignored during splitting logic.",
                UserWarning,
            )
        splitter = _CVWithGroups(splitter, groups)

    return splitter
