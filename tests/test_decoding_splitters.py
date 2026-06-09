import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from coco_pipe.decoding._splitters import (
    SimpleSplit,
    _CVWithGroups,
    cv_uses_groups,
    get_cv_splitter,
)
from coco_pipe.decoding.configs import CVConfig


def test_simple_split_basic():
    X = np.zeros((100, 10))
    y = np.zeros(100)
    ss = SimpleSplit(test_size=0.2, shuffle=True, random_state=42)

    splits = list(ss.split(X, y))
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert len(train_idx) == 80
    assert len(test_idx) == 20
    assert ss.get_n_splits() == 1
    assert ss._get_tags()["non_deterministic"] is True
    assert "SimpleSplit" in repr(ss)


def test_simple_split_invalid_size():
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        SimpleSplit(test_size=1.5)


def test_simple_split_stratify():
    X = np.zeros((10, 1))
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Stratify = True (uses y)
    ss = SimpleSplit(test_size=0.2, stratify=True)
    train_idx, test_idx = next(ss.split(X, y))
    assert y[test_idx].sum() == 1  # 20% of 5 is 1

    # Stratify = False
    ss_none = SimpleSplit(test_size=0.2, stratify=False)
    next(ss_none.split(X, y))

    # Stratify = array
    ss_arr = SimpleSplit(test_size=0.2, stratify=y)
    next(ss_arr.split(X, y))


def test_cv_with_groups_wrapper():
    base_cv = KFold(n_splits=5)
    groups = np.repeat(np.arange(10), 10)
    X = np.zeros((100, 10))

    wrapped = _CVWithGroups(base_cv, groups)
    assert wrapped.get_n_splits(X) == 5
    assert len(list(wrapped.split(X))) == 5
    assert "non_deterministic" in wrapped._get_tags()
    assert wrapped.get_params()["groups"] is groups
    assert "_CVWithGroups" in repr(wrapped)


def test_cv_with_groups_rejects_unsliced_nested_groups():
    base_cv = KFold(n_splits=2)
    groups = np.repeat(np.arange(5), 2)
    wrapped = _CVWithGroups(base_cv, groups)

    with pytest.raises(ValueError, match="Bound groups length does not match X"):
        list(wrapped.split(np.zeros((6, 2))))


def test_cv_uses_groups_detects_runtime_splitters():
    wrapped = _CVWithGroups(KFold(n_splits=2), np.arange(6))

    assert cv_uses_groups(GroupKFold(n_splits=2))
    assert cv_uses_groups(wrapped)
    assert not cv_uses_groups(KFold(n_splits=2))


def test_get_cv_splitter_factory():
    # KFold
    cfg = CVConfig(strategy="kfold", n_splits=5, shuffle=True, random_state=42)
    splitter = get_cv_splitter(cfg)
    assert isinstance(splitter, KFold)
    assert splitter.n_splits == 5

    # Stratified
    cfg_s = CVConfig(strategy="stratified", n_splits=3)
    splitter_s = get_cv_splitter(cfg_s, task="classification")
    assert isinstance(splitter_s, StratifiedKFold)

    # GroupKFold
    cfg_g = CVConfig(strategy="group_kfold", n_splits=5)
    groups = np.repeat(np.arange(5), 20)
    splitter_g = get_cv_splitter(cfg_g, groups=groups)
    assert isinstance(splitter_g, _CVWithGroups)
    assert isinstance(splitter_g.cv, GroupKFold)


def test_get_cv_splitter_errors():
    cfg_g = CVConfig(strategy="group_kfold")

    # Missing groups
    with pytest.raises(ValueError, match="requires groups"):
        get_cv_splitter(cfg_g, groups=None)

    # Stratified on regression
    cfg_s = CVConfig(strategy="stratified")
    with pytest.raises(ValueError, match="not supported for regression"):
        get_cv_splitter(cfg_s, task="regression")

    # Unknown strategy (Pydantic catches this at config level)
    with pytest.raises(ValidationError):
        CVConfig(strategy="invalid")

    # Bypassing Pydantic to hit the ValueError in the factory
    from types import SimpleNamespace

    fake_cfg = SimpleNamespace(
        strategy="invalid", n_splits=5, shuffle=True, random_state=42
    )
    with pytest.raises(ValueError, match="Unknown CV strategy"):
        get_cv_splitter(fake_cfg)


def test_get_cv_splitter_all_strategies():
    strategies = [
        ("stratified_group_kfold", 5),
        ("leave_p_out", 2),
        ("leave_one_group_out", 1),
        ("group_shuffle_split", 5),
        ("timeseries", 5),
        ("split", 1),
    ]
    groups = np.repeat(np.arange(10), 10)
    for strat, n in strategies:
        cfg = CVConfig(strategy=strat, n_splits=n)
        splitter = get_cv_splitter(cfg, groups=groups, task="classification")
        assert splitter is not None


def test_get_cv_splitter_unshuffled():
    # Timeseries or KFold without shuffle
    cfg = CVConfig(strategy="kfold", shuffle=False)
    splitter = get_cv_splitter(cfg)
    assert splitter.random_state is None

    cfg_t = CVConfig(strategy="timeseries")
    splitter_t = get_cv_splitter(cfg_t)
    assert splitter_t.n_splits == 5


def test_simple_split_shuffle_false():
    """Verify shuffle=False ignores random_state."""
    X = np.zeros((10, 2))
    y = np.zeros(10)
    ss = SimpleSplit(shuffle=False, random_state=42)
    train, test = next(ss.split(X, y))
    assert np.all(test == [8, 9])  # Deterministic split from the end


def test_cv_with_groups_pandas_alignment():
    """Verify group binding aligns to pandas index."""
    cv = KFold(n_splits=2)
    df = pd.DataFrame({"a": [1, 2]}, index=[2, 3])
    gp_ser = pd.Series([10, 20, 30, 40], index=[0, 1, 2, 3])
    wrapper_pd = _CVWithGroups(cv, gp_ser)
    aligned = wrapper_pd._get_effective_groups(df)
    assert np.all(aligned.values == [30, 40])

    # Matching explicit groups
    explicit = np.array([30, 40])
    assert np.all(wrapper_pd._get_effective_groups(df, groups=explicit) == explicit)


def test_get_cv_splitter_all_strategies_extended():
    """Verify factory handles all supported strategies."""
    # 1. Timeseries already tested in test_get_cv_splitter_unshuffled

    # 2. Leave P Out
    cfg = CVConfig(strategy="leave_p_out", n_splits=2)
    s = get_cv_splitter(cfg, groups=[1, 2, 3])
    assert s.cv.n_groups == 2

    # 3. Group Shuffle Split
    cfg = CVConfig(strategy="group_shuffle_split", n_splits=5, test_size=0.1)
    s = get_cv_splitter(cfg, groups=[1, 2, 3, 4, 5])
    assert s.cv.n_splits == 5

    # 4. Stratified Group KFold
    cfg = CVConfig(strategy="stratified_group_kfold", n_splits=3)
    s = get_cv_splitter(cfg, groups=[1, 2, 3])
    assert isinstance(s.cv, StratifiedGroupKFold)

    # 5. Leave One Group Out
    cfg = CVConfig(strategy="leave_one_group_out")
    s = get_cv_splitter(cfg, groups=[1, 2, 3])
    assert isinstance(s.cv, LeaveOneGroupOut)

    # 6. Split (SimpleSplit)
    cfg = CVConfig(strategy="split", test_size=0.3)
    s = get_cv_splitter(cfg)
    assert isinstance(s, SimpleSplit)
    assert s.test_size == 0.3

    # 7. Regression + Stratified -> ValueError
    cfg = CVConfig(strategy="stratified")
    with pytest.raises(ValueError, match="not supported for regression"):
        get_cv_splitter(cfg, task="regression")

    # 8. require_groups=False
    cfg = CVConfig(strategy="group_kfold")
    s = get_cv_splitter(cfg, groups=None, require_groups=False)
    assert s.n_splits == 5

    # 9. Warning for groups in non-group strategy (covers 322-323)
    cfg = CVConfig(strategy="kfold")
    with pytest.warns(UserWarning, match="not group-aware"):
        get_cv_splitter(cfg, groups=[1, 2, 3])


def test_cv_with_groups_more_coverage():
    """Hit the remaining edge cases in _CVWithGroups."""
    cv = KFold(n_splits=2)
    groups = np.array([1, 1, 2, 2])
    wrapper = _CVWithGroups(cv, groups)

    # X is None (covers 75 and 104)
    assert np.all(wrapper._get_effective_groups(None) == groups)
    assert wrapper.get_n_splits(X=None, groups=[1, 1, 2, 2]) == 2

    # Matching explicit groups (covers return on line 72 or similar)
    # Actually, let's re-verify line numbers.
    # In my local view, line 68 was the ValueError.

    # Explicit groups that match length (covers 72)
    explicit = np.array([3, 3, 4, 4])
    assert np.all(
        wrapper._get_effective_groups(np.zeros(4), groups=explicit) == explicit
    )

    # Mismatched explicit groups (covers 68-71)
    with pytest.raises(ValueError, match="Explicit groups length does not match X"):
        wrapper._get_effective_groups(np.zeros(4), groups=[1, 2])
