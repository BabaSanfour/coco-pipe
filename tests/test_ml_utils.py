import pytest
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    LeavePGroupsOut,
    LeaveOneGroupOut,
    GroupKFold,
)
from coco_pipe.ml.utils import get_cv_splitter, SimpleSplit
from coco_pipe.ml.config import DEFAULT_CV

def test_stratified_no_shuffle_behavior():
    # shuffle=False should force random_state=None
    splitter = get_cv_splitter(
        "stratified", n_splits=3, shuffle=False, random_state=123
    )
    assert isinstance(splitter, StratifiedKFold)
    assert splitter.n_splits == 3
    assert splitter.shuffle is False
    assert splitter.random_state is None

def test_stratified_with_shuffle_preserves_seed():
    splitter = get_cv_splitter(
        "stratified", n_splits=4, shuffle=True, random_state=99
    )
    assert isinstance(splitter, StratifiedKFold)
    assert splitter.n_splits == 4
    assert splitter.shuffle is True
    assert splitter.random_state == 99

def test_kfold_defaults_and_override():
    # default KFold
    kf = get_cv_splitter("kfold")
    assert isinstance(kf, KFold)
    assert kf.n_splits == DEFAULT_CV["n_splits"]
    assert kf.shuffle == DEFAULT_CV["shuffle"]
    # seed only when shuffle=True
    if DEFAULT_CV["shuffle"]:
        assert kf.random_state == DEFAULT_CV["random_state"]
    else:
        assert kf.random_state is None

    # override shuffle=False
    kf2 = get_cv_splitter("kfold", n_splits=5, shuffle=False, random_state=42)
    assert isinstance(kf2, KFold)
    assert kf2.n_splits == 5
    assert kf2.shuffle is False
    assert kf2.random_state is None

def test_group_kfold_n_splits():
    gk = get_cv_splitter("group_kfold", n_splits=7)
    assert isinstance(gk, GroupKFold)
    assert gk.n_splits == 7

def test_leave_p_out_requires_n_groups_and_behavior():
    with pytest.raises(ValueError):
        get_cv_splitter("leave_p_out")
    lp = get_cv_splitter("leave_p_out", n_groups=2)
    assert isinstance(lp, LeavePGroupsOut)
    # scikit-learn leaves n_groups stored privately, but repr shows the parameter
    assert "n_groups=2" in repr(lp)

def test_leave_one_out_behavior():
    loo = get_cv_splitter("leave_one_out")
    assert isinstance(loo, LeaveOneGroupOut)

def test_simple_split_default_vs_custom():
    # default test_size=0.2
    sp = get_cv_splitter("split")
    assert isinstance(sp, SimpleSplit)
    idx = np.arange(50)
    train_idx, test_idx = next(sp.split(idx, y=None, groups=None))
    assert len(test_idx) == int(0.2 * 50)
    assert len(train_idx) == 50 - len(test_idx)

    # custom test_size
    sp2 = get_cv_splitter("split", test_size=0.3, shuffle=False, random_state=123)
    assert isinstance(sp2, SimpleSplit)
    idx2 = np.arange(100)
    # no stratify by default
    train2, test2 = next(sp2.split(idx2))
    assert len(test2) == 30
    assert len(train2) == 70

def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        get_cv_splitter("this_does_not_exist")
