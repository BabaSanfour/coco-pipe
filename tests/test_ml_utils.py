import pytest
from sklearn.model_selection import LeaveOneGroupOut

from coco_pipe.ml.utils import get_cv_splitter
from coco_pipe.ml.config import DEFAULT_CV

def test_get_cv_splitter_utils():
    # stratified shuffle=False
    cs = get_cv_splitter('stratified', n_splits=3, shuffle=False, random_state=1)
    assert cs.n_splits == 3 and cs.shuffle is False and cs.random_state is None
    # leave_p_out
    lp = get_cv_splitter('leave_p_out', n_groups=1)
    assert lp.__class__.__name__ == 'LeaveOneGroupOut'
    lp2 = get_cv_splitter('leave_p_out', n_groups=2)
    assert lp2.__class__.__name__ == 'LeavePGroupsOut'
    # group_kfold
    gk = get_cv_splitter('group_kfold', n_splits=4)
    assert gk.n_splits == 4
    # split
    sp = get_cv_splitter('split', test_size=0.3, shuffle=True, random_state=42)
    train_idx, test_idx = sp(range(10))[0]
    assert len(train_idx) == 7 and len(test_idx) == 3
    # unknown
    with pytest.raises(ValueError):
        get_cv_splitter('unknown')

def test_stratified_splitter_default():
    splitter = get_cv_splitter('stratified')
    assert splitter.n_splits == DEFAULT_CV['n_splits']
    assert splitter.shuffle == DEFAULT_CV['shuffle']
    assert splitter.random_state == DEFAULT_CV['random_state']

def test_stratified_random_state_without_shuffle():
    with pytest.warns(UserWarning):
        splitter = get_cv_splitter('stratified', n_splits=3, shuffle=False, random_state=7)
    assert splitter.shuffle is False
    assert splitter.random_state is None

def test_stratified_random_state_with_shuffle():
    splitter = get_cv_splitter('stratified', n_splits=3, shuffle=True, random_state=7)
    assert splitter.shuffle is True
    assert splitter.random_state == 7

def test_stratified_n_splits():
    splitter = get_cv_splitter('stratified', n_splits=3)
    assert splitter.n_splits == 3
    assert splitter.shuffle is True

def test_leave_p_out_and_group_kfold():
    lo = get_cv_splitter('leave_p_out', n_groups=1)
    assert isinstance(lo, LeaveOneGroupOut)
    lp = get_cv_splitter('leave_p_out', n_groups=2)
    assert lp.__class__.__name__ == 'LeavePGroupsOut'
    gk = get_cv_splitter('group_kfold', n_splits=4)
    assert gk.n_splits == 4

def test_split_strategy():
    # Test with default parameters
    sp = get_cv_splitter('split')
    train_idx, test_idx = sp(range(100))[0]
    assert len(train_idx) == 80  # Default test_size=0.2
    assert len(test_idx) == 20
    
    # Test with custom parameters
    sp = get_cv_splitter('split', test_size=0.4, shuffle=True, random_state=42)
    train_idx, test_idx = sp(range(100))[0]
    assert len(train_idx) == 60
    assert len(test_idx) == 40

def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        get_cv_splitter('foobar')
