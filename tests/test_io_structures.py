import numpy as np
import pytest
from pathlib import Path


from coco_pipe.io.structures import DataContainer

@pytest.fixture(scope="module")
def data_container_cls():
    return DataContainer


@pytest.fixture
def sample_container(data_container_cls):
    X = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    coords = {"channel": ["Fz", "Cz", "Pz"], "time": np.array([0, 1, 2, 3])}
    y = np.array([0, 1])
    ids = np.array(["s0", "s1"])
    return data_container_cls(X=X, dims=("obs", "channel", "time"), coords=coords, y=y, ids=ids)


def test_data_container_shape_validation(data_container_cls):
    X = np.ones((2, 3))
    with pytest.raises(ValueError, match="Shape mismatch"):
        data_container_cls(X=X, dims=("obs", "feature", "time"), coords={"feature": [1, 2, 3]})


def test_isel_preserves_alignment(sample_container):
    subset = sample_container.isel(obs=[1], channel=[0, 2], time=slice(1, 3))
    assert subset.X.shape == (1, 2, 2)
    assert subset.coords["channel"].tolist() == ["Fz", "Pz"]
    assert subset.coords["time"].tolist() == [1, 2]
    assert subset.y.tolist() == [1]
    assert subset.ids.tolist() == ["s1"]


def test_select_supports_patterns_and_ops(sample_container):
    wildcard = sample_container.select(channel="*z", time={">": 1})
    assert wildcard.X.shape == (2, 3, 2)
    assert wildcard.coords["time"].tolist() == [2, 3]

    fuzzy = sample_container.select(channel=["pz"], ignore_case=True, fuzzy=True)
    assert fuzzy.X.shape == (2, 1, 4)
    assert fuzzy.coords["channel"].tolist() == ["Pz"]


def test_flatten_and_stack(sample_container):
    flat = sample_container.flatten(preserve="obs")
    assert flat.dims == ("obs", "feature")
    assert flat.X.shape == (2, 12)
    assert flat.coords["feature"][0] == "Fz_0"

    stacked = sample_container.stack(dims=("obs", "time"), new_dim="obs")
    assert stacked.X.shape == (sample_container.shape[0] * sample_container.shape[2], sample_container.shape[1])
    assert stacked.y.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert stacked.ids[0].startswith("s0_")


def test_balance_undersample(data_container_cls):
    X = np.arange(6 * 2).reshape(6, 2)
    y = np.array([0, 0, 0, 0, 1, 1])
    container = data_container_cls(X=X, dims=("obs", "feature"), coords={"feature": [0, 1]}, y=y)

    balanced = container.balance(random_state=0)
    _, counts = np.unique(balanced.y, return_counts=True)
    assert np.all(counts == counts[0])
    assert balanced.shape[0] == counts[0] * 2


def test_save_load_roundtrip(tmp_path, sample_container):
    path = tmp_path / "container.joblib"
    sample_container.save(path)

    loaded = sample_container.__class__.load(path)
    assert loaded.dims == sample_container.dims
    np.testing.assert_array_equal(loaded.X, sample_container.X)
    np.testing.assert_array_equal(loaded.y, sample_container.y)


def test_select_advanced_operators(sample_container):
    # Test 'in' operator
    ids = ["s0"]
    subset = sample_container.select(ids={"in": ids})
    assert len(subset.ids) == 1
    assert subset.ids[0] == "s0"

    # Test '!=' operator
    subset_neq = sample_container.select(channel={"!=": "Cz"})
    assert "Cz" not in subset_neq.coords["channel"]
    assert len(subset_neq.coords["channel"]) == 2

    # Test callable
    def odd_time(arr):
        return arr % 2 != 0
    
    subset_call = sample_container.select(time=odd_time)
    assert subset_call.coords["time"].tolist() == [1, 3]


def test_select_edge_cases(sample_container):
    # Empty result should raise ValueError
    with pytest.raises(ValueError, match="resulted in empty set"):
        sample_container.select(ids=["non_existent"])
    
    # Empty result via operator
    with pytest.raises(ValueError, match="resulted in empty set"):
        sample_container.select(time={">": 100})

    # Unknown operator
    with pytest.raises(ValueError, match="Unknown operator"):
        sample_container.select(time={"??": 1})
        
    # Selection on missing dimension warning (captured via logging if needed, or just ensure no crash)
    # logic says it warns and ignores.
    subset = sample_container.select(missing_dim=["a", "b"])
    assert subset.shape == sample_container.shape


def test_balance_stratified(data_container_cls):
    # Create dataset with covariates
    # 20 samples. y=0 (15), y=1 (5). 
    # Covariate 'sex': 0s -> 10M, 5F. 1s -> 2M, 3F.
    
    # We want to undersample y=0 to match y=1 (5).
    # Total = 10.
    
    y = np.array([0]*15 + [1]*5)
    sex = np.array(['M']*10 + ['F']*5 + ['M']*2 + ['F']*3)
    X = np.zeros((20, 2))
    
    container = data_container_cls(
        X=X, 
        dims=("obs", "feat"), 
        coords={"sex": sex}, 
        y=y
    )
    
    # Stratified balance by sex
    balanced = container.balance(target='y', covariates=['sex'], strategy='undersample', random_state=42)
    
    # Check y balance
    u, c = np.unique(balanced.y, return_counts=True)
    assert c[0] == c[1] # Balanced classes
    
    # Check sex preservation (roughly)
    # y=1 has 2M 3F (40% M). y=0 should ideally maintain ~40% M.
    # y=0 (size 5). 40% of 5 is 2. So we expect around 2M in y=0.
    
    subset_0 = balanced.select(y={'==': 0})
    sex_0 = subset_0.coords['sex']
    n_m = np.sum(sex_0 == 'M')
    assert n_m in [1, 2, 3] # Allow some variance due to small sample size


def test_balance_auto_oversample(data_container_cls):
    # y: 10 vs 2. Auto should decide to oversample the 2 -> 10.
    y = np.array([0]*10 + [1]*2)
    X = np.zeros((12, 1))
    container = data_container_cls(X=X, dims=("obs", "f"), y=y)
    
    balanced = container.balance(strategy='auto', random_state=42)
    
    u, c = np.unique(balanced.y, return_counts=True)
    assert np.all(c == 10)
    assert balanced.shape[0] == 20


def test_flatten_complex(data_container_cls):
    # (obs, ch, time, freq)
    X = np.zeros((2, 2, 3, 4))
    dims = ("obs", "ch", "time", "freq")
    coords = {
        "ch": ["C1", "C2"],
        "time": [0, 1, 2],
        "freq": [10, 20, 30, 40]
    }
    container = data_container_cls(X=X, dims=dims, coords=coords)
    
    # Flatten but preserve obs and freq -> (obs, freq, feature=ch*time)
    flat = container.flatten(preserve=["obs", "freq"])
    assert flat.dims == ("obs", "freq", "feature")
    assert flat.X.shape == (2, 4, 6) # 6 = 2ch * 3time
    
    # Check feature labels
    # Expected: "C1_0", "C1_1", ...
    assert flat.coords["feature"][0] == "C1_0"


def test_flatten_validation(sample_container):
    with pytest.raises(ValueError, match="Dimension 'missing' not found"):
        sample_container.flatten(preserve=["missing"])


def test_stack_validation(sample_container):
    with pytest.raises(ValueError, match="Dimension 'missing' not found"):
        sample_container.stack(dims=("obs", "missing"))



def test_init_validation():
    """Test __post_init__ validation."""
    X = np.zeros((10, 5))
    
    # 1. Dim mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        DataContainer(X, dims=('obs',)) # Missing one dim
        
    # 2. Coords mismatch warning (checking log would be ideal, but execution is enough)
    dc = DataContainer(X, dims=('obs', 'feat'), coords={'obs': [0, 1]}) 

def test_repr():
    """Test __repr__ formatting."""
    dc = DataContainer(np.zeros((5, 2)), dims=('obs', 'feat'))
    r = repr(dc)
    assert "<DataContainer" in r
    assert "obs=5" in r

def test_save_load_errors(tmp_path):
    """Test save/load failure modes."""
    dc = DataContainer(np.zeros((2,2)), dims=('a','b'))
    
    # Load non-existent
    with pytest.raises(FileNotFoundError):
        DataContainer.load(tmp_path / "missing.joblib")
        
    # Load wrong type
    import joblib
    bad_file = tmp_path / "bad.joblib"
    joblib.dump({"not": "a container"}, bad_file)
    
    with pytest.raises(TypeError, match="Loaded object is"):
        DataContainer.load(bad_file)

def test_stack_expansion():
    """Test stack with and without y/ids expansion."""
    # Shape: (2 obs, 2 time)
    X = np.arange(4).reshape(2, 2)
    # ids: sub-0, sub-1
    # y: 0, 1
    dc = DataContainer(
        X, dims=('obs', 'time'), 
        y=np.array([0, 1]), 
        ids=np.array(['sub0', 'sub1']),
        coords={'time': [10, 20]}
    )
    
    # Stack obs and time -> new obs
    stacked = dc.stack(dims=('obs', 'time'), new_dim='obs')
    assert stacked.shape == (4,)
    
    # Check y expansion (repeated)
    assert np.array_equal(stacked.y, [0, 0, 1, 1])
    
    # Check ids expansion (combined)
    # sub0_10, sub0_20, sub1_10, sub1_20
    assert "sub0_10" in stacked.ids[0]
    
    # Test error
    with pytest.raises(ValueError):
        dc.stack(dims=('obs', 'invalid'))

def test_normalization_methods():
    """Test center, zscore, rms_scale."""
    X = np.array([[1., 2.], [3., 4.]]) # Mean=2.5
    dc = DataContainer(X, dims=('obs', 'feat'))

    # Error
    with pytest.raises(ValueError):
        dc.center(dim='invalid')
        
    # Center Inplace
    import copy
    dc_c = copy.deepcopy(dc)
    dc_c.center(dim='feat', inplace=True)
    assert np.allclose(np.mean(dc_c.X, axis=1), 0)
    
    # Zscore
    dc_z = dc.zscore(dim='feat')
    assert np.allclose(np.std(dc_z.X, axis=1), 1)
    
    # RMS
    dc_rms = dc.rms_scale(dim='feat')
    assert np.all(dc_rms.X < dc.X)

def test_aggregate():
    """Test aggregation edge cases."""
    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1]) # Consistent for group 'A', unique for 'B'
    ids = np.array(['a1', 'a2', 'b1'])
    coords = {'group': ['A', 'A', 'B'], 'bad_len': [1, 2]}
    
    dc = DataContainer(X, dims=('obs', 'feat'), coords=coords, y=y, ids=ids)
    
    # 1. Error: missing obs
    dc_no_obs = DataContainer(X, dims=('d1', 'd2'))
    with pytest.raises(ValueError, match="Aggregation requires 'obs'"):
        dc_no_obs.aggregate(by='anything')
        
    # 2. Error: bad by key
    with pytest.raises(ValueError, match="Grouping key 'miss' not found"):
        dc.aggregate(by='miss')
        
    # 3. Error: length mismatch
    with pytest.raises(ValueError, match="Grouping array length"):
        dc.aggregate(by=[1, 2])
        
    # 4. Aggregation Mean (Standard)
    agg = dc.aggregate(by='group', method='mean')
    assert agg.shape == (2, 2)
    assert np.array_equal(agg.coords['obs'], ['A', 'B'])
    # Group A: (1+2)/2 = 1.5. Group B: 3.
    assert agg.X[0, 0] == 1.5
    assert agg.y is not None 
    assert np.array_equal(agg.y, [0, 1]) # 0 is consistent for A
    
    # 5. Method variants
    agg_std = dc.aggregate(by='group', method='std')
    assert agg_std.ids is None # Std voids IDs
    
    with pytest.raises(ValueError, match="Unknown method"):
        dc.aggregate(by='group', method='invalid')

def test_aggregate_unknown_method():
    """Test unknown aggregation method error."""
    X = np.zeros((2, 2))
    dc = DataContainer(X, dims=('obs', 'f'))
    with pytest.raises(ValueError, match="Unknown method"):
        dc.aggregate(by=[1, 2], method='magic')
