import numpy as np
import pytest

from coco_pipe.descriptors import DescriptorPipeline
from coco_pipe.io.structures import DataContainer


@pytest.fixture(scope="module")
def data_container_cls():
    return DataContainer


@pytest.fixture
def sample_container(data_container_cls):
    X = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    coords = {
        "channel": ["Fz", "Cz", "Pz"],
        "time": np.array([0, 1, 2, 3]),
        "Study ID": np.array(["S0", "S1"]),
        "group": np.array(["control", "patient"]),
    }
    y = np.array([0, 1])
    ids = np.array(["s0", "s1"])
    return data_container_cls(
        X=X, dims=("obs", "channel", "time"), coords=coords, y=y, ids=ids
    )


def test_data_container_shape_validation(data_container_cls):
    X = np.ones((2, 3))
    with pytest.raises(ValueError, match="Shape mismatch"):
        data_container_cls(
            X=X, dims=("obs", "feature", "time"), coords={"feature": [1, 2, 3]}
        )


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
    assert flat.coords["Study ID"].tolist() == ["S0", "S1"]
    assert flat.coords["group"].tolist() == ["control", "patient"]

    stacked = sample_container.stack(dims=("obs", "time"), new_dim="obs")
    assert stacked.X.shape == (
        sample_container.shape[0] * sample_container.shape[2],
        sample_container.shape[1],
    )
    assert stacked.y.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert stacked.ids[0].startswith("s0_")
    assert stacked.coords["Study ID"].tolist() == ["S0"] * 4 + ["S1"] * 4
    assert stacked.coords["group"].tolist() == ["control"] * 4 + ["patient"] * 4


def test_stack_unstack_round_trip_restores_metadata(sample_container):
    """unstack() must restore y, ids, and the stacked dim's coord."""
    stacked = sample_container.stack(dims=("obs", "time"), new_dim="obs")
    restored = stacked.unstack("obs")
    assert restored.X.shape == sample_container.X.shape
    assert restored.dims == sample_container.dims
    np.testing.assert_array_equal(restored.y, sample_container.y)
    np.testing.assert_array_equal(restored.ids, sample_container.ids)
    np.testing.assert_array_equal(
        restored.coords["time"], sample_container.coords["time"]
    )
    # The internal snapshot key is dropped after consumption
    assert "_stacked_snapshot" not in restored.meta


def test_with_features_replaces_feature_axis(sample_container):
    # Stack so we have (obs, feature_axis) shape; then re-attach a 3-D embedding
    stacked = sample_container.stack(dims=("obs", "time"), new_dim="obs")
    n_obs = stacked.X.shape[0]
    new_X = np.random.randn(n_obs, 3)
    out = stacked.with_features(
        new_X, names=["PC1", "PC2", "PC3"], new_dim_name="component"
    )
    assert out.X.shape == (n_obs, 3)
    assert out.dims == ("obs", "component")
    assert list(out.coords["component"]) == ["PC1", "PC2", "PC3"]
    # y, ids, and unrelated coords survive
    assert out.y is not None and out.y.shape == stacked.y.shape
    assert out.ids is not None and out.ids.shape == stacked.ids.shape
    assert "channel" not in out.coords  # old feature coord dropped


def test_with_features_default_names_and_same_dim(sample_container):
    new_X = np.random.randn(*sample_container.X.shape[:-1], 2)
    out = sample_container.with_features(new_X, feature_dim="time", new_dim_name="time")
    assert out.X.shape == (*sample_container.X.shape[:-1], 2)
    assert out.dims == sample_container.dims
    assert list(out.coords["time"]) == [0, 1]  # default integer names


def test_with_features_validates_shape_and_names(sample_container):
    # Wrong leading axes (matching ndim but mismatched obs/channel)
    with pytest.raises(ValueError, match="leading axes"):
        sample_container.with_features(np.random.randn(99, 3, 3), names=["a", "b", "c"])
    # Wrong number of dimensions
    with pytest.raises(ValueError, match="number of dimensions"):
        sample_container.with_features(np.random.randn(99, 3), names=["a", "b", "c"])
    # Wrong number of names
    with pytest.raises(ValueError, match="names"):
        sample_container.with_features(
            np.random.randn(*sample_container.X.shape[:-1], 3), names=["a", "b"]
        )


def test_balance_undersample(data_container_cls):
    X = np.arange(6 * 2).reshape(6, 2)
    y = np.array([0, 0, 0, 0, 1, 1])
    container = data_container_cls(
        X=X, dims=("obs", "feature"), coords={"feature": [0, 1]}, y=y
    )

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


def test_concat_two_flat_containers():
    first = DataContainer(
        X=np.array([[1.0, 2.0]]),
        dims=("obs", "feature"),
        coords={"feature": ["a", "b"], "group": ["control"]},
        y=np.array([0]),
        ids=np.array(["s1"]),
        meta={"condition": "baseline"},
    )
    second = DataContainer(
        X=np.array([[3.0, 4.0], [5.0, 6.0]]),
        dims=("obs", "feature"),
        coords={"feature": ["a", "b"], "group": ["case", "case"]},
        y=np.array([1, 1]),
        ids=np.array(["s2", "s3"]),
        meta={"condition": "active"},
    )

    combined = DataContainer.concat([first, second])

    assert combined.X.shape == (3, 2)
    assert combined.coords["feature"].tolist() == ["a", "b"]
    assert combined.coords["group"].tolist() == ["control", "case", "case"]
    assert combined.y.tolist() == [0, 1, 1]
    assert combined.ids.tolist() == ["s1", "s2", "s3"]


def test_concat_tensor_containers():
    first = DataContainer(
        X=np.ones((1, 2, 3)),
        dims=("obs", "sensor", "feature"),
        coords={
            "sensor": ["Fz", "Cz"],
            "feature": ["a", "b", "c"],
            "feature_family": ["band", "band", "complexity"],
        },
    )
    second = DataContainer(
        X=np.zeros((2, 2, 3)),
        dims=first.dims,
        coords=first.coords,
    )

    combined = DataContainer.concat(
        [first, second],
        fill_condition_from_meta=False,
    )

    assert combined.X.shape == (3, 2, 3)
    assert combined.coords["sensor"].tolist() == ["Fz", "Cz"]
    assert combined.coords["feature_family"].tolist() == [
        "band",
        "band",
        "complexity",
    ]


def test_concat_carries_non_obs_coords_generically():
    first = DataContainer(
        X=np.ones((2, 3)),
        dims=("obs", "component"),
        coords={
            "component": ["C1", "C2", "C3"],
            "basis_kind": ["spatial", "temporal", "spectral"],
            "subject": ["s1", "s2"],
        },
    )
    second = DataContainer(
        X=np.zeros((1, 3)),
        dims=("obs", "component"),
        coords={
            "component": ["C1", "C2", "C3"],
            "basis_kind": ["spatial", "temporal", "spectral"],
            "subject": ["s3"],
        },
    )

    combined = DataContainer.concat(
        [first, second],
        fill_condition_from_meta=False,
    )

    assert combined.coords["basis_kind"].tolist() == [
        "spatial",
        "temporal",
        "spectral",
    ]
    assert combined.coords["subject"].tolist() == ["s1", "s2", "s3"]


def test_concat_preserves_axis_coord_when_axis_length_matches_obs():
    first = DataContainer(
        X=np.ones((2, 2)),
        dims=("obs", "feature"),
        coords={"feature": ["a", "b"], "subject": ["s1", "s2"]},
    )
    second = DataContainer(
        X=np.zeros((1, 2)),
        dims=("obs", "feature"),
        coords={"feature": ["a", "b"], "subject": ["s3"]},
    )

    combined = DataContainer.concat(
        [first, second],
        fill_condition_from_meta=False,
    )

    assert combined.coords["feature"].tolist() == ["a", "b"]
    assert combined.coords["subject"].tolist() == ["s1", "s2", "s3"]


def test_concat_preserves_auxiliary_coord_when_length_matches_first_obs():
    first = DataContainer(
        X=np.ones((2, 2)),
        dims=("obs", "feature"),
        coords={
            "feature": ["a", "b"],
            "feature_kind": ["spectral", "complexity"],
            "subject": ["s1", "s2"],
        },
    )
    second = DataContainer(
        X=np.zeros((1, 2)),
        dims=("obs", "feature"),
        coords={
            "feature": ["a", "b"],
            "feature_kind": ["spectral", "complexity"],
            "subject": ["s3"],
        },
    )

    combined = DataContainer.concat(
        [first, second],
        fill_condition_from_meta=False,
    )

    assert combined.coords["feature_kind"].tolist() == [
        "spectral",
        "complexity",
    ]
    assert combined.coords["subject"].tolist() == ["s1", "s2", "s3"]


def test_concat_partial_coord_presence_fills_none():
    first = DataContainer(
        X=np.ones((2, 1)),
        dims=("obs", "feature"),
        coords={"feature": ["a"], "session": ["pre", "post"]},
    )
    second = DataContainer(
        X=np.zeros((1, 1)),
        dims=("obs", "feature"),
        coords={"feature": ["a"], "subject": ["s3"]},
    )

    combined = DataContainer.concat(
        [first, second],
        fill_condition_from_meta=False,
    )

    assert combined.coords["session"].tolist() == ["pre", "post", None]
    assert combined.coords["subject"].tolist() == [None, None, "s3"]


def test_concat_mismatched_dims_raises():
    flat = DataContainer(np.ones((1, 2)), dims=("obs", "feature"))
    tensor = DataContainer(np.ones((1, 1, 2)), dims=("obs", "sensor", "feature"))

    with pytest.raises(ValueError, match="matching dims"):
        DataContainer.concat([flat, tensor])


def test_concat_fills_condition_from_meta():
    first = DataContainer(
        np.ones((2, 1)),
        dims=("obs", "feature"),
        meta={"condition": "baseline"},
    )
    second = DataContainer(
        np.ones((1, 1)),
        dims=("obs", "feature"),
        meta={"condition": "active"},
    )

    combined = DataContainer.concat([first, second])

    assert combined.coords["condition"].tolist() == [
        "baseline",
        "baseline",
        "active",
    ]


def test_concat_single_container_is_noop():
    container = DataContainer(
        X=np.arange(6).reshape(3, 2),
        dims=("obs", "feature"),
        coords={"feature": ["a", "b"], "subject": ["1", "2", "3"]},
        y=np.array([0, 1, 1]),
        ids=np.array(["a", "b", "c"]),
    )

    combined = DataContainer.concat(
        [container],
        fill_condition_from_meta=False,
    )

    np.testing.assert_array_equal(combined.X, container.X)
    np.testing.assert_array_equal(combined.y, container.y)
    np.testing.assert_array_equal(combined.ids, container.ids)
    np.testing.assert_array_equal(
        combined.coords["subject"],
        container.coords["subject"],
    )


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

    # Selection on missing dimension warning (captured via logging if needed, or
    # just ensure no crash)
    # logic says it warns and ignores.
    subset = sample_container.select(missing_dim=["a", "b"])
    assert subset.shape == sample_container.shape


def test_balance_stratified(data_container_cls):
    # Create dataset with covariates
    # 20 samples. y=0 (15), y=1 (5).
    # Covariate 'sex': 0s -> 10M, 5F. 1s -> 2M, 3F.

    # We want to undersample y=0 to match y=1 (5).
    # Total = 10.

    y = np.array([0] * 15 + [1] * 5)
    sex = np.array(["M"] * 10 + ["F"] * 5 + ["M"] * 2 + ["F"] * 3)
    X = np.zeros((20, 2))

    container = data_container_cls(X=X, dims=("obs", "feat"), coords={"sex": sex}, y=y)

    # Stratified balance by sex
    balanced = container.balance(
        target="y", covariates=["sex"], strategy="undersample", random_state=42
    )

    # Check y balance
    _u, c = np.unique(balanced.y, return_counts=True)
    assert c[0] == c[1]  # Balanced classes

    # Check sex preservation (roughly)
    # y=1 has 2M 3F (40% M). y=0 should ideally maintain ~40% M.
    # y=0 (size 5). 40% of 5 is 2. So we expect around 2M in y=0.

    subset_0 = balanced.select(y={"==": 0})
    sex_0 = subset_0.coords["sex"]
    n_m = np.sum(sex_0 == "M")
    assert n_m in [1, 2, 3]  # Allow some variance due to small sample size


def test_balance_auto_oversample(data_container_cls):
    # y: 10 vs 2. Auto should decide to oversample the 2 -> 10.
    y = np.array([0] * 10 + [1] * 2)
    X = np.zeros((12, 1))
    container = data_container_cls(X=X, dims=("obs", "f"), y=y)

    balanced = container.balance(strategy="auto", random_state=42)

    _u, c = np.unique(balanced.y, return_counts=True)
    assert np.all(c == 10)
    assert balanced.shape[0] == 20


def test_flatten_complex(data_container_cls):
    # (obs, ch, time, freq)
    X = np.zeros((2, 2, 3, 4))
    dims = ("obs", "ch", "time", "freq")
    coords = {"ch": ["C1", "C2"], "time": [0, 1, 2], "freq": [10, 20, 30, 40]}
    container = data_container_cls(X=X, dims=dims, coords=coords)

    # Flatten but preserve obs and freq -> (obs, freq, feature=ch*time)
    flat = container.flatten(preserve=["obs", "freq"])
    assert flat.dims == ("obs", "freq", "feature")
    assert flat.X.shape == (2, 4, 6)  # 6 = 2ch * 3time

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
        DataContainer(X, dims=("obs",))  # Missing one dim

    # 2. Coords mismatch warning (checking log would be ideal, but execution is enough)
    DataContainer(X, dims=("obs", "feat"), coords={"obs": [0, 1]})


def test_repr():
    """Test __repr__ formatting."""
    dc = DataContainer(np.zeros((5, 2)), dims=("obs", "feat"))
    r = repr(dc)
    assert "<DataContainer" in r
    assert "obs=5" in r


def test_obs_table_exports_only_obs_aligned_vectors(sample_container):
    sample_container.coords["bad_matrix"] = np.array([[1, 2], [3, 4]])

    metadata = sample_container.obs_table(
        include_ids=True, include_y=True, y_col="target"
    )

    assert metadata.columns.tolist() == [
        "obs_id",
        "Study ID",
        "group",
        "target",
    ]
    assert metadata["obs_id"].tolist() == ["s0", "s1"]
    assert metadata["Study ID"].tolist() == ["S0", "S1"]
    assert metadata["group"].tolist() == ["control", "patient"]
    assert metadata["target"].tolist() == [0, 1]


def test_obs_table_requires_ids_when_requested(sample_container):
    sample_container.ids = None

    with pytest.raises(ValueError, match="include_ids=True"):
        sample_container.obs_table(include_ids=True)


def test_obs_table_validation_errors(sample_container):
    """Test alignment and dimensionality validation in obs_table."""
    # 1. No obs dimension
    X = np.zeros((10, 10))
    dc_no_obs = DataContainer(X, dims=("a", "b"))
    with pytest.raises(ValueError, match="requires an 'obs' dimension"):
        dc_no_obs.obs_table()

    # 2. Corrupted ids (2D)
    sample_container.ids = np.zeros((2, 2))
    with pytest.raises(ValueError, match="must be 1D and aligned"):
        sample_container.obs_table(include_ids=True)

    # 3. Corrupted y (aligned but 2D)
    sample_container.ids = ["s0", "s1"]
    sample_container.y = np.zeros((2, 2))
    with pytest.raises(ValueError, match="must be 1D and aligned"):
        sample_container.obs_table(include_y=True)


def test_obs_table_include_obs_coord(sample_container):
    """Verify include_obs_coord parameter."""
    sample_container.coords["obs"] = ["O1", "O2"]
    table = sample_container.obs_table(include_obs_coord=True)
    assert "obs" in table.columns
    assert table["obs"].tolist() == ["O1", "O2"]

    # Also check coordinate loop skip logic for 'obs'
    table2 = sample_container.obs_table(include_obs_coord=False)
    assert "obs" not in table2.columns


def test_save_load_errors(tmp_path):
    """Test save/load failure modes."""
    DataContainer(np.zeros((2, 2)), dims=("a", "b"))

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
        X,
        dims=("obs", "time"),
        y=np.array([0, 1]),
        ids=np.array(["sub0", "sub1"]),
        coords={"time": [10, 20]},
    )

    # Stack obs and time -> new obs
    stacked = dc.stack(dims=("obs", "time"), new_dim="obs")
    assert stacked.shape == (4,)

    # Check y expansion (repeated)
    assert np.array_equal(stacked.y, [0, 0, 1, 1])

    # Check ids expansion (combined)
    # sub0_10, sub0_20, sub1_10, sub1_20
    assert "sub0_10" in stacked.ids[0]

    # Test error
    with pytest.raises(ValueError):
        dc.stack(dims=("obs", "invalid"))


def test_normalization_methods():
    """Test center, zscore, rms_scale."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])  # Mean=2.5
    dc = DataContainer(X, dims=("obs", "feat"))

    # Error
    with pytest.raises(ValueError):
        dc.center(dim="invalid")

    # Center Inplace
    import copy

    dc_c = copy.deepcopy(dc)
    dc_c.center(dim="feat", inplace=True)
    assert np.allclose(np.mean(dc_c.X, axis=1), 0)

    # Zscore
    dc_z = dc.zscore(dim="feat")
    assert np.allclose(np.std(dc_z.X, axis=1), 1)

    # RMS
    dc_rms = dc.rms_scale(dim="feat")
    assert np.all(dc_rms.X < dc.X)


def test_aggregate():
    """Test aggregation edge cases."""
    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1])  # Consistent for group 'A', unique for 'B'
    ids = np.array(["a1", "a2", "b1"])
    coords = {
        "Study ID": ["A", "A", "B"],
        "site": ["north", "north", "south"],
        "mixed": ["x", "y", "z"],
        "bad_len": [1, 2],
    }

    dc = DataContainer(X, dims=("obs", "feat"), coords=coords, y=y, ids=ids)

    # 1. Error: missing obs
    dc_no_obs = DataContainer(X, dims=("d1", "d2"))
    with pytest.raises(ValueError, match="Aggregation requires 'obs'"):
        dc_no_obs.aggregate(by="anything")

    # 2. Error: bad by key
    with pytest.raises(ValueError, match="Grouping key 'miss' not found"):
        dc.aggregate(by="miss")

    # 3. Error: length mismatch
    with pytest.raises(ValueError, match="Grouping array length"):
        dc.aggregate(by=[1, 2])

    # 4. Aggregation Mean (Standard)
    agg = dc.aggregate(by="Study ID", stats="mean")
    assert agg.shape == (2, 2)
    assert np.array_equal(agg.coords["obs"], ["A", "B"])
    assert np.array_equal(agg.coords["Study ID"], ["A", "B"])
    assert np.array_equal(agg.coords["site"], ["north", "south"])
    assert "mixed" not in agg.coords
    assert np.array_equal(agg.coords["epoch_count"], [2, 1])
    # Group A: (1+2)/2 = 1.5. Group B: 3.
    assert agg.X[0, 0] == 1.5
    assert agg.y is not None
    assert np.array_equal(agg.y, [0, 1])  # 0 is consistent for A

    # 5. Method variants
    agg_std = dc.aggregate(by="Study ID", stats="std")
    assert np.array_equal(agg_std.ids, ["A", "B"])

    with pytest.raises(ValueError, match="Unknown stats"):
        dc.aggregate(by="Study ID", stats="invalid")


def test_aggregate_unknown_method():
    """Test unknown aggregation method error."""
    X = np.zeros((2, 2))
    dc = DataContainer(X, dims=("obs", "f"))
    with pytest.raises(ValueError, match="Unknown stats"):
        dc.aggregate(by=[1, 2], stats="magic")


def _make_descriptor_container(X, *, descriptor_names=None):
    return DataContainer(
        X=np.asarray(X, dtype=np.float32),
        dims=("obs", "feature"),
        coords={
            "feature": descriptor_names or ["alpha_ch-all", "beta_ch-all"],
        },
    )


def _make_grouped_descriptor_container():
    return _make_descriptor_container(
        [
            [np.nan, 1.0],
            [np.nan, 2.0],
            [3.0, 4.0],
            [np.nan, np.nan],
        ],
    )


def _make_signal_data():
    rng = np.random.default_rng(31)
    t = np.linspace(0, 1, 256, endpoint=False)
    X = rng.normal(scale=0.1, size=(6, 2, 256))
    X[:, 0, :] += np.sin(2 * np.pi * 10 * t)
    X[:, 1, :] += np.sin(2 * np.pi * 6 * t)
    return X


def _descriptor_result_container(result):
    # DescriptorPipeline.extract / pool_channels already return a DataContainer.
    return result


def test_aggregate_multiple_stats_insert_stat_dimension_in_requested_order():
    agg = _make_grouped_descriptor_container().aggregate(
        by=["g1", "g1", "g2", "g2"],
        stats=["mean", "std"],
    )

    assert agg.dims == ("obs", "stat", "feature")
    assert agg.coords["stat"].tolist() == ["mean", "std"]
    assert list(agg.coords["feature"]) == ["alpha_ch-all", "beta_ch-all"]
    assert agg.X.shape == (2, 2, 2)


def test_aggregate_group_ids_follow_first_appearance_order():
    agg = _make_descriptor_container(
        [[1.0], [2.0], [3.0], [4.0]],
        descriptor_names=["alpha_ch-all"],
    ).aggregate(by=["g2", "g1", "g2", "g3"])

    assert agg.coords["obs"].tolist() == ["g2", "g1", "g3"]
    assert agg.ids.tolist() == ["g2", "g1", "g3"]


def test_aggregate_count_sem_and_epoch_count_match_expected_values():
    count_agg = _make_grouped_descriptor_container().aggregate(
        by=["g1", "g1", "g2", "g2"],
        stats="count",
    )
    sem_agg = _make_grouped_descriptor_container().aggregate(
        by=["g1", "g1", "g2", "g2"],
        stats="sem",
    )

    assert count_agg.X[0].tolist() == [0.0, 2.0]
    assert count_agg.coords["epoch_count"].tolist() == [2, 2]
    expected_sem = np.nanstd([1.0, 2.0]) / np.sqrt(2)
    assert np.isclose(sem_agg.X[0, 1], expected_sem)


def test_aggregate_mad_and_iqr_ignore_nans_per_feature():
    agg = _make_grouped_descriptor_container().aggregate(
        by=["g1", "g1", "g2", "g2"],
        stats=["mad", "iqr"],
    )

    assert agg.dims == ("obs", "stat", "feature")
    assert agg.coords["stat"].tolist() == ["mad", "iqr"]
    assert np.isnan(agg.X[0, 0, 0])
    assert np.isnan(agg.X[0, 1, 0])
    assert np.isclose(agg.X[0, 0, 1], 0.5)
    assert np.isclose(agg.X[0, 1, 1], 0.5)
    assert np.isclose(agg.X[1, 0, 0], 0.0)
    assert np.isclose(agg.X[1, 1, 0], 0.0)
    assert np.isclose(agg.X[1, 0, 1], 0.0)
    assert np.isclose(agg.X[1, 1, 1], 0.0)


def test_aggregate_groups_applies_selected_stats_in_requested_order():
    container = _make_descriptor_container(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [3.0, 14.0, 120.0, 1100.0],
            [5.0, 18.0, 130.0, 1300.0],
            [7.0, 22.0, 150.0, 1500.0],
        ],
        descriptor_names=[
            "band_abs_alpha_ch-all",
            "band_log_abs_alpha_ch-all",
            "complexity_entropy_ch-all",
            "param_offset_ch-all",
        ],
    )

    agg = container.aggregate_groups(
        by=["s1", "s1", "s2", "s2"],
        groups=[
            {
                "stats": "mean",
                "exclude_prefixes": ["band_abs_"],
            },
            {
                "prefixes": ["band_log_abs_"],
                "stats": ["median", "iqr"],
            },
            {
                "prefixes": ["complexity_"],
                "stats": ["median", "mad"],
            },
            {
                "prefixes": ["param_"],
                "stats": ["median", "iqr"],
            },
        ],
    )

    assert agg.dims == ("obs", "feature")
    assert agg.coords["obs"].tolist() == ["s1", "s2"]
    assert agg.ids.tolist() == ["s1", "s2"]
    assert agg.coords["feature"].tolist() == [
        "mean_band_log_abs_alpha_ch-all",
        "mean_complexity_entropy_ch-all",
        "mean_param_offset_ch-all",
        "median_band_log_abs_alpha_ch-all",
        "iqr_band_log_abs_alpha_ch-all",
        "median_complexity_entropy_ch-all",
        "mad_complexity_entropy_ch-all",
        "median_param_offset_ch-all",
        "iqr_param_offset_ch-all",
    ]
    assert "mean_band_abs_alpha_ch-all" not in agg.coords["feature"].tolist()
    assert np.allclose(
        agg.X[0],
        [12.0, 110.0, 1050.0, 12.0, 2.0, 110.0, 10.0, 1050.0, 50.0],
    )
    assert np.allclose(
        agg.X[1],
        [20.0, 140.0, 1400.0, 20.0, 2.0, 140.0, 10.0, 1400.0, 100.0],
    )
    assert agg.meta["agg_stats"] == ["mean", "median", "iqr", "mad"]


def test_aggregate_groups_rejects_duplicate_output_feature_names():
    container = _make_descriptor_container(
        [[1.0], [2.0], [3.0], [4.0]],
        descriptor_names=["band_log_abs_alpha_ch-all"],
    )

    with pytest.raises(ValueError, match="duplicate feature names"):
        container.aggregate_groups(
            by=["s1", "s1", "s2", "s2"],
            groups=[
                {"stats": "mean", "names": ["band_log_abs_alpha_ch-all"]},
                {"stats": "mean", "prefixes": ["band_log_abs_"]},
            ],
        )


def test_aggregate_groups_selectors_and_validation(sample_container):
    """Test selector helpers and input validation in aggregate_groups."""
    sample_container.dims = ("obs", "feature")
    sample_container.coords["feature"] = ["a", "b"]
    sample_container.X = np.zeros((2, 2))

    # 1. Unknown keys
    with pytest.raises(ValueError, match="Unknown aggregate_groups keys"):
        groups = [{"stats": "mean", "invalid": 1}]
        sample_container.aggregate_groups(by="group", groups=groups)

    # 2. Missing stats
    with pytest.raises(ValueError, match="must include `stats`"):
        sample_container.aggregate_groups(by="group", groups=[{"names": ["a"]}])

    # 3. No match when skipping is disabled
    with pytest.raises(ValueError, match="matched no features"):
        sample_container.aggregate_groups(
            by="group",
            groups=[{"names": ["missing"], "stats": "mean"}],
            skip_empty=False,
        )

    agg_empty_skipped = sample_container.aggregate_groups(
        by=[0, 1],
        groups=[
            {"names": ["missing"], "stats": "mean"},
            {"names": ["a"], "stats": "mean"},
        ],
    )
    assert agg_empty_skipped.coords["feature"].tolist() == ["mean_a"]

    # Selector variety
    container = _make_descriptor_container(
        np.zeros((4, 4)), descriptor_names=["aa", "ab", "ba", "bb"]
    )
    # Check regex and contains in one go
    agg = container.aggregate_groups(
        by=[0, 0, 1, 1],
        groups=[
            {"contains": ["a"], "stats": "mean"},
            {"regex": ["^b"], "stats": "median"},
            {"suffixes": ["b"], "stats": "max"},
        ],
    )
    assert "mean_aa" in agg.coords["feature"]
    assert "median_ba" in agg.coords["feature"]
    assert "max_bb" in agg.coords["feature"]
    agg2 = container.aggregate_groups(
        by=[0, 0, 1, 1],
        groups=[
            {"contains": ["a"], "stats": "mean"},
            {"suffixes": ["b"], "stats": "median"},
        ],
    )
    assert "mean_aa" in agg2.coords["feature"]
    assert "median_ab" in agg2.coords["feature"]


def test_aggregate_groups_meta_and_consistency(sample_container):
    """Test meta propagation and consistency checks in aggregate_groups."""
    sample_container.dims = ("obs", "feature")
    sample_container.coords["feature"] = ["f1", "f2"]
    sample_container.X = np.zeros((2, 2))

    # 2. Failures collection
    sample_container.meta["aggregate_failures"] = [
        {
            "family": "bands",
            "obs_index": 0,
            "obs_id": "s0",
            "channel_index": 0,
            "channel_name": "ch0",
            "exception_type": "Error",
            "message": "msg",
        }
    ]
    # isel will keep meta. aggregate will keep meta.
    agg = sample_container.aggregate_groups(
        by=[0, 1], groups=[{"names": ["f1"], "stats": "mean", "name": "G1"}]
    )
    assert len(agg.meta.get("aggregate_failures", [])) > 0
    assert agg.meta["aggregate_failures"][0]["aggregate_group_name"] == "G1"


def test_aggregate_groups_consistency_checks(sample_container):
    """Verify mixed per-group stats are flattened into one feature axis."""
    sample_container.dims = ("obs", "feature")
    sample_container.coords["feature"] = ["f1", "f2"]
    sample_container.X = np.zeros((2, 2))

    agg = sample_container.aggregate_groups(
        by=[0, 1],
        groups=[
            {"names": ["f1"], "stats": ["mean", "median"]},
            {"names": ["f2"], "stats": "mean"},
        ],
    )

    assert agg.dims == ("obs", "feature")
    assert agg.coords["feature"].tolist() == ["mean_f1", "median_f1", "mean_f2"]


def test_aggregate_min_count_collect_policy_records_failure():
    agg = _make_grouped_descriptor_container().aggregate(
        by=["g1", "g1", "g2", "g2"],
        min_count=2,
        on_insufficient="collect",
    )

    assert len(agg.meta["aggregate_failures"]) == 1
    assert agg.meta["aggregate_failures"][0]["exception_type"] == (
        "InsufficientObservations"
    )
    assert agg.meta["aggregate_failures"][0]["valid_row_count"] == 1
    assert agg.meta["aggregate_failures"][0]["row_count"] == 2
    assert np.isnan(agg.X[1]).all()


def test_aggregate_min_count_warn_policy_emits_warning():
    with pytest.warns(UserWarning, match="requires at least 2"):
        agg = _make_grouped_descriptor_container().aggregate(
            by=["g1", "g1", "g2", "g2"],
            min_count=2,
            on_insufficient="warn",
        )

    assert np.isnan(agg.X[1]).all()
    assert agg.meta["aggregate_failures"][0]["exception_type"] == (
        "InsufficientObservations"
    )


def test_aggregate_descriptor_pipeline_output_can_be_grouped():
    X = _make_signal_data()
    result = DescriptorPipeline(
        {
            "families": {"bands": {"enabled": True, "outputs": ["absolute_power"]}},
        }
    ).extract(X=X, sfreq=256.0, channel_names=["Fz", "Cz"])
    agg = _descriptor_result_container(result).aggregate(
        by=["s1", "s1", "s1", "s2", "s2", "s2"],
        stats="mean",
    )

    names = list(result.coords["feature"])
    assert all("_global" not in name for name in names)
    assert any(name.endswith("_ch-Fz") for name in names)
    assert agg.X.shape == (2, result.X.shape[1])
    assert agg.dims == ("obs", "feature")


def test_aggregate_descriptor_pipeline_preserves_channel_group_tokens():
    X = _make_signal_data()
    pipe = DescriptorPipeline(
        {
            "families": {"bands": {"enabled": True, "outputs": ["absolute_power"]}},
        }
    )
    result = pipe.extract(
        X=X,
        sfreq=256.0,
        channel_names=["Fz", "Cz"],
    )
    result = pipe.pool_channels(result, {"Frontal": ["Fz", "Cz"]})
    agg = _descriptor_result_container(result).aggregate(
        by=["s1", "s1", "s1", "s2", "s2", "s2"],
        stats=["mean", "std"],
    )

    assert any(name.endswith("_chgrp-Frontal") for name in result.coords["feature"])
    assert agg.dims == ("obs", "stat", "feature")
    assert agg.coords["stat"].tolist() == ["mean", "std"]


def test_unstack_basic():
    # Shape: (10 trials, 500 time, 32 channels)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 500, 32))
    dims = ("trials", "time", "channels")
    container = DataContainer(X=X, dims=dims)

    # Stack 'trials' and 'time' into 'obs'
    # Result: (5000, 32) with dims ('obs', 'channels')
    stacked = container.stack(dims=("trials", "time"), new_dim="obs")
    assert stacked.shape == (5000, 32)
    assert stacked.dims == ("obs", "channels")

    # Unstack back (using stored metadata)
    unstacked = stacked.unstack("obs")

    assert unstacked.shape == (10, 500, 32)
    assert unstacked.dims == ("trials", "time", "channels")
    np.testing.assert_array_equal(unstacked.X, X)


def test_unstack_preserves_order():
    # Shape: (32 channels, 10 trials, 500 time)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((32, 10, 500))
    dims = ("channels", "trials", "time")
    container = DataContainer(X=X, dims=dims)

    # Stack 'trials' and 'time' -> 'obs'.
    stacked = container.stack(dims=("trials", "time"), new_dim="obs")
    assert stacked.shape == (5000, 32)

    # Unstack 'obs' -> ('trials', 'time'). stack() snapshots the original
    # dim ordering, so the round-trip restores it exactly.
    unstacked = stacked.unstack("obs")
    assert unstacked.dims == ("channels", "trials", "time")
    np.testing.assert_array_equal(unstacked.X, X)


def test_unstack_updates_metadata():
    X = np.zeros((100, 5))
    ids = np.array([f"id_{i}" for i in range(100)])
    # Manually inject metadata as if stacked
    container = DataContainer(
        X=X,
        dims=("obs", "feats"),
        ids=ids,
        coords={"obs": np.arange(100), "feats": np.arange(5)},
        meta={"stacked_from": ("a", "b"), "stacked_shapes": (10, 10)},
    )

    # Unstack 'obs' -> ('a', 'b') (10, 10) using injected metadata
    unstacked = container.unstack("obs")

    assert unstacked.shape == (10, 10, 5)
    assert unstacked.dims == ("a", "b", "feats")
    assert unstacked.ids is None  # Should be dropped as length mismatches
    assert "obs" not in unstacked.coords
    assert "feats" in unstacked.coords


@pytest.mark.parametrize(
    "dims, unstack_dim, match",
    [
        # No metadata provided -> cannot reconstruct the stacked axis.
        pytest.param(
            ("obs", "features"),
            "obs",
            "Cannot unstack: Metadata",
            id="missing_metadata",
        ),
        pytest.param(("a", "b"), "c", "Dimension 'c' not found", id="dim_not_found"),
    ],
)
def test_unstack_errors(dims, unstack_dim, match):
    container = DataContainer(X=np.zeros((10, 10)), dims=dims)
    with pytest.raises(ValueError, match=match):
        container.unstack(unstack_dim)


def test_aggregate_validation_errors(sample_container):
    """Test validation of min_count, on_insufficient, and empty stats."""
    with pytest.raises(ValueError, match="`min_count` must be at least 1"):
        sample_container.aggregate(by="group", min_count=0)
    with pytest.raises(ValueError, match="`on_insufficient` must be one of"):
        sample_container.aggregate(by="group", on_insufficient="invalid")
    with pytest.raises(ValueError, match="`stats` must not be empty"):
        sample_container.aggregate(by="group", stats=[])


def test_aggregate_by_y(sample_container):
    """Verify grouping using the target vector 'y'."""
    # sample_container.y is [0, 1]
    agg = sample_container.aggregate(by="y", stats="mean")
    assert agg.shape[0] == 2
    assert np.array_equal(agg.coords["obs"], [0, 1])


def test_aggregate_obs_idx_not_zero():
    """Verify aggregation when 'obs' is not at the first axis."""
    X = np.zeros((3, 5, 10))
    # obs at index 1
    dc = DataContainer(X, dims=("channel", "obs", "time"))
    # 5 observations grouped into 3: [G0, G0, G1, G1, G2]
    agg = dc.aggregate(by=[0, 0, 1, 1, 2], stats="mean")
    assert agg.dims == ("channel", "obs", "time")
    assert agg.X.shape == (3, 3, 10)


def test_aggregate_all_stats(sample_container):
    """Verify all supported statistical measures and aliases."""
    # Using legacy aliases to ensure normalization
    stats = [
        "obs-mean",
        "median",
        "std",
        "var",
        "sem",
        "obs-mad",
        "obs-iqr",
        "min",
        "max",
        "first",
        "count",
    ]
    # Reduce all observations to 1 group
    agg = sample_container.aggregate(by=[0, 0], stats=stats)
    assert agg.dims == ("obs", "stat", "channel", "time")
    assert agg.coords["stat"].tolist() == [
        "mean",
        "median",
        "std",
        "var",
        "sem",
        "mad",
        "iqr",
        "min",
        "max",
        "first",
        "count",
    ]


def test_aggregate_1d_data():
    """Test aggregation of 1D data (only obs dimension)."""
    X = np.array([1.0, 2.0, 3.0, 4.0])
    dc = DataContainer(X, dims=("obs",))
    agg = dc.aggregate(by=["A", "A", "B", "B"], stats="mean")
    assert agg.dims == ("obs",)
    assert agg.X.shape == (2,)
    assert np.allclose(agg.X, [1.5, 3.5])


def test_aggregate_insufficient_policy_raise():
    """Verify 'raise' policy for insufficient observations."""
    X = np.ones((1, 1))
    dc = DataContainer(X, dims=("obs", "f"))
    with pytest.raises(ValueError, match="has 1 valid rows, requires at least 2"):
        dc.aggregate(by=["G"], min_count=2, on_insufficient="raise")


def test_aggregate_y_inconsistency():
    """Verify 'y' is dropped if it varies within a group."""
    X = np.ones((2, 1))
    y = np.array([0, 1])
    dc = DataContainer(X, dims=("obs", "f"), y=y)
    # Group [0, 1] has inconsistent y
    agg = dc.aggregate(by=["Group", "Group"])
    assert agg.y is None


def test_normalization_inplace(sample_container):
    """Verify in-place variants for center, zscore, and rms_scale."""
    # sample_container uses int X, must cast to float for subtract/div
    sample_container.X = sample_container.X.astype(float)

    # center
    dc_c = sample_container.center(dim="time")
    assert not np.array_equal(dc_c.X, sample_container.X)

    import copy

    dc_c_in = copy.deepcopy(sample_container)
    dc_c_in.center(dim="time", inplace=True)
    assert np.allclose(np.nanmean(dc_c_in.X, axis=2), 0)

    # zscore
    dc_z_in = copy.deepcopy(sample_container)
    dc_z_in.zscore(dim="time", inplace=True)
    assert np.allclose(np.nanstd(dc_z_in.X, axis=2), 1)

    # rms_scale
    dc_rms_in = copy.deepcopy(sample_container)
    dc_rms_in.rms_scale(dim="time", inplace=True)
    # RMS should be 1
    rms = np.sqrt(np.mean(dc_rms_in.X**2, axis=2))
    assert np.allclose(rms, 1)


def test_isel_edge_cases(sample_container, caplog):
    """Cover untested lines in isel (warnings and errors)."""
    # 1. Unknown dim warning
    import logging

    with caplog.at_level(logging.WARNING):
        subset = sample_container.isel(unknown=[0])
    assert "Dimension unknown not in" in caplog.text
    assert subset.shape == sample_container.shape

    # 2. Slicing failure (e.g. out of bounds)
    with pytest.raises(IndexError):
        sample_container.isel(obs=[10])


def test_balance_complex_edge_cases(data_container_cls):
    """Cover untested lines in balance (strata fallback and cleaning)."""
    # 1. Target not found
    X = np.zeros((2, 1))
    dc = data_container_cls(X, dims=("obs", "f"))
    with pytest.raises(ValueError, match="Target 'missing' not found"):
        dc.balance(target="missing")

    # 2. Covariate not found
    y = np.array([0, 1])
    dc2 = data_container_cls(X, dims=("obs", "f"), y=y)
    with pytest.raises(ValueError, match="Covariate 'missing' not found"):
        dc2.balance(covariates=["missing"], target="y")

    # 3. Single-class stratum in oversample (fallback path)
    y3 = np.array([0, 0, 1])
    s3 = np.array(["A", "A", "B"])
    dc3 = data_container_cls(
        X=np.zeros((3, 1)), dims=("obs", "f"), y=y3, coords={"s": s3}
    )
    # Group 'B' has only class 1. Undersample would fail, so we oversample.
    balanced = dc3.balance(target="y", covariates=["s"], strategy="oversample")
    assert balanced.shape[0] > 0


def test_select_conflicting_selections(sample_container):
    """Verify that conflicting selections on same axis raise ValueError."""
    # time=1 AND time=2 -> empty set
    with pytest.raises(ValueError, match="resulted in empty set"):
        sample_container.select(time=1).select(time=2)


def test_select_aux_coord_no_match(data_container_cls, caplog):
    """Test selection on auxiliary coordinate that matches no dimension."""
    X = np.zeros((5, 10))
    # Aux coord with len 7 (matches neither 5 nor 10)
    coords = {"aux": np.arange(7)}
    dc = data_container_cls(X, dims=("obs", "feat"), coords=coords)

    import logging

    with caplog.at_level(logging.WARNING):
        subset = dc.select(aux=1)
    assert "matches no dimension" in caplog.text
    assert subset.shape == (5, 10)


def test_select_fuzzy_no_match(sample_container, caplog):
    """Verify warning when fuzzy matching fails to find candidates."""
    import logging

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(ValueError, match="resulted in empty set"),
    ):
        # xyz is very far from Fz, Cz, Pz.
        sample_container.select(channel=["xyz"], fuzzy=True)

    assert "No fuzzy match found" in caplog.text


def test_select_no_coords(data_container_cls, caplog):
    """Test selection on a dimension that has no defined coordinates."""
    X = np.zeros((2, 2))
    dc = data_container_cls(X, dims=("obs", "feat"), coords={})

    import logging

    with caplog.at_level(logging.WARNING):
        subset = dc.select(feat=[0])
    assert "is empty" in caplog.text.lower()
    assert subset.shape == (2, 2)


def test_select_conflicting_axes(sample_container):
    """Verify intersection logic for multiple selections on the same axis (y + obs)."""
    # y=0 matches obs index 0. ids='s1' matches obs index 1. Intersection is empty.
    with pytest.raises(ValueError, match="Conflicting selections"):
        sample_container.select(y=[0], ids=["s1"])


def test_unstack_shape_mismatch(sample_container):
    """Verify error when unstacking with corrupted shape metadata."""
    stacked = sample_container.stack(dims=("obs", "time"), new_dim="obs")
    # Manually corrupt shapes metadata: 10*10 = 100, but actual obs length is 8 (2*4)
    stacked.meta["stacked_shapes"] = (10, 10)
    with pytest.raises(ValueError, match="Shape mismatch"):
        stacked.unstack("obs")


def test_normalization_invalid_dims(sample_container):
    """Verify errors for invalid dimensions in zscore and rms_scale."""
    with pytest.raises(ValueError, match="not found"):
        sample_container.zscore(dim="invalid")
    with pytest.raises(ValueError, match="not found"):
        sample_container.rms_scale(dim="invalid")


def test_baseline_correction_alias(sample_container):
    """Verify that baseline_correction is a functional alias for center."""
    sample_container.X = sample_container.X.astype(float)
    dc = sample_container.baseline_correction(dim="time")
    assert np.allclose(np.nanmean(dc.X, axis=2), 0)


def test_aggregate_empty_feature_dim():
    """Verify valid_row_count calculation for data with no features."""
    X = np.zeros((2, 0))
    dc = DataContainer(X, dims=("obs", "feature"))
    # Should not raise even if min_count=1 because valid_row_count will match row_count
    agg = dc.aggregate(by=["A", "B"], min_count=1)
    assert agg.shape == (2, 0)


def test_observation_frame_extra():
    # no obs dim
    dc = DataContainer(X=np.zeros((2, 2)), dims=("a", "b"))
    with pytest.raises(ValueError, match="no 'obs' dimension"):
        dc.observation_frame()

    # no ids
    dc2 = DataContainer(X=np.zeros((2, 2)), dims=("obs", "feature"))
    df = dc2.observation_frame()
    assert "sample_id" in df.columns
    assert df["sample_id"].iloc[0] == "sample-000000"


def test_concat_extra():
    dc1 = DataContainer(X=np.zeros((2, 2)), dims=("a", "b"))
    # concat empty
    with pytest.raises(ValueError):
        DataContainer.concat([])

    # no obs in base
    with pytest.raises(ValueError, match="include an 'obs' dimension"):
        DataContainer.concat([dc1, dc1])

    # matching dims
    dc_obs = DataContainer(X=np.zeros((2, 2)), dims=("obs", "feature"))
    dc_diff = DataContainer(X=np.zeros((2, 2)), dims=("obs", "other"))
    with pytest.raises(ValueError, match="matching dims"):
        DataContainer.concat([dc_obs, dc_diff])

    # matching non-obs dims
    dc_shape = DataContainer(X=np.zeros((2, 3)), dims=("obs", "feature"))
    with pytest.raises(ValueError, match="matching non-obs dimensions"):
        DataContainer.concat([dc_obs, dc_shape])


def test_isel_does_not_misslice_obs_coord_when_dims_share_length(data_container_cls):
    """An obs-aligned aux coord must survive indexing of a same-length dim.

    Regression: ``isel`` previously matched aux coords by length per indexer,
    so slicing ``channel`` would also slice an obs-aligned coord when ``obs``
    and ``channel`` happened to share a length.
    """
    X = np.arange(3 * 3).reshape(3, 3)  # obs=3, channel=3 share a length
    container = data_container_cls(
        X=X,
        dims=("obs", "channel"),
        coords={
            "channel": ["c0", "c1", "c2"],
            "subject": np.array(["s0", "s1", "s2"]),  # obs-aligned aux coord
        },
        ids=np.array(["o0", "o1", "o2"]),
    )

    sub = container.isel(channel=[0, 1])

    # channel sliced to 2; obs (and its aligned aux coord/ids) untouched.
    assert sub.X.shape == (3, 2)
    assert list(sub.coords["channel"]) == ["c0", "c1"]
    np.testing.assert_array_equal(sub.coords["subject"], ["s0", "s1", "s2"])
    np.testing.assert_array_equal(sub.ids, ["o0", "o1", "o2"])


def _cov_container_2d():
    return DataContainer(np.random.randn(6, 3), dims=("obs", "feature"))


def _cov_feature_container():
    return DataContainer(
        X=np.random.rand(4, 3),
        dims=("obs", "feature"),
        coords={"feature": np.array(["band_a", "band_b", "param_c"], dtype=object)},
        y=np.array([0, 0, 1, 1]),
    )


def test_balance_without_y_raises():
    with pytest.raises(ValueError, match="no y data"):
        _cov_container_2d().balance(target="y")


def test_isel_no_indexers_returns_self():
    container = _cov_container_2d()
    assert container.isel() is container


def test_aggregate_groups_requires_feature_dim():
    container = DataContainer(np.random.rand(4, 2, 2), dims=("obs", "channel", "time"))
    with pytest.raises(ValueError, match="requires a 'feature' dimension"):
        container.aggregate_groups(
            by=np.array([0, 0, 1, 1]), groups=[{"stats": "mean"}]
        )


def test_aggregate_groups_requires_feature_coord():
    container = DataContainer(np.random.rand(4, 3), dims=("obs", "feature"))
    with pytest.raises(ValueError, match="requires a 'feature' coordinate"):
        container.aggregate_groups(
            by=np.array([0, 0, 1, 1]), groups=[{"stats": "mean"}]
        )


def test_aggregate_groups_empty_groups():
    with pytest.raises(ValueError, match="must not be empty"):
        _cov_feature_container().aggregate_groups(by="y", groups=[])


def test_aggregate_groups_entry_not_dict():
    with pytest.raises(ValueError, match="must be a dict"):
        _cov_feature_container().aggregate_groups(by="y", groups=[["nope"]])


def test_aggregate_groups_unknown_keys():
    with pytest.raises(ValueError, match="Unknown aggregate_groups keys"):
        _cov_feature_container().aggregate_groups(
            by="y", groups=[{"stats": "mean", "bogus": 1}]
        )


def test_aggregate_groups_missing_stats():
    with pytest.raises(ValueError, match="must include `stats`"):
        _cov_feature_container().aggregate_groups(
            by="y", groups=[{"prefixes": ["band"]}]
        )


def test_aggregate_groups_empty_stats():
    with pytest.raises(ValueError, match="at least one stat"):
        _cov_feature_container().aggregate_groups(
            by="y", groups=[{"stats": [], "prefixes": ["band"]}]
        )


def test_aggregate_groups_string_selector():
    # A string selector (not a list) exercises the scalar->tuple normalization.
    out = _cov_feature_container().aggregate_groups(
        by="y", groups=[{"stats": "mean", "prefixes": "band"}]
    )
    assert [str(f) for f in out.coords["feature"]] == ["mean_band_a", "mean_band_b"]


def test_observation_frame_and_flatten_noop():
    container = DataContainer(
        X=np.random.rand(3, 2),
        dims=("obs", "feature"),
        coords={"feature": ["f0", "f1"], "subject": np.array(["a", "b", "c"])},
        ids=np.array(["o0", "o1", "o2"]),
    )
    frame = container.observation_frame()
    assert list(frame["subject"]) == ["a", "b", "c"]
    assert list(frame["sample_id"]) == ["o0", "o1", "o2"]

    # flatten with nothing to collapse returns the same container.
    assert container.flatten(preserve=["obs", "feature"]) is container


def test_concat_skips_non_1d_aux_coord():
    container = DataContainer(
        X=np.zeros((2, 3)),
        dims=("obs", "feature"),
        coords={"feature": ["a", "b", "c"], "twod": np.zeros((2, 2))},
    )
    out = DataContainer.concat([container, container])
    assert out.X.shape[0] == 4
    assert "twod" not in out.coords  # 2-D aux coord is not propagated


def test_balance_stratified_single_class_strata_fallback():
    # Each stratum is single-class -> stratified path falls back to global balance.
    container = DataContainer(
        X=np.random.rand(4, 2),
        dims=("obs", "feature"),
        coords={"feature": ["f0", "f1"], "cov": np.array(["a", "a", "b", "b"])},
        y=np.array([0, 0, 1, 1]),
    )
    balanced = container.balance(target="y", covariates=["cov"], strategy="undersample")
    _, counts = np.unique(balanced.y, return_counts=True)
    assert counts[0] == counts[1]


def test_balance_on_coordinate_target():
    container = DataContainer(
        X=np.random.rand(6, 2),
        dims=("obs", "feature"),
        coords={
            "feature": ["f0", "f1"],
            "group": np.array(["a", "a", "a", "a", "b", "b"]),
        },
    )
    balanced = container.balance(target="group", strategy="undersample")
    _, counts = np.unique(balanced.coords["group"], return_counts=True)
    assert counts[0] == counts[1]


def test_select_on_aux_coordinate():
    container = DataContainer(
        X=np.arange(8).reshape(4, 2),
        dims=("obs", "feature"),
        coords={"feature": ["f0", "f1"], "group": np.array(["a", "a", "b", "b"])},
    )
    assert container.select(group=["a"]).X.shape[0] == 2


def test_select_unknown_key_is_ignored():
    container = DataContainer(np.arange(8).reshape(4, 2), dims=("obs", "feature"))
    # Unknown selection key is warned and ignored, leaving the container intact.
    assert container.select(nonexistent=["x"]).X.shape == (4, 2)


def test_stack_dim_without_coordinate_uses_range():
    container = DataContainer(
        X=np.arange(6).reshape(2, 3),
        dims=("obs", "time"),
        coords={"obs": np.array(["o0", "o1"])},  # no 'time' coord
        ids=np.array(["o0", "o1"]),
    )
    stacked = container.stack(dims=("obs", "time"), new_dim="obs")
    assert stacked.X.shape == (6,)
    assert len(stacked.ids) == 6
