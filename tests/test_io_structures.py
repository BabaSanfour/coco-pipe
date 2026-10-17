import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from coco_pipe.descriptors import DescriptorPipeline
from coco_pipe.io.structures import DataContainer


@pytest.fixture
def sample_container():
    return DataContainer(
        X=np.arange(2 * 3 * 4).reshape(2, 3, 4),
        dims=("obs", "channel", "time"),
        coords={
            "channel": ["Fz", "Cz", "Pz"],
            "time": np.array([0, 1, 2, 3]),
            "Study ID": np.array(["S0", "S1"]),
            "group": np.array(["control", "patient"]),
        },
        y=np.array([0, 1]),
        ids=np.array(["s0", "s1"]),
    )


def _descriptors(X, names=("alpha_ch-all", "beta_ch-all")):
    return DataContainer(
        X=np.asarray(X, dtype=np.float32),
        dims=("obs", "feature"),
        coords={"feature": list(names)},
    )


def _with_nans():
    return _descriptors([[np.nan, 1.0], [np.nan, 2.0], [3.0, 4.0], [np.nan, np.nan]])


# --- construction ------------------------------------------------------------


def test_construction_checks_shapes_and_lengths():
    X = np.zeros((10, 5))
    with pytest.raises(ValueError, match="Shape mismatch"):
        DataContainer(X, dims=("obs",))
    with pytest.raises(ValueError, match="Shape mismatch"):
        DataContainer(np.ones((2, 3)), dims=("obs", "feature", "time"))
    with pytest.raises(ValueError, match=r"conflicting sizes .* coordinate 'obs'"):
        DataContainer(X, dims=("obs", "feat"), coords={"obs": [0, 1]})
    with pytest.raises(ValueError, match=r"conflicting sizes .* coordinate 'y'"):
        DataContainer(X, dims=("obs", "feat"), y=np.zeros(3))

    text = repr(DataContainer(np.zeros((5, 2)), dims=("obs", "feat")))
    assert "<DataContainer" in text
    assert "obs=5" in text


# --- isel / select -----------------------------------------------------------


def test_isel_slices_data_and_aligned_metadata(sample_container):
    subset = sample_container.isel(obs=[1], channel=[0, 2], time=slice(1, 3))
    assert subset.X.shape == (1, 2, 2)
    assert subset.coords["channel"].tolist() == ["Fz", "Pz"]
    assert subset.coords["time"].tolist() == [1, 2]
    assert subset.y.tolist() == [1]
    assert subset.ids.tolist() == ["s1"]

    assert sample_container.isel() is sample_container
    with pytest.raises(ValueError, match="unknown"):
        sample_container.isel(unknown=[0])
    with pytest.raises(IndexError):
        sample_container.isel(obs=[10])


def test_isel_keeps_each_coordinate_on_its_own_dimension():
    # Every axis has length 2, so a coordinate cannot be matched by length.
    tensor = DataContainer(
        X=np.arange(8, dtype=float).reshape(2, 2, 2),
        dims=("obs", "sensor", "feature"),
        coords={
            "sensor": ["Fz", "Cz"],
            "feature": ["alpha", "beta"],
            "feature_family": ["band", "connectivity"],
        },
    )
    sliced = tensor.isel(sensor=0, feature=[0, 1])
    assert sliced.X.shape == (2, 1, 2)
    assert list(sliced.coords["feature_family"]) == ["band", "connectivity"]
    assert list(sliced.coords["sensor"]) == ["Fz"]

    flat = DataContainer(
        X=np.arange(9).reshape(3, 3),
        dims=("obs", "channel"),
        coords={"channel": ["c0", "c1", "c2"], "subject": np.array(["s0", "s1", "s2"])},
        ids=np.array(["o0", "o1", "o2"]),
    )
    sub = flat.isel(channel=[0, 1])
    assert list(sub.coords["channel"]) == ["c0", "c1"]
    np.testing.assert_array_equal(sub.coords["subject"], ["s0", "s1", "s2"])
    np.testing.assert_array_equal(sub.ids, ["o0", "o1", "o2"])


def test_select_matches_values_patterns_and_operators(sample_container):
    wildcard = sample_container.select(channel="*z", time={">": 1})
    assert wildcard.X.shape == (2, 3, 2)
    assert wildcard.coords["time"].tolist() == [2, 3]

    lowercase = sample_container.select(channel=["pz"], ignore_case=True)
    assert lowercase.coords["channel"].tolist() == ["Pz"]

    assert sample_container.select(ids={"in": ["s0"]}).ids.tolist() == ["s0"]
    assert sample_container.select(channel={"!=": "Cz"}).coords["channel"].tolist() == [
        "Fz",
        "Pz",
    ]
    odd = sample_container.select(time=lambda values: values % 2 != 0)
    assert odd.coords["time"].tolist() == [1, 3]
    assert sample_container.select(group=["patient"]).ids.tolist() == ["s1"]


def test_select_errors(sample_container):
    for empty in (
        lambda: sample_container.select(ids=["non_existent"]),
        lambda: sample_container.select(time={">": 100}),
        lambda: sample_container.select(time=1).select(time=2),
        lambda: sample_container.select(y=[0], ids=["s1"]),
    ):
        with pytest.raises(ValueError, match="resulted in empty set"):
            empty()
    with pytest.raises(ValueError, match="Unknown operator"):
        sample_container.select(time={"??": 1})
    with pytest.raises(KeyError, match="missing_dim"):
        sample_container.select(missing_dim=["a", "b"])


# --- flatten / stack / unstack ----------------------------------------------


def test_flatten_builds_feature_labels_and_keeps_coordinates(sample_container):
    flat = sample_container.flatten(preserve="obs")
    assert flat.dims == ("obs", "feature")
    assert flat.X.shape == (2, 12)
    assert flat.coords["feature"][0] == "Fz_0"
    assert flat.coords["group"].tolist() == ["control", "patient"]

    # coordinates of the flattened axes follow the Cartesian product
    product = DataContainer(
        X=np.zeros((1, 2, 2)),
        dims=("obs", "sensor", "feature"),
        coords={
            "sensor": ["Fz", "Cz"],
            "sensor_region": ["frontal", "central"],
            "feature": ["alpha", "beta"],
            "feature_family": ["band", "connectivity"],
        },
    ).flatten(preserve="obs", sep="-")
    assert product.coords["feature"].tolist() == [
        "Fz-alpha",
        "Fz-beta",
        "Cz-alpha",
        "Cz-beta",
    ]
    assert product.coords["sensor_region"].tolist() == [
        "frontal",
        "frontal",
        "central",
        "central",
    ]
    assert product.coords["feature_family"].tolist() == [
        "band",
        "connectivity",
        "band",
        "connectivity",
    ]

    partial = DataContainer(
        X=np.zeros((2, 2, 3, 4)),
        dims=("obs", "ch", "time", "freq"),
        coords={"ch": ["C1", "C2"], "time": [0, 1, 2], "freq": [10, 20, 30, 40]},
    ).flatten(preserve=["obs", "freq"])
    assert partial.dims == ("obs", "freq", "feature")
    assert partial.X.shape == (2, 4, 6)
    assert partial.coords["feature"][0] == "C1_0"

    assert flat.flatten(preserve=["obs", "feature"]) is flat
    with pytest.raises(ValueError, match="Dimension 'missing' not found"):
        sample_container.flatten(preserve=["missing"])


def test_flatten_warns_past_label_cap():
    container = DataContainer(
        X=np.zeros((1, 200000)),
        dims=("obs", "sample"),
        coords={"sample": np.arange(200000)},
    )
    with pytest.warns(UserWarning, match="exceeds the 200000 label cap"):
        flat = container.flatten(preserve="obs")
    assert "feature" not in flat.coords


def test_stack_expands_y_ids_and_observation_coordinates(sample_container):
    stacked = sample_container.stack(dims=("obs", "time"), new_dim="obs")
    assert stacked.X.shape == (8, 3)
    assert stacked.y.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert stacked.ids[0].startswith("s0_")
    assert stacked.coords["group"].tolist() == ["control"] * 4 + ["patient"] * 4

    # a dimension without a coordinate is numbered
    no_time = DataContainer(
        X=np.arange(6).reshape(2, 3),
        dims=("obs", "time"),
        ids=np.array(["o0", "o1"]),
    ).stack(dims=("obs", "time"), new_dim="obs")
    assert no_time.X.shape == (6,)
    assert len(no_time.ids) == 6

    with pytest.raises(ValueError, match="Dimension 'missing' not found"):
        sample_container.stack(dims=("obs", "missing"))


@pytest.mark.parametrize(
    "dims, shape",
    [
        (("trials", "time", "channels"), (10, 50, 4)),
        (("channels", "trials", "time"), (4, 10, 50)),
    ],
)
def test_stack_unstack_round_trip(dims, shape):
    X = np.random.default_rng(42).standard_normal(shape)
    container = DataContainer(X=X, dims=dims)
    stacked = container.stack(dims=("trials", "time"), new_dim="obs")
    assert stacked.shape == (500, 4)

    restored = stacked.unstack("obs")
    assert restored.dims == dims
    np.testing.assert_array_equal(restored.X, X)
    assert "_stacked_snapshot" not in restored.meta


def test_stack_unstack_restores_metadata(sample_container):
    restored = sample_container.stack(dims=("obs", "time"), new_dim="obs").unstack(
        "obs"
    )
    assert restored.dims == sample_container.dims
    np.testing.assert_array_equal(restored.X, sample_container.X)
    np.testing.assert_array_equal(restored.y, sample_container.y)
    np.testing.assert_array_equal(restored.ids, sample_container.ids)
    np.testing.assert_array_equal(
        restored.coords["time"], sample_container.coords["time"]
    )

    corrupted = sample_container.stack(dims=("obs", "time"), new_dim="obs")
    corrupted.meta["stacked_shapes"] = (10, 10)
    with pytest.raises(ValueError, match="Shape mismatch"):
        corrupted.unstack("obs")


def test_unstack_from_metadata_and_errors():
    container = DataContainer(
        X=np.zeros((100, 5)),
        dims=("obs", "feats"),
        ids=np.array([f"id_{i}" for i in range(100)]),
        coords={"obs": np.arange(100), "feats": np.arange(5)},
        meta={"stacked_from": ("a", "b"), "stacked_shapes": (10, 10)},
    )
    unstacked = container.unstack("obs")
    assert unstacked.shape == (10, 10, 5)
    assert unstacked.dims == ("a", "b", "feats")
    assert unstacked.ids is None
    assert "obs" not in unstacked.coords

    with pytest.raises(ValueError, match="Cannot unstack: Metadata"):
        DataContainer(X=np.zeros((10, 10)), dims=("obs", "features")).unstack("obs")
    with pytest.raises(ValueError, match="Dimension 'c' not found"):
        DataContainer(X=np.zeros((10, 10)), dims=("a", "b")).unstack("c")


# --- with_features -----------------------------------------------------------


def test_with_features_replaces_the_feature_axis(sample_container):
    stacked = sample_container.stack(dims=("obs", "time"), new_dim="obs")
    out = stacked.with_features(
        np.random.randn(8, 3), names=["PC1", "PC2", "PC3"], new_dim_name="component"
    )
    assert out.dims == ("obs", "component")
    assert list(out.coords["component"]) == ["PC1", "PC2", "PC3"]
    assert out.y.shape == stacked.y.shape
    assert out.ids.shape == stacked.ids.shape
    assert "channel" not in out.coords

    same = sample_container.with_features(
        np.random.randn(2, 3, 2), feature_dim="time", new_dim_name="time"
    )
    assert same.dims == sample_container.dims
    assert list(same.coords["time"]) == [0, 1]

    with pytest.raises(ValueError, match="leading axes"):
        sample_container.with_features(np.random.randn(99, 3, 3), names=["a", "b", "c"])
    with pytest.raises(ValueError, match="number of dimensions"):
        sample_container.with_features(np.random.randn(99, 3), names=["a", "b", "c"])
    with pytest.raises(ValueError, match="names"):
        sample_container.with_features(np.random.randn(2, 3, 3), names=["a", "b"])


# --- tables -------------------------------------------------------------------


def test_feature_schema():
    container = DataContainer(
        X=np.zeros((1, 2)),
        dims=("obs", "feature"),
        coords={
            "feature": ["opaque0", "opaque1"],
            "feature_family": ["band", "connectivity"],
            "feature_measure": ["alpha", "coherence"],
        },
    )
    assert container.feature_schema().to_dict(orient="list") == {
        "column": ["opaque0", "opaque1"],
        "family": ["band", "connectivity"],
        "measure": ["alpha", "coherence"],
    }
    no_feature = DataContainer(
        X=np.zeros((1, 2)), dims=("obs", "channel"), coords={"channel": ["Fz", "Cz"]}
    )
    assert no_feature.feature_schema() is None


def test_obs_table_exports_observation_vectors(sample_container):
    container = sample_container.replace(
        coords={
            **sample_container.coords,
            "bad_matrix": np.array([[1, 2], [3, 4]]),
            "obs": ["O1", "O2"],
        }
    )
    table = container.obs_table(include_ids=True, include_y=True, y_col="target")
    assert table.columns.tolist() == ["obs_id", "Study ID", "group", "target"]
    assert table["obs_id"].tolist() == ["s0", "s1"]
    assert table["target"].tolist() == [0, 1]
    assert container.obs_table(include_obs_coord=True)["obs"].tolist() == ["O1", "O2"]

    with pytest.raises(ValueError, match="include_ids=True"):
        sample_container.replace(ids=None).obs_table(include_ids=True)
    with pytest.raises(ValueError, match="requires an 'obs' dimension"):
        DataContainer(np.zeros((10, 10)), dims=("a", "b")).obs_table()


def test_observation_frame():
    container = DataContainer(
        X=np.random.rand(3, 3),
        dims=("obs", "feature"),
        coords={"feature": ["f0", "f1", "f2"], "subject": np.array(["a", "b", "c"])},
        ids=np.array(["o0", "o1", "o2"]),
    )
    frame = container.observation_frame()
    assert "feature" not in frame
    assert list(frame["subject"]) == ["a", "b", "c"]
    assert list(frame["sample_id"]) == ["o0", "o1", "o2"]

    unnamed = DataContainer(X=np.zeros((2, 2)), dims=("obs", "feature"))
    assert unnamed.observation_frame()["sample_id"].iloc[0] == "sample-000000"
    with pytest.raises(ValueError, match="no 'obs' dimension"):
        DataContainer(X=np.zeros((2, 2)), dims=("a", "b")).observation_frame()

    wide = DataContainer(
        X=np.zeros((2, 2)),
        dims=("obs", "feature"),
        coords={f"coord_{i}": np.asarray([i, i + 1]) for i in range(150)},
        ids=np.asarray(["a", "b"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        assert wide.observation_frame().shape[1] == 151


# --- save / load -------------------------------------------------------------


def test_save_load_round_trip(tmp_path, sample_container):
    path = tmp_path / "container.joblib"
    sample_container.save(path)

    loaded = DataContainer.load(path)
    assert loaded.dims == sample_container.dims
    np.testing.assert_array_equal(loaded.X, sample_container.X)
    np.testing.assert_array_equal(loaded.y, sample_container.y)
    np.testing.assert_array_equal(loaded.ids, sample_container.ids)
    assert loaded.coords.keys() == sample_container.coords.keys()
    # each coordinate comes back on the dimension it was on
    assert loaded.isel(obs=[1]).coords["group"].tolist() == ["patient"]
    assert loaded.isel(obs=[1]).coords["channel"].tolist() == ["Fz", "Cz", "Pz"]

    with pytest.raises(FileNotFoundError):
        DataContainer.load(tmp_path / "missing.joblib")
    joblib.dump({"not": "a container"}, tmp_path / "bad.joblib")
    with pytest.raises(ValueError, match="does not hold a saved DataContainer"):
        DataContainer.load(tmp_path / "bad.joblib")


# --- concat ------------------------------------------------------------------


def test_concat_stacks_observations_and_keeps_other_coordinates():
    # feature length equals the first container's observation count, so a
    # coordinate cannot be told apart by length
    first = DataContainer(
        X=np.ones((2, 2)),
        dims=("obs", "feature"),
        coords={
            "feature": ["a", "b"],
            "feature_kind": ["spectral", "complexity"],
            "subject": ["s1", "s2"],
        },
        y=np.array([0, 0]),
        ids=np.array(["i1", "i2"]),
    )
    second = DataContainer(
        X=np.zeros((1, 2)),
        dims=("obs", "feature"),
        coords={
            "feature": ["a", "b"],
            "feature_kind": ["spectral", "complexity"],
            "subject": ["s3"],
        },
        y=np.array([1]),
        ids=np.array(["i3"]),
    )
    combined = DataContainer.concat([first, second], fill_condition_from_meta=False)
    assert combined.X.shape == (3, 2)
    assert combined.coords["feature"].tolist() == ["a", "b"]
    assert combined.coords["feature_kind"].tolist() == ["spectral", "complexity"]
    assert combined.coords["subject"].tolist() == ["s1", "s2", "s3"]
    assert combined.y.tolist() == [0, 0, 1]
    assert combined.ids.tolist() == ["i1", "i2", "i3"]

    # "kind" has no dimension prefix: it is placed by its length
    tensor = DataContainer(
        X=np.ones((1, 2, 3)),
        dims=("obs", "sensor", "feature"),
        coords={
            "sensor": ["Fz", "Cz"],
            "feature": ["a", "b", "c"],
            "kind": ["spatial", "temporal", "spectral"],
        },
    )
    stacked = DataContainer.concat([tensor, tensor])
    assert stacked.X.shape == (2, 2, 3)
    assert stacked.coords["kind"].tolist() == ["spatial", "temporal", "spectral"]

    single = DataContainer.concat([first], fill_condition_from_meta=False)
    np.testing.assert_array_equal(single.X, first.X)
    np.testing.assert_array_equal(single.coords["subject"], first.coords["subject"])


def test_concat_fills_missing_coordinates():
    first = DataContainer(
        X=np.ones((2, 1)),
        dims=("obs", "feature"),
        coords={"feature": ["a"], "session": ["pre", "post"], "twod": np.zeros((2, 2))},
        meta={"condition": "baseline"},
    )
    second = DataContainer(
        X=np.zeros((1, 1)),
        dims=("obs", "feature"),
        coords={"feature": ["a"], "subject": ["s3"]},
        meta={"condition": "active"},
    )
    combined = DataContainer.concat([first, second])
    assert combined.coords["session"].tolist() == ["pre", "post", None]
    assert combined.coords["subject"].tolist() == [None, None, "s3"]
    assert combined.coords["condition"].tolist() == ["baseline", "baseline", "active"]
    assert "twod" not in combined.coords


def test_concat_matches_other_dimensions_by_label():
    def container(features, values, kinds):
        return DataContainer(
            X=np.array([values], dtype=float),
            dims=("obs", "feature"),
            coords={"feature": features, "feature_kind": kinds},
        )

    first = container(["alpha", "beta"], [1, 2], ["band", "band2"])
    second = container(["beta", "alpha"], [20, 10], ["band2", "band"])
    combined = DataContainer.concat([first, second], fill_condition_from_meta=False)
    assert combined.coords["feature"].tolist() == ["alpha", "beta"]
    assert combined.X.tolist() == [[1, 2], [10, 20]]
    assert combined.coords["feature_kind"].tolist() == ["band", "band2"]

    third = container(["beta", "gamma"], [200, 300], ["band2", "band3"])
    with pytest.raises(ValueError, match="different 'feature' labels"):
        DataContainer.concat([first, third])
    inner = DataContainer.concat([first, second, third], join="inner")
    assert inner.coords["feature"].tolist() == ["beta"]
    assert inner.X[:, 0].tolist() == [2, 20, 200]


def test_concat_errors():
    flat = DataContainer(np.zeros((2, 2)), dims=("obs", "feature"))
    with pytest.raises(ValueError):
        DataContainer.concat([])
    no_obs = DataContainer(np.zeros((2, 2)), dims=("a", "b"))
    with pytest.raises(ValueError, match="include an 'obs' dimension"):
        DataContainer.concat([no_obs, no_obs])
    with pytest.raises(ValueError, match="matching dims"):
        DataContainer.concat(
            [flat, DataContainer(np.ones((1, 1, 2)), dims=("obs", "sensor", "feature"))]
        )
    with pytest.raises(ValueError, match="matching non-obs dimensions"):
        DataContainer.concat(
            [flat, DataContainer(np.zeros((2, 3)), dims=("obs", "feature"))]
        )


# --- aggregate ---------------------------------------------------------------


def test_aggregate_groups_observations():
    container = DataContainer(
        np.array([[1, 1], [2, 2], [3, 3]]),
        dims=("obs", "feat"),
        coords={
            "Study ID": ["A", "A", "B"],
            "site": ["north", "north", "south"],
            "mixed": ["x", "y", "z"],
        },
        y=np.array([0, 0, 1]),
        ids=np.array(["a1", "a2", "b1"]),
    )
    agg = container.aggregate(by="Study ID", stats="mean")
    assert agg.shape == (2, 2)
    assert agg.X[0, 0] == 1.5
    assert agg.ids.tolist() == ["A", "B"]
    assert agg.coords["site"].tolist() == ["north", "south"]
    assert "mixed" not in agg.coords  # varies within a group
    assert agg.coords["epoch_count"].tolist() == [2, 1]
    assert agg.y.tolist() == [0, 1]

    # groups follow first appearance; y that varies within a group is dropped
    order = DataContainer(
        np.arange(4.0)[:, None], dims=("obs", "f"), y=np.array([0, 1, 1, 1])
    ).aggregate(by=["g2", "g1", "g2", "g3"])
    assert order.ids.tolist() == ["g2", "g1", "g3"]
    assert order.y is None

    by_y = DataContainer(
        np.zeros((2, 1)), dims=("obs", "f"), y=np.array([0, 1])
    ).aggregate(by="y")
    assert by_y.coords["obs"].tolist() == [0, 1]


def test_aggregate_statistics():
    agg = _with_nans().aggregate(by=["g1", "g1", "g2", "g2"], stats=["mad", "iqr"])
    assert agg.dims == ("obs", "stat", "feature")
    assert agg.coords["stat"].tolist() == ["mad", "iqr"]
    # NaNs are ignored feature by feature
    assert np.isnan(agg.X[0, :, 0]).all()
    np.testing.assert_allclose(agg.X[0, :, 1], [0.5, 0.5])
    np.testing.assert_allclose(agg.X[1], 0.0)

    count = _with_nans().aggregate(by=["g1", "g1", "g2", "g2"], stats="count")
    assert count.X[0].tolist() == [0.0, 2.0]
    assert count.coords["epoch_count"].tolist() == [2, 2]
    sem = _with_nans().aggregate(by=["g1", "g1", "g2", "g2"], stats="sem")
    assert np.isclose(sem.X[0, 1], np.nanstd([1.0, 2.0]) / np.sqrt(2))

    every = DataContainer(np.zeros((2, 3, 4)), dims=("obs", "channel", "time"))
    stats = ["obs-mean", "median", "std", "var", "sem", "obs-mad", "obs-iqr"]
    stats += ["min", "max", "first", "count"]
    named = every.aggregate(by=[0, 0], stats=stats).coords["stat"].tolist()
    assert named == [name.removeprefix("obs-") for name in stats]

    one_d = DataContainer(np.array([1.0, 2.0, 3.0, 4.0]), dims=("obs",))
    np.testing.assert_allclose(one_d.aggregate(by=["A", "A", "B", "B"]).X, [1.5, 3.5])
    obs_second = DataContainer(np.zeros((3, 5, 10)), dims=("channel", "obs", "time"))
    assert obs_second.aggregate(by=[0, 0, 1, 1, 2]).X.shape == (3, 3, 10)
    no_features = DataContainer(np.zeros((2, 0)), dims=("obs", "feature"))
    assert no_features.aggregate(by=["A", "B"], min_count=1).shape == (2, 0)


@pytest.mark.parametrize("policy", ["collect", "warn"])
def test_aggregate_records_groups_below_min_count(policy):
    context = (
        pytest.warns(UserWarning, match="requires at least 2")
        if policy == "warn"
        else warnings.catch_warnings()
    )
    with context:
        agg = _with_nans().aggregate(
            by=["g1", "g1", "g2", "g2"], min_count=2, on_insufficient=policy
        )
    (failure,) = agg.meta["aggregate_failures"]
    assert failure["exception_type"] == "InsufficientObservations"
    assert (failure["valid_row_count"], failure["row_count"]) == (1, 2)
    assert np.isnan(agg.X[1]).all()

    with pytest.raises(ValueError, match="has 1 valid rows, requires at least 2"):
        DataContainer(np.ones((1, 1)), dims=("obs", "f")).aggregate(
            by=["G"], min_count=2, on_insufficient="raise"
        )


def test_aggregate_errors(sample_container):
    with pytest.raises(ValueError, match="Aggregation requires 'obs'"):
        DataContainer(np.zeros((2, 2)), dims=("d1", "d2")).aggregate(by="anything")
    for kwargs, match in [
        ({"by": "miss"}, "Grouping key 'miss' not found"),
        ({"by": [1, 2, 3]}, "Grouping array length"),
        ({"by": "group", "stats": "invalid"}, "Unknown stats"),
        ({"by": "group", "stats": []}, "`stats` must not be empty"),
        ({"by": "group", "min_count": 0}, "`min_count` must be at least 1"),
        ({"by": "group", "on_insufficient": "x"}, "`on_insufficient` must be one of"),
    ]:
        with pytest.raises(ValueError, match=match):
            sample_container.aggregate(**kwargs)


def test_aggregate_keeps_descriptor_names():
    t = np.linspace(0, 1, 256, endpoint=False)
    X = np.random.default_rng(31).normal(scale=0.1, size=(6, 2, 256))
    X[:, 0] += np.sin(2 * np.pi * 10 * t)
    X[:, 1] += np.sin(2 * np.pi * 6 * t)
    pipe = DescriptorPipeline(
        {"families": {"bands": {"enabled": True, "outputs": ["absolute_power"]}}}
    )
    result = pipe.extract(X=X, sfreq=256.0, channel_names=["Fz", "Cz"])
    by = ["s1", "s1", "s1", "s2", "s2", "s2"]

    agg = result.aggregate(by=by, stats="mean")
    assert agg.X.shape == (2, result.X.shape[1])
    assert any(name.endswith("_ch-Fz") for name in agg.coords["feature"])

    pooled = pipe.pool_channels(result, {"Frontal": ["Fz", "Cz"]})
    agg = pooled.aggregate(by=by, stats=["mean", "std"])
    assert agg.dims == ("obs", "stat", "feature")
    assert any(name.endswith("_chgrp-Frontal") for name in agg.coords["feature"])


# --- aggregate_groups ----------------------------------------------------------


def test_aggregate_groups_applies_stats_per_selection():
    container = _descriptors(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [3.0, 14.0, 120.0, 1100.0],
            [5.0, 18.0, 130.0, 1300.0],
            [7.0, 22.0, 150.0, 1500.0],
        ],
        names=[
            "band_abs_alpha_ch-all",
            "band_log_abs_alpha_ch-all",
            "complexity_entropy_ch-all",
            "param_offset_ch-all",
        ],
    )
    agg = container.aggregate_groups(
        by=["s1", "s1", "s2", "s2"],
        groups=[
            {"stats": "mean", "exclude_prefixes": ["band_abs_"]},
            {"prefixes": ["band_log_abs_"], "stats": ["median", "iqr"]},
            {"prefixes": "complexity_", "stats": ["median", "mad"]},
            {"prefixes": ["param_"], "stats": ["median", "iqr"]},
        ],
    )
    assert agg.dims == ("obs", "feature")
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
    np.testing.assert_allclose(
        agg.X,
        [
            [12.0, 110.0, 1050.0, 12.0, 2.0, 110.0, 10.0, 1050.0, 50.0],
            [20.0, 140.0, 1400.0, 20.0, 2.0, 140.0, 10.0, 1400.0, 100.0],
        ],
    )
    assert agg.meta["agg_stats"] == ["mean", "median", "iqr", "mad"]


def test_aggregate_groups_selectors_and_failures():
    container = _descriptors(np.zeros((4, 4)), names=["aa", "ab", "ba", "bb"])
    agg = container.aggregate_groups(
        by=[0, 0, 1, 1],
        groups=[
            {"contains": ["a"], "stats": "mean"},
            {"regex": ["^b"], "stats": "median"},
            {"suffixes": ["b"], "stats": "max"},
            {"names": ["missing"], "stats": "mean"},  # skipped: matches nothing
        ],
    )
    features = agg.coords["feature"].tolist()
    assert {"mean_aa", "mean_ab", "mean_ba", "median_ba", "max_bb"} <= set(features)

    container.meta["aggregate_failures"] = [{"family": "bands", "obs_index": 0}]
    agg = container.aggregate_groups(
        by=[0, 0, 1, 1], groups=[{"names": ["aa"], "stats": "mean", "name": "G1"}]
    )
    assert agg.meta["aggregate_failures"][0]["aggregate_group_name"] == "G1"


def test_aggregate_groups_errors():
    container = DataContainer(
        X=np.random.rand(4, 3),
        dims=("obs", "feature"),
        coords={"feature": np.array(["band_a", "band_b", "param_c"], dtype=object)},
        y=np.array([0, 0, 1, 1]),
    )
    for groups, match in [
        ([], "must not be empty"),
        ([["nope"]], "must be a dict"),
        ([{"stats": "mean", "bogus": 1}], "Unknown aggregate_groups keys"),
        ([{"prefixes": ["band"]}], "must include `stats`"),
        ([{"stats": [], "prefixes": ["band"]}], "at least one stat"),
        (
            [
                {"stats": "mean", "names": ["band_a"]},
                {"stats": "mean", "prefixes": ["band_a"]},
            ],
            "duplicate feature names",
        ),
    ]:
        with pytest.raises(ValueError, match=match):
            container.aggregate_groups(by="y", groups=groups)
    with pytest.raises(ValueError, match="matched no features"):
        container.aggregate_groups(
            by="y", groups=[{"names": ["missing"], "stats": "mean"}], skip_empty=False
        )
    with pytest.raises(ValueError, match="requires a 'feature' coordinate"):
        DataContainer(np.random.rand(4, 3), dims=("obs", "feature")).aggregate_groups(
            by=[0, 0, 1, 1], groups=[{"stats": "mean"}]
        )
    with pytest.raises(ValueError, match="requires a 'feature' dimension"):
        DataContainer(
            np.random.rand(4, 2, 2), dims=("obs", "channel", "time")
        ).aggregate_groups(by=[0, 0, 1, 1], groups=[{"stats": "mean"}])


# --- balance -----------------------------------------------------------------


def test_balance_equalises_classes():
    container = DataContainer(
        X=np.zeros((12, 1)),
        dims=("obs", "f"),
        y=np.array([0] * 10 + [1] * 2),
        coords={"group": np.array(["a"] * 8 + ["b"] * 4)},
    )
    under = container.balance(strategy="undersample", random_state=0)
    assert np.bincount(under.y).tolist() == [2, 2]
    auto = container.balance(strategy="auto", random_state=42)
    assert np.bincount(auto.y).tolist() == [10, 10]
    by_coord = container.balance(target="group", strategy="undersample")
    assert np.unique(by_coord.coords["group"], return_counts=True)[1].tolist() == [4, 4]


def test_balance_keeps_covariate_mix():
    y = np.array([0] * 15 + [1] * 5)
    sex = np.array(["M"] * 10 + ["F"] * 5 + ["M"] * 2 + ["F"] * 3)
    container = DataContainer(
        X=np.zeros((20, 2)), dims=("obs", "feat"), coords={"sex": sex}, y=y
    )
    balanced = container.balance(
        target="y", covariates=["sex"], strategy="undersample", random_state=42
    )
    assert np.bincount(balanced.y).tolist() == [5, 5]
    # class 1 is 40% M, and class 0 keeps roughly that mix
    assert np.sum(balanced.select(y={"==": 0}).coords["sex"] == "M") in [1, 2, 3]

    # strata holding a single class fall back to balancing overall
    single = DataContainer(
        X=np.zeros((4, 1)),
        dims=("obs", "f"),
        coords={"cov": np.array(["a", "a", "b", "b"])},
        y=np.array([0, 0, 1, 1]),
    )
    for strategy in ("undersample", "oversample"):
        balanced = single.balance(target="y", covariates=["cov"], strategy=strategy)
        assert np.bincount(balanced.y).tolist() == [2, 2]


def test_balance_errors():
    no_y = DataContainer(np.zeros((2, 1)), dims=("obs", "f"))
    with pytest.raises(ValueError, match="no y data"):
        no_y.balance(target="y")
    with pytest.raises(ValueError, match="Target 'missing' not found"):
        no_y.balance(target="missing")
    with_y = DataContainer(np.zeros((2, 1)), dims=("obs", "f"), y=np.array([0, 1]))
    with pytest.raises(ValueError, match="Covariate 'missing' not found"):
        with_y.balance(covariates=["missing"], target="y")
