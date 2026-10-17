import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from coco_pipe.dim_reduction.evaluation.result import (
    EmbeddingQualityResult,
    TrajectoryResult,
    VelocityResult,
)
from coco_pipe.dim_reduction.evaluation.stats import permutation_null_separation_auc


@pytest.fixture
def dummy_trajectory_data():
    rng = np.random.default_rng(42)
    n_trials = 10
    n_times = 50
    n_dims = 3
    trajectories = rng.normal(size=(n_trials, n_times, n_dims))
    times = np.linspace(0, 1, n_times)
    subjects = np.array(
        [
            "sub01",
            "sub01",
            "sub01",
            "sub01",
            "sub01",
            "sub02",
            "sub02",
            "sub02",
            "sub02",
            "sub02",
        ]
    )
    conditions = np.array([1, 1, 2, 2, 2, 1, 1, 1, 2, 2])
    return trajectories, times, subjects, conditions


def test_trajectory_result_initialization(dummy_trajectory_data):
    trajectories, times, subjects, conditions = dummy_trajectory_data

    # Should initialize correctly
    res = TrajectoryResult(trajectories, times, subjects, conditions)
    assert res.trajectories.shape == trajectories.shape
    assert res.times.shape == times.shape

    # Invalid shape
    with pytest.raises(ValueError, match="must be"):
        TrajectoryResult(trajectories[0], times, subjects, conditions)


def test_trajectory_result_save_load(dummy_trajectory_data, tmp_path):
    res = TrajectoryResult(*dummy_trajectory_data)
    file_path = tmp_path / "test_res.pkl"
    res.save(file_path)

    loaded_res = TrajectoryResult.load(file_path)
    assert np.allclose(loaded_res.trajectories, res.trajectories)
    assert np.allclose(loaded_res.times, res.times)
    assert np.array_equal(loaded_res.subjects, res.subjects)
    assert np.array_equal(loaded_res.conditions, res.conditions)


def test_trajectory_result_get_per_trial_scalars(dummy_trajectory_data):
    res = TrajectoryResult(*dummy_trajectory_data)
    df = res.get_per_trial_scalars()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {"subject", "condition", "trial", "metric", "value"}
    metrics = df["metric"].unique()
    assert "mean_dispersion" in metrics
    assert "mean_jerk" in metrics
    assert expected_cols.issubset(df.columns)


def test_trajectory_result_get_per_condition_scalars(dummy_trajectory_data):
    res = TrajectoryResult(*dummy_trajectory_data)
    df = res.get_per_condition_scalars()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {"subject", "condition", "metric", "value"}
    assert expected_cols.issubset(df.columns)


def test_trajectory_result_get_separation_pair_scalars(dummy_trajectory_data):
    res = TrajectoryResult(*dummy_trajectory_data)
    df = res.get_separation_pair_scalars()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {
        "subject",
        "method",
        "pair",
        "label_a",
        "label_b",
        "metric",
        "value",
    }
    assert expected_cols.issubset(df.columns)


def test_trajectory_result_get_separation_timecourses(dummy_trajectory_data):
    res = TrajectoryResult(*dummy_trajectory_data)
    tc = res.get_separation_timecourses()

    assert isinstance(tc, dict)
    assert "centroid" in tc
    assert (1, 2) in tc["centroid"]
    assert tc["centroid"][(1, 2)].shape == dummy_trajectory_data[1].shape


def test_permutation_null_separation_auc(dummy_trajectory_data):
    res = TrajectoryResult(*dummy_trajectory_data)
    obs, null = permutation_null_separation_auc(
        res, group_a=[1], group_b=[2], n_perm=10
    )

    assert isinstance(obs, float)
    assert isinstance(null, np.ndarray)
    assert len(null) <= 10


@pytest.fixture
def dummy_quality_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 10))
    Z = rng.normal(size=(50, 2))
    return X, Z


def test_embedding_quality_result(dummy_quality_data, tmp_path):
    X, Z = dummy_quality_data
    res = EmbeddingQualityResult(X, Z)

    # Test property Q
    assert res.Q.shape == (49, 49)

    # Test methods
    k_vals = [5, 10]
    df_trust = res.get_trustworthiness(k_vals)
    assert len(df_trust) == 2

    df_sum = res.summary(k_vals)
    assert (
        len(df_sum) == 10
    )  # 5 metrics (trust, cont, lcmc, mrre_int, mrre_ext) * 2 k_values = 10 rows
    assert "metric" in df_sum.columns

    # Test shepard diagram
    d_orig, d_emb = res.get_shepard_diagram_data(sample_size=10, random_state=0)
    assert len(d_orig) == 45
    assert len(d_emb) == 45

    # Test save/load
    p = tmp_path / "qual.pkl"
    res.save(p)
    res2 = EmbeddingQualityResult.load(p)
    assert np.allclose(res2.X, res.X)


@pytest.fixture
def dummy_velocity_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 10))
    Z = rng.normal(size=(50, 2))
    times = np.arange(50)
    groups = np.zeros(50)
    return X, Z, times, groups


def test_velocity_result(dummy_velocity_data, tmp_path):
    X, Z, times, groups = dummy_velocity_data
    res = VelocityResult(X, Z, times, groups)

    # Test methods
    V = res.get_velocity_fields(delta_t=1, n_neighbors=5)
    assert V.shape == Z.shape

    # Test save/load
    p = tmp_path / "vel.pkl"
    res.save(p)
    res2 = VelocityResult.load(p)
    assert np.allclose(res2.X, res.X)


def test_trajectory_result_slice_time(dummy_trajectory_data):
    res = TrajectoryResult(*dummy_trajectory_data)
    sliced = res.slice_time(0.2, 0.8)

    assert sliced.times.min() >= 0.2
    assert sliced.times.max() <= 0.8
    assert sliced.trajectories.shape[1] == len(sliced.times)
    assert sliced.trajectories.shape[0] == res.trajectories.shape[0]


def test_trajectory_result_filter(dummy_trajectory_data):
    res = TrajectoryResult(*dummy_trajectory_data)
    filt = res.filter(subjects=["sub01"], conditions=[1])

    assert np.all(filt.subjects == "sub01")
    assert np.all(filt.conditions == 1)
    assert filt.trajectories.shape[0] < res.trajectories.shape[0]
    assert filt.trajectories.shape[1] == res.trajectories.shape[1]


def test_trajectory_result_kinematic_timecourses(dummy_trajectory_data):
    res = TrajectoryResult(*dummy_trajectory_data)
    df = res.get_kinematic_timecourses(["speed", "curvature"])

    assert "metric" in df.columns
    assert "time" in df.columns
    assert set(df["metric"].unique()) == {"speed", "curvature"}
    assert len(df) > 0


def test_trajectory_result_exceptions():
    traj = np.zeros((2, 3, 2))
    times = np.array([0, 1, 2])
    subjects = np.array([1, 1])
    conditions = np.array([1, 1])

    res = TrajectoryResult(traj, times, subjects, conditions)

    def mock_raise(*args, **kwargs):
        raise RuntimeError("Mock error")

    original_reducers = TrajectoryResult._REDUCERS
    TrajectoryResult._REDUCERS = [("mock_metric", mock_raise)]
    try:
        df = res.get_per_trial_scalars()
        assert np.isnan(df["value"].iloc[0])
    finally:
        TrajectoryResult._REDUCERS = original_reducers


def test_trajectory_result_condition_scalars_skip():
    traj = np.zeros((2, 3, 2))
    subjects = np.array([1, 2])
    conditions = np.array([1, 2])
    res = TrajectoryResult(traj, np.array([0, 1, 2]), subjects, conditions)
    df = res.get_per_condition_scalars()
    assert len(df) == 4

    # original test for mock exception
    traj2 = np.zeros((1, 3, 2))
    subjects2 = np.array([1])
    conditions2 = np.array([1])
    res2 = TrajectoryResult(traj2, np.array([0, 1, 2]), subjects2, conditions2)
    with patch(
        "coco_pipe.dim_reduction.evaluation.result.trajectory_cohesion",
        side_effect=Exception("Mock"),
    ):
        df2 = res2.get_per_condition_scalars()
        assert np.isnan(df2[df2["metric"] == "mean_cohesion"]["value"].iloc[0])


def test_trajectory_result_separation_pair_scalars():
    traj = np.zeros((2, 3, 2))
    res = TrajectoryResult(
        traj, np.array([0, 1, 2]), np.array([1, 1]), np.array([1, 1])
    )
    df = res.get_separation_pair_scalars()
    assert len(df) == 0

    TrajectoryResult(traj, np.array([0, 1, 2]), np.array([1, 1]), np.array([1, 2]))
    traj_nan = np.full((2, 3, 2), np.nan)
    res_nan = TrajectoryResult(
        traj_nan, np.array([0, 1, 2]), np.array([1, 1]), np.array([1, 2])
    )
    df_nan = res_nan.get_separation_pair_scalars(["centroid"])
    # an undefined curve is reported, as NaN, rather than dropped
    assert df_nan["metric"].tolist() == [
        "peak_separation",
        "peak_separation_time",
        "auc_separation",
    ]
    assert df_nan["value"].isna().all()


def test_trajectory_result_filter_and_slice():
    traj = np.zeros((2, 3, 2))
    times = np.array([0, 1, 2])
    res = TrajectoryResult(traj, times, np.array([1, 2]), np.array([1, 2]))

    with pytest.raises(ValueError, match="No timepoints found"):
        res.slice_time(5, 6)

    with pytest.raises(ValueError, match="0 trials"):
        res.filter(subjects=[3])


def test_trajectory_result_kinematic_timecourses_unknown_metric():
    traj = np.zeros((2, 3, 2))
    times = np.array([0, 1, 2])
    res = TrajectoryResult(traj, times, np.array([1, 2]), np.array([1, 2]))
    with pytest.raises(ValueError, match="Unknown metric"):
        res.get_kinematic_timecourses(["unknown_metric"])


def test_result_loads_type_error():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "bad.pkl")
        import joblib

        joblib.dump("not a result", p)

        with pytest.raises(TypeError, match="not a TrajectoryResult"):
            TrajectoryResult.load(p)

        with pytest.raises(TypeError, match="not an EmbeddingQualityResult"):
            EmbeddingQualityResult.load(p)

        with pytest.raises(TypeError, match="not a VelocityResult"):
            VelocityResult.load(p)


# --- condition labels -----------------------------------------------------------

# sorted, so the integer labels 3, 4, 5 below take the names' order
NAMES = np.array(["fast-easy-right", "fast-hard-left", "slow-hard-left"])
TO_NAME = {3: NAMES[0], 4: NAMES[1], 5: NAMES[2]}


def _named_and_integer_results():
    rng = np.random.default_rng(0)
    codes = np.tile(np.repeat([0, 1, 2], 4), 2)  # 2 subjects x 3 conditions x 4
    subjects = np.repeat(["s1", "s2"], 12)
    traj = rng.normal(size=(codes.size, 20, 3)) + codes[:, None, None]
    times = np.linspace(0, 1, 20)
    named = TrajectoryResult(traj, times, subjects, NAMES[codes])
    integer = TrajectoryResult(traj, times, subjects, codes + 3)
    return named, integer


def test_named_conditions_give_the_numbers_of_integer_ones():
    named, integer = _named_and_integer_results()

    per_cond_named = named.get_per_condition_scalars()
    per_cond_int = integer.get_per_condition_scalars()
    assert per_cond_named["condition"].tolist() == [
        TO_NAME[c] for c in per_cond_int["condition"]
    ]
    np.testing.assert_array_equal(per_cond_named["value"], per_cond_int["value"])

    pairs_named = named.get_separation_pair_scalars()
    pairs_int = integer.get_separation_pair_scalars()
    assert pairs_named["pair"].tolist() == [
        f"{TO_NAME[a]}_vs_{TO_NAME[b]}"
        for a, b in zip(pairs_int["label_a"], pairs_int["label_b"], strict=True)
    ]
    assert set(pairs_named["label_a"]) <= set(NAMES)
    np.testing.assert_array_equal(pairs_named["value"], pairs_int["value"])

    curves_named = named.get_separation_timecourses(methods=["centroid"])["centroid"]
    curves_int = integer.get_separation_timecourses(methods=["centroid"])["centroid"]
    assert list(curves_named) == [(TO_NAME[a], TO_NAME[b]) for a, b in curves_int]
    for (a, b), curve in curves_int.items():
        np.testing.assert_array_equal(curves_named[(TO_NAME[a], TO_NAME[b])], curve)

    obs_named, null_named = permutation_null_separation_auc(
        named,
        group_a=[NAMES[0]],
        group_b=NAMES[1:],
        n_perm=20,
        rng=np.random.default_rng(1),
        within_subject=True,
    )
    obs_int, null_int = permutation_null_separation_auc(
        integer,
        group_a=[3],
        group_b=[4, 5],
        n_perm=20,
        rng=np.random.default_rng(1),
        within_subject=True,
    )
    assert obs_named == obs_int
    np.testing.assert_array_equal(null_named, null_int)

    kept = named.filter(conditions=[NAMES[2]])
    assert set(kept.conditions) == {NAMES[2]}
    np.testing.assert_array_equal(
        kept.trajectories, integer.filter(conditions=[5]).trajectories
    )


def test_integer_conditions_keep_integer_labels():
    _, integer = _named_and_integer_results()
    pairs = integer.get_separation_pair_scalars(methods=["centroid"])
    assert pairs["pair"].unique().tolist() == ["3_vs_4", "3_vs_5", "4_vs_5"]
    assert {type(label) for label in pairs["label_a"]} == {int}
    curves = integer.get_separation_timecourses(methods=["centroid"])["centroid"]
    assert list(curves) == [(3, 4), (3, 5), (4, 5)]


def test_unknown_conditions_are_rejected():
    named, _ = _named_and_integer_results()
    with pytest.raises(
        ValueError, match=r"Unknown condition\(s\) \['fast-easy-rigth'\]"
    ):
        named.filter(conditions=["fast-easy-rigth"])
    with pytest.raises(ValueError, match="Unknown condition"):
        permutation_null_separation_auc(named, group_a=[3], group_b=[NAMES[1]])


def test_labels_must_match_the_trajectories():
    with pytest.raises(ValueError, match="one entry per trajectory"):
        TrajectoryResult(np.zeros((4, 5, 2)), np.arange(5), ["s1"] * 4, [0, 1, 0])


# --- grid constructor -----------------------------------------------------------


def test_from_grid_flattens_subject_major():
    rng = np.random.default_rng(0)
    grid = rng.normal(size=(3, 2, 10, 4))
    times = np.linspace(0, 1, 10)
    res = TrajectoryResult.from_grid(grid, times, ["s1", "s2", "s3"], ["easy", "hard"])

    assert res.subjects.tolist() == ["s1", "s1", "s2", "s2", "s3", "s3"]
    assert res.conditions.tolist() == ["easy", "hard"] * 3
    for i in range(3):
        for j in range(2):
            np.testing.assert_array_equal(res.trajectories[2 * i + j], grid[i, j])

    by_hand = TrajectoryResult(
        np.concatenate([grid[:, 0], grid[:, 1]]),
        times,
        ["s1", "s2", "s3"] * 2,
        ["easy"] * 3 + ["hard"] * 3,
    )
    pd.testing.assert_frame_equal(
        res.get_per_condition_scalars(), by_hand.get_per_condition_scalars()
    )


def test_from_grid_checks_its_shape():
    grid = np.zeros((3, 2, 10, 4))
    with pytest.raises(ValueError, match="n_subjects, n_conditions"):
        TrajectoryResult.from_grid(grid[0], np.arange(10), ["s1", "s2"], ["a", "b"])
    with pytest.raises(ValueError, match="Expected 3 subject and 2 condition"):
        TrajectoryResult.from_grid(grid, np.arange(10), ["s1", "s2"], ["a", "b"])


def test_a_trial_with_a_gap_reports_nan_for_the_metrics_that_use_it():
    traj = np.cumsum(np.random.default_rng(0).normal(size=(2, 10, 2)), axis=1)
    traj[1, 4] = np.nan
    res = TrajectoryResult(traj, np.linspace(0, 1, 10), ["s1", "s1"], ["a", "b"])
    values = res.get_per_trial_scalars().pivot(
        index="metric", columns="trial", values="value"
    )
    assert values[0].notna().all()
    # only the start-to-end distance does not pass through the missing point
    assert values.index[values[1].notna()].tolist() == ["straight_distance"]
    assert values.loc["straight_distance", 1] == pytest.approx(
        np.linalg.norm(traj[1, -1] - traj[1, 0])
    )
