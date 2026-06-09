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
