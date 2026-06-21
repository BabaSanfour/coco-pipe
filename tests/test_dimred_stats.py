import numpy as np
import pandas as pd

from coco_pipe.dim_reduction.evaluation.result import TrajectoryResult
from coco_pipe.dim_reduction.evaluation.stats import (
    grouped_condition_stats,
    paired_condition_stats,
    permutation_null_separation_auc,
)


def test_paired_condition_stats():
    data = []
    # metric, condition, subject, value
    for sub in range(4):
        data.append(
            {"metric": "speed", "condition": "A", "subject": f"s{sub}", "value": sub}
        )
        data.append(
            {
                "metric": "speed",
                "condition": "B",
                "subject": f"s{sub}",
                "value": sub + 1,
            }
        )
        data.append(
            {"metric": "acc", "condition": "A", "subject": f"s{sub}", "value": sub * 2}
        )
        data.append(
            {
                "metric": "acc",
                "condition": "B",
                "subject": f"s{sub}",
                "value": sub * 2 + 0.5,
            }
        )

    df = pd.DataFrame(data)
    result = paired_condition_stats(df, conditions=["A", "B"])

    assert len(result) == 2
    assert set(result["metric"]) == {"speed", "acc"}
    speed_res = result[result["metric"] == "speed"].iloc[0]
    # A - B = sub - (sub + 1) = -1
    assert speed_res["mean_diff"] == -1.0
    assert "p_fdr" in result.columns


def test_grouped_condition_stats():
    data = []
    for sub in range(4):
        # A1 and A2 will be grouped into G1
        data.append(
            {"metric": "speed", "condition": "A1", "subject": f"s{sub}", "value": sub}
        )
        data.append(
            {
                "metric": "speed",
                "condition": "A2",
                "subject": f"s{sub}",
                "value": sub + 2,
            }
        )
        # B will be G2
        data.append(
            {
                "metric": "speed",
                "condition": "B",
                "subject": f"s{sub}",
                "value": sub + 0.5,
            }
        )

    df = pd.DataFrame(data)
    condition_sets = {
        "set1": {
            "G1": ["A1", "A2"],
            "G2": ["B"],
        }
    }
    result = grouped_condition_stats(df, condition_sets=condition_sets)
    assert len(result) == 1
    # G1 mean is (sub + sub + 2) / 2 = sub + 1
    # G2 mean is sub + 0.5
    # diff = (sub + 1) - (sub + 0.5) = 0.5
    assert result.iloc[0]["mean_diff"] == 0.5
    assert result.iloc[0]["set"] == "set1"


def test_grouped_condition_stats_min_pairs():
    df = pd.DataFrame(
        {
            "subject": [1, 2],
            "condition": ["A", "B"],
            "metric": ["speed", "speed"],
            "value": [1.0, 2.0],
        }
    )
    sets = {"Set1": {"G1": ["A"], "G2": ["B"]}}
    out = grouped_condition_stats(df, sets, min_pairs=3)
    assert len(out) == 0


def test_permutation_null_separation_auc_missing():
    traj = np.zeros((2, 3, 2))
    times = np.array([0, 1, 2])
    subjects = np.array([1, 1])
    conditions = np.array([1, 2])
    res = TrajectoryResult(traj, times, subjects, conditions)
    # The groups passed are not present
    obs, _null = permutation_null_separation_auc(res, group_a=[3], group_b=[4])
    assert np.isnan(obs)

    rng = np.random.default_rng(0)
    obs, _null = permutation_null_separation_auc(
        res, group_a=[1], group_b=[2], n_perm=10, rng=rng
    )


def test_paired_condition_stats_missing_conditions():
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3"],
            "condition": ["A", "A", "A"],
            "metric": ["speed"] * 3,
            "value": [1.0, 2.0, 3.0],
        }
    )
    out = paired_condition_stats(df, conditions=["A", "B"])
    assert len(out) == 0


def test_paired_condition_stats_min_pairs():
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s1", "s2"],
            "condition": ["A", "A", "B", "B"],
            "metric": ["speed"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = paired_condition_stats(df, conditions=["A", "B"], min_pairs=3)
    assert len(out) == 0


def test_grouped_condition_stats_missing_conditions():
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3"],
            "condition": ["A", "A", "A"],
            "metric": ["speed"] * 3,
            "value": [1.0, 2.0, 3.0],
        }
    )
    sets = {"Set1": {}}
    out = grouped_condition_stats(df, sets)
    assert len(out) == 0

    sets2 = {"Set1": {"G1": ["A"], "G2": ["B"]}}
    out2 = grouped_condition_stats(df, sets2)
    assert len(out2) == 0


def test_permutation_null_separation_auc_window():
    traj = np.zeros((4, 3, 2))
    traj[0, :, 0] = 1  # group 1
    traj[1, :, 0] = 1  # group 1
    traj[2, :, 0] = 5  # group 2
    traj[3, :, 0] = 5  # group 2
    times = np.array([0.0, 1.0, 2.0])
    subjects = np.array([1, 2, 3, 4])
    conditions = np.array([1, 1, 2, 2])
    res = TrajectoryResult(traj, times, subjects, conditions)
    obs, null = permutation_null_separation_auc(
        res, group_a=[1], group_b=[2], n_perm=10, window=(0.5, 1.5)
    )
    assert np.isfinite(obs)
    assert len(null) > 0
