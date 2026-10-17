import numpy as np
import pandas as pd
import pytest

from coco_pipe.dim_reduction.evaluation.result import TrajectoryResult
from coco_pipe.dim_reduction.evaluation.stats import (
    grouped_condition_stats,
    paired_condition_stats,
    permutation_familywise_separation_auc,
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
    with pytest.raises(ValueError, match=r"Unknown condition\(s\) \[3\]"):
        permutation_null_separation_auc(res, group_a=[3], group_b=[2])

    rng = np.random.default_rng(0)
    permutation_null_separation_auc(res, group_a=[1], group_b=[2], n_perm=10, rng=rng)


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
    obs, null, _ = permutation_null_separation_auc(
        res, group_a=[1], group_b=[2], n_perm=10, window=(0.5, 1.5)
    )
    assert np.isfinite(obs)
    assert null.shape == (10,)


# --- permutation p-values -------------------------------------------------------

CONTRASTS = [(["famous", "unfamiliar"], "scrambled"), ("famous", "unfamiliar")]


def _band_family(effect, seed=0, n_subjects=8, n_trials=3):
    """Two bands of the same trials; ``scrambled`` sits ``effect`` away in alpha."""
    rng = np.random.default_rng(seed)
    conditions = np.tile(
        np.repeat(["famous", "unfamiliar", "scrambled"], n_trials), n_subjects
    )
    subjects = np.repeat([f"s{i}" for i in range(n_subjects)], 3 * n_trials)
    times = np.linspace(0, 1, 15)

    def band(scale):
        traj = rng.normal(size=(conditions.size, times.size, 3))
        traj[conditions == "scrambled", :, 0] += scale * effect
        return TrajectoryResult(traj, times, subjects, conditions)

    return {"alpha": band(1.0), "beta": band(0.0)}


def test_a_known_separation_gets_the_smallest_p():
    family = _band_family(effect=3.0)
    table, max_null = permutation_familywise_separation_auc(
        family, CONTRASTS, n_perm=99
    )
    p = table.set_index(["key", "contrast"])["p_fwer"]
    assert p["alpha", "famous+unfamiliar_vs_scrambled"] == pytest.approx(1 / 100)
    assert p["beta", "famous_vs_unfamiliar"] > 0.2
    assert max_null.shape == (99,)

    _, _, p_single = permutation_null_separation_auc(
        family["alpha"],
        ["famous", "unfamiliar"],
        ["scrambled"],
        n_perm=99,
        within_subject=True,
    )
    assert p_single == pytest.approx(1 / 100)


def test_p_values_are_uniform_under_exchangeable_labels():
    smallest_fwer, single = [], []
    for seed in range(100):
        family = _band_family(effect=0.0, seed=seed)
        rng = np.random.default_rng(seed)
        table, _ = permutation_familywise_separation_auc(
            family, CONTRASTS, n_perm=49, rng=rng
        )
        smallest_fwer.append(table["p_fwer"].min())
        single.append(
            permutation_null_separation_auc(
                family["alpha"], "famous", "unfamiliar", n_perm=49, rng=rng
            )[2]
        )
    # 100 draws: the mean of uniform p-values has sd 0.029, P(p <= 0.1) sd 0.03
    for p_values in (np.array(smallest_fwer), np.array(single)):
        assert 0.41 < p_values.mean() < 0.59
        assert 0.01 < np.mean(p_values <= 0.1) < 0.19


def test_one_relabelling_per_permutation_is_shared_across_the_family():
    alpha = _band_family(effect=1.0)["alpha"]
    # two identical members: shared relabellings give them identical AUCs
    family = {
        "a": alpha,
        "b": TrajectoryResult(
            alpha.trajectories.copy(), alpha.times, alpha.subjects, alpha.conditions
        ),
    }
    _, max_null = permutation_familywise_separation_auc(
        family, CONTRASTS, n_perm=30, rng=np.random.default_rng(5)
    )
    nulls = [
        permutation_null_separation_auc(
            alpha, a, b, n_perm=30, rng=np.random.default_rng(5), within_subject=True
        )[1]
        for a, b in CONTRASTS
    ]
    np.testing.assert_array_equal(max_null, np.maximum(*nulls))


def test_familywise_results_must_be_aligned():
    family = _band_family(effect=0.0)
    beta = family["beta"]
    order = np.argsort(beta.conditions, kind="stable")
    family["beta"] = TrajectoryResult(
        beta.trajectories[order],
        beta.times,
        beta.subjects[order],
        beta.conditions[order],
    )
    with pytest.raises(ValueError, match="'beta' does not hold the subjects"):
        permutation_familywise_separation_auc(family, CONTRASTS, n_perm=5)
    with pytest.raises(ValueError, match="Unknown condition"):
        permutation_familywise_separation_auc(
            _band_family(effect=0.0), [("famous", "scrambeld")], n_perm=5
        )
