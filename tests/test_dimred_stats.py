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


SPEED = (["fast-easy", "fast-hard"], ["slow-easy", "slow-hard"])


def _speed_by_class(speed_effect, seed=0, scale=1.0, class_effect=3.0):
    """Speed x class conditions with a large class effect.

    Returns the result, the subject of each trial and its subject x class
    stratum, inside which only the speed label can move.
    """
    rng = np.random.default_rng(seed)
    names = np.array(["fast-easy", "fast-hard", "slow-easy", "slow-hard"])
    conditions = np.tile(np.repeat(names, 3), 8)
    subjects = np.repeat([f"s{i}" for i in range(8)], 12)
    speed, cls = np.array([c.split("-") for c in conditions]).T
    traj = rng.normal(size=(conditions.size, 15, 3))
    traj[cls == "hard", :, 0] += class_effect
    traj[speed == "slow", :, 1] += speed_effect
    result = TrajectoryResult(traj * scale, np.linspace(0, 1, 15), subjects, conditions)
    return result, subjects, np.char.add(subjects, cls)


def test_a_known_separation_gets_the_smallest_p():
    family = _band_family(effect=3.0)
    subjects = family["alpha"].subjects
    table, max_null, null_aucs = permutation_familywise_separation_auc(
        family, CONTRASTS, strata=subjects, n_perm=99
    )
    p = table.set_index(["key", "contrast"])["p_fwer"]
    assert p["alpha", "famous+unfamiliar_vs_scrambled"] == pytest.approx(1 / 100)
    assert p["beta", "famous_vs_unfamiliar"] > 0.2
    assert max_null.shape == (99,)
    assert null_aucs.shape == (99, 2, 2)

    _, _, p_single = permutation_null_separation_auc(
        family["alpha"],
        ["famous", "unfamiliar"],
        ["scrambled"],
        n_perm=99,
        strata=subjects,
    )
    assert p_single == pytest.approx(1 / 100)


def test_p_values_are_uniform_under_exchangeable_labels():
    smallest_fwer, single = [], []
    for seed in range(100):
        family = _band_family(effect=0.0, seed=seed)
        rng = np.random.default_rng(seed)
        table, _, _ = permutation_familywise_separation_auc(
            family, CONTRASTS, strata=family["alpha"].subjects, n_perm=49, rng=rng
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


def test_strata_keep_the_other_factors_fixed():
    # no speed effect: swapping speed across classes puts the class effect in
    # the null, which then is too wide; within subject x class it is not
    p_by_subject, p_by_class = [], []
    for seed in range(60):
        result, subjects, subject_class = _speed_by_class(0.0, seed=seed)
        for strata, p_values in ((subjects, p_by_subject), (subject_class, p_by_class)):
            p_values.append(
                permutation_null_separation_auc(
                    result,
                    *SPEED,
                    n_perm=49,
                    strata=strata,
                    rng=np.random.default_rng(seed),
                )[2]
            )
    # measured: mean p 0.80 within subject, 0.51 within subject x class
    assert np.mean(p_by_subject) > 0.7
    assert 0.41 < np.mean(p_by_class) < 0.59

    # and a small speed effect is found only with the right strata
    result, subjects, subject_class = _speed_by_class(0.5)
    p_subject = permutation_null_separation_auc(
        result, *SPEED, n_perm=199, strata=subjects
    )[2]
    p_class = permutation_null_separation_auc(
        result, *SPEED, n_perm=199, strata=subject_class
    )[2]
    assert p_class == pytest.approx(1 / 200)
    assert p_subject > 0.1


def test_standardising_lets_every_member_reach_significance():
    # one PCA per band: alpha carries the effect, beta is noise ten times larger
    alpha, _, subject_class = _speed_by_class(1.0, seed=1)
    beta, _, _ = _speed_by_class(0.0, seed=2, scale=10.0)
    family = {"alpha": alpha, "beta": beta}

    def p_alpha(standardize):
        table, _, _ = permutation_familywise_separation_auc(
            family, [SPEED], strata=subject_class, n_perm=199, standardize=standardize
        )
        return table.set_index("key").loc["alpha", "p_fwer"]

    assert p_alpha(standardize=True) == pytest.approx(1 / 200)
    assert p_alpha(standardize=False) > 0.5  # beta's raw AUCs set the maximum


def test_the_max_null_is_the_maximum_over_standardised_member_nulls():
    family = _band_family(effect=1.0)
    table, max_null, null_aucs = permutation_familywise_separation_auc(
        family, CONTRASTS, strata=family["alpha"].subjects, n_perm=30
    )
    z_null = (null_aucs - null_aucs.mean(axis=0)) / null_aucs.std(axis=0, ddof=1)
    np.testing.assert_allclose(max_null, z_null.max(axis=(1, 2)))
    np.testing.assert_allclose(
        table["z"], (table["observed_auc"] - table["null_mean"]) / table["null_sd"]
    )


def test_one_relabelling_per_permutation_is_shared_across_the_family():
    alpha = _band_family(effect=1.0)["alpha"]
    # two identical members: shared relabellings give them identical AUCs
    family = {
        "a": alpha,
        "b": TrajectoryResult(
            alpha.trajectories.copy(), alpha.times, alpha.subjects, alpha.conditions
        ),
    }
    _, max_null, null_aucs = permutation_familywise_separation_auc(
        family,
        CONTRASTS,
        strata=alpha.subjects,
        n_perm=30,
        rng=np.random.default_rng(5),
        standardize=False,
    )
    np.testing.assert_array_equal(null_aucs[:, 0], null_aucs[:, 1])
    nulls = [
        permutation_null_separation_auc(
            alpha, a, b, n_perm=30, rng=np.random.default_rng(5), strata=alpha.subjects
        )[1]
        for a, b in CONTRASTS
    ]
    np.testing.assert_array_equal(max_null, np.maximum(*nulls))


def test_strata_per_contrast_draw_one_relabelling_per_contrast():
    alpha = _band_family(effect=1.0)["alpha"]
    twice = [CONTRASTS[0], CONTRASTS[0]]
    _, _, shared = permutation_familywise_separation_auc(
        {"alpha": alpha}, twice, strata=alpha.subjects, n_perm=20
    )
    _, _, separate = permutation_familywise_separation_auc(
        {"alpha": alpha}, twice, strata=[alpha.subjects, alpha.subjects], n_perm=20
    )
    np.testing.assert_array_equal(shared[:, 0, 0], shared[:, 0, 1])
    assert not np.array_equal(separate[:, 0, 0], separate[:, 0, 1])
    with pytest.raises(ValueError, match="one row per contrast"):
        permutation_familywise_separation_auc(
            {"alpha": alpha}, twice, strata=[alpha.subjects] * 3, n_perm=5
        )


def test_per_subject_averages_the_subjects_separation_curves():
    result, subjects, subject_class = _speed_by_class(1.0)
    observed, _, _ = permutation_null_separation_auc(
        result, *SPEED, n_perm=5, strata=subject_class, per_subject=True
    )
    slow = np.char.startswith(result.conditions, "slow")
    curves = [
        np.linalg.norm(
            result.trajectories[(subjects == s) & ~slow].mean(axis=0)
            - result.trajectories[(subjects == s) & slow].mean(axis=0),
            axis=-1,
        )
        for s in np.unique(subjects)
    ]
    assert observed == pytest.approx(
        np.trapezoid(np.mean(curves, axis=0), result.times)
    )

    with pytest.raises(ValueError, match="strata nested in subjects"):
        permutation_null_separation_auc(result, *SPEED, n_perm=5, per_subject=True)
    by_class_only = [condition.split("-")[1] for condition in result.conditions]
    with pytest.raises(ValueError, match="strata nested in subjects"):
        permutation_familywise_separation_auc(
            {"r": result}, [SPEED], strata=by_class_only, n_perm=5, per_subject=True
        )


def test_a_missing_point_raises_naming_the_member_and_contrast():
    family = _band_family(effect=0.0)
    family["beta"].trajectories[0, 3] = np.nan
    with pytest.raises(
        ValueError, match=r"of 'beta', 'famous\+unfamiliar_vs_scrambled'"
    ):
        permutation_familywise_separation_auc(
            family, CONTRASTS, strata=family["alpha"].subjects, n_perm=5
        )
    with pytest.raises(ValueError, match="AUC of the contrast is NaN"):
        permutation_null_separation_auc(family["beta"], "famous", "scrambled", n_perm=5)


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
    strata = family["alpha"].subjects
    with pytest.raises(ValueError, match="'beta' does not hold the subjects"):
        permutation_familywise_separation_auc(
            family, CONTRASTS, strata=strata, n_perm=5
        )
    with pytest.raises(ValueError, match="Unknown condition"):
        permutation_familywise_separation_auc(
            _band_family(effect=0.0), [("famous", "scrambeld")], strata=strata, n_perm=5
        )
    with pytest.raises(ValueError, match="one value per trajectory"):
        permutation_familywise_separation_auc(
            _band_family(effect=0.0), CONTRASTS, strata=strata[:-1], n_perm=5
        )
