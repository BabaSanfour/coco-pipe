import warnings

import numpy as np
import pandas as pd
import pytest

from coco_pipe.decoding import (
    Experiment,
    ExperimentConfig,
    paired_prediction_test,
    permute_labels_within_strata,
)
from coco_pipe.decoding.configs import (
    ChanceAssessmentConfig,
    ClassicalModelConfig,
    CVConfig,
    DummyClassifierConfig,
    LogisticRegressionConfig,
    StatisticalAssessmentConfig,
    TemporalDecoderConfig,
)
from coco_pipe.decoding.stats import (
    aggregate_predictions_for_inference,
    apply_multiple_comparison_correction,
    assess_paired_comparison,
    assess_post_hoc_permutation,
    benjamini_hochberg,
    binomial_accuracy_test,
    correct_sweep_pvalues,
    run_paired_permutation_assessment,
)

GROUPS = np.repeat(np.arange(8), 3)


def _grouped_data():
    y = GROUPS % 2
    X = np.random.default_rng(0).normal(size=(24, 3))
    X[:, 0] += y
    return X, y


def _assessment(unit=None, column=None, **chance):
    return StatisticalAssessmentConfig(
        enabled=True,
        metrics=["accuracy"],
        unit_of_inference=unit,
        custom_unit_column=column,
        chance=ChanceAssessmentConfig(**chance),
    )


def _experiment(models, assessment=None, cv=None):
    return Experiment(
        ExperimentConfig(
            task="classification",
            models=models,
            metrics=["accuracy"],
            cv=cv or CVConfig(strategy="stratified_group_kfold", n_splits=2),
            statistical_assessment=assessment or StatisticalAssessmentConfig(),
            n_jobs=1,
            verbose=False,
        )
    )


def _temporal_run(assessment):
    rng = np.random.default_rng(42)
    y = np.array([0, 1] * 8)
    X = rng.normal(scale=0.1, size=(16, 3, 4))
    X[y == 1, 0, :2] += 1.0
    base = ClassicalModelConfig(estimator="LogisticRegression")
    experiment = _experiment(
        {"sl": TemporalDecoderConfig(wrapper="sliding", base=base)},
        assessment,
        cv=CVConfig(strategy="stratified", n_splits=2),
    )
    return experiment.run(X, y, time_axis=np.array([0.0, 0.1, 0.2, 0.3]))


def _accuracy_by_time(result):
    predictions = result.get_predictions()
    hits = predictions["y_true"] == predictions["y_pred"]
    return hits.groupby(predictions["Time"]).mean().to_numpy()


# --- assessment in a run -------------------------------------------------------


def test_permutation_assessment_of_a_temporal_decoder():
    result = _temporal_run(
        _assessment(
            method="permutation",
            n_permutations=9,
            temporal_correction="max_stat",
            store_null_distribution=True,
        )
    )
    rows = result.get_statistical_assessment().sort_values("Time")
    observed = rows["Observed"].to_numpy()
    np.testing.assert_allclose(observed, _accuracy_by_time(result))

    # p-values follow from the stored null: per time, and against the
    # maximum over time for the max-stat correction
    null = np.asarray(result.get_statistical_nulls("sl")["sl"]["accuracy"]["values"])
    n = len(null)
    raw = ((null >= observed).sum(axis=0) + 1) / (n + 1)
    max_stat = ((null.max(axis=1)[:, None] >= observed).sum(axis=0) + 1) / (n + 1)
    np.testing.assert_allclose(rows["PValue"], raw)
    np.testing.assert_allclose(rows["CorrectedPValue"], max_stat)
    assert (rows["CILower"] <= rows["Observed"]).all()
    assert (rows["Observed"] <= rows["CIUpper"]).all()


def test_fdr_correction_and_other_metrics_over_time():
    from scipy.stats import false_discovery_control

    assessment = _assessment(
        method="permutation", n_permutations=9, temporal_correction="fdr_bh"
    )
    assessment.metrics = ["accuracy", "balanced_accuracy"]
    rows = _temporal_run(assessment).get_statistical_assessment()
    accuracy = rows[rows["Metric"] == "accuracy"].sort_values("Time")
    np.testing.assert_allclose(
        accuracy["CorrectedPValue"], false_discovery_control(accuracy["PValue"])
    )
    assert len(rows[rows["Metric"] == "balanced_accuracy"]) == 4


def test_binomial_assessment_of_a_temporal_decoder():
    result = _temporal_run(_assessment(method="binomial", p0=0.5))
    rows = result.get_statistical_assessment().sort_values("Time")
    assert len(rows) == 4
    np.testing.assert_allclose(rows["Observed"], _accuracy_by_time(result))


@pytest.mark.parametrize(
    ("unit", "column", "n_units", "label"),
    [
        (None, None, 24, "sample"),
        ("group_mean", None, 8, "group"),
        ("group_majority", None, 8, "group"),
        ("custom", "site", 4, "custom"),
    ],
)
def test_permutation_assessment_counts_the_chosen_units(unit, column, n_units, label):
    X, y = _grouped_data()
    metadata = pd.DataFrame({"Subject": GROUPS, "Session": "01", "site": GROUPS % 4})
    experiment = _experiment(
        {"lr": LogisticRegressionConfig()},
        _assessment(unit, column, method="permutation", n_permutations=5),
    )
    result = experiment.run(X, y, groups=GROUPS, sample_metadata=metadata)
    (row,) = result.get_statistical_assessment().to_dict("records")
    assert (row["NEff"], row["InferentialUnit"]) == (n_units, label)


def test_permutation_assessment_needs_one_label_per_unit():
    X, y = _grouped_data()
    # each "site" holds subjects of both classes
    metadata = pd.DataFrame({"Subject": GROUPS, "Session": "01", "site": GROUPS // 2})
    experiment = _experiment(
        {"lr": LogisticRegressionConfig()},
        _assessment("custom", "site", method="permutation", n_permutations=3),
    )
    with pytest.raises(ValueError, match="label varies within inference unit"):
        experiment.run(X, y, groups=GROUPS, sample_metadata=metadata)


def test_permutation_on_small_grouped_data_with_one_class_test_folds():
    # 18 subjects, one sample per class; shuffled labels leave some of the
    # four-sample test folds with a single class (issue #21)
    groups = np.repeat(np.arange(18), 2)
    y = np.tile([0, 1], 18)
    X = np.random.default_rng(0).normal(size=(36, 3))
    X[:, 0] += y
    cv = CVConfig(strategy="group_kfold", n_splits=9)
    result = _experiment(
        {"lr": LogisticRegressionConfig()},
        _assessment(method="permutation", n_permutations=10),
        cv=cv,
    ).run(X, y, groups=groups)
    assert 0 < result.get_statistical_assessment()["PValue"].iloc[0] <= 1

    auc = StatisticalAssessmentConfig(
        enabled=True,
        metrics=["roc_auc"],
        chance=ChanceAssessmentConfig(method="permutation", n_permutations=10),
    )
    experiment = Experiment(
        ExperimentConfig(
            task="classification",
            models={"lr": LogisticRegressionConfig()},
            metrics=["roc_auc"],
            cv=cv,
            statistical_assessment=auc,
            n_jobs=1,
            verbose=False,
        )
    )
    with pytest.raises(RuntimeError, match="need both classes"):
        experiment.run(X, y, groups=groups)


def test_permutation_assessment_inputs():
    X, y = _grouped_data()
    # neither groups nor metadata: every permutation reruns without them
    plain = _experiment(
        {"lr": LogisticRegressionConfig()},
        _assessment(method="permutation", n_permutations=3),
        cv=CVConfig(strategy="stratified", n_splits=2),
    )
    assert len(plain.run(X, y).get_statistical_assessment()) == 1

    no_column = _experiment(
        {"lr": LogisticRegressionConfig()},
        _assessment("custom", method="permutation", n_permutations=3),
    )
    with pytest.raises(ValueError, match="requires custom_unit_column"):
        no_column.run(X, y, groups=GROUPS)
    missing = _experiment(
        {"lr": LogisticRegressionConfig()},
        _assessment("custom", "nope", method="permutation", n_permutations=3),
    )
    with pytest.raises(ValueError, match="'nope' not found"):
        missing.run(X, y, groups=GROUPS)


# --- comparing models ------------------------------------------------------------


def test_paired_assessment_pairs_samples_across_runs():
    X, y = _grouped_data()
    runs = [
        _experiment({"m": model}, cv=CVConfig(n_splits=2, random_state=seed)).run(X, y)
        for model, seed in [
            (LogisticRegressionConfig(), 0),
            (DummyClassifierConfig(), 1),
        ]
    ]
    config = _assessment("sample", n_permutations=10)
    (row,) = run_paired_permutation_assessment(*runs, "m", "accuracy", config).to_dict(
        "records"
    )
    accuracy = [
        (run.get_predictions()["y_true"] == run.get_predictions()["y_pred"]).mean()
        for run in runs
    ]
    # different folds, same samples: every sample is paired
    assert row["NEff"] == 24
    assert row["Observed"] == pytest.approx(accuracy[0] - accuracy[1])

    # by group: units are swapped whole; a custom unit needs its column
    groups = _experiment({"m": LogisticRegressionConfig()}).run(X, y, groups=GROUPS)
    (row,) = run_paired_permutation_assessment(
        groups, groups, "m", "accuracy", _assessment("group_mean", n_permutations=5)
    ).to_dict("records")
    assert row["Observed"] == 0.0
    with pytest.raises(ValueError, match="requires custom_unit_column"):
        run_paired_permutation_assessment(
            *runs, "m", "accuracy", _assessment("custom", n_permutations=5)
        )
    with pytest.raises(ValueError, match="not present in the predictions"):
        run_paired_permutation_assessment(
            *runs, "m", "accuracy", _assessment("custom", "site", n_permutations=5)
        )

    both = _experiment(
        {"lr": LogisticRegressionConfig(), "dummy": DummyClassifierConfig()}
    ).run(X, y, groups=GROUPS)
    assert not both.compare_models(n_permutations=5, random_state=42).empty


def _paired_predictions(folds, y_pred, n=8):
    return pd.DataFrame(
        {
            "Model": "m",
            "Fold": folds,
            "SampleID": [f"s{i}" for i in range(n)],
            "Group": [f"g{i // 2}" for i in range(n)],
            "y_true": [0, 1] * (n // 2),
            "y_pred": y_pred,
        }
    )


def test_paired_prediction_test():
    a = _paired_predictions([0, 0, 1, 1, 0, 0, 1, 1], [0, 1] * 4)
    b = _paired_predictions([1, 1, 0, 0, 1, 1, 0, 0], [0] * 8)
    result = paired_prediction_test(
        a, b, metric="accuracy", n_permutations=20, random_state=0
    )
    assert (result.loc[0, "ScoreA"], result.loc[0, "ScoreB"]) == (1.0, 0.5)
    assert result.loc[0, "NUnits"] == 4
    with pytest.raises(ValueError, match="same samples"):
        paired_prediction_test(a, b.iloc[:6], metric="accuracy", n_permutations=5)


def test_assess_paired_comparison():
    temporal = pd.DataFrame(
        {
            "SampleID": [0, 1, 0, 1],
            "Time": [0.0, 0.0, 1.0, 1.0],
            "y_true": [1, 0, 1, 0],
            "y_pred_A": [1, 0, 1, 0],
            "y_pred_B": [0, 1, 0, 1],
        }
    )
    result = assess_paired_comparison(
        temporal, metric="accuracy", unit="sample", n_permutations=10
    )
    assert result["Difference"].tolist() == [1.0, 1.0]

    wide = {
        "SampleID": [0, 1],
        "y_true": [0, 1],
        "Fold": [0, 0],
        "y_pred_A": [0, 1],
        "y_pred_B": [1, 1],
    }
    for c in range(150):
        wide[f"y_proba_{c}_A"] = wide[f"y_proba_{c}_B"] = [0.5, 0.5]
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        result = assess_paired_comparison(
            pd.DataFrame(wide),
            metric="accuracy",
            unit="sample",
            n_permutations=2,
            random_state=0,
        )
    assert (result.loc[0, "ScoreA"], result.loc[0, "ScoreB"]) == (1.0, 0.5)


def test_assess_post_hoc_permutation():
    result = {
        "predictions": [
            {
                "y_true": np.array([0, 0, 1, 1]),
                "y_pred": np.array([0, 1, 1, 0]),
                "group": np.array([0, 0, 1, 1]),
                "sample_index": np.array([0, 1, 2, 3]),
            }
        ]
    }
    frame = assess_post_hoc_permutation(
        result, metric="accuracy", unit="group", n_permutations=10
    )
    assert frame["Observed"].iloc[0] == 0.5


# --- building blocks --------------------------------------------------------------


def test_binomial_accuracy_test():
    result = binomial_accuracy_test(
        [1, 1, 1, 0], [1, 1, 0, 0], p0=0.5, ci_method="clopper_pearson"
    )
    assert result["observed"] == 0.75
    assert result["ci_lower"] < 0.75 < result["ci_upper"]
    none_right = binomial_accuracy_test([1] * 10, [0] * 10, p0=0.5)
    all_right = binomial_accuracy_test(
        [1] * 10, [1] * 10, p0=0.5, ci_method="clopper_pearson"
    )
    assert none_right["ci_lower"] == 0.0
    assert all_right["ci_upper"] == 1.0
    assert "chance_threshold" in binomial_accuracy_test([0] * 100, [0] * 100, p0=0.5)

    with pytest.raises(ValueError, match="requires an explicit p0"):
        binomial_accuracy_test([1], [1], p0=None)
    with pytest.raises(ValueError, match="zero predictions"):
        binomial_accuracy_test([], [], p0=0.5)


def test_aggregate_predictions_for_inference():
    regression = pd.DataFrame(
        {
            "Subject": ["S1", "S1", "S2"],
            "y_true": [10.0, 10.0, 20.0],
            "y_pred": [12.0, 8.0, 22.0],
            "SampleID": [0, 1, 2],
        }
    )
    pooled = aggregate_predictions_for_inference(
        regression, metric="mse", task="regression", unit_of_inference="Subject"
    ).set_index("InferentialUnitID")
    assert pooled.loc["S1", "y_pred"] == 10.0

    sessions = pd.DataFrame(
        {
            "Session": ["A", "A", "B"],
            "y_true": [1, 1, 0],
            "y_pred": [1, 0, 0],
            "y_proba_0": [0.2, 0.8, 0.9],
            "y_proba_1": [0.8, 0.2, 0.1],
            "SampleID": [0, 1, 2],
        }
    )
    mean = aggregate_predictions_for_inference(
        sessions, "accuracy", unit_of_inference="custom", custom_unit_column="Session"
    ).set_index("InferentialUnitID")
    assert mean.loc["A", "y_pred"] == 0  # probabilities are averaged
    majority = aggregate_predictions_for_inference(
        sessions.assign(Session=["A", "A", "B"]),
        "accuracy",
        unit_of_inference="Session",
        custom_aggregation="majority",
    )
    assert len(majority) == 2
    assert aggregate_predictions_for_inference(pd.DataFrame(), "accuracy").empty


def test_aggregate_predictions_for_inference_errors():
    frame = pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 0], "Subject": ["S1", "S1"]})
    with pytest.raises(ValueError, match="Inference unit 'Site' not found"):
        aggregate_predictions_for_inference(frame, "accuracy", unit_of_inference="Site")
    with pytest.raises(ValueError, match="only valid for classification"):
        aggregate_predictions_for_inference(
            frame,
            "neg_mean_squared_error",
            task="regression",
            unit_of_inference="Subject",
            custom_aggregation="majority",
        )
    with pytest.raises(ValueError, match="mean aggregation requires probability"):
        aggregate_predictions_for_inference(
            frame, "accuracy", unit_of_inference="Subject", custom_aggregation="mean"
        )


def test_multiple_comparison_corrections():
    adjusted, rejected = benjamini_hochberg([0.01, 0.04, 0.03, 0.005, np.nan])
    assert rejected[0]
    assert np.isnan(adjusted[-1])
    adjusted, rejected = benjamini_hochberg([np.nan, np.nan])
    assert np.isnan(adjusted).all()
    assert not rejected.any()

    sweep = pd.DataFrame(
        {"target": ["A", "A", "B", "B"], "p_value": [0.01, 0.04, 0.001, 0.5]}
    )
    corrected = correct_sweep_pvalues(sweep, family_columns=["target"])
    assert "p_value_fdr" in corrected.columns
    with pytest.raises(KeyError, match="Missing p-value column"):
        correct_sweep_pvalues(sweep, p_column="missing")

    single = apply_multiple_comparison_correction(pd.DataFrame({"PValue": [0.01]}))
    assert single["Significant"].iloc[0]


# --- permutation schemes ------------------------------------------------------------


def test_units_scheme_matches_the_pinned_null():
    # nulls drawn before permutation schemes existed; the default must not move
    X, y = _grouped_data()
    pinned = {
        None: (24, [12, 13, 13, 8, 11, 9, 12, 10]),
        "group_mean": (8, [5, 6, 5, 2, 5, 5, 3, 4]),
    }
    for unit, (n_units, counts) in pinned.items():
        result = _experiment(
            {"lr": LogisticRegressionConfig()},
            _assessment(unit, n_permutations=8, store_null_distribution=True),
        ).run(X, y, groups=GROUPS)
        null = result.get_statistical_nulls("lr")["lr"]["accuracy"]["values"]
        np.testing.assert_array_equal(np.ravel(null) * n_units, counts)
        meta = result.meta["statistical_assessment"]
        assert meta["permutation_scheme"] == "units"
        assert meta["strata_column"] is None and meta["block_column"] is None


SUBJECTS = np.repeat(np.arange(6), 12)
RUNS = np.tile(np.repeat(np.arange(4), 3), 6)


def _within_subject_data(effect_subjects=(), effect=3.0, seed=0, blocked=False):
    """Six subjects of 12 trials whose label varies inside every subject.

    Unblocked, labels alternate trial by trial and subject 5 is imbalanced;
    blocked, each of the four runs of a subject holds one label.
    """
    rng = np.random.default_rng(seed)
    if blocked:
        y = np.tile(np.repeat([0, 1, 1, 0], 3), 6)
    else:
        y = np.tile([0, 1], 36)
        y[SUBJECTS == 5] = [0] * 8 + [1] * 4
    X = rng.normal(size=(len(y), 3))
    planted = np.isin(SUBJECTS, effect_subjects)
    X[planted, 0] += effect * (2 * y[planted] - 1)
    metadata = pd.DataFrame({"Subject": SUBJECTS, "Session": "01", "Run": RUNS})
    return X, y, metadata


def _scheme_run(X, y, metadata, cv=None, **chance):
    assessment = StatisticalAssessmentConfig(
        enabled=True,
        metrics=["accuracy"],
        chance=ChanceAssessmentConfig(store_null_distribution=True, **chance),
    )
    return _experiment(
        {"lr": LogisticRegressionConfig()},
        assessment,
        cv=cv or CVConfig(strategy="stratified", n_splits=3),
    ).run(X, y, sample_metadata=metadata)


def _null(result):
    return np.ravel(result.get_statistical_nulls("lr")["lr"]["accuracy"]["values"])


def test_within_strata_shuffles_keep_each_strata_class_counts():
    _, y, _ = _within_subject_data()
    moved = 0
    for seed in range(50):
        y_perm = permute_labels_within_strata(y, SUBJECTS, random_state=seed)
        for subject in np.unique(SUBJECTS):
            rows = subject == SUBJECTS
            np.testing.assert_array_equal(
                np.bincount(y_perm[rows], minlength=2), np.bincount(y[rows])
            )
        moved += (y_perm != y).any()
    assert moved == 50
    # text labels and strata work the same way
    labels = np.where(y == 1, "right", "left")
    shuffled = permute_labels_within_strata(labels, SUBJECTS.astype(str))
    assert sorted(shuffled) == sorted(labels)


def test_block_shuffles_move_whole_blocks_inside_their_stratum():
    _, y, _ = _within_subject_data(blocked=True)
    # unequal blocks: the last run of subject 0 is one trial longer
    runs = RUNS.copy()
    runs[8] = 3
    y = y.copy()
    y[8] = y[9]
    moved = 0
    for seed in range(50):
        y_perm = permute_labels_within_strata(
            y, SUBJECTS, blocks=runs, random_state=seed
        )
        for subject in np.unique(SUBJECTS):
            rows = subject == SUBJECTS
            block_labels = {}
            for run in np.unique(runs[rows]):
                block = rows & (runs == run)
                (label,) = np.unique(y_perm[block])  # constant within the block
                block_labels[run] = label
            original = [np.unique(y[rows & (runs == r)])[0] for r in block_labels]
            # the stratum keeps its multiset of block labels: none crossed over
            assert sorted(block_labels.values()) == sorted(original)
        moved += (y_perm != y).any()
    assert moved == 50


def test_permuters_reject_degenerate_and_incomplete_strata():
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="null is degenerate"):
        permute_labels_within_strata(y, ["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="null is degenerate"):
        permute_labels_within_strata(y, ["a", "a", "b", "b"], blocks=[0, 0, 1, 1])
    with pytest.raises(ValueError, match="1 samples have no stratum"):
        permute_labels_within_strata(y, ["a", None, "b", "b"])
    with pytest.raises(ValueError, match="1 samples have no block"):
        permute_labels_within_strata(y, ["a"] * 4, blocks=[0, 0, np.nan, 1])


def test_within_strata_null_is_centred_on_chance_for_random_features():
    X, y, metadata = _within_subject_data()
    result = _scheme_run(
        X,
        y,
        metadata,
        n_permutations=30,
        permutation_scheme="within_strata",
        strata_column="Subject",
    )
    null = _null(result)
    assert len(null) == 30
    assert abs(null.mean() - 0.5) < 0.05
    (row,) = result.get_statistical_assessment().to_dict("records")
    assert abs(row["NullMedian"] - 0.5) < 0.05
    assert (row["InferentialUnit"], row["NEff"]) == ("sample", 72)


def test_within_strata_detects_an_effect_planted_in_some_subjects():
    X, y, metadata = _within_subject_data(effect_subjects=(0, 2, 4))
    result = _scheme_run(
        X,
        y,
        metadata,
        n_permutations=19,
        permutation_scheme="within_strata",
        strata_column="subject",  # matched ignoring case
    )
    (row,) = result.get_statistical_assessment().to_dict("records")
    assert row["Observed"] > 0.7
    assert row["PValue"] == pytest.approx(1 / 20)

    (raw,) = result.raw["lr"]["statistical_assessment"]
    assert "shuffled within each 'subject' stratum" in raw["Assumptions"]
    meta = result.meta["statistical_assessment"]
    assert meta["permutation_scheme"] == "within_strata"
    assert (meta["strata_column"], meta["block_column"]) == ("subject", None)
    assert meta["resolved_unit_of_inference"] == "sample"


def test_block_scheme_detects_an_effect_planted_in_some_subjects():
    X, y, metadata = _within_subject_data(effect_subjects=(0, 1, 2, 3), blocked=True)
    result = _scheme_run(
        X,
        y,
        metadata,
        cv=CVConfig(strategy="leave_one_group_out", group_key="Subject"),
        n_permutations=19,
        permutation_scheme="blocks_within_strata",
        strata_column="Subject",
        block_column="Run",
    )
    (row,) = result.get_statistical_assessment().to_dict("records")
    assert row["Observed"] > 0.8
    assert row["PValue"] == pytest.approx(1 / 20)
    assert len(_null(result)) == 19
    (raw,) = result.raw["lr"]["statistical_assessment"]
    assert "'Run' block labels shuffled within each 'Subject'" in raw["Assumptions"]
    meta = result.meta["statistical_assessment"]
    assert (meta["permutation_scheme"], meta["block_column"]) == (
        "blocks_within_strata",
        "Run",
    )


def test_block_scheme_names_a_block_holding_two_labels():
    X, y, metadata = _within_subject_data(blocked=True)
    y = y.copy()
    y[SUBJECTS == 2] = np.tile([0, 1], 6)  # subject 2 alternates inside runs
    with pytest.raises(ValueError, match="Block 0 of stratum 2 holds more than one"):
        _scheme_run(
            X,
            y,
            metadata,
            n_permutations=3,
            permutation_scheme="blocks_within_strata",
            strata_column="Subject",
            block_column="Run",
        )


def test_schemes_check_their_columns_and_scoring_unit():
    X, y, metadata = _within_subject_data()
    with pytest.raises(ValueError, match="Strata column 'Site' not found"):
        _scheme_run(
            X,
            y,
            metadata,
            n_permutations=3,
            permutation_scheme="within_strata",
            strata_column="Site",
        )
    # a subject-level score takes one label per subject, which a within-subject
    # shuffle does not keep
    assessment = StatisticalAssessmentConfig(
        enabled=True,
        metrics=["accuracy"],
        unit_of_inference="group_mean",
        chance=ChanceAssessmentConfig(
            n_permutations=3,
            permutation_scheme="within_strata",
            strata_column="Subject",
        ),
    )
    experiment = _experiment({"lr": LogisticRegressionConfig()}, assessment)
    with pytest.raises(ValueError, match="spans more than one sample"):
        experiment.run(X, y, groups=SUBJECTS)


def test_strata_can_be_the_cv_groups():
    X, y, _ = _within_subject_data(effect_subjects=(0, 1))
    assessment = StatisticalAssessmentConfig(
        enabled=True,
        metrics=["accuracy"],
        chance=ChanceAssessmentConfig(
            n_permutations=5,
            permutation_scheme="within_strata",
            strata_column="group",
            store_null_distribution=True,
        ),
    )
    result = _experiment(
        {"lr": LogisticRegressionConfig()},
        assessment,
        cv=CVConfig(strategy="leave_one_group_out"),
    ).run(X, y, groups=SUBJECTS)
    assert len(_null(result)) == 5


def test_stratified_null_is_reproducible_from_the_seed():
    X, y, metadata = _within_subject_data()
    chance = {
        "n_permutations": 6,
        "permutation_scheme": "within_strata",
        "strata_column": "Subject",
    }
    first = _null(_scheme_run(X, y, metadata, **chance))
    np.testing.assert_array_equal(first, _null(_scheme_run(X, y, metadata, **chance)))
    np.testing.assert_array_equal(
        permute_labels_within_strata(y, SUBJECTS, random_state=3),
        permute_labels_within_strata(y, SUBJECTS, random_state=3),
    )

    assessment = StatisticalAssessmentConfig(
        enabled=True,
        metrics=["accuracy"],
        random_state=7,
        chance=ChanceAssessmentConfig(store_null_distribution=True, **chance),
    )
    other = _experiment(
        {"lr": LogisticRegressionConfig()},
        assessment,
        cv=CVConfig(strategy="stratified", n_splits=3),
    ).run(X, y, sample_metadata=metadata)
    assert not np.array_equal(first, _null(other))

    # parallel workers draw from the same per-permutation seeds
    parallel = Experiment(
        ExperimentConfig(
            task="classification",
            models={"lr": LogisticRegressionConfig()},
            metrics=["accuracy"],
            cv=CVConfig(strategy="stratified", n_splits=3),
            statistical_assessment=StatisticalAssessmentConfig(
                enabled=True,
                metrics=["accuracy"],
                chance=ChanceAssessmentConfig(store_null_distribution=True, **chance),
            ),
            n_jobs=2,
            verbose=False,
        )
    ).run(X, y, sample_metadata=metadata)
    np.testing.assert_array_equal(first, _null(parallel))


def test_stratified_scheme_over_time_keeps_the_temporal_correction():
    X, y, metadata = _within_subject_data(effect_subjects=(0, 2, 4))
    X = np.repeat(X[:, :, None], 3, axis=2)
    X[:, 0, 2] = np.random.default_rng(1).normal(size=len(y))  # no effect at t=2
    base = ClassicalModelConfig(estimator="LogisticRegression")
    assessment = StatisticalAssessmentConfig(
        enabled=True,
        metrics=["accuracy"],
        chance=ChanceAssessmentConfig(
            n_permutations=9,
            temporal_correction="max_stat",
            permutation_scheme="within_strata",
            strata_column="Subject",
            store_null_distribution=True,
        ),
    )
    result = _experiment(
        {"sl": TemporalDecoderConfig(wrapper="sliding", base=base)},
        assessment,
        cv=CVConfig(strategy="stratified", n_splits=3),
    ).run(X, y, sample_metadata=metadata, time_axis=np.array([0.0, 0.1, 0.2]))
    rows = result.get_statistical_assessment().sort_values("Time")
    null = np.asarray(result.get_statistical_nulls("sl")["sl"]["accuracy"]["values"])
    observed = rows["Observed"].to_numpy()
    max_stat = ((null.max(axis=1)[:, None] >= observed).sum(axis=0) + 1) / 10
    np.testing.assert_allclose(rows["CorrectedPValue"], max_stat)
    assert rows["CorrectedPValue"].iloc[0] == pytest.approx(0.1)


# --- group-level sign-flip test -------------------------------------------------------


def _sign_flip_run(X, y, groups, metric="accuracy", time_axis=None, **chance):
    base = ClassicalModelConfig(estimator="LogisticRegression")
    model = (
        LogisticRegressionConfig()
        if X.ndim == 2
        else TemporalDecoderConfig(wrapper="sliding", base=base)
    )
    assessment = StatisticalAssessmentConfig(
        enabled=True,
        metrics=[metric],
        chance=ChanceAssessmentConfig(
            method="sign_flip", store_null_distribution=True, **chance
        ),
    )
    return Experiment(
        ExperimentConfig(
            task="classification",
            models={"m": model},
            metrics=[metric],
            cv=CVConfig(strategy="leave_one_group_out"),
            statistical_assessment=assessment,
            n_jobs=1,
            verbose=False,
        )
    ).run(X, y, groups=groups, time_axis=time_axis)


def test_sign_flip_flips_per_group_scores_around_chance():
    X, y, _ = _within_subject_data(effect_subjects=(0, 1, 2, 3, 4, 5), effect=1.0)
    X = np.repeat(X[:, :, None], 2, axis=2)
    X[:, 0, 1] = np.random.default_rng(1).normal(size=len(y))
    result = _sign_flip_run(
        X, y, SUBJECTS, n_permutations=200, time_axis=np.array([0.0, 0.1])
    )
    rows = result.get_statistical_assessment().sort_values("Time")
    assert (rows["NullMethod"] == "sign_flip_group").all()
    assert (rows["NEff"] == 6).all() and (rows["InferentialUnit"] == "group").all()

    predictions = result.get_predictions()
    hits = predictions["y_true"] == predictions["y_pred"]
    per_group = hits.groupby([predictions["Group"], predictions["Time"]]).mean()
    scores = per_group.unstack("Time").to_numpy()  # groups x times
    np.testing.assert_allclose(rows["Observed"], scores.mean(axis=0))

    null = np.asarray(result.get_statistical_nulls("m")["m"]["accuracy"]["values"])
    assert null.shape == (200, 2)
    # every draw is chance plus a mean of sign-flipped deviations
    deviations = scores - 0.5
    sign_seed, _ = np.random.SeedSequence(42).spawn(2)
    signs = np.random.default_rng(sign_seed).choice([-1.0, 1.0], size=(200, 6))
    np.testing.assert_allclose(null, 0.5 + signs @ deviations / 6)
    assert abs(null.mean() - 0.5) < 0.02

    # p-values rank one-sample t values, so the max-stat correction compares
    # each observed t with the largest t of every draw (Nichols & Holmes 2002)
    flipped = signs[:, :, None] * deviations[None]
    t_null = flipped.mean(1) / (flipped.std(1, ddof=1) / np.sqrt(6))
    t_obs = deviations.mean(0) / (deviations.std(0, ddof=1) / np.sqrt(6))
    tied = 1e-9 * np.abs(t_obs)  # draws tying with the observed t count
    raw = (t_null >= t_obs - tied).sum(0) + 1
    max_stat = (t_null.max(axis=1)[:, None] >= t_obs - tied).sum(0) + 1
    np.testing.assert_allclose(rows["PValue"], raw / 201)
    np.testing.assert_allclose(rows["CorrectedPValue"], max_stat / 201)
    assert rows["CorrectedPValue"].iloc[0] < 0.05
    assert all(row["P0"] == 0.5 for row in result.raw["m"]["statistical_assessment"])


def test_sign_flip_needs_one_group_per_fold_and_a_chance_level():
    X, y, _ = _within_subject_data()
    with pytest.raises(ValueError, match="one group per test fold"):
        _experiment(
            {"lr": LogisticRegressionConfig()},
            _assessment(method="sign_flip", n_permutations=10),
            cv=CVConfig(strategy="group_kfold", n_splits=3),
        ).run(X, y, groups=SUBJECTS)
    with pytest.raises(ValueError, match="no known chance level for 'f1'"):
        _sign_flip_run(X, y, SUBJECTS, metric="f1", n_permutations=10)
    result = _sign_flip_run(X, y, SUBJECTS, metric="f1", p0=0.5, n_permutations=10)
    (row,) = result.raw["m"]["statistical_assessment"]
    assert row["P0"] == 0.5
