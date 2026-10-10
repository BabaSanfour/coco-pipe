import warnings

import numpy as np
import pandas as pd
import pytest

from coco_pipe.decoding import Experiment, ExperimentConfig, paired_prediction_test
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
