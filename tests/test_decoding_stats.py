import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import (
    ChanceAssessmentConfig,
    CVConfig,
    DummyClassifierConfig,
    LogisticRegressionConfig,
    StatisticalAssessmentConfig,
)
from coco_pipe.decoding.stats import (
    _accuracy_ci,
    _coord_dict,
    _correct_p_values,
    _empirical_p_values,
    _run_binomial_assessment,
    aggregate_predictions_for_inference,
    apply_multiple_comparison_correction,
    assess_paired_comparison,
    assess_post_hoc_permutation,
    binomial_accuracy_test,
    run_paired_permutation_assessment,
)

# --- Unit Tests for Core Stats Functions ---


def test_aggregate_predictions_regression_mean():
    df = pd.DataFrame(
        {
            "Subject": ["S1", "S1", "S2"],
            "y_true": [10.0, 10.0, 20.0],
            "y_pred": [12.0, 8.0, 22.0],
            "SampleID": [0, 1, 2],
        }
    )
    res = aggregate_predictions_for_inference(
        df,
        metric="mse",
        task="regression",
        unit_of_inference="Subject",
        custom_aggregation="mean",
    )
    assert len(res) == 2
    assert res[res["InferentialUnitID"] == "S1"]["y_pred"].iloc[0] == 10.0


def test_aggregate_predictions_custom_unit():
    df = pd.DataFrame(
        {
            "Session": ["A", "A", "B"],
            "y_true": [1, 1, 0],
            "y_pred": [1, 0, 0],
            "y_proba_0": [0.2, 0.8, 0.9],
            "y_proba_1": [0.8, 0.2, 0.1],
            "SampleID": [0, 1, 2],
        }
    )
    res = aggregate_predictions_for_inference(
        df,
        metric="accuracy",
        task="classification",
        unit_of_inference="custom",
        custom_unit_column="Session",
        custom_aggregation="mean",
    )
    assert len(res) == 2
    assert "InferentialUnitID" in res.columns
    assert res[res["InferentialUnitID"] == "A"]["y_pred"].iloc[0] == 0


def test_aggregate_predictions_empty():
    df = pd.DataFrame()
    res = aggregate_predictions_for_inference(df, metric="accuracy")
    assert res.empty


def test_binomial_accuracy_test_clopper():
    y_true = [1, 1, 1, 0]
    y_pred = [1, 1, 0, 0]  # 3/4 correct
    res = binomial_accuracy_test(y_true, y_pred, p0=0.5, ci_method="clopper_pearson")
    assert res["observed"] == 0.75
    assert "ci_lower" in res
    assert "ci_upper" in res


def test_binomial_accuracy_test_errors():
    with pytest.raises(ValueError, match="requires an explicit p0"):
        binomial_accuracy_test([1], [1], p0=None)
    with pytest.raises(ValueError, match="zero predictions"):
        binomial_accuracy_test([], [], p0=0.5)


def test_empirical_p_values_two_sided():
    observed = np.array([0.8, 0.2])
    null = np.array([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]])
    p_vals = _empirical_p_values(observed, null, greater_is_better=True, two_sided=True)
    assert len(p_vals) == 2
    assert np.all(p_vals <= 1.0)


def test_correct_p_values_all_methods():
    observed = np.array([0.9, 0.1])
    null = np.array([[0.5, 0.5], [0.8, 0.8]])
    p_vals_raw = np.array([0.1, 0.1])

    # Bonferroni
    p_bonf = _correct_p_values(
        observed, null, p_vals_raw, method="bonferroni", greater_is_better=True
    )
    assert p_bonf[0] == 0.2

    # Max-Stat
    p_max = _correct_p_values(
        observed, null, p_vals_raw, method="max_stat", greater_is_better=True
    )
    assert p_max[0] == pytest.approx(1 / 3)

    # FDR
    p_fdr = _correct_p_values(
        observed, null, p_vals_raw, method="fdr_bh", greater_is_better=True
    )
    assert len(p_fdr) == 2


def test_accuracy_ci_boundary():
    # k=0
    low, high = _accuracy_ci(np.array([0]), np.array([10]), 0.05, "clopper_pearson")
    assert low[0] == 0.0
    # k=n
    low, high = _accuracy_ci(np.array([10]), np.array([10]), 0.05, "clopper_pearson")
    assert high[0] == 1.0


def test_coord_dict_temporal():
    res = _coord_dict((10.0, 20.0), ["TrainTime", "TestTime"])
    assert res["TrainTime"] == 10.0
    assert res["TestTime"] == 20.0
    assert res["Time"] is None


def test_apply_multiple_comparison_correction_short():
    df = pd.DataFrame({"PValue": [0.01]})
    res = apply_multiple_comparison_correction(df)
    assert "Significant" in res.columns
    assert res["Significant"].iloc[0]


# --- Integration Tests for Statistical Assessment ---


def test_run_binomial_assessment_temporal():
    predictions = pd.DataFrame(
        {
            "SampleID": [0, 1, 0, 1],
            "Time": [0.0, 0.0, 1.0, 1.0],
            "y_true": [1, 0, 1, 0],
            "y_pred": [1, 0, 0, 1],  # Time 0: 100%, Time 1: 0%
        }
    )
    # Use proper ChanceAssessmentConfig
    chance_cfg = ChanceAssessmentConfig(p0=0.5)
    config = StatisticalAssessmentConfig(chance=chance_cfg)
    rows = _run_binomial_assessment(
        model="LR",
        metric="accuracy",
        predictions=predictions,
        task="classification",
        config=config,
        unit="sample",
    )
    assert len(rows) == 2  # One per timepoint
    assert rows[0]["Time"] == 0.0
    assert rows[1]["Time"] == 1.0
    assert rows[0]["Observed"] == 1.0
    assert rows[1]["Observed"] == 0.0


def test_assess_paired_comparison_temporal():
    df = pd.DataFrame(
        {
            "SampleID": [0, 1, 0, 1],
            "Time": [0.0, 0.0, 1.0, 1.0],
            "y_true": [1, 0, 1, 0],
            "y_pred_A": [1, 0, 1, 0],
            "y_pred_B": [0, 1, 0, 1],
        }
    )
    res = assess_paired_comparison(
        df, metric="accuracy", unit="sample", n_permutations=10
    )
    assert len(res) == 2  # One per timepoint
    assert "Difference" in res.columns
    assert res.iloc[0]["Difference"] == 1.0


def test_correct_p_values_less_is_better():
    observed = np.array([0.1, 0.2])  # less-is-better
    null = np.array([[0.5, 0.5], [0.1, 0.1]])
    p_vals_raw = np.array([0.1, 0.1])

    # Max-Stat with greater_is_better=False
    p_max = _correct_p_values(
        observed, null, p_vals_raw, method="max_stat", greater_is_better=False
    )
    assert p_max[0] == pytest.approx(2 / 3)


def test_assess_post_hoc_permutation_with_groups():
    res = {
        "predictions": [
            {
                "y_true": np.array([0, 0, 1, 1]),
                "y_pred": np.array([0, 1, 1, 0]),
                "group": np.array([0, 0, 1, 1]),
                "sample_index": np.array([0, 1, 2, 3]),
            }
        ]
    }
    df = assess_post_hoc_permutation(
        res, metric="accuracy", unit="group", n_permutations=10
    )
    assert "Observed" in df.columns
    assert df["Observed"].iloc[0] == 0.5


def test_run_paired_permutation_assessment_full():
    from unittest.mock import MagicMock

    res_a = MagicMock()
    res_b = MagicMock()

    df_a = pd.DataFrame(
        {
            "Model": ["m"],
            "Fold": [0],
            "SampleID": [0],
            "Group": [0],
            "y_true": [1],
            "y_pred": [1],
            "InferentialUnitID": [0],
        }
    )
    res_a.get_predictions.return_value = df_a
    res_b.get_predictions.return_value = df_a.copy()

    config = StatisticalAssessmentConfig(
        chance=ChanceAssessmentConfig(n_permutations=10), unit_of_inference="sample"
    )
    res = run_paired_permutation_assessment(res_a, res_b, "m", "accuracy", config)
    assert not res.empty
    assert res.iloc[0]["Observed"] == 0.0


def test_aggregate_predictions_error_paths():
    df = pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 0]})

    # Missing unit column
    with pytest.raises(ValueError, match="Inference unit 'Subject' not found"):
        aggregate_predictions_for_inference(df, "accuracy", unit_of_inference="Subject")

    # Majority aggregation for regression
    df_reg = pd.DataFrame(
        {"y_true": [1.0, 2.0], "y_pred": [1.1, 1.9], "Subject": ["S1", "S1"]}
    )
    with pytest.raises(
        ValueError, match="majority aggregation is only valid for classification"
    ):
        aggregate_predictions_for_inference(
            df_reg,
            "neg_mean_squared_error",
            task="regression",
            unit_of_inference="Subject",
            custom_aggregation="majority",
        )

    # Mean aggregation without probas
    df_class = pd.DataFrame(
        {"y_true": [0, 1], "y_pred": [0, 0], "Subject": ["S1", "S1"]}
    )
    with pytest.raises(
        ValueError, match="mean aggregation requires probability columns"
    ):
        aggregate_predictions_for_inference(
            df_class,
            "accuracy",
            task="classification",
            unit_of_inference="Subject",
            custom_aggregation="mean",
        )


def test_binomial_test_hardening():
    # k_alpha adjustment path (line 236)
    res = binomial_accuracy_test([0] * 100, [0] * 100, p0=0.5, alpha=0.05)
    assert "chance_threshold" in res


def test_run_paired_permutation_assessment_hardening():
    # Create two results to compare
    X, y = make_classification(n_samples=20, random_state=42)
    config = ExperimentConfig(
        task="classification",
        models={
            "lr": LogisticRegressionConfig(),
            "dummy": DummyClassifierConfig(),
        },
        cv=CVConfig(n_splits=2),
    )
    res = Experiment(config).run(X, y)

    # Note: run_paired_permutation_assessment expects ExperimentResult objects
    # But it is often called via result.compare_models
    paired = res.compare_models(n_permutations=5, random_state=42)
    assert not paired.empty
