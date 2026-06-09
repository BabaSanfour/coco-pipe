import numpy as np
import pandas as pd
import pytest

from coco_pipe.descriptors.qc import (
    KNOWN_FAMILY_TOKENS,
    aggregate_family_qc,
    classify_descriptor_columns,
    compute_family_constant_summary,
    compute_family_missingness,
)


def test_classify_sensor_columns():
    names = [
        "band_abs_alpha_ch-Fz",
        "param_aperiodic_slope_ch-Cz",
    ]

    result = classify_descriptor_columns(names)

    assert result.loc[0, "family"] == "band"
    assert result.loc[0, "scope"] == "sensor"
    assert result.loc[0, "channel"] == "Fz"
    assert result.loc[0, "measure"] == "abs_alpha"
    assert result.loc[1, "family"] == "param"
    assert result.loc[1, "scope"] == "sensor"
    assert result.loc[1, "channel"] == "Cz"
    assert result.loc[1, "measure"] == "aperiodic_slope"


def test_classify_sensor_group_columns():
    result = classify_descriptor_columns(
        ["band_abs_alpha_chgrp-frontal"],
    )

    assert result.loc[0, "family"] == "band"
    assert result.loc[0, "scope"] == "sensor_group"
    assert result.loc[0, "channel"] == "frontal"
    assert result.loc[0, "measure"] == "abs_alpha"


def test_classify_cross_channel_uses_rightmost_scope():
    result = classify_descriptor_columns(
        ["band_cross_spectrum_ch-Fz_ch-Pz"],
    )
    row = result.iloc[0]

    assert row["family"] == "band"
    assert row["scope"] == "sensor"
    assert row["channel"] == "Pz"
    assert row["measure"] == "cross_spectrum_ch-Fz"


@pytest.mark.parametrize(
    ("column", "expected_scope", "expected_channel"),
    [
        ("band_abs_alpha_ch-_reference", "sensor", "_reference"),
        (
            "band_abs_alpha_chgrp-_frontal",
            "sensor_group",
            "_frontal",
        ),
    ],
)
def test_classify_scope_label_may_start_with_underscore(
    column,
    expected_scope,
    expected_channel,
):
    row = classify_descriptor_columns([column]).iloc[0]

    assert row["family"] == "band"
    assert row["measure"] == "abs_alpha"
    assert row["scope"] == expected_scope
    assert row["channel"] == expected_channel


def test_classify_no_scope_token():
    result = classify_descriptor_columns(["band_global_rms"])
    row = result.iloc[0]

    assert row["family"] == "band"
    assert row["scope"] == ""
    assert row["channel"] == ""
    assert row["measure"] == "global_rms"


def test_classify_unknown_family_is_none():
    result = classify_descriptor_columns(
        ["subject", "session", "condition"],
    )

    assert result["family"].isna().all()
    assert result["scope"].tolist() == ["", "", ""]
    assert result["measure"].tolist() == ["subject", "session", "condition"]


def test_classify_mixed_families():
    names = [
        "band_abs_alpha_ch-Fz",
        "param_aperiodic_slope_ch-Cz",
        "complexity_sample_entropy_ch-Pz",
        "subject",
    ]

    result = classify_descriptor_columns(names)

    assert result["family"].tolist() == [
        "band",
        "param",
        "complexity",
        None,
    ]
    assert KNOWN_FAMILY_TOKENS == ("band", "param", "complexity")


def test_classify_aggregated_descriptor_column():
    row = classify_descriptor_columns(["mean_band_log_abs_alpha_chgrp-frontal"]).iloc[0]

    assert row["family"] == "band"
    assert row["measure"] == "mean_log_abs_alpha"
    assert row["scope"] == "sensor_group"
    assert row["channel"] == "frontal"


def test_classify_empty_list():
    result = classify_descriptor_columns([])

    assert result.empty
    assert result.columns.tolist() == [
        "column",
        "family",
        "scope",
        "channel",
        "measure",
    ]


def test_classify_returns_independent_copies():
    names = ["band_abs_alpha_ch-Fz"]

    first = classify_descriptor_columns(names)
    second = classify_descriptor_columns(names)

    assert first is not second
    pd.testing.assert_frame_equal(first, second)


def test_classify_mutation_does_not_corrupt_cache():
    names = ["band_abs_alpha_ch-Fz"]
    first = classify_descriptor_columns(names)
    first.loc[0, "family"] = "CORRUPTED"

    second = classify_descriptor_columns(names)

    assert second.loc[0, "family"] == "band"


def test_family_missingness_rates():
    df = pd.DataFrame(
        {
            "band_abs_alpha_ch-Fz": [1.0, float("nan"), 3.0, 4.0],
            "param_aperiodic_slope_ch-Fz": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = compute_family_missingness(df, list(df.columns))
    band_row = result[result["family"] == "band"].iloc[0]
    param_row = result[result["family"] == "param"].iloc[0]

    assert band_row["missing_rate"] == pytest.approx(0.25)
    assert band_row["measure"] == "abs_alpha"
    assert param_row["missing_rate"] == pytest.approx(0.0)


def test_family_missingness_empty_descriptor_names():
    result = compute_family_missingness(
        pd.DataFrame({"subject": ["s1"]}),
        [],
    )

    assert result.empty
    assert result.columns.tolist() == [
        "column",
        "missing_count",
        "missing_rate",
        "nonfinite_count",
        "nonfinite_rate",
        "family",
        "scope",
        "channel",
        "measure",
    ]


def test_family_constant_summary_metadata():
    df = pd.DataFrame(
        {
            "complexity_sample_entropy_ch-Pz": [1.0, 1.0, 1.0],
        }
    )

    result = compute_family_constant_summary(df, list(df.columns))
    row = result.iloc[0]

    assert bool(row["is_constant"])
    assert row["family"] == "complexity"
    assert row["scope"] == "sensor"
    assert row["channel"] == "Pz"
    assert row["measure"] == "sample_entropy"


def test_family_constant_summary_all_nan_is_not_constant():
    df = pd.DataFrame(
        {
            "band_abs_alpha_ch-Fz": [np.nan, np.nan],
        }
    )

    result = compute_family_constant_summary(df, list(df.columns))
    row = result.iloc[0]

    assert bool(row["is_all_nan"])
    assert not bool(row["is_constant"])


def test_family_constant_summary_empty_descriptor_names():
    result = compute_family_constant_summary(
        pd.DataFrame({"subject": ["s1"]}),
        [],
    )

    assert result.empty
    assert result.columns.tolist() == [
        "column",
        "std",
        "is_all_nan",
        "is_constant",
        "family",
        "scope",
        "channel",
        "measure",
    ]


def test_aggregate_family_qc_counts():
    df = pd.DataFrame(
        {
            "band_abs_alpha_ch-Fz": [1.0, float("nan"), 3.0],
            "band_abs_beta_ch-Fz": [1.0, 1.0, 1.0],
            "param_aperiodic_slope_ch-Fz": [1.0, 2.0, 3.0],
        }
    )

    result = aggregate_family_qc(df, list(df.columns))
    band = result[result["family"] == "band"].iloc[0]
    param = result[result["family"] == "param"].iloc[0]

    assert band["n_features"] == 2
    assert band["n_constant_features"] == 1
    assert band["n_all_nan_features"] == 0
    assert band["missing_rate_mean"] == pytest.approx(1 / 6)
    assert band["missing_rate_max"] == pytest.approx(1 / 3)
    assert param["n_features"] == 1
    assert param["n_constant_features"] == 0


def test_aggregate_family_qc_failure_rate():
    df = pd.DataFrame(
        {
            "band_abs_alpha_ch-Fz": [1.0, 2.0, 3.0],
        }
    )
    failures = pd.DataFrame({"family": ["band", "band"]})

    result = aggregate_family_qc(
        df,
        list(df.columns),
        failures_df=failures,
    )
    band = result[result["family"] == "band"].iloc[0]

    assert band["failure_count"] == 2
    assert band["failure_rate"] == pytest.approx(2 / 3)


def test_aggregate_family_qc_normalizes_extractor_failure_names():
    df = pd.DataFrame(
        {
            "band_abs_alpha_ch-Fz": [1.0, 2.0],
            "param_aperiodic_slope_ch-Fz": [1.0, 2.0],
            "complexity_sample_entropy_ch-Fz": [1.0, 2.0],
        }
    )
    failures = pd.DataFrame(
        {
            "family": ["bands", "parametric", "complexity"],
        }
    )

    result = aggregate_family_qc(
        df,
        list(df.columns),
        failures_df=failures,
    ).set_index("family")

    assert result.loc["band", "failure_count"] == 1
    assert result.loc["param", "failure_count"] == 1
    assert result.loc["complexity", "failure_count"] == 1


def test_aggregate_family_qc_failure_rate_nan_when_df_empty():
    df = pd.DataFrame(
        {
            "band_abs_alpha_ch-Fz": pd.Series([], dtype=float),
        }
    )
    failures = pd.DataFrame({"family": ["band"]})

    result = aggregate_family_qc(
        df,
        list(df.columns),
        failures_df=failures,
    )
    band = result[result["family"] == "band"].iloc[0]

    assert band["failure_count"] == 1
    assert np.isnan(band["failure_rate"])


def test_aggregate_family_qc_empty_descriptors():
    df = pd.DataFrame({"subject": ["s1", "s2"]})

    result = aggregate_family_qc(df, [])

    assert result.empty
    assert result.columns.tolist() == [
        "family",
        "n_features",
        "missing_rate_mean",
        "missing_rate_max",
        "nonfinite_rate_mean",
        "n_all_nan_features",
        "n_constant_features",
        "failure_count",
        "failure_rate",
    ]


def test_family_missingness_inf_not_nan():
    df = pd.DataFrame(
        {
            "band_abs_alpha_ch-Fz": [1.0, float("inf"), 3.0, 4.0],
        }
    )

    result = compute_family_missingness(df, list(df.columns))
    row = result.iloc[0]

    assert row["missing_rate"] == pytest.approx(0.0)
    assert row["nonfinite_rate"] == pytest.approx(0.25)


def test_aggregate_unknown_family_columns_excluded():
    df = pd.DataFrame(
        {
            "band_abs_alpha_ch-Fz": [1.0, 2.0],
            "subject": ["s1", "s2"],
        }
    )

    result = aggregate_family_qc(
        df,
        ["band_abs_alpha_ch-Fz", "subject"],
    )

    assert result["family"].tolist() == ["band"]
