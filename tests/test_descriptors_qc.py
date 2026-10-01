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
from coco_pipe.io.structures import DataContainer


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
        "subfamily",
        "descriptor",
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


def test_classify_descriptor_columns_passthrough():
    schema = pd.DataFrame(
        {
            "column": ["f1", "f0"],
            "family": ["connectivity", "band"],
            "scope": ["sensor", "sensor"],
            "channel": ["Pz", "Fz"],
            "measure": ["coherence", "alpha"],
        }
    )

    result = classify_descriptor_columns(["f0", "f1"], feature_schema=schema)

    assert result["column"].tolist() == ["f0", "f1"]
    # Schema-provided columns pass through verbatim (names are opaque, so a
    # value here proves parsing was bypassed for those fields).
    assert result["family"].tolist() == ["band", "connectivity"]
    assert result["measure"].tolist() == ["alpha", "coherence"]
    # Absent columns (subfamily/descriptor) are enriched from the resolved
    # family/measure rather than left null.
    assert result["subfamily"].notna().all()
    assert result["descriptor"].tolist() == ["alpha", "coherence"]
    assert result.columns.tolist() == [
        "column",
        "family",
        "scope",
        "channel",
        "measure",
        "subfamily",
        "descriptor",
    ]


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


def test_aggregate_family_qc_uses_feature_schema():
    df = pd.DataFrame(
        {
            "f0": [1.0, float("nan"), 3.0],
            "f1": [1.0, 1.0, 1.0],
        }
    )
    schema = pd.DataFrame(
        {
            "column": ["f0", "f1"],
            "family": ["band", "band"],
            "scope": ["sensor", "sensor"],
            "channel": ["Fz", "Fz"],
            "measure": ["alpha", "beta"],
        }
    )

    result = aggregate_family_qc(df, list(df.columns), feature_schema=schema)
    band = result[result["family"] == "band"].iloc[0]

    assert band["n_features"] == 2
    assert band["n_constant_features"] == 1


def test_aggregate_family_qc_schema_supports_unknown_families():
    df = pd.DataFrame(
        {
            "f0": [1.0, 2.0, 3.0],
            "f1": [1.0, 1.0, 2.0],
        }
    )
    schema = pd.DataFrame(
        {
            "column": ["f0", "f1"],
            "family": ["connectivity", "band"],
            "measure": ["coherence", "alpha"],
        }
    )

    result = aggregate_family_qc(df, list(df.columns), feature_schema=schema)

    assert result["family"].tolist() == ["band", "connectivity"]


def test_flattened_axis_metadata_feeds_family_qc():
    container = DataContainer(
        X=np.asarray(
            [
                [1.0, 1.0],
                [np.nan, 1.0],
                [3.0, 1.0],
            ]
        ).reshape(3, 1, 2),
        dims=("obs", "sensor", "feature"),
        coords={
            "sensor": ["Fz"],
            "feature": ["m0", "m1"],
            "feature_family": ["connectivity", "band"],
        },
    )
    flat = container.flatten(preserve="obs")
    schema = flat.feature_schema()
    df = pd.DataFrame(flat.X, columns=flat.coords["feature"])

    result = aggregate_family_qc(df, list(df.columns), feature_schema=schema)

    assert schema is not None
    assert flat.coords["feature"] == ["Fz_m0", "Fz_m1"]
    assert schema["family"].tolist() == ["connectivity", "band"]
    assert result["family"].tolist() == ["band", "connectivity"]
    assert result.set_index("family").loc["band", "n_constant_features"] == 1
    assert result.set_index("family").loc["connectivity", "missing_rate_max"] == 1 / 3


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


def test_classify_descriptor_columns_remainder_parsing():
    result = classify_descriptor_columns(["CZ_band_abs_alpha"])
    assert result.loc[0, "family"] == "band"
    assert result.loc[0, "measure"] == "CZ_abs_alpha"
    assert result.loc[0, "scope"] == ""
    assert result.loc[0, "channel"] == ""


def test_summarize_failures_empty_and_normal():
    from coco_pipe.descriptors.qc import summarize_failures

    # 1. Empty DataFrame
    res_empty = summarize_failures(pd.DataFrame())
    assert res_empty["by_family"].empty
    assert res_empty["by_channel"].empty
    assert res_empty["by_exception_type"].empty
    assert res_empty["by_condition"].empty
    assert res_empty["by_family_channel"].empty
    assert res_empty["combined"].empty

    # 2. DataFrame with missing columns
    res_missing_cols = summarize_failures(pd.DataFrame({"dummy": [1]}))
    assert res_missing_cols["by_family"].empty
    assert res_missing_cols["combined"].empty

    # 3. Normal DataFrame
    fail_df = pd.DataFrame(
        {
            "family": ["band", None, "param"],
            "channel_name": ["Fz", "Cz", None],
            "exception_type": ["ValueError", "TypeError", "ValueError"],
            "condition": ["rest", "task", "rest"],
        }
    )
    res = summarize_failures(fail_df)
    assert not res["by_family"].empty
    assert not res["by_channel"].empty
    assert not res["by_exception_type"].empty
    assert not res["by_condition"].empty
    assert not res["by_family_channel"].empty
    assert not res["combined"].empty

    # Verify "combined" columns
    assert "group" in res["combined"].columns
    assert set(res["combined"]["group"].unique()) == {
        "family",
        "channel",
        "exception_type",
        "condition",
    }


def test_add_family_diagnostics():
    from coco_pipe.descriptors.qc import add_family_diagnostics

    # 1. Empty input
    empty_df = pd.DataFrame()
    res_empty = add_family_diagnostics(empty_df, pd.DataFrame(), pd.DataFrame())
    assert res_empty.empty

    # 2. Band, Param, Complexity families
    family_summary = pd.DataFrame(
        [
            {"family": "band", "missing_rate_max": 0.1, "nonfinite_rate": 0.05},
            {"family": "param", "missing_rate_max": 0.2, "nonfinite_rate": 0.0},
            {"family": "complexity", "missing_rate_max": 0.3, "nonfinite_rate": 0.1},
        ]
    )

    feature_missingness = pd.DataFrame(
        [
            {"column": "band_abs_alpha", "family": "band", "missing_rate": 0.1},
            {"column": "band_rel_beta", "family": "band", "missing_rate": 0.0},
            {"column": "band_corr_rel_theta", "family": "band", "missing_rate": 0.0},
            {"column": "ratio_gamma", "family": "band", "missing_rate": 0.2},
            {"column": "param_r_squared_1", "family": "param", "missing_rate": 0.2},
            {"column": "param_fit_error_1", "family": "param", "missing_rate": 0.2},
            {"column": "peak_freq", "family": "param", "missing_rate": 0.2},
            {"column": "alpha_peak_freq", "family": "param", "missing_rate": 0.2},
            {
                "column": "complexity_entropy",
                "family": "complexity",
                "missing_rate": 0.3,
            },
        ]
    )

    feature_df = pd.DataFrame(
        {
            "band_abs_alpha": [-0.5, 1.0, 2.0],
            "band_rel_beta": [-0.1, 0.5, 1.2],
            "band_corr_rel_theta": [0.1, 0.9, 1.5],
            "ratio_gamma": [1.0, np.nan, 3.0],
            "param_r_squared_1": [0.8, 0.9, 0.95],
            "param_fit_error_1": [0.01, 0.05, 0.1],
            "peak_freq": [10.0, np.nan, 12.0],
            "alpha_peak_freq": [9.0, np.nan, 10.0],
            "complexity_entropy": [1.2, 1.5, 1.8],
        }
    )

    res = add_family_diagnostics(family_summary, feature_missingness, feature_df)
    assert "band_abs_negative_rate" in res.columns
    assert "param_r_squared_median" in res.columns
    assert "complexity_measure_missingness_max" in res.columns


def test_descriptor_subfamily_derivation():
    from coco_pipe.descriptors.qc import descriptor_subfamily

    # band: output type, robust to stat prefix and band suffix
    assert descriptor_subfamily("band", "log_abs_alpha") == "log_abs"
    assert descriptor_subfamily("band", "median_log_abs_alpha") == "log_abs"
    assert descriptor_subfamily("band", "iqr_corr_rel_beta") == "corr_rel"
    assert descriptor_subfamily("band", "corr_log_abs_gamma") == "corr_log_abs"
    assert descriptor_subfamily("band", "abs_delta") == "abs"
    assert descriptor_subfamily("band", "agg_ratio_theta_beta") == "ratio"
    assert descriptor_subfamily("band", "agg_corr_ratio_theta_beta") == "corr_ratio"
    # param: aperiodic / peaks / fit_quality
    assert descriptor_subfamily("param", "offset") == "aperiodic"
    assert descriptor_subfamily("param", "median_alpha_peak_freq") == "peaks"
    assert descriptor_subfamily("param", "r_squared") == "fit_quality"
    # complexity: curated 3-way map
    assert descriptor_subfamily("complexity", "sample_entropy") == "entropy"
    assert descriptor_subfamily("complexity", "higuchi_fd") == "fractal_complexity"
    assert descriptor_subfamily("complexity", "hjorth_mobility") == "signal_dynamics"
    # unknowns fall back
    assert descriptor_subfamily(None, "x") == "unknown"
    assert descriptor_subfamily("complexity", "made_up") == "complexity_other"
    assert descriptor_subfamily("band", "made_up") == "band_other"
    assert descriptor_subfamily("param", "made_up") == "param_other"
    # an entirely unknown family echoes back as its own label
    assert descriptor_subfamily("custom", "anything") == "custom"


def test_select_viable_feature_columns_rejects_bad_missing_rate():
    from coco_pipe.descriptors.qc import select_viable_feature_columns

    frame = pd.DataFrame({"band_a_ch-Fz": [1.0, 2.0]})
    with pytest.raises(ValueError, match="max_missing_rate"):
        select_viable_feature_columns(frame, ["band_a_ch-Fz"], max_missing_rate=1.5)


def test_classify_adds_subfamily_column():
    result = classify_descriptor_columns(
        ["band_log_abs_alpha_ch-Fz", "complexity_sample_entropy_ch-Fz"]
    )
    assert result["subfamily"].tolist() == ["log_abs", "entropy"]


def test_select_viable_feature_columns_row_budget():
    from coco_pipe.descriptors.qc import select_viable_feature_columns

    df = pd.DataFrame(
        {
            "band_log_abs_alpha_ch-Fz": [1.0, 2.0, 3.0, np.nan, 5.0],  # 20% missing
            "band_log_abs_beta_ch-Fz": [5.0, 4.0, 3.0, 2.0, 1.0],  # clean
        }
    )
    cols = list(df.columns)
    # Without a budget the 20%-missing column survives (not over threshold)...
    surviving, _ = select_viable_feature_columns(df, cols)
    assert set(surviving) == set(cols)
    # ...with budget 0 it is dropped (worst-NaN first) to preserve all rows.
    surviving, drop_log = select_viable_feature_columns(df, cols, max_row_drop_rate=0.0)
    assert surviving == ["band_log_abs_beta_ch-Fz"]
    assert "row_preserving" in set(drop_log["drop_reason"])
