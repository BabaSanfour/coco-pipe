import warnings

import numpy as np
import pandas as pd
import pytest

from coco_pipe.io.quality import (
    CheckResult,
    EpochDropRecord,
    QCResult,
    SubjectDropRecord,
    _container_to_feature_df,
    _filter_observations,
    _get_subject_ids,
    _numeric_values,
    check_constant_columns,
    check_flatline,
    check_missingness,
    check_outliers_zscore,
    compute_row_outlier_scores,
    compute_subject_outlier_burden,
    drop_epoch_outliers,
    drop_subject_outliers,
    group_labels,
    make_qc_flag,
    resolve_qc_status,
    run_qc,
)
from coco_pipe.io.structures import DataContainer
from coco_pipe.io.utils import row_quality_score


def test_check_result_is_issue():
    ok = CheckResult("test", "OK", "msg", 0)
    warn = CheckResult("test", "WARN", "msg", 5)
    fail = CheckResult("test", "FAIL", "msg", 10)
    assert not ok.is_issue
    assert warn.is_issue
    assert fail.is_issue


def test_make_qc_flag_and_resolve_status():
    pass_flag = make_qc_flag("pass", "complete", "All checks passed.")
    warn_flag = make_qc_flag(
        "warn",
        "high_missingness",
        "Missingness exceeded the warning threshold.",
        value=0.25,
        threshold=0.20,
        scope="band",
    )

    assert pass_flag == {
        "level": "pass",
        "code": "complete",
        "message": "All checks passed.",
        "value": None,
        "threshold": None,
        "scope": "",
    }
    assert warn_flag["scope"] == "band"
    assert resolve_qc_status([]) == "pass"
    assert resolve_qc_status([pass_flag, warn_flag]) == "warn"
    assert (
        resolve_qc_status(
            [warn_flag, make_qc_flag("fail", "invalid", "Invalid output.")]
        )
        == "fail"
    )

    with pytest.raises(ValueError, match="level must be"):
        make_qc_flag("invalid", "bad", "bad")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("level", "status", "severity"),
    [
        ("pass", "OK", 0),
        ("warn", "WARN", 5),
        ("fail", "FAIL", 9),
    ],
)
def test_check_result_from_flag_dict(level, status, severity):
    flag = make_qc_flag(
        level,
        "feature_qc",
        "Feature QC result.",
        value=3,
        threshold=2,
        scope="complexity",
    )

    result = CheckResult.from_flag_dict(flag)

    assert result.check_name == "feature_qc"
    assert result.status == status
    assert result.severity == severity
    assert result.metric_name == "complexity"
    assert result.metric_value == 3


def test_row_scores_extreme_row_flagged():
    df = pd.DataFrame(
        {
            "f1": [1.0, 1.1, 0.9, 500.0],
            "f2": [2.0, 2.1, 1.9, 500.0],
        }
    )

    result = compute_row_outlier_scores(
        df,
        ["f1", "f2"],
        z_threshold=3.0,
    )

    assert result.iloc[3]["outlier_fraction"] == pytest.approx(1.0)
    assert (result.iloc[:3]["outlier_fraction"] == 0.0).all()


def test_row_scores_zero_mad_no_deviation():
    result = compute_row_outlier_scores(
        pd.DataFrame({"f1": [5.0, 5.0, 5.0]}),
        ["f1"],
    )

    assert (result["outlier_fraction"] == 0.0).all()


def test_row_scores_zero_mad_with_deviation():
    result = compute_row_outlier_scores(
        pd.DataFrame({"f1": [5.0, 5.0, 5.0, 6.0]}),
        ["f1"],
        z_threshold=3.0,
    )

    assert result.iloc[3]["outlier_fraction"] == pytest.approx(1.0)
    assert result.iloc[3]["mad_z_max"] > 0
    assert (result.iloc[:3]["outlier_fraction"] == 0.0).all()


def test_row_scores_has_stable_schema_and_index():
    df = pd.DataFrame(
        {"f1": [1.0, 2.0], "f2": [1.0, 2.0]},
        index=[10, 20],
    )

    result = compute_row_outlier_scores(df, ["f1", "f2"])

    assert result.columns.tolist() == [
        "outlier_fraction",
        "n_outlier_features",
        "mad_z_max",
    ]
    assert result.index.tolist() == [10, 20]


def test_compute_subject_outlier_burden_uses_featurewise_mad():
    df = pd.DataFrame(
        {
            "subject": [f"s{i}" for i in range(6)],
            "constant_with_outlier": [1.0, 1.0, 1.0, 1.0, 1.0, 100.0],
            "stable": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )

    burden = compute_subject_outlier_burden(
        df,
        ["constant_with_outlier", "stable"],
    )

    assert burden["subject"].tolist() == [f"s{i}" for i in range(6)]
    assert burden["n_outlier_features"].tolist() == [0, 0, 0, 0, 0, 1]
    assert burden["outlier_fraction"].tolist() == [0.0] * 5 + [0.5]
    assert burden["n_epochs"].tolist() == [1] * 6


def test_subject_burden_scores_epochs_before_aggregation():
    df = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "f1": [0.0, 0.0, 0.0, 0.0, -100.0, 100.0],
        }
    )

    burden = compute_subject_outlier_burden(df, ["f1"])
    s3 = burden.loc[burden["subject"] == "s3"].iloc[0]

    assert s3["outlier_fraction"] == pytest.approx(1.0)
    assert s3["n_outlier_features"] == pytest.approx(1.0)
    assert s3["n_epochs"] == 2


def test_compute_subject_outlier_burden_validation_and_empty_features():
    df = pd.DataFrame({"participant": ["s1", "s2"], "value": [1.0, 2.0]})

    burden = compute_subject_outlier_burden(
        df,
        [],
        subject_col="participant",
    )
    assert burden.to_dict(orient="list") == {
        "participant": ["s1", "s2"],
        "outlier_fraction": [0.0, 0.0],
        "n_outlier_features": [0.0, 0.0],
        "n_epochs": [1, 1],
    }

    with pytest.raises(ValueError, match="Subject column"):
        compute_subject_outlier_burden(df, ["value"])
    with pytest.raises(ValueError, match="z_threshold must be positive"):
        compute_subject_outlier_burden(
            df,
            ["value"],
            subject_col="participant",
            z_threshold=0,
        )
    with pytest.raises(ValueError, match="Feature columns not found"):
        compute_subject_outlier_burden(df, ["missing"], subject_col="participant")


def test_compute_subject_outlier_burden_ignores_nonfinite_features():
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3"],
            "all_nan": [np.nan, np.nan, np.nan],
            "partly_infinite": [1.0, np.inf, 1.0],
        }
    )

    burden = compute_subject_outlier_burden(
        df,
        ["all_nan", "partly_infinite"],
    )

    assert burden["n_outlier_features"].tolist() == [0, 0, 0]
    assert burden["outlier_fraction"].tolist() == [0.0, 0.0, 0.0]
    assert burden["n_epochs"].tolist() == [1, 1, 1]


def test_numeric_values_dataframe():
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"], "C": [3.5, np.nan]})
    arr = _numeric_values(df)
    assert arr.shape == (2, 2)
    assert np.isnan(arr[1, 1])


def test_numeric_values_ndarray():
    arr = np.array([1, 2, 3])
    res = _numeric_values(arr)
    assert res.dtype == float
    assert res.tolist() == [1.0, 2.0, 3.0]

    # Mixed object array with strings
    obj_arr = np.array(["a", "b", "c"], dtype=object)
    res_obj = _numeric_values(obj_arr)
    assert res_obj.size == 0


def test_check_missingness_empty():
    res = check_missingness(np.array([]))
    assert res.status == "WARN"
    assert "empty" in res.message


def test_check_missingness_ndarray():
    arr = np.array([1, np.nan, 3, np.nan])
    res = check_missingness(arr, threshold_warn=0.1, threshold_fail=0.4)
    assert res.status == "FAIL"
    assert res.metric_value == 0.5


def test_check_missingness_dataframe():
    df = pd.DataFrame({"A": [1, 2, np.nan], "B": [np.nan, 5, 6]})
    res = check_missingness(df, threshold_warn=0.2, threshold_fail=0.5)
    assert res.status == "WARN"  # 2/6 = 33% > 20%
    assert np.isclose(res.metric_value, 1 / 3)


def test_check_constant_columns_empty():
    assert check_constant_columns(np.array([])) == []
    assert check_constant_columns(pd.DataFrame()) == []


def test_check_constant_columns_ndarray():
    # 2D array
    arr = np.array([[1, 2, 3], [1, 4, 3], [1, 5, 3]])
    res = check_constant_columns(arr)
    assert len(res) == 1
    assert "2 constant features" in res[0].message


def test_check_constant_columns_dataframe():
    df = pd.DataFrame({"A": [1, 1, 1], "B": [1, 2, 3], "C": ["x", "x", "x"]})
    res = check_constant_columns(df)
    assert len(res) == 1
    assert "constant columns: A" in res[0].message


def test_check_outliers_zscore_empty_or_nan():
    assert check_outliers_zscore(np.array([])) is None
    assert check_outliers_zscore(np.array([np.nan, np.nan])) is None


def test_check_outliers_zscore_no_variance():
    assert check_outliers_zscore(np.array([5, 5, 5])) is None


def test_check_outliers_zscore_detection():
    arr = np.array([1.0] * 50 + [100.0])
    res = check_outliers_zscore(arr, sigma=3.0)
    assert res is not None
    assert res.status == "WARN"
    assert res.metric_value > 3.0


def test_check_flatline_empty():
    res = check_flatline(np.array([]))
    assert res.status == "WARN"

    res_non_num = check_flatline(np.array(["a", "b"]))
    assert res_non_num.status == "OK"  # Not numeric, handled gracefully


def test_check_flatline_detection():
    # Standard dev is zero
    arr = np.array([5, 5, 5])
    res = check_flatline(arr, threshold=1e-5)
    assert res.status == "FAIL"
    assert res.metric_value == 0

    arr2 = np.array([1, 2, 3])
    res2 = check_flatline(arr2, threshold=1e-5)
    assert res2.status == "OK"


def test_check_constant_columns_strings_only():
    df = pd.DataFrame({"A": ["x", "y"], "B": ["a", "b"]})
    res = check_constant_columns(df)
    assert res == []

    arr = np.array([["a", "b"], ["c", "d"]], dtype=object)
    res_arr = check_constant_columns(arr)
    assert res_arr == []


def test_check_constant_columns_many():
    df = pd.DataFrame({"A": [1, 1], "B": [2, 2], "C": [3, 3], "D": [4, 4], "E": [5, 6]})
    res = check_constant_columns(df)
    assert len(res) == 1
    assert "..." in res[0].message


def test_check_constant_columns_none():
    df = pd.DataFrame({"A": [1, 2], "B": [2, 3]})
    assert check_constant_columns(df) == []


def test_check_outliers_zscore_all_nan():
    arr = np.array([np.nan, np.nan, np.nan])
    assert check_outliers_zscore(arr) is None


def _make_container(
    X: np.ndarray,
    subject_ids: list[str],
    *,
    obs_ids: list[str] | None = None,
) -> DataContainer:
    n_obs, n_features = X.shape
    return DataContainer(
        X=X,
        dims=("obs", "feature"),
        coords={
            "feature": [f"f{index}" for index in range(n_features)],
            "subject": subject_ids,
            "condition": ["baseline"] * n_obs,
        },
        y=np.arange(n_obs),
        ids=np.asarray(obs_ids or [f"obs-{index}" for index in range(n_obs)]),
        meta={"source": "synthetic"},
    )


def _clean_container(
    n_subjects: int = 10,
    n_features: int = 20,
) -> DataContainer:
    rng = np.random.default_rng(42)
    subjects = [f"sub-{index:02d}" for index in range(n_subjects)]
    return _make_container(
        rng.standard_normal((n_subjects, n_features)),
        subjects,
    )


def _container_with_outlier_subject(
    n_subjects: int = 10,
) -> DataContainer:
    rng = np.random.default_rng(0)
    subjects = [f"sub-{index:02d}" for index in range(n_subjects)]
    values = rng.standard_normal((n_subjects, 20))
    values[-1, :] = 1000.0
    return _make_container(values, subjects)


def test_qc_result_summary_and_retention_rate():
    result = QCResult(
        n_rows_entering_qc=13,
        n_dropped_nan_inf=2,
        n_dropped_extreme=1,
        n_obs_in=10,
        n_obs_out=8,
        n_subjects_in=5,
        n_subjects_out=4,
        epochs_dropped=[
            EpochDropRecord(1, "obs-1", 0.5, 10.0),
        ],
        subjects_dropped=[
            SubjectDropRecord("sub-4", 0.75, 3),
        ],
    )

    assert result.retention_rate == pytest.approx(0.8)
    assert result.n_epochs_dropped == 1
    assert result.n_subjects_dropped == 1
    assert result.total_dropped == 5
    assert set(result.summary()) == {
        "n_rows_entering_qc",
        "n_dropped_nan_inf",
        "n_dropped_extreme",
        "n_obs_in",
        "n_obs_out",
        "n_epochs_dropped",
        "n_subjects_in",
        "n_subjects_out",
        "n_subjects_dropped",
        "retention_rate",
        "epoch_drop_threshold",
        "subject_drop_threshold",
        "subject_outlier_fraction_threshold",
    }


def test_row_scores_by_family_separates_bad_family():
    names = ["band_alpha_ch-Fz", "complexity_entropy_ch-Fz"]
    df = pd.DataFrame(
        {
            names[0]: [0.0, 0.0, 0.0, 100.0],
            names[1]: [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = compute_row_outlier_scores(
        df, names, z_threshold=3.0, descriptor_names=names, group_by="family"
    )

    assert result.loc[3, "outlier_fraction_band"] == 1.0
    assert result.loc[3, "outlier_fraction_complexity"] == 0.0


def test_row_scores_by_family_uses_feature_schema():
    df = pd.DataFrame(
        {
            "f0": [0.0, 0.0, 0.0, 100.0],
            "f1": [1.0, 1.0, 1.0, 1.0],
        }
    )
    schema = pd.DataFrame(
        {
            "column": ["f0", "f1"],
            "family": ["connectivity", "band"],
        }
    )

    result = compute_row_outlier_scores(
        df,
        ["f0", "f1"],
        z_threshold=3.0,
        group_by="family",
        feature_schema=schema,
    )

    assert result.loc[3, "outlier_fraction_connectivity"] == 1.0
    assert result.loc[3, "outlier_fraction_band"] == 0.0


def test_row_scores_partial_schema_enriches_missing_measure():
    # Schema carries family but not measure; group_by='measure' must still split
    # alpha vs. beta by enriching the missing column from the parseable names.
    df = pd.DataFrame(
        {
            "band_abs_alpha_ch-Fz": [0.0, 0.0, 0.0, 100.0],
            "band_abs_beta_ch-Fz": [1.0, 1.0, 1.0, 1.0],
        }
    )
    schema = pd.DataFrame(
        {
            "column": list(df.columns),
            "family": ["band", "band"],
        }
    )

    result = compute_row_outlier_scores(
        df,
        list(df.columns),
        z_threshold=3.0,
        group_by="measure",
        feature_schema=schema,
    )

    assert result.loc[3, "outlier_fraction_abs_alpha"] == 1.0
    assert result.loc[3, "outlier_fraction_abs_beta"] == 0.0


def test_row_scores_many_groups_avoids_fragmentation_warning():
    columns = [f"feature_{index}" for index in range(120)]
    df = pd.DataFrame(
        np.vstack(
            [
                np.zeros(len(columns)),
                np.zeros(len(columns)),
                np.zeros(len(columns)),
                np.full(len(columns), 100.0),
            ]
        ),
        columns=columns,
    )
    schema = pd.DataFrame(
        {
            "column": columns,
            "measure": [f"measure_{index}" for index in range(len(columns))],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.PerformanceWarning)
        result = compute_row_outlier_scores(
            df,
            columns,
            z_threshold=3.0,
            group_by="measure",
            feature_schema=schema,
        )

    assert result.loc[3, "outlier_fraction_measure_0"] == 1.0
    assert not any(
        isinstance(item.message, pd.errors.PerformanceWarning) for item in caught
    )


def test_drop_subject_outliers_per_family_returns_masks():
    names = ["band_alpha_ch-Fz", "complexity_entropy_ch-Fz"]
    container = DataContainer(
        X=np.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [100.0, 1.0]]),
        dims=("obs", "feature"),
        coords={"feature": names, "subject": ["s1", "s2", "s3", "s4"]},
        ids=np.asarray(["o1", "o2", "o3", "o4"]),
    )

    masks, result = drop_subject_outliers(
        container,
        z_threshold=3.0,
        outlier_fraction_threshold=0.5,
        descriptor_names=names,
        group_by="family",
    )

    assert masks["band"].tolist() == [True, True, True, False]
    assert masks["complexity"].all()
    assert result.per_family_dropped["band"][0].subject_id == "s4"


def test_drop_epoch_outliers_min_obs_gate():
    container = _clean_container(n_subjects=4, n_features=2)
    container.X[-1, :] = 1000.0
    with pytest.raises(RuntimeError, match="minimum required"):
        drop_epoch_outliers(
            container,
            z_threshold=3.0,
            outlier_fraction_threshold=0.5,
            min_obs=4,
        )


def test_drop_epoch_clean_data_drops_nothing():
    container = _clean_container()

    clean, result = drop_epoch_outliers(container)

    assert clean.X.shape == container.X.shape
    assert result.n_epochs_dropped == 0


def test_drop_epoch_removes_outlier_and_filters_metadata():
    container = _clean_container()
    container.X[3, :] = 500.0

    clean, result = drop_epoch_outliers(
        container,
        z_threshold=3.0,
        outlier_fraction_threshold=0.5,
    )

    assert clean.X.shape[0] == 9
    assert clean.y.tolist() == [0, 1, 2, 4, 5, 6, 7, 8, 9]
    assert clean.ids.tolist() == [
        "obs-0",
        "obs-1",
        "obs-2",
        "obs-4",
        "obs-5",
        "obs-6",
        "obs-7",
        "obs-8",
        "obs-9",
    ]
    assert len(clean.coords["condition"]) == 9
    assert clean.coords["feature"] == container.coords["feature"]
    assert clean.meta == container.meta
    assert result.epochs_dropped[0].obs_index == 3
    assert result.epochs_dropped[0].mad_z_max > 3.0


def test_drop_subject_removes_outlier():
    container = _container_with_outlier_subject()

    clean, result = drop_subject_outliers(
        container,
        z_threshold=3.0,
        outlier_fraction_threshold=0.5,
    )

    assert clean.X.shape[0] == 9
    assert result.n_subjects_dropped == 1
    assert result.subjects_dropped[0].subject_id == "sub-09"
    assert result.subject_outlier_burden is not None


def test_drop_subject_aggregates_repeated_observations():
    rng = np.random.default_rng(5)
    subject_ids = [
        subject
        for subject in [f"sub-{index:02d}" for index in range(6)]
        for _ in range(2)
    ]
    values = rng.standard_normal((12, 8))
    values[-2:, :] = 1000.0
    container = _make_container(values, subject_ids)

    clean, result = drop_subject_outliers(
        container,
        z_threshold=3.0,
        outlier_fraction_threshold=0.5,
    )

    assert clean.X.shape[0] == 10
    assert result.n_subjects_dropped == 1
    assert result.subjects_dropped[0].subject_id == "sub-05"
    assert len(result.subject_outlier_burden) == 6


def test_run_qc_can_skip_each_level():
    container = _clean_container()

    _, no_epoch = run_qc(container, epoch_z_threshold=None)
    _, no_subject = run_qc(container, subject_z_threshold=None)

    assert no_epoch.epoch_drop_threshold is None
    assert no_epoch.epoch_outlier_fraction_threshold is None
    assert no_subject.subject_drop_threshold is None
    assert no_subject.subject_outlier_fraction_threshold is None


def test_run_qc_combines_counts_and_missingness():
    container = _container_with_outlier_subject()

    clean, result = run_qc(
        container,
        epoch_z_threshold=None,
        subject_z_threshold=3.0,
        subject_outlier_fraction_threshold=0.5,
    )

    assert result.n_obs_in == 10
    assert result.n_obs_out == clean.X.shape[0] == 9
    assert result.retention_rate == pytest.approx(0.9)
    assert result.feature_missingness is not None
    assert len(result.feature_missingness) == container.X.shape[1]


def test_run_qc_surfaces_level1_counts_and_family_qc_is_attachable():
    container = _clean_container()
    container.meta.update(
        {
            "n_rows_entering_qc": 15,
            "n_dropped_nan_inf": 3,
            "dropped_extreme_rows": 2,
        }
    )

    _, result = run_qc(
        container,
        epoch_z_threshold=None,
        subject_z_threshold=None,
    )

    assert result.n_rows_entering_qc == 15
    assert result.n_dropped_nan_inf == 3
    assert result.n_dropped_extreme == 2
    assert result.family_qc is None

    result.family_qc = pd.DataFrame({"family": ["band"], "n_features": [10]})
    assert result.family_qc["family"].tolist() == ["band"]


def test_run_qc_defaults_missing_level1_counts():
    _, result = run_qc(
        _clean_container(),
        epoch_z_threshold=None,
        subject_z_threshold=None,
    )

    assert result.n_rows_entering_qc is None
    assert result.n_dropped_nan_inf == 0
    assert result.n_dropped_extreme == 0


def test_run_qc_can_skip_missingness():
    _, result = run_qc(
        _clean_container(),
        compute_missingness=False,
    )

    assert result.feature_missingness is None


def test_filter_removes_stale_dropped_extreme_rows_from_meta():
    """Filtered container must NOT carry the original load's drop count."""
    container = _make_container(
        np.ones((5, 3)),
        [f"s{i}" for i in range(5)],
    )
    container.meta.update(
        {
            "dropped_extreme_rows": 7,
            "n_rows_entering_qc": 20,
        }
    )
    keep = np.array([True, True, False, True, True])
    filtered = _filter_observations(container, keep)

    assert "dropped_extreme_rows" not in filtered.meta
    # non-stale key must survive the copy
    assert filtered.meta.get("n_rows_entering_qc") == 20


def test_filter_meta_is_deep_copied():
    """Mutating the original container's nested meta must not affect the slice."""
    container = _make_container(
        np.ones((4, 3)),
        [f"s{i}" for i in range(4)],
    )
    container.meta["nested"] = [1, 2, 3]
    keep = np.ones(4, dtype=bool)
    filtered = _filter_observations(container, keep)

    # mutate the original
    container.meta["nested"].append(99)

    assert filtered.meta["nested"] == [1, 2, 3]


def test_total_dropped_simplified_formula():
    """
    total_dropped == n_dropped_nan_inf + n_dropped_extreme + (n_obs_in - n_obs_out).
    """
    result = QCResult(
        n_dropped_nan_inf=4,
        n_dropped_extreme=2,
        n_obs_in=30,
        n_obs_out=21,
        n_subjects_in=6,
        n_subjects_out=6,
    )
    # epochs_dropped is 0 here; subject-level accounts for the remainder
    assert result.total_dropped == 4 + 2 + (30 - 21)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_drop_outliers_rejects_invalid_fraction_threshold(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        drop_epoch_outliers(
            _clean_container(),
            outlier_fraction_threshold=threshold,
        )


def test_quality_functions_require_flat_feature_container():
    container = DataContainer(
        X=np.zeros((2, 3, 4)),
        dims=("obs", "channel", "time"),
    )

    with pytest.raises(ValueError, match="flat 2D"):
        run_qc(container)


def test_check_missingness_extra():
    df = pd.DataFrame({"A": [1, 2, 3]})
    res = check_missingness(df, threshold_warn=0.1)
    assert res.status == "OK"


def test_check_constant_columns_extra():
    # ndarray ndim != 2
    res1 = check_constant_columns(np.array([1, 2, 3]))
    assert res1 == []

    # Empty numeric cols
    df = pd.DataFrame({"A": ["a", "b"]})
    res2 = check_constant_columns(df)
    assert res2 == []


def test_check_outliers_zscore_extra():
    df = pd.DataFrame({"A": [1, 1, 1]})
    res = check_outliers_zscore(df, sigma=1.0)
    assert res is None  # std == 0 returns None


def test_check_flatline_extra():
    # Only inf
    res = check_flatline(np.array([np.inf, -np.inf]))
    assert res.status == "WARN"


def test_row_quality_score_extra():
    df = pd.DataFrame({"A": [np.nan, 2, 0], "B": [1, np.inf, 1], "C": ["x", "y", "z"]})

    scores = row_quality_score(df, exclude_cols=["C"], count_zero=True, normalize=False)
    assert list(scores) == [1, 1, 1]

    scores_norm = row_quality_score(
        df, exclude_cols=["C"], count_zero=True, normalize=True
    )
    assert list(scores_norm) == [0.5, 0.5, 0.5]

    # no numeric
    df2 = pd.DataFrame({"C": ["x", "y", "z"]})
    assert list(row_quality_score(df2, normalize=False)) == [0, 0, 0]
    assert list(row_quality_score(df2, normalize=True)) == [0.0, 0.0, 0.0]


def test_drop_epoch_outliers_extra():
    dc = DataContainer(X=np.zeros((2, 2)), dims=("obs", "feature"))
    with pytest.raises(ValueError):
        drop_epoch_outliers(dc, z_threshold=0)


def test_get_subject_ids_extra():
    # coords bad length
    dc = DataContainer(
        X=np.zeros((2, 2)), dims=("obs", "feature"), coords={"subject": [1]}
    )
    with pytest.raises(ValueError):
        _get_subject_ids(dc, "subject")

    # ids bad length
    dc2 = DataContainer(X=np.zeros((2, 2)), dims=("obs", "feature"), ids=[1])
    with pytest.raises(ValueError):
        _get_subject_ids(dc2, "subject")

    # fallback to index
    dc3 = DataContainer(X=np.zeros((2, 2)), dims=("obs", "feature"))
    assert _get_subject_ids(dc3, "subject") == ["0", "1"]


def test_container_to_feature_df_extra():
    # Missing feature coords (auto generate)
    dc = DataContainer(X=np.zeros((2, 2)), dims=("obs", "feature"))
    df = _container_to_feature_df(dc, None)
    assert list(df.columns) == ["f0", "f1"]

    # Bad names length
    dc2 = DataContainer(
        X=np.zeros((2, 2)), dims=("obs", "feature"), coords={"feature": ["f0"]}
    )
    with pytest.raises(ValueError):
        _container_to_feature_df(dc2, None)

    # missing required feature cols
    dc3 = DataContainer(
        X=np.zeros((2, 2)), dims=("obs", "feature"), coords={"feature": ["f0", "f1"]}
    )
    with pytest.raises(ValueError):
        _container_to_feature_df(dc3, ["f2"])


def test_filter_observations_extra():
    dc = DataContainer(X=np.zeros((2, 2)), dims=("obs", "feature"))
    with pytest.raises(ValueError):
        _filter_observations(dc, np.array([True]))


def _cov_flat_container():
    return DataContainer(
        X=np.random.rand(6, 4),
        dims=("obs", "feature"),
        coords={
            "feature": ["f0", "f1", "f2", "f3"],
            "subject": np.array(["a", "a", "b", "b", "c", "c"]),
        },
        ids=np.arange(6).astype(str),
    )


def test_qcresult_retention_rate_zero_input():
    assert np.isnan(QCResult().retention_rate)


def test_qcresult_summary_reports_dropped_columns():
    result = QCResult(feature_columns_dropped=pd.DataFrame({"column": ["a", "b"]}))
    assert result.summary()["n_feature_columns_dropped"] == 2


def test_check_constant_columns_non_2d_array():
    assert check_constant_columns(np.array([1, 2, 3])) == []


def test_check_constant_columns_2d_no_constant():
    assert check_constant_columns(np.array([[1.0, 2.0], [3.0, 4.0]])) == []


def test_check_outliers_zscore_zero_std():
    assert check_outliers_zscore(np.ones((4, 2))) is None


def test_check_outliers_zscore_no_outlier():
    assert check_outliers_zscore(np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]])) is None


def test_compute_row_outlier_scores_descriptor_names_misaligned():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match="descriptor_names must align"):
        compute_row_outlier_scores(
            df, ["a", "b"], group_by="family", descriptor_names=["only_one"]
        )


def test_drop_epoch_outliers_invalid_group_by():
    with pytest.raises(ValueError, match="group_by must be"):
        drop_epoch_outliers(_cov_flat_container(), group_by="bogus")


def test_drop_epoch_outliers_invalid_min_obs():
    with pytest.raises(ValueError, match="min_obs"):
        drop_epoch_outliers(_cov_flat_container(), min_obs=0)


def test_drop_epoch_outliers_group_by_feature():
    masks, result = drop_epoch_outliers(_cov_flat_container(), group_by="feature")
    assert isinstance(masks, dict)
    assert result.thresholds["group_by"] == "feature"


def test_drop_epoch_outliers_group_by_reads_container_feature_schema():
    X = np.ones((6, 2))
    X[0, 0] = 1000.0
    container = DataContainer(
        X=X,
        dims=("obs", "feature"),
        coords={
            "feature": ["f0", "f1"],
            "feature_family": ["connectivity", "band"],
            "subject": np.array(["a"] * 6),
        },
        ids=np.arange(6).astype(str),
    )

    masks, result = drop_epoch_outliers(
        container,
        group_by="family",
        outlier_fraction_threshold=0.0,
    )

    assert masks["connectivity"].tolist() == [False, True, True, True, True, True]
    assert masks["band"].all()
    assert result.per_family_dropped["connectivity"][0].obs_id == "0"


def test_drop_epoch_outliers_group_by_min_obs_raises():
    with pytest.raises(RuntimeError, match="remain for family"):
        drop_epoch_outliers(_cov_flat_container(), group_by="feature", min_obs=999)


def test_drop_epoch_outliers_group_by_records_drop():
    # One strong per-feature outlier is dropped, exercising record dedup.
    X = np.ones((6, 2))
    X[0, 0] = 1000.0
    container = DataContainer(
        X=X,
        dims=("obs", "feature"),
        coords={"feature": ["f0", "f1"], "subject": np.array(["a"] * 6)},
        ids=np.arange(6).astype(str),
    )
    _, result = drop_epoch_outliers(
        container, group_by="feature", outlier_fraction_threshold=0.0
    )
    assert result.n_epochs_dropped >= 1
    assert result.epochs_dropped[0].obs_index == 0


def test_run_qc_without_subject_ids_uses_range():
    # No subject coord and no ids -> subject ids fall back to positional range.
    container = DataContainer(
        X=np.random.rand(4, 3),
        dims=("obs", "feature"),
        coords={"feature": ["f0", "f1", "f2"]},
    )
    _, result = run_qc(container, subject_z_threshold=None, epoch_z_threshold=None)
    assert result.n_subjects_in == 4


def test_run_qc_subject_ids_from_container_ids():
    # No subject coord but ids present -> subject ids come from container.ids.
    container = DataContainer(
        X=np.random.rand(4, 2),
        dims=("obs", "feature"),
        coords={"feature": ["f0", "f1"]},
        ids=np.array(["s1", "s1", "s2", "s2"]),
    )
    _, result = run_qc(container, subject_z_threshold=None, epoch_z_threshold=None)
    assert result.n_subjects_in == 2


def test_run_qc_with_feature_cols_subset():
    container = _cov_flat_container()
    cleaned, _ = run_qc(
        container,
        epoch_z_threshold=None,
        subject_z_threshold=None,
        feature_cols=["f0", "f1"],
    )
    assert cleaned.X.shape[0] == container.X.shape[0]


def test_group_labels_from_structured_schema():
    # Opaque feature names: labels must come from the structured schema, not
    # from parsing the column strings.
    container = DataContainer(
        X=np.zeros((1, 3)),
        dims=("obs", "feature"),
        coords={
            "feature": ["c0", "c1", "c2"],
            "feature_family": ["band", "band", "connectivity"],
            "feature_measure": ["alpha", "beta", "coherence"],
        },
    )

    assert group_labels(container, "family") == ["band", "connectivity"]
    assert group_labels(container, "measure") == ["alpha", "beta", "coherence"]
    # group_by="feature" falls back to the column id per GROUP_BY_COLUMN.
    assert group_labels(container, "feature") == ["c0", "c1", "c2"]


def test_group_labels_enriches_partial_schema_for_measure():
    # Schema carries only family; group_by="measure" must still split via the
    # parser enrichment instead of collapsing to a single label.
    container = DataContainer(
        X=np.zeros((1, 2)),
        dims=("obs", "feature"),
        coords={
            "feature": ["band_abs_alpha_ch-Fz", "band_abs_beta_ch-Fz"],
            "feature_family": ["band", "band"],
        },
    )

    assert group_labels(container, "family") == ["band"]
    assert group_labels(container, "measure") == ["abs_alpha", "abs_beta"]


def test_group_labels_no_feature_axis_returns_empty():
    container = DataContainer(
        X=np.zeros((1, 2)),
        dims=("obs", "channel"),
        coords={"channel": ["Fz", "Cz"]},
    )

    assert group_labels(container, "family") == []
    with pytest.raises(ValueError, match="group_by must be one of"):
        group_labels(container, "nonsense")
