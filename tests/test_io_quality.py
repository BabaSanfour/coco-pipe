import warnings

import numpy as np
import pandas as pd
import pytest

from coco_pipe.io.quality import (
    CheckResult,
    EpochDropRecord,
    QCResult,
    SubjectDropRecord,
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


def _container(X, subjects, *, ids=None, **coords):
    n_obs, n_features = X.shape
    return DataContainer(
        X=X,
        dims=("obs", "feature"),
        coords={
            "feature": [f"f{index}" for index in range(n_features)],
            "subject": subjects,
            "condition": ["baseline"] * n_obs,
            **coords,
        },
        y=np.arange(n_obs),
        ids=np.asarray(ids or [f"obs-{index}" for index in range(n_obs)]),
        meta={"source": "synthetic"},
    )


def _clean(n_subjects=10, n_features=20):
    X = np.random.default_rng(42).standard_normal((n_subjects, n_features))
    return _container(X, [f"sub-{index:02d}" for index in range(n_subjects)])


def _outlier_subject(n_subjects=10):
    X = np.random.default_rng(0).standard_normal((n_subjects, 20))
    X[-1] = 1000.0
    return _container(X, [f"sub-{index:02d}" for index in range(n_subjects)])


# --- flags and results ---------------------------------------------------------


def test_qc_flags_and_status():
    pass_flag = make_qc_flag("pass", "complete", "All checks passed.")
    warn_flag = make_qc_flag(
        "warn", "high_missingness", "Too many NaNs.", value=0.25, threshold=0.2
    )
    assert pass_flag == {
        "level": "pass",
        "code": "complete",
        "message": "All checks passed.",
        "value": None,
        "threshold": None,
        "scope": "",
    }
    assert resolve_qc_status([]) == "pass"
    assert resolve_qc_status([pass_flag, warn_flag]) == "warn"
    fail_flag = make_qc_flag("fail", "invalid", "Invalid output.")
    assert resolve_qc_status([warn_flag, fail_flag]) == "fail"
    with pytest.raises(ValueError, match="level must be"):
        make_qc_flag("invalid", "bad", "bad")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("level", "status", "severity"),
    [("pass", "OK", 0), ("warn", "WARN", 5), ("fail", "FAIL", 9)],
)
def test_check_result_from_flag(level, status, severity):
    flag = make_qc_flag(
        level, "feature_qc", "Feature QC.", value=3, threshold=2, scope="complexity"
    )
    result = CheckResult.from_flag_dict(flag)
    assert result.check_name == "feature_qc"
    assert result.status == status
    assert result.severity == severity
    assert result.metric_name == "complexity"
    assert result.metric_value == 3
    assert result.is_issue == (level != "pass")


def test_qc_result_counts():
    result = QCResult(
        n_rows_entering_qc=13,
        n_dropped_nan_inf=2,
        n_dropped_extreme=1,
        n_obs_in=10,
        n_obs_out=8,
        n_subjects_in=5,
        n_subjects_out=4,
        epochs_dropped=[EpochDropRecord(1, "obs-1", 0.5, 10.0)],
        subjects_dropped=[SubjectDropRecord("sub-4", 0.75, 3)],
        feature_columns_dropped=pd.DataFrame({"column": ["a", "b"]}),
    )
    assert result.retention_rate == pytest.approx(0.8)
    assert result.n_epochs_dropped == 1
    assert result.n_subjects_dropped == 1
    # NaN/inf rows + extreme rows + observations dropped by QC
    assert result.total_dropped == 2 + 1 + (10 - 8)
    summary = result.summary()
    assert summary["n_feature_columns_dropped"] == 2
    assert {"retention_rate", "n_rows_entering_qc", "epoch_drop_threshold"} <= set(
        summary
    )
    assert np.isnan(QCResult().retention_rate)


# --- simple checks ---------------------------------------------------------------


def test_check_missingness():
    assert check_missingness(np.array([])).status == "WARN"
    array = check_missingness(np.array([1, np.nan, 3, np.nan]), threshold_fail=0.4)
    assert (array.status, array.metric_value) == ("FAIL", 0.5)
    frame = pd.DataFrame({"A": [1, 2, np.nan], "B": [np.nan, 5, 6]})
    result = check_missingness(frame, threshold_warn=0.2, threshold_fail=0.5)
    assert result.status == "WARN"
    assert np.isclose(result.metric_value, 1 / 3)
    assert check_missingness(pd.DataFrame({"A": [1, 2]})).status == "OK"


def test_check_constant_columns():
    (array,) = check_constant_columns(np.array([[1, 2, 3], [1, 4, 3], [1, 5, 3]]))
    assert "2 constant features" in array.message
    frame = pd.DataFrame({"A": [1, 1, 1], "B": [1, 2, 3], "C": ["x", "x", "x"]})
    (named,) = check_constant_columns(frame)
    assert "constant columns: A" in named.message
    many = pd.DataFrame({name: [1, 1] for name in "ABCD"} | {"E": [5, 6]})
    assert "..." in check_constant_columns(many)[0].message
    for nothing in (
        np.array([]),
        pd.DataFrame(),
        np.array([1, 2, 3]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        pd.DataFrame({"A": ["x", "y"]}),
        pd.DataFrame({"A": [1, 2], "B": [2, 3]}),
        np.array([["a", "b"], ["c", "d"]], dtype=object),
    ):
        assert check_constant_columns(nothing) == []


def test_check_outliers_zscore():
    result = check_outliers_zscore(np.array([1.0] * 50 + [100.0]), sigma=3.0)
    assert result.status == "WARN"
    assert result.metric_value > 3.0
    for nothing in (
        np.array([]),
        np.array([np.nan, np.nan]),
        np.array([5, 5, 5]),
        pd.DataFrame({"A": [1, 1, 1]}),
        np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]),
    ):
        assert check_outliers_zscore(nothing) is None


def test_check_flatline():
    flat = check_flatline(np.array([5, 5, 5]), threshold=1e-5)
    assert (flat.status, flat.metric_value) == ("FAIL", 0)
    assert check_flatline(np.array([1, 2, 3]), threshold=1e-5).status == "OK"
    assert check_flatline(np.array([])).status == "WARN"
    assert check_flatline(np.array([np.inf, -np.inf])).status == "WARN"
    assert check_flatline(np.array(["a", "b"])).status == "OK"


# --- outlier scores --------------------------------------------------------------


def test_row_outlier_scores():
    frame = pd.DataFrame(
        {"f1": [1.0, 1.1, 0.9, 500.0], "f2": [2.0, 2.1, 1.9, 500.0]}, index=[5, 6, 7, 8]
    )
    scores = compute_row_outlier_scores(frame, ["f1", "f2"], z_threshold=3.0)
    assert scores.columns.tolist() == [
        "outlier_fraction",
        "n_outlier_features",
        "mad_z_max",
    ]
    assert scores.index.tolist() == [5, 6, 7, 8]
    assert scores["outlier_fraction"].tolist() == [0.0, 0.0, 0.0, 1.0]

    # a column with zero MAD (a count that is nearly always the same) is not scored
    one_off = compute_row_outlier_scores(
        pd.DataFrame({"f1": [5.0, 5.0, 5.0, 6.0], "f2": [1.0, 1.1, 0.9, 50.0]}),
        ["f1", "f2"],
        z_threshold=3.0,
    )
    assert one_off["outlier_fraction"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert one_off["n_outlier_features"].tolist() == [0, 0, 0, 1]

    # the fraction is over the finite values of the row: a missing feature does
    # not dilute an outlier
    missing = compute_row_outlier_scores(
        pd.DataFrame({"f1": [1.0, 1.1, 0.9, 50.0], "f2": [2.0, 2.1, 1.9, np.nan]}),
        ["f1", "f2"],
        z_threshold=3.0,
    )
    assert missing["outlier_fraction"].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_row_outlier_scores_per_group():
    names = ["band_abs_alpha_ch-Fz", "complexity_entropy_ch-Fz"]
    frame = pd.DataFrame(
        {names[0]: [1.0, 1.1, 0.9, 100.0], names[1]: [1.0, 1.1, 0.9, 1.0]}
    )
    by_name = compute_row_outlier_scores(
        frame, names, z_threshold=3.0, descriptor_names=names, group_by="family"
    )
    assert by_name.loc[3, "outlier_fraction_band"] == 1.0
    assert by_name.loc[3, "outlier_fraction_complexity"] == 0.0

    # a schema with only `family` is completed from the names for `measure`
    schema = pd.DataFrame({"column": names, "family": ["connectivity", "band"]})
    by_schema = compute_row_outlier_scores(
        frame, names, z_threshold=3.0, group_by="family", feature_schema=schema
    )
    assert by_schema.loc[3, "outlier_fraction_connectivity"] == 1.0
    by_measure = compute_row_outlier_scores(
        frame, names, z_threshold=3.0, group_by="measure", feature_schema=schema
    )
    assert by_measure.loc[3, "outlier_fraction_abs_alpha"] == 1.0
    assert by_measure.loc[3, "outlier_fraction_entropy"] == 0.0

    columns = [f"feature_{index}" for index in range(120)]
    wide = pd.DataFrame(
        np.tile([[1.0], [1.1], [0.9], [100.0]], (1, 120)), columns=columns
    )
    measures = pd.DataFrame({"column": columns, "measure": columns})
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        many = compute_row_outlier_scores(
            wide, columns, z_threshold=3.0, group_by="measure", feature_schema=measures
        )
    assert many.loc[3, "outlier_fraction_feature_0"] == 1.0

    with pytest.raises(ValueError, match="descriptor_names must align"):
        compute_row_outlier_scores(
            frame, names, group_by="family", descriptor_names=["only_one"]
        )


def test_subject_outlier_burden():
    frame = pd.DataFrame(
        {
            "subject": [f"s{i}" for i in range(8)],
            "spiky": [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 100.0],
            "stable": [2.0] * 8,
            "all_nan": [np.nan] * 8,
            "partly_infinite": [1.0, np.inf] + [1.0] * 6,
        }
    )
    burden = compute_subject_outlier_burden(
        frame, ["spiky", "stable", "all_nan", "partly_infinite"]
    )
    assert burden["n_outlier_features"].tolist() == [0] * 7 + [1]
    assert burden["n_epochs"].tolist() == [1] * 8

    # epochs are scored before they are averaged per subject
    epochs = pd.DataFrame(
        {
            "subject": [f"s{i}" for i in range(1, 9) for _ in range(2)],
            "f1": [0.1, -0.1] * 7 + [-100.0, 100.0],
        }
    )
    s8 = compute_subject_outlier_burden(epochs, ["f1"]).set_index("subject").loc["s8"]
    assert s8["outlier_fraction"] == pytest.approx(1.0)
    assert s8["n_epochs"] == 2


def test_subject_outlier_burden_arguments():
    frame = pd.DataFrame({"participant": ["s1", "s2"], "value": [1.0, 2.0]})
    empty = compute_subject_outlier_burden(frame, [], subject_col="participant")
    assert empty.to_dict(orient="list") == {
        "participant": ["s1", "s2"],
        "outlier_fraction": [0.0, 0.0],
        "n_outlier_features": [0.0, 0.0],
        "n_epochs": [1, 1],
    }
    with pytest.raises(ValueError, match="Subject column"):
        compute_subject_outlier_burden(frame, ["value"])
    with pytest.raises(ValueError, match="z_threshold must be positive"):
        compute_subject_outlier_burden(
            frame, ["value"], subject_col="participant", z_threshold=0
        )
    with pytest.raises(ValueError, match="Feature columns not found"):
        compute_subject_outlier_burden(frame, ["missing"], subject_col="participant")


# --- dropping ---------------------------------------------------------------------


def test_drop_epoch_outliers_filters_rows_and_metadata():
    container = _clean()
    clean, result = drop_epoch_outliers(container)
    assert clean.X.shape == container.X.shape
    assert result.n_epochs_dropped == 0

    container.X[3] = 500.0
    container.meta.update(
        {"dropped_extreme_rows": 7, "n_rows_entering_qc": 20, "nested": [1, 2]}
    )
    clean, result = drop_epoch_outliers(
        container, z_threshold=3.0, outlier_fraction_threshold=0.5
    )
    assert clean.y.tolist() == [0, 1, 2, 4, 5, 6, 7, 8, 9]
    assert clean.ids.tolist() == [f"obs-{i}" for i in (0, 1, 2, 4, 5, 6, 7, 8, 9)]
    assert len(clean.coords["condition"]) == 9
    assert clean.coords["feature"].tolist() == container.coords["feature"].tolist()
    assert result.epochs_dropped[0].obs_index == 3
    assert result.epochs_dropped[0].mad_z_max > 3.0
    # the load's drop count does not describe the filtered rows; meta is a copy
    assert "dropped_extreme_rows" not in clean.meta
    assert clean.meta["n_rows_entering_qc"] == 20
    container.meta["nested"].append(99)
    assert clean.meta["nested"] == [1, 2]


def test_drop_epoch_outliers_per_group():
    X = 1.0 + 0.01 * np.arange(16.0).reshape(8, 2)
    X[0, 0] = 1000.0
    container = DataContainer(
        X=X,
        dims=("obs", "feature"),
        coords={
            "feature": ["f0", "f1"],
            "feature_family": ["connectivity", "band"],
            "subject": np.array(["a"] * 8),
        },
        ids=np.arange(8).astype(str),
    )
    masks, result = drop_epoch_outliers(
        container, group_by="family", outlier_fraction_threshold=0.0
    )
    assert masks["connectivity"].tolist() == [False] + [True] * 7
    assert masks["band"].all()
    assert result.per_family_dropped["connectivity"][0].obs_id == "0"

    masks, result = drop_epoch_outliers(
        container, group_by="feature", outlier_fraction_threshold=0.0
    )
    assert set(masks) == {"f0", "f1"}
    assert result.thresholds["group_by"] == "feature"
    assert result.epochs_dropped[0].obs_index == 0


def test_drop_epoch_outliers_errors():
    container = _clean(n_subjects=4, n_features=2)
    container.X[-1] = 1000.0
    with pytest.raises(RuntimeError, match="minimum required"):
        drop_epoch_outliers(
            container, z_threshold=3.0, outlier_fraction_threshold=0.5, min_obs=4
        )
    with pytest.raises(RuntimeError, match="remain for family"):
        drop_epoch_outliers(container, group_by="feature", min_obs=999)
    for kwargs, match in [
        ({"outlier_fraction_threshold": -0.1}, "between 0 and 1"),
        ({"outlier_fraction_threshold": 1.1}, "between 0 and 1"),
        ({"z_threshold": 0}, "z_threshold"),
        ({"group_by": "bogus"}, "group_by must be"),
        ({"min_obs": 0}, "min_obs"),
    ]:
        with pytest.raises(ValueError, match=match):
            drop_epoch_outliers(container, **kwargs)


def test_drop_subject_outliers():
    clean, result = drop_subject_outliers(
        _outlier_subject(), z_threshold=3.0, outlier_fraction_threshold=0.5
    )
    assert clean.X.shape[0] == 9
    assert result.subjects_dropped[0].subject_id == "sub-09"
    assert result.subject_outlier_burden is not None

    # a subject's observations are averaged before it is scored
    X = np.random.default_rng(5).standard_normal((12, 8))
    X[-2:] = 1000.0
    subjects = [f"sub-{index:02d}" for index in range(6) for _ in range(2)]
    clean, result = drop_subject_outliers(
        _container(X, subjects), z_threshold=3.0, outlier_fraction_threshold=0.5
    )
    assert clean.X.shape[0] == 10
    assert result.subjects_dropped[0].subject_id == "sub-05"
    assert len(result.subject_outlier_burden) == 6

    names = ["band_alpha_ch-Fz", "complexity_entropy_ch-Fz"]
    container = DataContainer(
        X=np.asarray([[0.0, 1.0], [0.1, 1.1], [-0.1, 0.9], [100.0, 1.0]]),
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


def test_subject_column_must_exist():
    # 3 subjects of 4 epochs, one bad epoch in sub-01
    X = np.random.default_rng(0).standard_normal((12, 6))
    X[5] += 50.0
    subjects = [f"sub-{index:02d}" for index in range(3) for _ in range(4)]
    container = _container(X, subjects)

    _, result = drop_subject_outliers(container)
    assert result.n_subjects_in == 3
    assert [record.subject_id for record in result.subjects_dropped] == ["sub-01"]

    # a misspelled column used to make every epoch its own subject
    with pytest.raises(ValueError, match=r"'participant' not found.*'subject'"):
        drop_subject_outliers(container, subject_col="participant")
    with pytest.raises(ValueError, match="'participant' not found"):
        run_qc(container, subject_col="participant")

    # epoch QC only counts subjects, so it runs and reports them as unknown
    clean, result = drop_epoch_outliers(container, subject_col="participant")
    assert clean.X.shape[0] == 11
    assert (result.n_subjects_in, result.n_subjects_out) == (None, None)
    _, result = run_qc(container, subject_z_threshold=None, subject_col="participant")
    assert result.n_subjects_in is None


# --- run_qc -----------------------------------------------------------------------


def test_run_qc_combines_levels():
    container = _outlier_subject()
    clean, result = run_qc(
        container,
        epoch_z_threshold=None,
        subject_z_threshold=3.0,
        subject_outlier_fraction_threshold=0.5,
    )
    assert result.n_obs_in == 10
    assert result.n_obs_out == clean.X.shape[0] == 9
    assert result.retention_rate == pytest.approx(0.9)
    assert len(result.feature_missingness) == container.X.shape[1]

    _, no_epoch = run_qc(container, epoch_z_threshold=None)
    assert no_epoch.epoch_drop_threshold is None
    assert no_epoch.epoch_outlier_fraction_threshold is None
    _, no_subject = run_qc(container, subject_z_threshold=None)
    assert no_subject.subject_drop_threshold is None
    _, no_missing = run_qc(container, compute_missingness=False)
    assert no_missing.feature_missingness is None


def test_run_qc_reports_counts_from_loading():
    container = _clean()
    _, result = run_qc(container, epoch_z_threshold=None, subject_z_threshold=None)
    assert result.n_rows_entering_qc is None
    assert (result.n_dropped_nan_inf, result.n_dropped_extreme) == (0, 0)

    container.meta.update(
        {"n_rows_entering_qc": 15, "n_dropped_nan_inf": 3, "dropped_extreme_rows": 2}
    )
    _, result = run_qc(container, epoch_z_threshold=None, subject_z_threshold=None)
    assert (result.n_rows_entering_qc, result.n_dropped_nan_inf) == (15, 3)
    assert result.n_dropped_extreme == 2
    assert result.family_qc is None


def test_run_qc_inputs():
    off = {"epoch_z_threshold": None, "subject_z_threshold": None}
    # without a subject coordinate the subjects are unknown, never the ids
    no_subjects = DataContainer(
        X=np.random.rand(4, 2),
        dims=("obs", "feature"),
        ids=np.array(["s1", "s1", "s2", "s2"]),
    )
    _, result = run_qc(no_subjects, **off)
    assert (result.n_subjects_in, result.n_subjects_out) == (None, None)
    with pytest.raises(ValueError, match="Subject column 'subject' not found"):
        run_qc(no_subjects, epoch_z_threshold=None)

    container = _clean(n_subjects=6, n_features=4)
    cleaned, _ = run_qc(container, feature_cols=["f0", "f1"], **off)
    assert cleaned.X.shape[0] == 6
    with pytest.raises(ValueError):
        run_qc(container, feature_cols=["missing"], **off)
    with pytest.raises(ValueError, match="flat 2D"):
        run_qc(DataContainer(X=np.zeros((2, 3, 4)), dims=("obs", "channel", "time")))


def test_group_labels():
    container = DataContainer(
        X=np.zeros((1, 3)),
        dims=("obs", "feature"),
        coords={
            "feature": ["c0", "c1", "c2"],
            "feature_family": ["band", "band", "connectivity"],
            "feature_measure": ["alpha", "beta", "coherence"],
        },
    )
    # opaque names: labels come from the schema, not from parsing
    assert group_labels(container, "family") == ["band", "connectivity"]
    assert group_labels(container, "measure") == ["alpha", "beta", "coherence"]
    assert group_labels(container, "feature") == ["c0", "c1", "c2"]

    partial = DataContainer(
        X=np.zeros((1, 2)),
        dims=("obs", "feature"),
        coords={
            "feature": ["band_abs_alpha_ch-Fz", "band_abs_beta_ch-Fz"],
            "feature_family": ["band", "band"],
        },
    )
    assert group_labels(partial, "measure") == ["abs_alpha", "abs_beta"]

    channels = DataContainer(
        X=np.zeros((1, 2)), dims=("obs", "channel"), coords={"channel": ["Fz", "Cz"]}
    )
    assert group_labels(channels, "family") == []
    with pytest.raises(ValueError, match="group_by must be one of"):
        group_labels(channels, "nonsense")
