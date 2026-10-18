import numpy as np
import pandas as pd
import pytest

from coco_pipe.descriptors import (
    check_feature_column_consistency,
    load_descriptor_table,
    merge_descriptor_tables,
    save_descriptor_table,
)
from coco_pipe.descriptors.qc import select_viable_feature_columns
from coco_pipe.io import iter_analysis_units


@pytest.fixture
def descriptor_files(tmp_path):
    feature_columns = [
        "band_abs_alpha_ch-Fz",
        "band_abs_alpha_ch-Cz",
        "complexity_sample_entropy_ch-Fz",
        "complexity_sample_entropy_ch-Cz",
    ]
    table = pd.DataFrame(
        {
            "obs_id": ["obs-1", "obs-2", "obs-3"],
            "subject": ["sub-1", "0002", "3"],
            "condition": ["baseline", "baseline", "active"],
            "target": ["control", "case", "case"],
            feature_columns[0]: [1.0, 2.0, 3.0],
            feature_columns[1]: [1.1, 2.1, 3.1],
            feature_columns[2]: [0.1, 0.2, 0.3],
            feature_columns[3]: [0.4, 0.5, 0.6],
        }
    )
    return _save(
        table, tmp_path / "descriptors.parquet", feature_columns
    ), feature_columns


def _save(table, path, feature_columns):
    """Write *table* as a descriptor table carrying *feature_columns*."""
    save_descriptor_table(table, path.with_suffix(""), feature_columns=feature_columns)
    return path.with_suffix(".parquet")


def test_load_descriptor_table_flat(descriptor_files):
    table_path, feature_columns = descriptor_files

    container = load_descriptor_table(
        table_path,
        condition="baseline",
        target_col="target",
        subjects=["1", "sub-2"],
    )

    assert container.dims == ("obs", "feature")
    assert container.X.shape == (2, 4)
    assert container.coords["feature"].tolist() == feature_columns
    assert container.ids.tolist() == ["obs-1", "obs-2"]
    assert container.y.tolist() == ["control", "case"]


def test_load_descriptor_table_flat_exposes_full_schema(descriptor_files):
    table_path, feature_columns = descriptor_files

    container = load_descriptor_table(table_path, condition="baseline")
    schema = container.feature_schema()

    assert schema is not None
    assert schema["column"].tolist() == feature_columns
    assert schema["family"].tolist() == ["band", "band", "complexity", "complexity"]
    # measure resolves the descriptor identity, so QC can split e.g. alpha vs.
    # entropy without re-parsing the flat column names.
    assert schema["measure"].tolist() == [
        "abs_alpha",
        "abs_alpha",
        "sample_entropy",
        "sample_entropy",
    ]
    assert schema["channel"].tolist() == ["Fz", "Cz", "Fz", "Cz"]
    assert schema["scope"].tolist() == ["sensor"] * 4
    assert schema["subfamily"].notna().all()
    assert schema["descriptor"].notna().all()


def test_load_descriptor_table_family_filter(descriptor_files):
    table_path, _ = descriptor_files

    container = load_descriptor_table(
        table_path,
        descriptor_families=["complexity"],
    )

    assert container.X.shape == (3, 2)
    assert container.coords["feature"].tolist() == [
        "complexity_sample_entropy_ch-Fz",
        "complexity_sample_entropy_ch-Cz",
    ]
    assert "band_abs_alpha_ch-Fz" not in container.coords["feature"]
    assert container.coords["band_abs_alpha_ch-Fz"].tolist() == [1.0, 2.0, 3.0]


def test_load_descriptor_table_drops_nan(descriptor_files):
    table_path, feature_columns = descriptor_files
    table = pd.read_parquet(table_path)
    table.loc[1, "band_abs_alpha_ch-Fz"] = np.nan
    _save(table, table_path, feature_columns)

    container = load_descriptor_table(table_path)

    assert container.ids.tolist() == ["obs-1", "obs-3"]
    assert container.meta["n_rows_entering_qc"] == 3
    assert container.meta["n_dropped_nan_inf"] == 1


def test_select_viable_feature_columns_and_loader_prune(tmp_path):
    columns = [
        "band_alpha_ch-Fz",
        "complexity_entropy_ch-Fz",
        "complexity_constant_ch-Fz",
    ]
    frame = pd.DataFrame(
        {
            "obs_id": ["o1", "o2", "o3"],
            columns[0]: [1.0, 2.0, 3.0],
            columns[1]: [np.nan, np.nan, np.nan],
            columns[2]: [5.0, 5.0, 5.0],
        }
    )
    surviving, log = select_viable_feature_columns(frame, columns)
    assert surviving == [columns[0]]
    assert set(log["drop_reason"]) == {"all_nan", "constant"}

    loader_frame = frame.copy()
    loader_frame[columns[1]] = [np.nan, np.nan, 7.0]
    table_path = _save(loader_frame, tmp_path / "features.parquet", columns)
    container = load_descriptor_table(
        table_path,
        drop_degenerate_columns=True,
    )
    assert container.X.shape == (3, 1)
    assert container.meta["n_dropped_nan_inf"] == 0
    assert len(container.meta["dropped_feature_columns"]) == 2


def test_flat_loader_units_do_not_recreate_pruned_sensor_feature_pairs(tmp_path):
    columns = [
        "param_mean_alpha_peak_freq_ch-Fz",
        "param_mean_alpha_peak_freq_ch-Cz",
        "complexity_mean_sample_entropy_ch-Fz",
        "complexity_mean_sample_entropy_ch-Cz",
    ]
    table = pd.DataFrame(
        {
            "obs_id": [f"o{index}" for index in range(5)],
            columns[0]: [np.nan] * 5,
            columns[1]: [8.0, 8.5, 9.0, 9.5, 10.0],
            columns[2]: [0.1, 0.2, 0.3, 0.4, 0.5],
            columns[3]: [0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    table_path = _save(table, tmp_path / "features.parquet", columns)

    container = load_descriptor_table(
        table_path,
        drop_degenerate_columns=True,
        max_row_drop_rate=0.0,
    )

    assert container.dims == ("obs", "feature")
    assert columns[0] not in container.coords["feature"]
    assert np.isfinite(container.X).all()

    for mode in (
        "flat",
        "sensor",
        "subfamily",
        "sensor_within_subfamily",
        "descriptor",
        "descriptor_sensor",
    ):
        units = list(iter_analysis_units(container, mode, "descriptors"))
        assert units
        assert all(np.isfinite(unit["container"].X).all() for unit in units)

    descriptor_sensor_keys = {
        unit["unit_key"]
        for unit in iter_analysis_units(container, "descriptor_sensor", "descriptors")
    }
    assert "alpha_peak_freq_Cz" in descriptor_sensor_keys
    assert "alpha_peak_freq_Fz" not in descriptor_sensor_keys


def test_select_viable_feature_columns_missingness_boundary():
    columns = ["band_keep_ch-Fz", "band_drop_ch-Fz"]
    frame = pd.DataFrame(
        {
            columns[0]: [np.nan, 1.0, 2.0, 3.0, 4.0],
            columns[1]: [np.nan, np.nan, 2.0, 3.0, 4.0],
        }
    )

    surviving, log = select_viable_feature_columns(
        frame,
        columns,
        max_missing_rate=0.20,
        drop_constant=False,
    )

    assert surviving == [columns[0]]
    assert log["column"].tolist() == [columns[1]]
    assert log["drop_reason"].tolist() == ["missing_rate"]


def test_load_descriptor_table_drops_extreme(descriptor_files):
    table_path, feature_columns = descriptor_files
    table = pd.read_parquet(table_path)
    table.loc[2, "band_abs_alpha_ch-Fz"] = 1000.0
    _save(table, table_path, feature_columns)

    container = load_descriptor_table(
        table_path,
        descriptor_max_abs_value=100.0,
    )

    assert container.ids.tolist() == ["obs-1", "obs-2"]
    assert container.meta["n_rows_entering_qc"] == 3
    assert container.meta["n_dropped_nan_inf"] == 0
    assert container.meta["dropped_extreme_rows"] == 1


def test_load_descriptor_table_empty_after_filter_raises(descriptor_files):
    table_path, _ = descriptor_files

    with pytest.raises(RuntimeError, match="No rows survived filtering"):
        load_descriptor_table(table_path, condition="missing")


def test_load_descriptor_table_rejects_nonpositive_threshold(descriptor_files):
    table_path, _ = descriptor_files

    with pytest.raises(ValueError, match="must be positive"):
        load_descriptor_table(
            table_path,
            descriptor_max_abs_value=0,
        )


def test_save_descriptor_table_writes_one_self_describing_file(tmp_path):
    df = pd.DataFrame({"obs_id": ["o1"], "band_a_ch-Fz": [1.0]})
    save_descriptor_table(df, tmp_path / "test", feature_columns=["band_a_ch-Fz"])
    assert [path.name for path in tmp_path.iterdir()] == ["test.parquet"]
    assert pd.read_parquet(tmp_path / "test.parquet").attrs["feature_columns"] == [
        "band_a_ch-Fz"
    ]


def test_check_feature_column_consistency_extra():
    acc = {}
    check_feature_column_consistency(["a"], acc, "key1", "first.parquet")
    assert acc["key1"] == ["a"]
    with pytest.raises(ValueError, match=r"second\.parquet"):
        check_feature_column_consistency(["b"], acc, "key1", "second.parquet")


def test_load_descriptor_table_errors(tmp_path):
    columns = ["band_feat_ch-s1"]
    df = pd.DataFrame({"obs_id": ["1", "2"]})

    # No feature-column list in the table
    save_descriptor_table(df, tmp_path / "plain")
    with pytest.raises(ValueError, match="carries no feature-column list"):
        load_descriptor_table(tmp_path / "plain.parquet")

    table = _save(df, tmp_path / "tbl.parquet", columns)
    # Condition column not found
    with pytest.raises(ValueError):
        load_descriptor_table(table, condition="C")
    # Subject filter column not found
    with pytest.raises(ValueError):
        load_descriptor_table(table, subjects=["sub-1"])
    # Columns not found
    with pytest.raises(ValueError):
        load_descriptor_table(table)

    # Target column not found
    df["band_feat_ch-s1"] = [1, 2]
    table = _save(df, tmp_path / "tbl.parquet", columns)
    with pytest.raises(ValueError):
        load_descriptor_table(table, target_col="missing")
    # A family the table lacks, alone or next to one it has
    for families in (["other"], ["band", "other"]):
        with pytest.raises(ValueError, match=r"\['other'\] not found.*\['band'\]"):
            load_descriptor_table(table, descriptor_families=families)

    # No rows survived NaN/Inf filtering
    df["band_feat_ch-s1"] = [np.nan, np.inf]
    with pytest.raises(RuntimeError):
        load_descriptor_table(_save(df, tmp_path / "tbl.parquet", columns))

    # No rows survived extreme-value filtering
    df["band_feat_ch-s1"] = [100, 200]
    with pytest.raises(RuntimeError):
        load_descriptor_table(
            _save(df, tmp_path / "tbl.parquet", columns), descriptor_max_abs_value=50
        )

    # Cannot infer obs IDs
    df_noid = pd.DataFrame({"band_feat_ch-s1": [1, 2]})
    with pytest.raises(ValueError):
        load_descriptor_table(_save(df_noid, tmp_path / "tbl2.parquet", columns))


def test_load_descriptor_table_location_statistic(tmp_path):
    feature_columns = [
        "mean_band_log_abs_alpha_ch-Fz",
        "median_band_log_abs_alpha_ch-Fz",
        "iqr_band_log_abs_alpha_ch-Fz",
        "agg_band_ratio_theta_beta_ch-Fz",
    ]
    table = pd.DataFrame(
        {
            "obs_id": ["o1", "o2", "o3"],
            "subject": ["1", "2", "3"],
            "condition": ["b", "b", "b"],
            feature_columns[0]: [1.0, 2.0, 3.0],
            feature_columns[1]: [1.0, 2.0, 3.0],
            feature_columns[2]: [0.1, 0.2, 0.3],
            feature_columns[3]: [0.5, 0.6, 0.7],
        }
    )
    table_path = _save(table, tmp_path / "d.parquet", feature_columns)

    # median selected -> mean dropped; spread (iqr) and ratio kept
    feats = (
        load_descriptor_table(table_path, location_statistic="median")
        .coords["feature"]
        .tolist()
    )
    assert "median_band_log_abs_alpha_ch-Fz" in feats
    assert "mean_band_log_abs_alpha_ch-Fz" not in feats
    assert "iqr_band_log_abs_alpha_ch-Fz" in feats
    assert "agg_band_ratio_theta_beta_ch-Fz" in feats

    # mean selected -> median dropped
    feats_m = (
        load_descriptor_table(table_path, location_statistic="mean")
        .coords["feature"]
        .tolist()
    )
    assert "mean_band_log_abs_alpha_ch-Fz" in feats_m
    assert "median_band_log_abs_alpha_ch-Fz" not in feats_m

    # default (None) keeps both location statistics
    assert len(load_descriptor_table(table_path).coords["feature"].tolist()) == 4

    with pytest.raises(ValueError, match="location_statistic"):
        load_descriptor_table(table_path, location_statistic="mode")


def test_load_descriptor_table_exclude_subfamilies_partial(descriptor_files):
    from coco_pipe.descriptors.qc import descriptor_subfamily

    table_path, _ = descriptor_files
    drop = descriptor_subfamily("complexity", "sample_entropy")
    container = load_descriptor_table(table_path, exclude_subfamilies=[drop])
    features = [str(f) for f in container.coords["feature"]]
    assert all("sample_entropy" not in f for f in features)
    assert any("abs_alpha" in f for f in features)


def test_load_descriptor_table_exclude_all_subfamilies_raises(descriptor_files):
    from coco_pipe.descriptors.qc import descriptor_subfamily

    table_path, _ = descriptor_files
    excluded = [
        descriptor_subfamily("band", "abs_alpha"),
        descriptor_subfamily("complexity", "sample_entropy"),
    ]
    with pytest.raises(RuntimeError, match="excluding sub-families"):
        load_descriptor_table(table_path, exclude_subfamilies=excluded)


def test_load_descriptor_table_missing_id_column(tmp_path):
    feature_columns = ["band_abs_alpha_ch-Fz"]
    table = pd.DataFrame({"condition": ["x", "y"], feature_columns[0]: [1.0, 2.0]})
    table_path = _save(table, tmp_path / "no_id.parquet", feature_columns)
    with pytest.raises(ValueError, match="Cannot infer obs IDs"):
        load_descriptor_table(table_path)


def test_load_descriptor_table_all_columns_degenerate(tmp_path):
    feature_columns = ["band_abs_alpha_ch-Fz", "band_abs_beta_ch-Fz"]
    table = pd.DataFrame(
        {
            "obs_id": ["o1", "o2", "o3"],
            "subject": ["1", "2", "3"],
            feature_columns[0]: [1.0, 1.0, 1.0],  # constant -> degenerate
            feature_columns[1]: [2.0, 2.0, 2.0],  # constant -> degenerate
        }
    )
    table_path = _save(table, tmp_path / "degenerate.parquet", feature_columns)
    with pytest.raises(RuntimeError, match="survived column pruning"):
        load_descriptor_table(table_path, drop_degenerate_columns=True)


# --------------------------------------------------------------------------- #
# Column-contract relocation (parser now lives in descriptors.naming)
# --------------------------------------------------------------------------- #
def test_descriptors_qc_does_not_import_io_descriptors():
    # The relocation must break the io <-> descriptors cycle: descriptors.qc
    # parses via the domain module, not back up through io.descriptors.
    import inspect

    from coco_pipe.descriptors import qc

    assert "coco_pipe.io.descriptors" not in inspect.getsource(qc)


# --------------------------------------------------------------------------- #
# merge_descriptor_tables
# --------------------------------------------------------------------------- #
def test_merge_descriptor_tables_concats_and_saves(tmp_path):
    cols = ["band_abs_alpha_ch-Fz"]
    t1 = _save(
        pd.DataFrame({"subject": ["1"], cols[0]: [1.0]}), tmp_path / "s1.parquet", cols
    )
    t2 = _save(
        pd.DataFrame({"subject": ["2"], cols[0]: [2.0]}), tmp_path / "s2.parquet", cols
    )

    combined, feature_columns = merge_descriptor_tables(
        [t1, t2], out_base_path=tmp_path / "combined" / "sensor_epoch_features"
    )

    assert len(combined) == 2
    assert feature_columns == cols
    out = pd.read_parquet(tmp_path / "combined" / "sensor_epoch_features.parquet")
    assert out.attrs["feature_columns"] == cols
    assert [path.name for path in (tmp_path / "combined").iterdir()] == [
        "sensor_epoch_features.parquet"
    ]


def test_merge_descriptor_tables_checks_the_feature_columns(tmp_path):
    with pytest.raises(ValueError, match="at least one table path"):
        merge_descriptor_tables([])

    t1 = _save(
        pd.DataFrame({"x": [1.0]}), tmp_path / "a.parquet", ["band_abs_alpha_ch-Fz"]
    )
    t2 = _save(
        pd.DataFrame({"x": [2.0]}), tmp_path / "b.parquet", ["band_abs_beta_ch-Fz"]
    )
    with pytest.raises(ValueError, match="Feature column mismatch"):
        merge_descriptor_tables([t1, t2])

    save_descriptor_table(pd.DataFrame({"x": [1.0]}), tmp_path / "plain")
    with pytest.raises(ValueError, match="carries no feature-column list"):
        merge_descriptor_tables([tmp_path / "plain.parquet"])
