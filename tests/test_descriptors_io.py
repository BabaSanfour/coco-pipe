import json

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
    table_path = tmp_path / "descriptors.csv"
    columns_path = tmp_path / "feature_columns.json"
    table.to_csv(table_path, index=False)
    columns_path.write_text(json.dumps(feature_columns), encoding="utf-8")
    return table_path, columns_path, feature_columns


def test_load_descriptor_table_flat(descriptor_files):
    table_path, columns_path, feature_columns = descriptor_files

    container = load_descriptor_table(
        table_path,
        columns_path,
        condition="baseline",
        target_col="target",
        subjects=["1", "sub-2"],
    )

    assert container.dims == ("obs", "feature")
    assert container.X.shape == (2, 4)
    assert container.coords["feature"].tolist() == feature_columns
    assert container.ids.tolist() == ["obs-1", "obs-2"]
    assert container.y.tolist() == ["control", "case"]


def test_load_descriptor_table_detects_csv_delimiter(tmp_path):
    columns = ["band_abs_alpha_ch-Fz"]
    table_path = tmp_path / "descriptors.csv"
    columns_path = tmp_path / "feature_columns.json"
    table_path.write_text(
        "obs_id;subject;condition;band_abs_alpha_ch-Fz;Unnamed: 4\n"
        "obs-1;sub-1;baseline;1.25;\n",
        encoding="utf-8",
    )
    columns_path.write_text(json.dumps(columns), encoding="utf-8")

    container = load_descriptor_table(table_path, columns_path)

    assert container.X.tolist() == [[1.25]]
    assert "Unnamed: 4" not in container.coords


def test_load_descriptor_table_sensor_mode(descriptor_files):
    table_path, columns_path, _ = descriptor_files

    container = load_descriptor_table(
        table_path,
        columns_path,
        condition="baseline",
        analysis_mode="sensor",
    )

    assert container.dims == ("obs", "sensor", "feature")
    assert container.X.shape == (2, 2, 2)
    assert container.coords["sensor"].tolist() == ["Fz", "Cz"]
    assert container.coords["feature"].tolist() == [
        "abs_alpha",
        "sample_entropy",
    ]
    assert container.coords["feature_family"].tolist() == [
        "band",
        "complexity",
    ]
    np.testing.assert_allclose(container.X[0], [[1.0, 0.1], [1.1, 0.4]])


def test_load_descriptor_table_family_filter(descriptor_files):
    table_path, columns_path, _ = descriptor_files

    container = load_descriptor_table(
        table_path,
        columns_path,
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
    table_path, columns_path, _ = descriptor_files
    table = pd.read_csv(table_path)
    table.loc[1, "band_abs_alpha_ch-Fz"] = np.nan
    table.to_csv(table_path, index=False)

    container = load_descriptor_table(table_path, columns_path)

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

    table_path = tmp_path / "features.csv"
    columns_path = tmp_path / "columns.json"
    loader_frame = frame.copy()
    loader_frame[columns[1]] = [np.nan, np.nan, 7.0]
    loader_frame.to_csv(table_path, index=False)
    columns_path.write_text(json.dumps(columns), encoding="utf-8")
    container = load_descriptor_table(
        table_path,
        columns_path,
        drop_degenerate_columns=True,
    )
    assert container.X.shape == (3, 1)
    assert container.meta["n_dropped_nan_inf"] == 0
    assert len(container.meta["dropped_feature_columns"]) == 2


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
    table_path, columns_path, _ = descriptor_files
    table = pd.read_csv(table_path)
    table.loc[2, "band_abs_alpha_ch-Fz"] = 1000.0
    table.to_csv(table_path, index=False)

    container = load_descriptor_table(
        table_path,
        columns_path,
        descriptor_max_abs_value=100.0,
    )

    assert container.ids.tolist() == ["obs-1", "obs-2"]
    assert container.meta["n_rows_entering_qc"] == 3
    assert container.meta["n_dropped_nan_inf"] == 0
    assert container.meta["dropped_extreme_rows"] == 1


def test_load_descriptor_table_empty_after_filter_raises(descriptor_files):
    table_path, columns_path, _ = descriptor_files

    with pytest.raises(RuntimeError, match="No rows survived filtering"):
        load_descriptor_table(table_path, columns_path, condition="missing")


def test_load_descriptor_table_rejects_nonpositive_threshold(descriptor_files):
    table_path, columns_path, _ = descriptor_files

    with pytest.raises(ValueError, match="must be positive"):
        load_descriptor_table(
            table_path,
            columns_path,
            descriptor_max_abs_value=0,
        )


def test_save_descriptor_table_extra(tmp_path):
    df = pd.DataFrame({"a": [1]})
    # Default writes parquet only (csv is opt-in to halve the footprint).
    save_descriptor_table(df, tmp_path / "test", feature_columns=["a"])
    assert (tmp_path / "test.parquet").exists()
    assert not (tmp_path / "test.csv").exists()
    assert (tmp_path / "test_feature_columns.json").exists()

    # Opt in to the human-readable csv alongside parquet.
    save_descriptor_table(df, tmp_path / "both", formats=("parquet", "csv"))
    assert (tmp_path / "both.parquet").exists()
    assert (tmp_path / "both.csv").exists()


def test_save_descriptor_table_rejects_bad_formats(tmp_path):
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="parquet"):
        save_descriptor_table(df, tmp_path / "x", formats=())
    with pytest.raises(ValueError, match="parquet"):
        save_descriptor_table(df, tmp_path / "x", formats=("feather",))


def test_check_feature_column_consistency_extra(tmp_path):
    d1 = tmp_path / "d1"
    d1.mkdir()
    (d1 / "f.json").write_text('["a"]')

    acc = {}
    check_feature_column_consistency(d1, "f.json", acc, "key1")
    assert acc["key1"] == ["a"]

    (d1 / "f2.json").write_text('["b"]')
    with pytest.raises(ValueError):
        check_feature_column_consistency(d1, "f2.json", acc, "key1")


def test_load_descriptor_table_errors(tmp_path):
    df = pd.DataFrame({"obs_id": ["1", "2"]})
    df.to_csv(tmp_path / "tbl.csv", index=False)

    feat_json = tmp_path / "tbl_feature_columns.json"
    feat_json.write_text('["band_feat_ch-s1"]')

    # Condition column not found
    with pytest.raises(ValueError):
        load_descriptor_table(tmp_path / "tbl.csv", feat_json, condition="C")

    # Subject filter column not found
    with pytest.raises(ValueError):
        load_descriptor_table(tmp_path / "tbl.csv", feat_json, subjects=["sub-1"])

    # Expected JSON list
    feat_json.write_text("{}")
    with pytest.raises(ValueError):
        load_descriptor_table(tmp_path / "tbl.csv", feat_json)

    # Columns not found
    feat_json.write_text('["band_feat_ch-s1"]')
    with pytest.raises(ValueError):
        load_descriptor_table(tmp_path / "tbl.csv", feat_json)

    # Target column not found
    df["band_feat_ch-s1"] = [1, 2]
    df.to_csv(tmp_path / "tbl.csv", index=False)
    with pytest.raises(ValueError):
        load_descriptor_table(tmp_path / "tbl.csv", feat_json, target_col="missing")

    # No features matched descriptor_families
    with pytest.raises(RuntimeError):
        load_descriptor_table(
            tmp_path / "tbl.csv", feat_json, descriptor_families=["other"]
        )

    # No rows survived NaN/Inf filtering
    df["band_feat_ch-s1"] = [np.nan, np.inf]
    df.to_csv(tmp_path / "tbl.csv", index=False)
    with pytest.raises(RuntimeError):
        load_descriptor_table(tmp_path / "tbl.csv", feat_json)

    # No rows survived extreme-value filtering
    df["band_feat_ch-s1"] = [100, 200]
    df.to_csv(tmp_path / "tbl.csv", index=False)
    with pytest.raises(RuntimeError):
        load_descriptor_table(
            tmp_path / "tbl.csv", feat_json, descriptor_max_abs_value=50
        )

    # Cannot infer obs IDs
    df_noid = pd.DataFrame({"band_feat_ch-s1": [1, 2]})
    df_noid.to_csv(tmp_path / "tbl2.csv", index=False)
    with pytest.raises(ValueError):
        load_descriptor_table(tmp_path / "tbl2.csv", feat_json)


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
    table_path = tmp_path / "d.csv"
    cols_path = tmp_path / "c.json"
    table.to_csv(table_path, index=False)
    cols_path.write_text(json.dumps(feature_columns), encoding="utf-8")

    # median selected -> mean dropped; spread (iqr) and ratio kept
    feats = (
        load_descriptor_table(table_path, cols_path, location_statistic="median")
        .coords["feature"]
        .tolist()
    )
    assert "median_band_log_abs_alpha_ch-Fz" in feats
    assert "mean_band_log_abs_alpha_ch-Fz" not in feats
    assert "iqr_band_log_abs_alpha_ch-Fz" in feats
    assert "agg_band_ratio_theta_beta_ch-Fz" in feats

    # mean selected -> median dropped
    feats_m = (
        load_descriptor_table(table_path, cols_path, location_statistic="mean")
        .coords["feature"]
        .tolist()
    )
    assert "mean_band_log_abs_alpha_ch-Fz" in feats_m
    assert "median_band_log_abs_alpha_ch-Fz" not in feats_m

    # default (None) keeps both location statistics
    assert (
        len(load_descriptor_table(table_path, cols_path).coords["feature"].tolist())
        == 4
    )

    with pytest.raises(ValueError, match="location_statistic"):
        load_descriptor_table(table_path, cols_path, location_statistic="mode")


def test_load_descriptor_table_exclude_subfamilies_partial(descriptor_files):
    from coco_pipe.descriptors.qc import descriptor_subfamily

    table_path, columns_path, _ = descriptor_files
    drop = descriptor_subfamily("complexity", "sample_entropy")
    container = load_descriptor_table(
        table_path, columns_path, exclude_subfamilies=[drop]
    )
    features = [str(f) for f in container.coords["feature"]]
    assert all("sample_entropy" not in f for f in features)
    assert any("abs_alpha" in f for f in features)


def test_load_descriptor_table_exclude_all_subfamilies_raises(descriptor_files):
    from coco_pipe.descriptors.qc import descriptor_subfamily

    table_path, columns_path, _ = descriptor_files
    excluded = [
        descriptor_subfamily("band", "abs_alpha"),
        descriptor_subfamily("complexity", "sample_entropy"),
    ]
    with pytest.raises(RuntimeError, match="excluding sub-families"):
        load_descriptor_table(table_path, columns_path, exclude_subfamilies=excluded)


def test_load_descriptor_table_missing_id_column(tmp_path):
    feature_columns = ["band_abs_alpha_ch-Fz"]
    table = pd.DataFrame({"condition": ["x", "y"], feature_columns[0]: [1.0, 2.0]})
    table_path = tmp_path / "no_id.csv"
    cols_path = tmp_path / "no_id_cols.json"
    table.to_csv(table_path, index=False)
    cols_path.write_text(json.dumps(feature_columns), encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot infer obs IDs"):
        load_descriptor_table(table_path, cols_path)


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
    table_path = tmp_path / "degenerate.csv"
    cols_path = tmp_path / "degenerate_cols.json"
    table.to_csv(table_path, index=False)
    cols_path.write_text(json.dumps(feature_columns), encoding="utf-8")
    with pytest.raises(RuntimeError, match="survived column pruning"):
        load_descriptor_table(table_path, cols_path, drop_degenerate_columns=True)


# --------------------------------------------------------------------------- #
# Column-contract relocation (parser now lives in descriptors.naming)
# --------------------------------------------------------------------------- #
def test_descriptors_qc_does_not_import_io_descriptors():
    # The relocation must break the io <-> descriptors cycle: descriptors.qc
    # parses via the domain module, not back up through io.descriptors.
    import inspect

    import coco_pipe.descriptors.qc as qc

    assert "coco_pipe.io.descriptors" not in inspect.getsource(qc)


# --------------------------------------------------------------------------- #
# merge_descriptor_tables
# --------------------------------------------------------------------------- #
def _write_shard(tmp_path, name, df, feature_columns):
    base = tmp_path / name
    df.to_parquet(base.with_suffix(".parquet"), index=False)
    sidecar = tmp_path / f"{name}_feature_columns.json"
    sidecar.write_text(json.dumps(feature_columns), encoding="utf-8")
    return base.with_suffix(".parquet"), sidecar


def test_merge_descriptor_tables_concats_and_saves(tmp_path):
    cols = ["band_abs_alpha_ch-Fz"]
    t1, s1 = _write_shard(
        tmp_path, "shard1", pd.DataFrame({"subject": ["1"], cols[0]: [1.0]}), cols
    )
    t2, s2 = _write_shard(
        tmp_path, "shard2", pd.DataFrame({"subject": ["2"], cols[0]: [2.0]}), cols
    )

    combined, feature_columns = merge_descriptor_tables(
        [t1, t2],
        [s1, s2],
        out_base_path=tmp_path / "combined" / "sensor_epoch_features",
    )

    assert len(combined) == 2
    assert feature_columns == cols
    out = tmp_path / "combined" / "sensor_epoch_features.parquet"
    assert out.exists()
    assert (
        tmp_path / "combined" / "sensor_epoch_features_feature_columns.json"
    ).exists()


def test_merge_descriptor_tables_without_sidecars(tmp_path):
    df = pd.DataFrame({"subject": ["1"], "band_abs_alpha_ch-Fz": [1.0]})
    t1, _ = _write_shard(tmp_path, "a", df, ["band_abs_alpha_ch-Fz"])
    t2, _ = _write_shard(tmp_path, "b", df, ["band_abs_alpha_ch-Fz"])
    combined, feature_columns = merge_descriptor_tables([t1, t2])
    assert len(combined) == 2
    assert feature_columns is None


def test_merge_descriptor_tables_mismatched_sidecars_raise(tmp_path):
    t1, s1 = _write_shard(
        tmp_path, "a", pd.DataFrame({"x": [1.0]}), ["band_abs_alpha_ch-Fz"]
    )
    t2, s2 = _write_shard(
        tmp_path, "b", pd.DataFrame({"x": [2.0]}), ["band_abs_beta_ch-Fz"]
    )
    with pytest.raises(ValueError, match="Feature column mismatch"):
        merge_descriptor_tables([t1, t2], [s1, s2])


def test_merge_descriptor_tables_validates_inputs(tmp_path):
    with pytest.raises(ValueError, match="at least one table path"):
        merge_descriptor_tables([])

    t1, s1 = _write_shard(tmp_path, "a", pd.DataFrame({"x": [1.0]}), ["c"])
    with pytest.raises(ValueError, match="must align"):
        merge_descriptor_tables([t1], [s1, s1])
