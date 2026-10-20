"""Tests for the shared serialization helpers."""

import numpy as np
import pandas as pd
import pytest

from coco_pipe.io._serialization import (
    atomic_path,
    load_object,
    read_json,
    read_table,
    save_npz,
    save_object,
    write_json,
)


@pytest.mark.parametrize("atomic", [True, False])
def test_json_round_trip(tmp_path, atomic):
    path = tmp_path / "sub" / "payload.json"
    payload = {"b": [1, 2, 3], "a": {"nested": True}}
    assert write_json(path, payload, atomic=atomic, sort_keys=True) == path
    write_json(path, payload, atomic=atomic, sort_keys=True)  # overwrite
    assert read_json(path) == payload
    text = path.read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"b"')
    assert [p.name for p in path.parent.iterdir()] == ["payload.json"]


def test_writers_of_one_file_do_not_share_a_temporary_file(tmp_path):
    path = tmp_path / "dataset_description.json"
    with atomic_path(path) as first:
        first.write_text("first", encoding="utf-8")
        with atomic_path(path) as second:
            second.write_text("second", encoding="utf-8")
        assert path.read_text(encoding="utf-8") == "second"
    assert path.read_text(encoding="utf-8") == "first"
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_object_round_trip(tmp_path):
    path = tmp_path / "nested" / "obj.joblib"
    assert save_object({"array": np.arange(5), "label": "x"}, path) == path
    loaded = load_object(path, expected_type=dict)
    np.testing.assert_array_equal(loaded["array"], np.arange(5))

    save_object([1, 2, 3], tmp_path / "list.joblib")
    with pytest.raises(TypeError, match="expected dict"):
        load_object(tmp_path / "list.joblib", expected_type=dict)
    with pytest.raises(FileNotFoundError):
        load_object(tmp_path / "nope.joblib")


@pytest.mark.parametrize("atomic", [True, False])
def test_npz_round_trip(tmp_path, atomic):
    path = tmp_path / "arr.npz"
    save_npz(path, atomic=atomic, a=np.arange(3))
    save_npz(
        path,
        atomic=atomic,
        a=np.arange(3) + 10,
        payload=np.asarray([{"k": "v"}], dtype=object),
    )
    with np.load(path, allow_pickle=True) as npz:
        np.testing.assert_array_equal(npz["a"], np.arange(3) + 10)
        assert dict(npz["payload"][0]) == {"k": "v"}
    assert [p.name for p in tmp_path.iterdir()] == ["arr.npz"]


def test_read_table(tmp_path):
    csv = tmp_path / "features.csv"
    csv.write_text("subject;feature\nsub-1;1.5\nsub-2;2.5\n", encoding="utf-8")
    assert read_table(csv).to_dict(orient="records") == [
        {"subject": "sub-1", "feature": 1.5},
        {"subject": "sub-2", "feature": 2.5},
    ]
    csv.write_text("subject|feature\nsub-1|1.5\n", encoding="utf-8")
    assert read_table(csv, sep="|").columns.tolist() == ["subject", "feature"]
    # unnamed and empty columns are dropped
    csv.write_text("subject,feature,Unnamed: 2,empty\nsub-1,1.5,,\n", encoding="utf-8")
    assert read_table(csv).columns.tolist() == ["subject", "feature"]

    parquet = tmp_path / "features.parquet"
    expected = pd.DataFrame({"subject": ["sub-1"], "feature": [1.5]})
    expected.to_parquet(parquet, index=False)
    pd.testing.assert_frame_equal(read_table(parquet), expected)

    (tmp_path / "features.tsv").touch()
    with pytest.raises(ValueError, match=r"Expected \.csv or \.parquet"):
        read_table(tmp_path / "features.tsv")
