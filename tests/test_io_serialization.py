"""Unit tests for the shared low-level serialization primitives."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from coco_pipe.io import _serialization as serialization_mod
from coco_pipe.io._serialization import (
    default_id_extractor,
    load_object,
    read_json,
    read_table,
    save_npz,
    save_object,
    smart_reader,
    write_json,
)


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def test_write_read_json_roundtrip(tmp_path):
    payload = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    path = tmp_path / "sub" / "payload.json"
    returned = write_json(path, payload)
    assert returned == path
    assert path.exists()
    assert read_json(path) == payload


def test_write_json_atomic_leaves_no_tmp(tmp_path):
    path = tmp_path / "p.json"
    write_json(path, {"x": 1}, atomic=True)
    write_json(path, {"x": 2}, atomic=True)
    assert read_json(path) == {"x": 2}
    # No leftover temp siblings.
    assert [p.name for p in tmp_path.iterdir()] == ["p.json"]


def test_write_json_sort_keys(tmp_path):
    path = tmp_path / "sorted.json"
    write_json(path, {"b": 1, "a": 2}, sort_keys=True)
    assert path.read_text(encoding="utf-8").index('"a"') < path.read_text(
        encoding="utf-8"
    ).index('"b"')


# --------------------------------------------------------------------------- #
# joblib object persistence
# --------------------------------------------------------------------------- #
def test_save_load_object_roundtrip(tmp_path):
    obj = {"array": np.arange(5), "label": "x"}
    path = tmp_path / "nested" / "obj.joblib"
    assert save_object(obj, path) == path
    loaded = load_object(path)
    assert loaded["label"] == "x"
    np.testing.assert_array_equal(loaded["array"], np.arange(5))


def test_load_object_expected_type_ok(tmp_path):
    path = tmp_path / "d.joblib"
    save_object({"k": 1}, path)
    assert load_object(path, expected_type=dict) == {"k": 1}


def test_load_object_expected_type_mismatch(tmp_path):
    path = tmp_path / "lst.joblib"
    save_object([1, 2, 3], path)
    with pytest.raises(TypeError, match="expected dict"):
        load_object(path, expected_type=dict)


def test_load_object_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_object(tmp_path / "nope.joblib")


# --------------------------------------------------------------------------- #
# NPZ
# --------------------------------------------------------------------------- #
def test_save_npz_atomic_overwrite(tmp_path):
    path = tmp_path / "arr.npz"
    save_npz(path, a=np.arange(3))
    save_npz(path, a=np.arange(3) + 10)
    with np.load(path) as npz:
        np.testing.assert_array_equal(npz["a"], np.arange(3) + 10)
    # No leftover temp siblings.
    assert [p.name for p in tmp_path.iterdir()] == ["arr.npz"]


def test_save_npz_object_payload_roundtrip(tmp_path):
    path = tmp_path / "diag.npz"
    save_npz(path, payload=np.asarray([{"k": "v"}], dtype=object))
    with np.load(path, allow_pickle=True) as npz:
        assert dict(npz["payload"][0]) == {"k": "v"}


# --------------------------------------------------------------------------- #
# Tabular reader
# --------------------------------------------------------------------------- #
def test_read_table_csv(tmp_path):
    path = tmp_path / "features.csv"
    path.write_text("subject;feature\nsub-1;1.5\nsub-2;2.5\n", encoding="utf-8")
    table = read_table(path)
    assert table.to_dict(orient="records") == [
        {"subject": "sub-1", "feature": 1.5},
        {"subject": "sub-2", "feature": 2.5},
    ]


def test_read_table_parquet(monkeypatch, tmp_path):
    path = tmp_path / "features.parquet"
    expected = pd.DataFrame({"subject": ["sub-1"], "feature": [1.5]})
    read_parquet = MagicMock(return_value=expected)
    monkeypatch.setattr(serialization_mod.pd, "read_parquet", read_parquet)
    table = read_table(path)
    read_parquet.assert_called_once_with(path)
    pd.testing.assert_frame_equal(table, expected)


def test_read_table_drops_unnamed_and_empty_columns(tmp_path):
    path = tmp_path / "features.csv"
    path.write_text("subject,feature,Unnamed: 2,empty\nsub-1,1.5,,\n", encoding="utf-8")
    table = read_table(path)
    assert table.columns.tolist() == ["subject", "feature"]


def test_read_table_explicit_separator(tmp_path):
    path = tmp_path / "features.csv"
    path.write_text("subject|feature\nsub-1|1.5\n", encoding="utf-8")
    table = read_table(path, sep="|")
    assert table.to_dict(orient="records") == [{"subject": "sub-1", "feature": 1.5}]


def test_read_table_unsupported_format_raises(tmp_path):
    path = tmp_path / "features.tsv"
    path.touch()
    with pytest.raises(ValueError, match=r"Expected \.csv or \.parquet"):
        read_table(path)


# --------------------------------------------------------------------------- #
# Blob reader + id extraction
# --------------------------------------------------------------------------- #
def test_default_id_extractor(tmp_path):
    assert default_id_extractor(tmp_path / "sub-01_task-rest.pkl") == "01"
    assert default_id_extractor(tmp_path / "patient_x.pkl") == "patient_x"


def test_smart_reader(tmp_path):
    import pickle

    p_pkl = tmp_path / "test.pkl"
    with open(p_pkl, "wb") as f:
        pickle.dump({"a": 1}, f)
    assert smart_reader(p_pkl) == {"a": 1}

    p_npy = tmp_path / "test.npy"
    np.save(p_npy, np.array([1, 2]))
    assert np.array_equal(smart_reader(p_npy), [1, 2])

    p_json = tmp_path / "test.json"
    p_json.write_text('{"key": "val"}')
    assert smart_reader(p_json) == {"key": "val"}

    p_bad = tmp_path / "test.xyz"
    p_bad.touch()
    with pytest.raises(ValueError, match="Unsupported extension"):
        smart_reader(p_bad)

    # H5 (mocked to avoid the dependency in this case).
    p_h5 = tmp_path / "test.h5"
    p_h5.touch()
    with patch.dict(sys.modules, {"h5py": MagicMock()}):
        m_h5 = sys.modules["h5py"]
        m_file = m_h5.File.return_value.__enter__.return_value
        m_file.keys.return_value = ["data"]
        m_file.__getitem__.return_value.__getitem__.return_value = "h5_data"
        assert smart_reader(p_h5) == "h5_data"


def test_smart_reader_h5(tmp_path):
    import h5py

    p1 = tmp_path / "test1.h5"
    with h5py.File(p1, "w") as f:
        f.create_dataset("embeddings", data=np.array([1, 2]))
    assert len(smart_reader(p1)) == 2

    p2 = tmp_path / "test2.h5"
    with h5py.File(p2, "w") as f:
        f.create_dataset("data", data=np.array([1, 2]))
    assert len(smart_reader(p2)) == 2

    p3 = tmp_path / "test3.h5"
    with h5py.File(p3, "w") as f:
        f.create_dataset("random", data=np.array([1, 2]))
    assert len(smart_reader(p3)) == 2

    p4 = tmp_path / "test4.h5"
    with h5py.File(p4, "w") as f:
        f.create_dataset("a", data=np.array([1]))
        f.create_dataset("b", data=np.array([2]))
    with pytest.raises(ValueError):
        smart_reader(p4)

    p5 = tmp_path / "test5.unknown"
    with pytest.raises(ValueError):
        smart_reader(p5)


def test_default_id_extractor_path_helper():
    assert default_id_extractor(Path("sub-0007_task-rest_emb.pkl")) == "0007"
    assert default_id_extractor(Path("plainfile.npy")) == "plainfile"


def test_write_json_non_atomic(tmp_path):
    path = tmp_path / "p_nonatomic.json"
    write_json(path, {"x": 1}, atomic=False)
    assert read_json(path) == {"x": 1}


def test_save_npz_non_atomic(tmp_path):
    path = tmp_path / "a_nonatomic.npz"
    save_npz(path, atomic=False, a=np.arange(3))
    with np.load(path) as npz:
        np.testing.assert_array_equal(npz["a"], np.arange(3))
