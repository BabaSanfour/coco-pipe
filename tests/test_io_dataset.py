import importlib
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import coco_pipe.io.dataset as dataset_mod
from coco_pipe.io.config import BIDSConfig, DatasetConfig, TabularConfig


def test_dataset_config_discriminator_and_defaults(tmp_path):
    tab_cfg = DatasetConfig(
        **{
            "dataset": {
                "mode": "tabular",
                "path": tmp_path / "table.csv",
                "select_kwargs": {"keep": ["f1"]},
            }
        }
    )
    assert isinstance(tab_cfg.dataset, TabularConfig)
    assert tab_cfg.dataset.sep == "\t"
    assert tab_cfg.dataset.select_kwargs == {"keep": ["f1"]}

    bids_cfg = DatasetConfig(**{"dataset": {"mode": "bids", "path": tmp_path}})
    assert isinstance(bids_cfg.dataset, BIDSConfig)
    assert bids_cfg.dataset.loading_mode == "epochs"

    with pytest.raises(ValidationError):
        DatasetConfig(**{"dataset": {"path": tmp_path}})


def test_tabular_dataset_columns_to_dims(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "target": [0, 1],
            "F1_T1": [1.0, 2.0],
            "F1_T2": [3.0, 4.0],
            "F2_T1": [5.0, 6.0],
            "F2_T2": [7.0, 8.0],
        }
    )

    file_path = tmp_path / "data.csv"
    file_path.write_text("placeholder")

    monkeypatch.setattr(dataset_mod.pd, "read_csv", lambda *args, **kwargs: df.copy())

    ds = dataset_mod.TabularDataset(
        file_path,
        target_col="target",
        columns_to_dims=["freq", "time"],
        col_sep="_",
    )
    container = ds.load()

    assert container.dims == ("obs", "freq", "time")
    assert container.coords["freq"].tolist() == ["F1", "F2"]
    assert container.coords["time"].tolist() == ["T1", "T2"]
    np.testing.assert_array_equal(container.y, df["target"].values)
    assert container.X.shape == (2, 2, 2)
    assert container.X[0, 0, 0] == 1.0
    assert container.X[1, 1, 1] == 8.0


def test_embedding_dataset_glob_concat(tmp_path):
    arr1 = np.arange(6).reshape(2, 3)
    arr2 = np.ones(
        (3,)
    )  # shape (3,) -> treated as single observation (1, 3) -> id "subjB"
    np.save(tmp_path / "subjA.npy", arr1)
    np.save(tmp_path / "subjB.npy", arr2)

    ds = dataset_mod.EmbeddingDataset(
        tmp_path,
        pattern="*.npy",
        dims=("feature",),
        reader=np.load,
        id_fn=lambda p: p.stem,
    )
    container = ds.load()

    assert container.dims == ("obs", "feature")
    assert container.X.shape == (3, 3)
    assert container.ids.tolist() == ["subjA_0", "subjA_1", "subjB"]


def test_bids_dataset_with_mocks(monkeypatch, tmp_path):
    fake_mne = types.SimpleNamespace()

    class FakeBIDSPath:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_mne_bids = types.SimpleNamespace(
        BIDSPath=FakeBIDSPath, read_raw_bids=lambda *a, **k: None
    )

    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    monkeypatch.setitem(sys.modules, "mne_bids", fake_mne_bids)

    dataset = importlib.reload(dataset_mod)

    monkeypatch.setattr(dataset, "detect_subjects", lambda root: ["01"])
    monkeypatch.setattr(dataset, "detect_sessions", lambda root, sub: [])
    monkeypatch.setattr(
        dataset, "load_participants_tsv", lambda root: {"01": {"age": 30}}
    )

    def fake_read_bids_entry(
        bids_path, is_pre_epoched, is_evoked, mode, window_length, stride
    ):
        data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        times = np.linspace(0, 1, 4)
        ch = np.array(["C1", "C2", "C3"])
        return data, times, ch, 100.0

    monkeypatch.setattr(dataset, "read_bids_entry", fake_read_bids_entry)

    ds = dataset.BIDSDataset(root=tmp_path, task="rest")
    container = ds.load()

    assert container.dims == ("obs", "channel", "time")
    assert container.X.shape == (2, 3, 4)
    assert container.coords["channel"].tolist() == ["C1", "C2", "C3"]
    assert container.coords["time"].tolist() == [
        0.0,
        0.3333333333333333,
        0.6666666666666666,
        1.0,
    ]
    assert container.ids.tolist() == ["01_0", "01_1"]
    assert container.meta["sfreq"] == 100.0
    assert container.meta["sfreq"] == 100.0
    assert container.coords["age"].tolist() == [30, 30]


def test_tabular_dataset_cleaning(tmp_path):
    # Create CSV with NaNs and Inf
    df = pd.DataFrame(
        {
            "good": [1.0, 2.0, 3.0],
            "bad_nan": [1.0, np.nan, 3.0],
            "bad_inf": [1.0, np.inf, 3.0],
        }
    )
    path = tmp_path / "dirty.csv"
    df.to_csv(path, index=False, sep="\t")

    # 1. Clean 'any' (default) -> should drop bad_nan and bad_inf
    ds = dataset_mod.TabularDataset(path, clean=True, clean_kwargs={"mode": "any"})
    container = ds.load()
    assert container.X.shape == (3, 1)
    assert container.coords["feature"][0] == "good"
    assert "dropped_columns" in container.meta["cleaning_report"]
    assert "bad_nan" in container.meta["cleaning_report"]["dropped_columns"]


def test_tabular_dataset_meta_columns(tmp_path):
    df = pd.DataFrame(
        {"feat1": [1, 2], "feat2": [3, 4], "age": [25, 30], "sex": ["M", "F"]}
    )
    path = tmp_path / "meta.csv"
    df.to_csv(path, index=False, sep="\t")

    ds = dataset_mod.TabularDataset(path, meta_columns=["age", "sex"])
    container = ds.load()

    # Features should just be feat1, feat2
    assert container.X.shape == (2, 2)
    assert "age" in container.coords
    assert "sex" in container.coords
    assert container.coords["age"].tolist() == [25, 30]


def test_embedding_dataset_recursive_and_mixed(tmp_path):
    # Structure:
    # root/
    #   sub1.pkl
    #   nested/
    #     sub2.pkl

    (tmp_path / "nested").mkdir()

    # Mock efficient reader by saving numpy arrays
    np.save(tmp_path / "sub1.npy", np.zeros((5, 10)))
    np.save(tmp_path / "nested" / "sub2.npy", np.zeros((5, 10)))

    ds = dataset_mod.EmbeddingDataset(
        tmp_path,
        pattern="*.npy",
        dims=("time", "channel"),  # 5x10
        reader=np.load,
        id_fn=lambda p: p.stem,
    )

    container = ds.load()
    assert container.X.shape == (2, 5, 10)
    # Check IDs (order might vary on FS, but both should be there)
    ids_set = set(container.ids)
    assert "sub1" in ids_set
    assert "sub2" in ids_set


def test_bids_dataset_mismatches(monkeypatch, tmp_path):
    # Test warnings for channel mismatch

    # Mock deps
    monkeypatch.setitem(sys.modules, "mne", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "mne_bids",
        types.SimpleNamespace(BIDSPath=lambda **k: k, read_raw_bids=lambda *a: None),
    )

    dataset = importlib.reload(dataset_mod)

    # One subject, two sessions
    monkeypatch.setattr(dataset, "detect_subjects", lambda root: ["01"])
    monkeypatch.setattr(dataset, "detect_sessions", lambda root, sub: ["ses1", "ses2"])
    monkeypatch.setattr(dataset, "load_participants_tsv", lambda root: {})

    # Mock reader to return different channels for ses2
    def fake_read(bids_path, **kwargs):
        ses = bids_path["session"]
        if ses == "ses1":
            ch = ["C1", "C2"]
            data = np.zeros((1, 2, 10))
        else:
            ch = ["C1", "C3"]  # Mismatch
            data = np.zeros((1, 2, 10))
        return data, np.arange(10), np.array(ch), 100.0

    monkeypatch.setattr(dataset, "read_bids_entry", fake_read)

    ds = dataset.BIDSDataset(root=tmp_path)
    container = ds.load()

    assert container.X.shape == (2, 2, 10)
    # First channel names prevail
    assert container.coords["channel"].tolist() == ["C1", "C2"]


def test_dataset_factory_errors(tmp_path):
    # Tabular: unknown extension
    # Create the file so exists() passes
    (tmp_path / "unknown.xyz").touch()
    with pytest.raises(ValueError, match="Unsupported file extension"):
        ds = dataset_mod.TabularDataset(tmp_path / "unknown.xyz")
        ds.load()

    # Embedding: no files found
    with pytest.raises(FileNotFoundError, match="No files matched"):
        ds = dataset_mod.EmbeddingDataset(tmp_path, pattern="*.nonexistent")
        ds.load()


# --- TabularDataset Tests ---


def test_tabular_excel_support(monkeypatch, tmp_path):
    """Test Excel file loading path."""
    # We mock read_excel to avoid needing openpyxl/xlrd
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: df)

    # Create dummy xlsx file
    p = tmp_path / "data.xlsx"
    p.touch()

    ds = dataset_mod.TabularDataset(p, target_col="b")
    container = ds.load()

    assert container.X.shape == (2, 1)  # col 'a'
    assert np.array_equal(container.y, [3, 4])
    assert container.meta["filename"] == "data.xlsx"


def test_tabular_non_numeric_warning(tmp_path, caplog):
    """Test warning for non-numeric columns."""
    df = pd.DataFrame({"a": [1, 2], "bad": ["x", "y"]})
    p = tmp_path / "warn.csv"
    df.to_csv(p, index=False)

    ds = dataset_mod.TabularDataset(p)
    with caplog.at_level("WARNING"):
        ds.load()

    assert "non-numeric columns" in caplog.text
    assert "bad" in caplog.text


def test_tabular_reshaping_errors(tmp_path):
    """Test errors during reshaping logic."""
    # Case 1: No columns match pattern
    df = pd.DataFrame({"A": [1], "B": [2]})
    p = tmp_path / "reshape_fail.csv"
    df.to_csv(p, index=False, sep="\t")

    ds = dataset_mod.TabularDataset(
        p,
        columns_to_dims=["dim1", "dim2"],
        col_sep="_",  # Expects A_B format, but cols are A, B
    )
    with pytest.raises(ValueError, match="No columns matched"):
        ds.load()

    # Case 2: Missing combinations (Shape mismatch)
    df2 = pd.DataFrame({"A_1": [1], "A_2": [2], "B_1": [3]})  # Missing B_2
    p2 = tmp_path / "reshape_incomplete.csv"
    df2.to_csv(p2, index=False, sep="\t")

    ds2 = dataset_mod.TabularDataset(
        p2, columns_to_dims=["L1", "L2"], col_sep="_"  # L1={A,B}, L2={1,2}
    )
    with pytest.raises(ValueError, match="Reshaping failed"):
        ds2.load()


def test_tabular_cleaning_advanced(tmp_path):
    """Test min_abs_value and sensor_wide cleaning."""
    # 1. min_abs_value
    df = pd.DataFrame({"big": [1.0, -1.0], "tiny": [1e-10, -1e-10]})
    X_clean, _ = dataset_mod.TabularDataset.clean(df, min_abs_value=1e-5)
    assert "tiny" not in X_clean.columns
    assert "big" in X_clean.columns

    # 2. sensor_wide mode
    df_sens = pd.DataFrame(
        {"C1_T1": [1.0], "C1_T2": [np.nan], "C2_T1": [1.0], "C2_T2": [1.0]}
    )

    X_clean_s, report = dataset_mod.TabularDataset.clean(
        df_sens,
        mode="sensor_wide",
        sep="_",
        reverse=True,  # C1_T1 -> split causes T1, C1. Feat=C1.
    )
    assert "C1_T1" not in X_clean_s.columns
    assert "C1_T2" not in X_clean_s.columns
    assert "C2_T1" in X_clean_s.columns
    assert "dropped_features" in report
    assert "C1" in report["dropped_features"]


# --- EmbeddingDataset Tests ---


def test_embedding_legacy_pattern():
    """Test BIDS-like pattern construction."""
    ds = dataset_mod.EmbeddingDataset("dummy", task="rest", run="01", processing="norm")
    assert "task-rest" in ds.pattern
    assert "run-01" in ds.pattern
    assert "embeddingsnorm.pkl" in ds.pattern


def test_embedding_filtering(tmp_path):
    """Test subject filtering."""
    # Create 3 files: sub-01, sub-02, sub-03
    for i in range(1, 4):
        p = tmp_path / f"sub-0{i}.pkl"
        p.touch()

    def mock_reader(p):
        return np.array([1, 2])  # 1 obs

    # 1. Select first 2 (int)
    ds_int = dataset_mod.EmbeddingDataset(
        tmp_path, dims=("f",), reader=mock_reader, subjects=2, id_fn=lambda p: p.stem
    )
    c_int = ds_int.load()
    assert len(c_int.ids) == 2

    # 2. Select specific list
    ds_list = dataset_mod.EmbeddingDataset(
        tmp_path,
        dims=("f",),
        reader=mock_reader,
        subjects=["sub-03"],  # Note: id_fn usually returns string
        id_fn=lambda p: p.stem,
    )
    c_list = ds_list.load()
    assert len(c_list.ids) == 1
    assert "sub-03" in c_list.ids[0]


def test_embedding_dict_loading(tmp_path):
    """Test loading dictionary content (multi-segment)."""
    import pickle

    data = {"seg1": np.zeros((2, 5)), "seg2": np.zeros((2, 5))}  # 2 obs per segment
    p = tmp_path / "sub-dict.pkl"
    with open(p, "wb") as f:
        pickle.dump(data, f)

    ds = dataset_mod.EmbeddingDataset(
        tmp_path,
        dims=("feature",),  # 5 features
        pattern="*.pkl",
        id_fn=lambda p: p.stem,
    )

    container = ds.load()
    assert container.X.shape == (4, 5)
    assert "sub-dict_seg1_0" in container.ids
    assert "sub-dict_seg2_1" in container.ids


def test_embedding_shape_mismatch(tmp_path, caplog):
    """Test warning on shape mismatch."""
    p = tmp_path / "bad_shape.npy"
    np.save(p, np.zeros((10, 10)))  # Shape (10, 10)

    p2 = tmp_path / "really_bad.npy"
    np.save(p2, np.zeros((2, 2, 2)))

    ds = dataset_mod.EmbeddingDataset(
        tmp_path, dims=("f",), pattern="really_bad.npy", reader=np.load
    )

    with caplog.at_level("WARNING"):
        try:
            ds.load()
        except RuntimeError:
            pass

    assert "Shape mismatch" in caplog.text


# --- BIDSDataset Tests ---


def test_bids_concatenation_failure(monkeypatch, tmp_path):
    """Test failure when subjects have different shapes."""
    monkeypatch.setattr(dataset_mod, "detect_subjects", lambda r: ["01", "02"])
    monkeypatch.setattr(dataset_mod, "detect_sessions", lambda r, s: [None])
    monkeypatch.setattr(dataset_mod, "load_participants_tsv", lambda r: {})

    # Mock read_bids_entry to return different shapes
    def fake_read(bids_path, **k):
        sub = bids_path.subject
        if sub == "01":
            return np.zeros((1, 5, 10)), [0], ["c"], 100
        else:
            return np.zeros((1, 6, 10)), [0], ["c"], 100  # Different channels count

    monkeypatch.setattr(dataset_mod, "read_bids_entry", fake_read)

    # Patch BIDSPath validation to avoid real FS checks
    class MockBIDSPath:
        def __init__(self, subject, **kwargs):
            self.subject = subject
            self.kwargs = kwargs

    monkeypatch.setattr(dataset_mod, "BIDSPath", MockBIDSPath)
    monkeypatch.setitem(sys.modules, "mne", MagicMock())
    monkeypatch.setitem(sys.modules, "mne_bids", MagicMock())

    ds = dataset_mod.BIDSDataset(tmp_path)

    with pytest.raises(ValueError, match="Concatenation failed"):
        ds.load()


def test_bids_time_warning(monkeypatch, tmp_path, caplog):
    """Test warning for time length mismatch."""
    monkeypatch.setattr(dataset_mod, "detect_subjects", lambda r: ["01", "02"])
    monkeypatch.setattr(dataset_mod, "detect_sessions", lambda r, s: [None])
    monkeypatch.setattr(dataset_mod, "load_participants_tsv", lambda r: {})

    # Mock read_bids_entry
    def fake_read(bids_path, **k):
        sub = bids_path.subject
        if sub == "01":
            return np.zeros((1, 1, 10)), np.arange(10), ["c"], 100
        else:
            # Different time length
            return np.zeros((1, 1, 11)), np.arange(11), ["c"], 100

    monkeypatch.setattr(dataset_mod, "read_bids_entry", fake_read)

    class MockBIDSPath:
        def __init__(self, subject, **kwargs):
            self.subject = subject
            self.kwargs = kwargs

    monkeypatch.setattr(dataset_mod, "BIDSPath", MockBIDSPath)
    monkeypatch.setitem(sys.modules, "mne", MagicMock())
    monkeypatch.setitem(sys.modules, "mne_bids", MagicMock())

    ds = dataset_mod.BIDSDataset(tmp_path)

    with pytest.raises(ValueError):
        ds.load()

    assert "Time length mismatch" in caplog.text
