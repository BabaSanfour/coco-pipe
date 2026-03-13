import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import coco_pipe.io.dataset as dataset_mod
import coco_pipe.io.load as load_mod


def test_load_data_auto_mode(monkeypatch, tmp_path):
    """Test auto mode inference."""
    # 1. Tabular inference
    p_csv = tmp_path / "data.csv"
    p_csv.touch()

    with patch("coco_pipe.io.load.TabularDataset") as mock_tab:
        mock_tab.return_value.load.return_value = "loaded_tab"
        res = load_mod.load_data(p_csv, mode="auto")
        assert res == "loaded_tab"
        mock_tab.assert_called_once()

    # 2. Embedding inference (unknown suffix)
    p_pkl = tmp_path / "data.pkl"
    p_pkl.touch()

    with patch("coco_pipe.io.load.EmbeddingDataset") as mock_emb:
        mock_emb.return_value.load.return_value = "loaded_emb"
        res = load_mod.load_data(p_pkl, mode="auto")
        assert res == "loaded_emb"

    # 3. BIDS inference (directory with dataset_description.json)
    p_bids = tmp_path / "bids_root"
    p_bids.mkdir()
    (p_bids / "dataset_description.json").touch()

    with patch("coco_pipe.io.load.BIDSDataset") as mock_bids:
        mock_bids.return_value.load.return_value = "loaded_bids"
        res = load_mod.load_data(p_bids, mode="auto")
        assert res == "loaded_bids"


def test_load_data_explicit_modes():
    """Test explicit mode dispatch."""
    p = Path("dummy")

    # Tabular
    with patch("coco_pipe.io.load.TabularDataset") as mock_tab:
        load_mod.load_data(p, mode="tabular", sep=",")
        mock_tab.assert_called_with(
            path=p,
            target_col=None,
            index_col=None,
            sep=",",
            header=0,
            sheet_name=0,
            columns_to_dims=None,
            col_sep="_",
            meta_columns=None,
            clean=False,
            clean_kwargs=None,
        )

    # BIDS
    with patch("coco_pipe.io.load.BIDSDataset") as mock_bids:
        load_mod.load_data(p, mode="bids", task="rest")
        mock_bids.assert_called_with(
            root=p,
            mode="epochs",
            task="rest",
            session=None,
            datatype="eeg",
            suffix=None,
            target_col=None,
            window_length=None,
            stride=None,
            subject_metadata_df=None,
            subject_key=None,
            subjects=None,
        )

    # Embedding
    with patch("coco_pipe.io.load.EmbeddingDataset") as mock_emb:
        load_mod.load_data(p, mode="embedding", pattern="*.npy")
        mock_emb.assert_called_with(
            path=p,
            pattern="*.npy",
            dims=("obs", "feature"),
            coords=None,
            reader=None,
            id_fn=None,
            subjects=None,
        )


def test_load_data_error():
    """Test invalid mode error."""
    with pytest.raises(ValueError, match="Unknown mode"):
        load_mod.load_data("dummy", mode="invalid")


def test_load_data_bids_pre_epoched_load_existing(monkeypatch, tmp_path):
    epo_path = tmp_path / "sub-0001" / "eeg" / "sub-0001_task-rest_epo.fif"
    epo_path.parent.mkdir(parents=True)
    epo_path.touch()
    meta_df = np.array([["0001", 42, "case"]], dtype=object)

    monkeypatch.setattr(dataset_mod, "detect_subjects", lambda root: ["0001"])
    monkeypatch.setattr(dataset_mod, "detect_sessions", lambda root, sub: [])
    monkeypatch.setattr(
        dataset_mod,
        "detect_runs",
        lambda root, sub, ses, task, datatype: [None],
    )
    monkeypatch.setattr(dataset_mod, "load_participants_tsv", lambda root: {})
    monkeypatch.setattr(
        dataset_mod,
        "_get_bids_path",
        lambda: (
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("BIDSPath should not be used for precomputed epochs")
            )
        ),
    )

    def fake_read_bids_entry(bids_path, **kwargs):
        assert bids_path.fpath == epo_path
        assert kwargs["is_pre_epoched"] is True
        data = np.zeros((2, 1, 4))
        times = np.arange(4)
        return data, times, ["C1"], 100.0, np.array([7, 7])

    monkeypatch.setattr(dataset_mod, "read_bids_entry", fake_read_bids_entry)

    container = load_mod.load_data(
        tmp_path,
        mode="bids",
        task="rest",
        suffix="epo",
        loading_mode="load_existing",
        subject_metadata_df=dataset_mod.pd.DataFrame(
            meta_df, columns=["Study ID", "age", "group"]
        ),
        subject_key="Study ID",
    )

    assert container.X.shape == (2, 1, 4)
    assert container.ids.tolist() == ["0001_0", "0001_1"]
    assert container.y.tolist() == [7, 7]
    assert container.coords["Study ID"].tolist() == ["0001", "0001"]
    assert container.coords["age"].tolist() == [42, 42]
    assert container.coords["group"].tolist() == ["case", "case"]


def test_load_data_bids_target_col_from_metadata(monkeypatch, tmp_path):
    epo_path = tmp_path / "sub-0001" / "eeg" / "sub-0001_task-rest_epo.fif"
    epo_path.parent.mkdir(parents=True)
    epo_path.touch()
    meta_df = np.array([["0001", "stimulant"]], dtype=object)

    monkeypatch.setattr(dataset_mod, "detect_subjects", lambda root: ["0001"])
    monkeypatch.setattr(dataset_mod, "detect_sessions", lambda root, sub: [])
    monkeypatch.setattr(
        dataset_mod,
        "detect_runs",
        lambda root, sub, ses, task, datatype: [None],
    )
    monkeypatch.setattr(dataset_mod, "load_participants_tsv", lambda root: {})
    monkeypatch.setattr(
        dataset_mod,
        "_get_bids_path",
        lambda: (
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("BIDSPath should not be used for precomputed epochs")
            )
        ),
    )

    def fake_read_bids_entry(bids_path, **kwargs):
        assert bids_path.fpath == epo_path
        return np.zeros((2, 1, 4)), np.arange(4), ["C1"], 100.0, np.array([7, 7])

    monkeypatch.setattr(dataset_mod, "read_bids_entry", fake_read_bids_entry)

    container = load_mod.load_data(
        tmp_path,
        mode="bids",
        task="rest",
        suffix="epo",
        loading_mode="load_existing",
        subject_metadata_df=dataset_mod.pd.DataFrame(
            meta_df, columns=["Study ID", "psychostimulant_category"]
        ),
        subject_key="Study ID",
        target_col="psychostimulant_category",
    )

    assert container.coords["psychostimulant_category"].tolist() == [
        "stimulant",
        "stimulant",
    ]
    assert container.y.tolist() == ["stimulant", "stimulant"]


def test_io_import_is_lightweight(monkeypatch):
    module_names = [
        "coco_pipe.io",
        "coco_pipe.io.load",
        "coco_pipe.io.structures",
        "coco_pipe.io.utils",
    ]
    cached_modules = {name: sys.modules.get(name) for name in module_names}

    for module_name in module_names:
        sys.modules.pop(module_name, None)

    monkeypatch.setitem(sys.modules, "mne", None)
    monkeypatch.setitem(sys.modules, "mne_bids", None)

    try:
        io_mod = importlib.import_module("coco_pipe.io")
        assert hasattr(io_mod, "DataContainer")
        assert callable(io_mod.load_data)
        assert "load" not in getattr(io_mod, "__all__", [])
    finally:
        for module_name, module_obj in cached_modules.items():
            if module_obj is not None:
                sys.modules[module_name] = module_obj
