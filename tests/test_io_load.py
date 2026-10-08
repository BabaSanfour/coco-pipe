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
            clean_kwargs={},
            select_kwargs={},
        )

    # BIDS
    with patch("coco_pipe.io.load.BIDSDataset") as mock_bids:
        load_mod.load_data(p, mode="bids", task="rest")
        mock_bids.assert_called_with(
            root=p,
            mode="epochs",
            task="rest",
            session=None,
            runs=None,
            datatype="eeg",
            suffix=None,
            target_col=None,
            window_length=None,
            stride=None,
            event_id=None,
            tmin=-0.2,
            tmax=0.5,
            baseline=None,
            units=None,
            description=None,
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
            task=None,
            run=None,
            processing=None,
            reader=None,
            id_fn=None,
            subjects=None,
        )


def test_load_data_error():
    """Test invalid mode error."""
    with pytest.raises(ValueError, match="Unknown mode"):
        load_mod.load_data("dummy", mode="invalid")


def test_load_data_bids_saved_epochs(tmp_path, write_bids_epochs):
    write_bids_epochs(tmp_path, "0001", description="base")
    meta_df = dataset_mod.pd.DataFrame(
        {"Study ID": [1], "age": [42], "group": ["case"]}
    )

    container = load_mod.load_data(
        tmp_path,
        mode="bids",
        task="rest",
        suffix="epo",
        loading_mode="load_existing",
        description="base",
        subject_metadata_df=meta_df,
        subject_key="Study ID",
        target_col="group",
    )

    assert container.X.shape == (2, 2, 20)
    assert container.ids.tolist() == [
        "sub-0001_task-rest_epoch-0000",
        "sub-0001_task-rest_epoch-0001",
    ]
    assert container.coords["Study ID"].tolist() == [1, 1]
    assert container.coords["age"].tolist() == [42, 42]
    assert container.y.tolist() == ["case", "case"]

    with pytest.raises(ValueError, match="subject_key must be provided"):
        load_mod.load_data(
            tmp_path, mode="bids", suffix="epo", subject_metadata_df=meta_df
        )


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


# --------------------------------------------------------------------------- #
# config <-> load_data wiring
# --------------------------------------------------------------------------- #
def _write_tabular_csv(tmp_path):
    import pandas as pd

    path = tmp_path / "feat.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "y": [0, 1, 0]}).to_csv(
        path, index=False
    )
    return path


def test_load_data_config_object_matches_kwargs(tmp_path):
    """The config= path produces the same container as the kwargs path."""
    from coco_pipe.io.config import DatasetConfig, TabularConfig

    path = _write_tabular_csv(tmp_path)

    via_kwargs = load_mod.load_data(path, mode="tabular", sep=",", target_col="y")
    via_config = load_mod.load_data(
        config=TabularConfig(path=path, sep=",", target_col="y")
    )
    via_wrapper = load_mod.load_data(
        config=DatasetConfig(
            dataset={
                "mode": "tabular",
                "path": str(path),
                "sep": ",",
                "target_col": "y",
            }
        )
    )

    for container in (via_kwargs, via_config, via_wrapper):
        assert container.dims == ("obs", "feature")
        assert list(container.coords["feature"]) == ["a", "b"]
        np.testing.assert_array_equal(container.y, [0, 1, 0])


def test_load_data_invalid_kwargs_raise_validation_error(tmp_path):
    from pydantic import ValidationError

    path = _write_tabular_csv(tmp_path)
    with pytest.raises(ValidationError):
        load_mod.load_data(path, mode="tabular", clean_kwargs="not_a_dict")


def test_load_data_requires_path_without_config():
    with pytest.raises(ValueError, match="`path` is required"):
        load_mod.load_data(mode="tabular")


def test_load_data_auto_embedding_dir(tmp_path):
    """A non-BIDS directory with blob files is inferred as embedding."""
    import pickle

    (tmp_path / "rec01_emb.pkl").write_bytes(pickle.dumps(np.ones((1, 4))))
    container = load_mod.load_data(tmp_path, mode="auto", dims=("feature",))
    assert container.dims == ("obs", "feature")


def test_load_data_tabular_index_col(tmp_path):
    """``index_col`` is surfaced as the obs coordinate."""
    import pandas as pd

    path = tmp_path / "t.csv"
    pd.DataFrame({"id": ["a", "b"], "f0": [1.0, 2.0]}).to_csv(path, index=False)
    container = load_mod.load_data(path, mode="tabular", sep=",", index_col="id")
    assert list(container.coords["obs"]) == ["a", "b"]
