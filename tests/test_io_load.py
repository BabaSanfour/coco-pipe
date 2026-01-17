from pathlib import Path
from unittest.mock import patch

import pytest

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
            window_length=None,
            stride=None,
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
