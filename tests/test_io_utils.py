import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import coco_pipe.io.utils as utils_mod

# --- DataFrame Utilities ---


def test_row_quality_score():
    """Test badness score calculation."""
    df = pd.DataFrame({"a": [1, np.nan, 0, np.inf], "b": [1, 1, 1, 1]})
    # Row 0: a=1, b=1 -> score 0
    # Row 1: a=NaN -> score 1
    # Row 2: a=0 -> score 1 (if count_zero=True)
    # Row 3: a=Inf -> score 1

    scores = utils_mod.row_quality_score(df, count_zero=True)
    assert np.array_equal(scores, [0, 1, 1, 1])

    scores_nz = utils_mod.row_quality_score(df, count_zero=False)
    assert np.array_equal(scores_nz, [0, 1, 0, 1])


def test_make_strata():
    """Test stratification label generation."""
    df = pd.DataFrame({"num": [1, 10, 100], "cat": ["a", "b", "a"]})
    # Numeric binning (3 bins) + cat
    # 1 -> bin0, 10 -> bin1, 100 -> bin2 (roughly)

    strata = utils_mod.make_strata(df, covariates=["num", "cat"], n_bins=3)
    assert len(strata) == 3
    assert len(strata.unique()) == 3  # All distinct combos


def test_sample_indices():
    """Test index sampling logic."""
    df = pd.DataFrame({"target": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
    rng = np.random.default_rng(42)
    size_map = {"A": 1, "B": 2}  # Downsample A, Keep B

    idx = utils_mod.sample_indices(
        df, "target", size_map, rng, replace=False, prefer_clean=True, exclude=[]
    )
    assert len(idx) == 3
    # Check coverage
    sampled_rows = df.loc[idx]
    assert sampled_rows["target"].value_counts()["A"] == 1
    assert sampled_rows["target"].value_counts()["B"] == 2


# --- File/String Utilities ---


def test_split_column():
    """Test column splitting."""
    # Normal
    res = utils_mod.split_column("unit_feat", "_", False)
    assert res == ("unit", "feat")

    # Reverse
    res_rev = utils_mod.split_column("unit_feat", "_", True)
    assert res_rev == ("feat", "unit")

    # No sep
    res_none = utils_mod.split_column("unit", "_", False)
    assert res_none == ("", "unit")


def test_default_id_extractor(tmp_path):
    """Test ID extraction heuristics."""
    # BIDS-like
    p1 = tmp_path / "sub-01_task-rest.pkl"
    assert utils_mod.default_id_extractor(p1) == "01"

    # Plain
    p2 = tmp_path / "patient_x.pkl"
    assert utils_mod.default_id_extractor(p2) == "patient_x"


def test_smart_reader(tmp_path):
    """Test smart file reader."""
    # Pickle
    import pickle

    p_pkl = tmp_path / "test.pkl"
    with open(p_pkl, "wb") as f:
        pickle.dump({"a": 1}, f)
    assert utils_mod.smart_reader(p_pkl) == {"a": 1}

    # NPY
    p_npy = tmp_path / "test.npy"
    np.save(p_npy, np.array([1, 2]))
    assert np.array_equal(utils_mod.smart_reader(p_npy), [1, 2])

    # JSON

    p_json = tmp_path / "test.json"
    p_json.write_text('{"key": "val"}')
    assert utils_mod.smart_reader(p_json) == {"key": "val"}

    # Unsupported
    p_bad = tmp_path / "test.xyz"
    p_bad.touch()
    with pytest.raises(ValueError, match="Unsupported extension"):
        utils_mod.smart_reader(p_bad)

    # H5 (Mocked to avoid dep)
    p_h5 = tmp_path / "test.h5"
    p_h5.touch()
    with patch.dict(sys.modules, {"h5py": MagicMock()}):
        m_h5 = sys.modules["h5py"]
        m_file = m_h5.File.return_value.__enter__.return_value
        m_file.keys.return_value = ["data"]
        m_file.__getitem__.return_value.__getitem__.return_value = "h5_data"
        assert utils_mod.smart_reader(p_h5) == "h5_data"


# --- BIDS Utilities ---


def test_read_bids_entry(monkeypatch, tmp_path):
    """Test BIDS reading dispatch."""
    mne_mock = MagicMock()
    mne_mock.__file__ = "mock_mne.py"
    read_raw_mock = MagicMock()

    monkeypatch.setattr(utils_mod, "mne", mne_mock)
    monkeypatch.setattr(utils_mod, "read_raw_bids", read_raw_mock)

    bids_path = MagicMock()
    bids_path.fpath = tmp_path / "dummy_eeg.vhdr"

    # 1. Pre-epoched
    mne_mock.read_epochs.return_value.get_data.return_value = np.zeros((1, 1, 10))
    mne_mock.read_epochs.return_value.times = np.arange(10)
    mne_mock.read_epochs.return_value.ch_names = ["C1"]
    mne_mock.read_epochs.return_value.info = {"sfreq": 100}

    d, t, c, s = utils_mod.read_bids_entry(
        bids_path,
        is_pre_epoched=True,
        is_evoked=False,
        mode="epochs",
        window_length=None,
        stride=None,
    )
    assert d.shape == (1, 1, 10)

    # 2. Raw (Continuous match)
    raw = read_raw_mock.return_value
    raw.get_data.return_value = np.zeros((2, 100))
    raw.times = np.arange(100)
    raw.ch_names = ["C1", "C2"]
    raw.info = {"sfreq": 100}

    d_cont, t_cont, c_cont, s_cont = utils_mod.read_bids_entry(
        bids_path,
        is_pre_epoched=False,
        is_evoked=False,
        mode="continuous",
        window_length=None,
        stride=None,
    )
    assert d_cont.shape == (1, 2, 100)  # Added batch dim


def test_participants_tsv(tmp_path):
    """Test TSV parsing."""
    p_tsv = tmp_path / "participants.tsv"

    # Valid
    df = pd.DataFrame({"participant_id": ["sub-01", "sub-02"], "age": [20, 30]})
    df.to_csv(p_tsv, sep="\t", index=False)

    lookup = utils_mod.load_participants_tsv(tmp_path)
    assert "01" in lookup
    assert lookup["01"]["age"] == 20

    # Missing
    assert utils_mod.load_participants_tsv(tmp_path / "nowhere") == {}
