import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import coco_pipe.io.utils as utils_mod
from coco_pipe.io.quality import row_quality_score
from coco_pipe.io.utils import (
    _get_bids_path,
    _get_mne,
    _get_read_raw_bids,
    detect_runs,
    detect_sessions,
    detect_subjects,
    load_participants_tsv,
    make_strata,
    sample_indices,
    smart_reader,
)


def test_row_quality_score():
    """Test badness score calculation."""
    df = pd.DataFrame({"a": [1, np.nan, 0, np.inf], "b": [1, 1, 1, 1]})
    # Row 0: a=1, b=1 -> score 0
    # Row 1: a=NaN -> score 1
    # Row 2: a=0 -> score 1 (if count_zero=True)
    # Row 3: a=Inf -> score 1

    scores = row_quality_score(df, count_zero=True)
    assert np.array_equal(scores, [0, 1, 1, 1])

    scores_nz = row_quality_score(df, count_zero=False)
    assert np.array_equal(scores_nz, [0, 1, 0, 1])


def test_row_quality_score_normalized():
    df = pd.DataFrame(
        {
            "a": [0.0, np.nan, 1.0],
            "b": [1.0, np.inf, 2.0],
            "label": ["x", "y", "z"],
        },
        index=[10, 20, 30],
    )

    scores = row_quality_score(df, normalize=True)

    assert scores.index.tolist() == [10, 20, 30]
    assert scores.tolist() == [0.5, 1.0, 0.0]

    no_numeric = row_quality_score(
        df[["label"]],
        normalize=True,
    )
    assert no_numeric.tolist() == [0.0, 0.0, 0.0]
    assert no_numeric.dtype == float


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("42", "0042"),
        ("sub-42", "0042"),
        (42, "0042"),
        ("P001", "P001"),
    ],
)
def test_normalize_subject_value(value, expected):
    assert utils_mod.normalize_subject_value(value) == expected


def test_read_table_csv(tmp_path):
    path = tmp_path / "features.csv"
    path.write_text("subject;feature\nsub-1;1.5\nsub-2;2.5\n", encoding="utf-8")

    table = utils_mod.read_table(path)

    assert table.to_dict(orient="records") == [
        {"subject": "sub-1", "feature": 1.5},
        {"subject": "sub-2", "feature": 2.5},
    ]


def test_read_table_parquet(monkeypatch, tmp_path):
    path = tmp_path / "features.parquet"
    expected = pd.DataFrame({"subject": ["sub-1"], "feature": [1.5]})
    read_parquet = MagicMock(return_value=expected)
    monkeypatch.setattr(utils_mod.pd, "read_parquet", read_parquet)

    table = utils_mod.read_table(path)

    read_parquet.assert_called_once_with(path)
    pd.testing.assert_frame_equal(table, expected)


def test_read_table_drops_unnamed_and_empty_columns(tmp_path):
    path = tmp_path / "features.csv"
    path.write_text(
        "subject,feature,Unnamed: 2,empty\nsub-1,1.5,,\n",
        encoding="utf-8",
    )

    table = utils_mod.read_table(path)

    assert table.columns.tolist() == ["subject", "feature"]


def test_read_table_explicit_separator(tmp_path):
    path = tmp_path / "features.csv"
    path.write_text("subject|feature\nsub-1|1.5\n", encoding="utf-8")

    table = utils_mod.read_table(path, sep="|")

    assert table.to_dict(orient="records") == [{"subject": "sub-1", "feature": 1.5}]


def test_read_table_unsupported_format_raises(tmp_path):
    path = tmp_path / "features.tsv"
    path.touch()

    with pytest.raises(ValueError, match="Expected .csv or .parquet"):
        utils_mod.read_table(path)


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
    mne_mock.read_epochs.return_value.events = np.array([[0, 0, 1]])
    mne_mock.read_epochs.return_value.event_id = {"stim": 1}

    d, t, c, s, labels = utils_mod.read_bids_entry(
        bids_path,
        is_pre_epoched=True,
        is_evoked=False,
        mode="epochs",
        window_length=None,
        stride=None,
    )
    assert d.shape == (1, 1, 10)
    assert np.array_equal(labels, [1])

    # 2. Raw (Continuous match)
    raw = read_raw_mock.return_value
    raw.get_data.return_value = np.zeros((2, 100))
    raw.times = np.arange(100)
    raw.ch_names = ["C1", "C2"]
    raw.info = {"sfreq": 100}

    d_cont, t_cont, c_cont, s_cont, labels_cont = utils_mod.read_bids_entry(
        bids_path,
        is_pre_epoched=False,
        is_evoked=False,
        mode="continuous",
        window_length=None,
        stride=None,
    )
    assert d_cont.shape == (1, 2, 100)  # Added batch dim
    assert labels_cont is None


def test_read_bids_entry_pre_epoched_event_id_filters(monkeypatch, tmp_path):
    """Test event_id filtering for precomputed epochs."""

    class FakeEpochs:
        def __init__(self, data, event_codes, event_id):
            self._data = data
            self.events = np.column_stack(
                [
                    np.arange(len(event_codes)),
                    np.zeros(len(event_codes), dtype=int),
                    np.asarray(event_codes),
                ]
            )
            self.event_id = event_id
            self.times = np.arange(data.shape[-1])
            self.ch_names = ["C1"]
            self.info = {"sfreq": 100}

        def __len__(self):
            return len(self._data)

        def __getitem__(self, item):
            if isinstance(item, list):
                keep_codes = [self.event_id[name] for name in item]
                mask = np.isin(self.events[:, -1], keep_codes)
            else:
                mask = np.asarray(item, dtype=bool)
            return FakeEpochs(
                self._data[mask],
                self.events[mask, -1],
                self.event_id,
            )

        def get_data(self, copy=False):
            return self._data

    mne_mock = MagicMock()
    monkeypatch.setattr(utils_mod, "mne", mne_mock)

    bids_path = MagicMock()
    bids_path.fpath = tmp_path / "dummy_eeg.epo.fif"
    bids_path.fpath.touch()

    epochs = FakeEpochs(
        np.arange(3 * 1 * 5).reshape(3, 1, 5),
        [1, 2, 1],
        {"left": 1, "right": 2},
    )
    mne_mock.read_epochs.return_value = epochs

    data, _, _, _, labels = utils_mod.read_bids_entry(
        bids_path,
        is_pre_epoched=True,
        is_evoked=False,
        mode="epochs",
        window_length=None,
        stride=None,
        event_id={"target": 2},
    )

    assert data.shape == (1, 1, 5)
    assert np.array_equal(labels, [2])

    with pytest.raises(ValueError, match="No epochs remain after filtering"):
        utils_mod.read_bids_entry(
            bids_path,
            is_pre_epoched=True,
            is_evoked=False,
            mode="epochs",
            window_length=None,
            stride=None,
            event_id={"missing": 99},
        )


@pytest.mark.parametrize("event_id", ["EO_baseline", ["EO_baseline"]])
def test_read_bids_entry_pre_epoched_event_name_filters(
    monkeypatch, tmp_path, event_id
):
    """Test event name filtering for precomputed epochs."""

    class FakeEpochs:
        def __init__(self, data, event_codes, event_id_map):
            self._data = data
            self.events = np.column_stack(
                [
                    np.arange(len(event_codes)),
                    np.zeros(len(event_codes), dtype=int),
                    np.asarray(event_codes),
                ]
            )
            self.event_id = event_id_map
            self.times = np.arange(data.shape[-1])
            self.ch_names = ["C1"]
            self.info = {"sfreq": 100}

        def __len__(self):
            return len(self._data)

        def __getitem__(self, item):
            if isinstance(item, list):
                keep_codes = [self.event_id[name] for name in item]
                mask = np.isin(self.events[:, -1], keep_codes)
            else:
                mask = np.asarray(item, dtype=bool)
            return FakeEpochs(
                self._data[mask],
                self.events[mask, -1],
                self.event_id,
            )

        def get_data(self, copy=False):
            return self._data

    mne_mock = MagicMock()
    monkeypatch.setattr(utils_mod, "mne", mne_mock)

    bids_path = MagicMock()
    bids_path.fpath = tmp_path / "dummy_eeg.epo.fif"
    bids_path.fpath.touch()

    mne_mock.read_epochs.return_value = FakeEpochs(
        np.arange(3 * 1 * 5).reshape(3, 1, 5),
        [11, 12, 11],
        {"EO_baseline": 11, "EC_baseline": 12},
    )

    data, _, _, _, labels = utils_mod.read_bids_entry(
        bids_path,
        is_pre_epoched=True,
        is_evoked=False,
        mode="epochs",
        window_length=None,
        stride=None,
        event_id=event_id,
    )

    assert data.shape == (2, 1, 5)
    assert np.array_equal(labels, [11, 11])


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


def test_lazy_imports():
    utils_mod.mne = None
    assert _get_mne() is not None
    utils_mod.read_raw_bids = None
    assert _get_read_raw_bids() is not None


def test_make_strata_extra():
    df = pd.DataFrame({"num": [1, 2, 3, 4, 5], "num_dup": [1, 1, 1, 1, 1]})

    # uniform
    res = make_strata(df, covariates=["num"], n_bins=2, binning="uniform")
    assert len(res) == 5

    # qcut exception fallback to cut
    res2 = make_strata(df, covariates=["num_dup"], n_bins=2, binning="quantile")
    assert len(res2) == 5


def test_sample_indices_extra():
    df = pd.DataFrame({"target": ["A", "A", "B"], "feat": [1, 2, 3]})
    rng = np.random.default_rng(42)

    # n <= 0
    res = sample_indices(
        df, "target", {"A": 0}, rng, replace=False, prefer_clean=False, exclude=[]
    )
    assert len(res) == 0

    # prefer_clean with replace
    res2 = sample_indices(
        df, "target", {"A": 3}, rng, replace=True, prefer_clean=True, exclude=[]
    )
    assert len(res2) == 3


def test_load_participants_tsv_exception(tmp_path):
    p = tmp_path / "participants.tsv"
    p.write_text("invalid\ttsv\ncontent")
    # Actually just missing columns or something might not raise exception
    # unless it's malformed.
    # To force pd.read_csv to raise, we can make it a directory
    p.unlink()
    p.mkdir()
    res = load_participants_tsv(tmp_path)
    assert res == {}


def test_detect_subjects_sessions(tmp_path):
    (tmp_path / "sub-01").mkdir()
    (tmp_path / "sub-02").mkdir()
    (tmp_path / "sub-01" / "ses-A").mkdir()

    subs = detect_subjects(tmp_path)
    assert "01" in subs and "02" in subs

    sess = detect_sessions(tmp_path, "01")
    assert "A" in sess

    sess2 = detect_sessions(tmp_path, "03")  # doesn't exist
    assert len(sess2) == 0


def test_detect_runs(tmp_path):
    # This requires mne_bids matching logic which might be hard to mock
    # without actual files
    # I will skip deep mocking if it's complicated, but let's try a simple mock
    class MockMatch:
        run = "01"

    class MockBIDSPath:
        def __init__(self, **kwargs):
            pass

        def match(self):
            return [MockMatch()]

    orig = utils_mod.BIDSPath
    utils_mod.BIDSPath = MockBIDSPath
    try:
        runs = detect_runs(tmp_path, "01")
        assert "01" in runs
    finally:
        utils_mod.BIDSPath = orig


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


def test_get_bids_path():
    utils_mod.BIDSPath = None
    assert _get_bids_path() is not None


def test_sample_indices_not_clean():
    df = pd.DataFrame({"target": ["A", "A", "B"], "feat": [1, 2, 3]})
    rng = np.random.default_rng(42)

    # n <= len and not replace
    res = sample_indices(
        df, "target", {"A": 2}, rng, replace=False, prefer_clean=False, exclude=[]
    )
    assert len(res) == 2

    # n > len or replace
    res2 = sample_indices(
        df, "target", {"A": 3}, rng, replace=True, prefer_clean=False, exclude=[]
    )
    assert len(res2) == 3


def test_read_bids_entry_evoked(monkeypatch):
    class DummyEvoked:
        def __init__(self):
            self.data = np.zeros((2, 10))
            self.times = np.zeros(10)
            self.ch_names = ["C1", "C2"]
            self.info = {"sfreq": 100.0}

    class MockMNE:
        def read_evokeds(self, *args, **kwargs):
            return [DummyEvoked()]

    monkeypatch.setattr(utils_mod, "_get_mne", lambda: MockMNE())

    class MockBIDSPath:
        fpath = Path("fake")

        def match(self):
            return [Path("fake2")]

    data, times, ch_names, sfreq, labels = utils_mod.read_bids_entry(
        MockBIDSPath(), False, True, "continuous", None, None
    )
    assert data.shape == (1, 2, 10)


def test_read_bids_entry_raw_continuous(monkeypatch):
    class DummyRaw:
        def __init__(self):
            self.times = np.zeros(10)
            self.ch_names = ["C1", "C2"]
            self.info = {"sfreq": 100.0}

        def load_data(self):
            pass

        def pick_types(self, **kwargs):
            pass

        def get_data(self):
            return np.zeros((2, 10))

    monkeypatch.setattr(
        utils_mod, "_get_read_raw_bids", lambda: lambda *args, **kwargs: DummyRaw()
    )

    data, times, ch_names, sfreq, labels = utils_mod.read_bids_entry(
        None, False, False, "continuous", None, None
    )
    assert data.shape == (1, 2, 10)


def test_read_bids_entry_raw_fixed_epochs(monkeypatch):
    class DummyRaw:
        def __init__(self):
            self.times = np.zeros(10)
            self.ch_names = ["C1", "C2"]
            self.info = {"sfreq": 100.0}

        def load_data(self):
            pass

        def pick_types(self, **kwargs):
            pass

        def get_data(self):
            return np.zeros((2, 10))

    class DummyEpochs:
        def __init__(self):
            self.times = np.zeros(10)
            self.events = np.zeros((5, 3))

        def get_data(self, **kwargs):
            return np.zeros((5, 2, 10))

    class MockMNE:
        def make_fixed_length_epochs(self, *args, **kwargs):
            return DummyEpochs()

    monkeypatch.setattr(
        utils_mod, "_get_read_raw_bids", lambda: lambda *args, **kwargs: DummyRaw()
    )
    monkeypatch.setattr(utils_mod, "_get_mne", lambda: MockMNE())

    data, times, ch_names, sfreq, labels = utils_mod.read_bids_entry(
        None, False, False, "epochs", 1.0, 0.5
    )
    assert data.shape == (5, 2, 10)


def test_read_bids_entry_raw_event_epochs(monkeypatch):
    class DummyRaw:
        def __init__(self):
            self.times = np.zeros(10)
            self.ch_names = ["C1", "C2"]
            self.info = {"sfreq": 100.0}

        def load_data(self):
            pass

        def pick_types(self, **kwargs):
            pass

        def get_data(self):
            return np.zeros((2, 10))

    class DummyEpochs:
        def __init__(self):
            self.times = np.zeros(10)
            self.events = np.zeros((5, 3))

        def get_data(self, **kwargs):
            return np.zeros((5, 2, 10))

    class MockMNE:
        def Epochs(*args, **kwargs):
            return DummyEpochs()

        def events_from_annotations(self, *args, **kwargs):
            return np.zeros((5, 3)), {"A": 1}

    monkeypatch.setattr(
        utils_mod, "_get_read_raw_bids", lambda: lambda *args, **kwargs: DummyRaw()
    )
    monkeypatch.setattr(utils_mod, "_get_mne", lambda: MockMNE())

    data, times, ch_names, sfreq, labels = utils_mod.read_bids_entry(
        None, False, False, "epochs", None, None, event_id={"A": 1}
    )
    assert data.shape == (5, 2, 10)


def test_read_bids_entry_raw_no_length(monkeypatch):
    class DummyRaw:
        def __init__(self):
            self.times = np.zeros(10)
            self.ch_names = ["C1", "C2"]
            self.info = {"sfreq": 100.0}

        def load_data(self):
            pass

        def pick_types(self, **kwargs):
            pass

        def get_data(self):
            return np.zeros((2, 10))

    monkeypatch.setattr(
        utils_mod, "_get_read_raw_bids", lambda: lambda *args, **kwargs: DummyRaw()
    )

    data, times, ch_names, sfreq, labels = utils_mod.read_bids_entry(
        None, False, False, "epochs", None, None
    )
    assert data.shape == (1, 2, 10)


def test_io_init_getattr():
    import coco_pipe.io as coco_io

    # 1. Valid attributes
    assert coco_io.BIDSDataset is not None
    assert coco_io.EmbeddingDataset is not None
    assert coco_io.TabularDataset is not None

    # 2. Invalid attribute raises AttributeError
    with pytest.raises(AttributeError, match="has no attribute 'invalid_attr'"):
        _ = coco_io.invalid_attr
