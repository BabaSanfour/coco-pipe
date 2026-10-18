import pickle

import numpy as np
import pandas as pd
import pytest

from coco_pipe.io.dataset import BIDSDataset, EmbeddingDataset, TabularDataset


def _tsv(path, frame):
    frame.to_csv(path, index=False, sep="\t")
    return path


# --- tabular -------------------------------------------------------------------


def test_tabular_dataset_loads_features_target_and_metadata(tmp_path, caplog):
    path = _tsv(
        tmp_path / "meta.tsv",
        pd.DataFrame(
            {
                "feat1": [1, 2],
                "feat2": [3, 4],
                "age": [25, 30],
                "sex": ["M", "F"],
                "label": [0, 1],
            }
        ),
    )
    container = TabularDataset(path, target_col="label", meta_columns=["age"]).load()
    assert container.X.shape == (2, 3)
    assert container.y.tolist() == [0, 1]
    assert container.coords["age"].tolist() == [25, 30]
    assert "non-numeric columns" in caplog.text and "sex" in caplog.text

    with pytest.raises(FileNotFoundError):
        TabularDataset(tmp_path / "missing.csv").load()
    (tmp_path / "unknown.xyz").touch()
    with pytest.raises(ValueError, match="Unsupported file extension"):
        TabularDataset(tmp_path / "unknown.xyz").load()


def test_tabular_dataset_reads_excel(monkeypatch, tmp_path):
    # read_excel needs openpyxl; the dataset only has to route .xlsx to it
    monkeypatch.setattr(
        pd, "read_excel", lambda *a, **k: pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    )
    (tmp_path / "data.xlsx").touch()
    container = TabularDataset(tmp_path / "data.xlsx", target_col="b").load()
    assert container.X.shape == (2, 1)
    assert container.y.tolist() == [3, 4]
    assert container.meta["filename"] == "data.xlsx"


def test_tabular_dataset_reshapes_columns_into_dimensions(tmp_path):
    frame = pd.DataFrame(
        {"target": [0, 1], "F1_T1": [1.0, 2.0], "F1_T2": [3.0, 4.0]}
        | {"F2_T1": [5.0, 6.0], "F2_T2": [7.0, 8.0]}
    )
    container = TabularDataset(
        _tsv(tmp_path / "data.tsv", frame),
        target_col="target",
        columns_to_dims=["freq", "time"],
    ).load()
    assert container.dims == ("obs", "freq", "time")
    assert container.coords["freq"].tolist() == ["F1", "F2"]
    assert container.coords["time"].tolist() == ["T1", "T2"]
    assert container.X[0, 0, 0] == 1.0
    assert container.X[1, 1, 1] == 8.0

    unmatched = _tsv(tmp_path / "flat.tsv", pd.DataFrame({"A": [1], "B": [2]}))
    with pytest.raises(ValueError, match="No columns matched"):
        TabularDataset(unmatched, columns_to_dims=["d1", "d2"]).load()
    incomplete = _tsv(
        tmp_path / "gap.tsv", pd.DataFrame({"A_1": [1], "A_2": [2], "B_1": [3]})
    )
    with pytest.raises(ValueError, match="Reshaping failed"):
        TabularDataset(incomplete, columns_to_dims=["L1", "L2"]).load()


def test_tabular_dataset_cleaning(tmp_path):
    dirty = _tsv(
        tmp_path / "dirty.tsv",
        pd.DataFrame(
            {"good": [1.0, 2.0, 3.0], "nan": [1.0, np.nan, 3.0]}
            | {"inf": [1.0, np.inf, 3.0]}
        ),
    )
    container = TabularDataset(dirty, clean=True, clean_kwargs={"mode": "any"}).load()
    assert container.coords["feature"].tolist() == ["good"]
    assert container.meta["cleaning_report"]["dropped_columns"] == ["inf", "nan"]

    tiny = pd.DataFrame({"big": [1.0, -1.0], "tiny": [1e-10, -1e-10]})
    cleaned, _ = TabularDataset.clean(tiny, min_abs_value=1e-5)
    assert cleaned.columns.tolist() == ["big"]
    # columns with at least half tiny values go, as do empty text columns
    half = pd.DataFrame({"A": [1.0, 3.0], "B": [3.0, 4.0], "C": [None, None]})
    cleaned, _ = TabularDataset.clean(half, min_abs_value=2, min_abs_fraction=0.5)
    assert cleaned.columns.tolist() == ["B"]

    # sensor_wide drops every column of a feature that is bad at any sensor
    sensors = pd.DataFrame(
        {"C1_T1": [1.0], "C1_T2": [np.nan], "C2_T1": [1.0], "C2_T2": [1.0]}
    )
    cleaned, report = TabularDataset.clean(
        sensors, mode="sensor_wide", sep="_", reverse=True
    )
    assert cleaned.columns.tolist() == ["C2_T1", "C2_T2"]
    assert report["dropped_features"] == ["C1"]

    assert TabularDataset.clean(pd.DataFrame())[0].empty
    with pytest.raises(ValueError):
        TabularDataset.clean(tiny, mode="invalid")


# --- embeddings ------------------------------------------------------------------


def test_embedding_dataset_stacks_files(tmp_path):
    (tmp_path / "nested").mkdir()
    np.save(tmp_path / "subjA.npy", np.arange(6).reshape(2, 3))
    np.save(tmp_path / "nested" / "subjB.npy", np.ones(3))  # one observation
    container = EmbeddingDataset(
        tmp_path, pattern="*.npy", dims=("feature",), reader=np.load
    ).load()
    assert container.dims == ("obs", "feature")
    # files are read in path order
    assert container.ids.tolist() == ["subjB", "subjA_0", "subjA_1"]

    # a dict holds one array per segment, of one or several observations
    with open(tmp_path / "sub-dict.pkl", "wb") as f:
        pickle.dump({"seg1": np.zeros((2, 3)), "seg2": [1, 2, 3]}, f)
    segments = EmbeddingDataset(tmp_path, pattern="*.pkl", dims=("feature",)).load()
    assert segments.ids.tolist() == [
        "sub-dict_seg1_0",
        "sub-dict_seg1_1",
        "sub-dict_seg2",
    ]


def test_embedding_dataset_selects_subjects(tmp_path):
    for i in range(1, 4):
        (tmp_path / f"sub-0{i}.pkl").touch()

    def load(subjects):
        return EmbeddingDataset(
            tmp_path, dims=("f",), reader=lambda p: np.array([1, 2]), subjects=subjects
        ).load()

    assert load(2).ids.tolist() == ["sub-01", "sub-02"]
    assert load(["sub-03"]).ids.tolist() == ["sub-03"]


def test_embedding_dataset_records_files_it_cannot_load(tmp_path, caplog):
    np.save(tmp_path / "s1.npy", np.ones((3, 4)))
    np.save(tmp_path / "s2.npy", np.ones((3, 4)))
    (tmp_path / "s3.npy").write_bytes(b"not a numpy file")

    with caplog.at_level("WARNING"):
        container = EmbeddingDataset(tmp_path, pattern="*.npy").load()

    assert container.X.shape == (6, 4)
    (failure,) = container.meta["load_failures"]
    assert failure["path"].endswith("s3.npy") and failure["error"]
    assert "Could not load s3.npy" in caplog.text


def test_embedding_dataset_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="No files matched"):
        EmbeddingDataset(tmp_path, pattern="*.npy").load()

    np.save(tmp_path / "cube.npy", np.zeros((2, 2, 2)))
    with pytest.raises(RuntimeError, match=r"the first being .*cube.npy .*fit"):
        EmbeddingDataset(tmp_path, pattern="*.npy", dims=("f",), reader=np.load).load()

    with open(tmp_path / "1.pkl", "wb") as f:
        pickle.dump({"seg1": [1, 2], "seg2": [1, 2, 3]}, f)
    with pytest.raises(ValueError, match="Concatenation failed"):
        EmbeddingDataset(tmp_path, pattern="*.pkl", dims=("feature",)).load()


# --- BIDS ------------------------------------------------------------------------


def _load_epochs(root, **kwargs):
    return BIDSDataset(root, suffix="epo", mode="load_existing", **kwargs).load()


def test_bids_dataset_identifies_recordings_by_entities(tmp_path, write_bids_epochs):
    write_bids_epochs(tmp_path, "01", session="A", run="1")
    write_bids_epochs(tmp_path, "01", session="A", run="10")
    write_bids_epochs(tmp_path, "02", session="A", run="1")
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tage\nsub-01\t30\nsub-02\t40\n"
    )

    container = _load_epochs(tmp_path)

    assert container.dims == ("obs", "channel", "time")
    assert container.X.shape == (6, 2, 20)
    assert container.X.dtype == np.float32
    assert container.meta["sfreq"] == 100.0
    assert container.coords["channel"].tolist() == ["C1", "C2"]
    assert container.coords["recording_id"].tolist() == [
        "sub-01_ses-A_task-rest_run-1",
        "sub-01_ses-A_task-rest_run-1",
        "sub-01_ses-A_task-rest_run-10",
        "sub-01_ses-A_task-rest_run-10",
        "sub-02_ses-A_task-rest_run-1",
        "sub-02_ses-A_task-rest_run-1",
    ]
    assert container.ids[:2].tolist() == [
        "sub-01_ses-A_task-rest_run-1_epoch-0000",
        "sub-01_ses-A_task-rest_run-1_epoch-0001",
    ]
    assert container.coords["subject"].tolist() == ["01"] * 4 + ["02"] * 2
    assert container.coords["run"].tolist() == ["1", "1", "10", "10", "1", "1"]
    assert container.coords["age"].tolist() == [30] * 4 + [40] * 2

    # reading with threads keeps the recording order
    threaded = _load_epochs(tmp_path, n_jobs=3)
    assert threaded.ids.tolist() == container.ids.tolist()
    np.testing.assert_array_equal(threaded.X, container.X)

    # an epoch keeps its number in the recording whichever events are selected
    one_run = _load_epochs(tmp_path, subjects="01", runs="1", event_id="right")
    assert one_run.ids.tolist() == ["sub-01_ses-A_task-rest_run-1_epoch-0001"]
    assert one_run.coords["epoch"].tolist() == [1]


def test_bids_dataset_description_selects_the_file(tmp_path, write_bids_epochs):
    write_bids_epochs(tmp_path, "01", description="base", value=1.0)
    write_bids_epochs(tmp_path, "01", description="alt", value=2.0)

    assert np.all(_load_epochs(tmp_path, description="base", units="V").X[:, 0] == 1)
    assert np.all(_load_epochs(tmp_path, description="alt", units="V").X[:, 0] == 2)
    with pytest.raises(ValueError, match="Several files match recording sub-01"):
        _load_epochs(tmp_path)


def test_bids_dataset_recordings_must_agree(tmp_path, write_bids_epochs):
    write_bids_epochs(tmp_path, "01", ch_names=("C1", "C2"))
    write_bids_epochs(tmp_path, "02", ch_names=("C2", "C1"))

    container = _load_epochs(tmp_path, units="V")
    assert container.coords["channel"].tolist() == ["C1", "C2"]
    assert container.X[:, :, 0].tolist() == [[0, 1], [0, 1], [1, 0], [1, 0]]

    write_bids_epochs(tmp_path, "03", ch_names=("C1", "C3"))
    with pytest.raises(ValueError, match=r"ch_names.* must match"):
        _load_epochs(tmp_path, subjects=["01", "03"])
    write_bids_epochs(tmp_path, "04", sfreq=200.0)
    with pytest.raises(ValueError, match=r"sfreq.* must match"):
        _load_epochs(tmp_path, subjects=["01", "04"])


def test_bids_dataset_stacks_recordings_that_number_events_differently(
    tmp_path, write_bids_epochs
):
    write_bids_epochs(tmp_path, "01")
    write_bids_epochs(tmp_path, "02", event_id={"left": 2, "right": 1}, bads=["C2"])

    container = _load_epochs(tmp_path)

    # one code per event name across recordings
    left = container.meta["event_id"]["left"]
    assert container.y.tolist() == [left, 3 - left] * 2
    assert container.coords["bad_channels"].tolist() == ["", "", "C2", "C2"]

    # codes passed in `event_id` are kept, whatever each file and the order of
    # the events say
    for codes in ({"left": 3, "right": 10}, {"left": 2, "right": 1}):
        container = _load_epochs(tmp_path, event_id=codes)
        assert container.meta["event_id"] == codes
        assert container.y.tolist() == [codes["left"], codes["right"]] * 2

    # only the names that were found are listed
    container = _load_epochs(tmp_path, event_id={"up": 4, "right": 10})
    assert container.meta["event_id"] == {"right": 10}
    assert container.y.tolist() == [10, 10]
    assert container.coords["epoch"].tolist() == [1, 1]


@pytest.mark.parametrize("event_id", [None, ["right", "left"]])
def test_bids_dataset_numbers_events_without_codes(
    tmp_path, write_bids_epochs, event_id
):
    write_bids_epochs(tmp_path, "01", event_id={"left": 5, "right": 7})

    container = _load_epochs(tmp_path, event_id=event_id)

    # numbered from 1 in the order the names appear in the recordings
    assert container.meta["event_id"] == {"left": 1, "right": 2}
    assert container.y.tolist() == [1, 2]


def test_bids_dataset_raw_recordings(tmp_path):
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    raw = mne.io.RawArray(
        np.zeros((2, 300)), mne.create_info(["M1", "M2"], 100.0, "mag"), verbose=False
    )
    raw.set_meas_date(0)
    raw.set_annotations(
        mne.Annotations([0.5, 1.5, 2.5], 0.0, ["right", "left", "right"])
    )
    for subject in ("01", "02"):
        write_raw_bids(
            raw,
            BIDSPath(root=tmp_path, subject=subject, task="rest", datatype="meg"),
            allow_preload=True,
            format="FIF",
            verbose=False,
        )

    container = BIDSDataset(
        tmp_path, task="rest", datatype="meg", window_length=1.0
    ).load()

    assert container.X.shape == (6, 2, 100)
    assert container.ids[:2].tolist() == [
        "sub-01_task-rest_epoch-0000",
        "sub-01_task-rest_epoch-0001",
    ]
    assert set(container.coords["recording_id"]) == {
        "sub-01_task-rest",
        "sub-02_task-rest",
    }

    # epoched around events, with the codes passed in `event_id`
    codes = {"left": 3, "right": 10}
    container = BIDSDataset(
        tmp_path, task="rest", datatype="meg", event_id=codes, tmin=0.0, tmax=0.2
    ).load()
    assert container.meta["event_id"] == codes
    assert container.y.tolist() == [10, 3, 10] * 2
    assert container.coords["time"][0] == 0.0

    with pytest.raises(RuntimeError, match="No valid data found"):
        BIDSDataset(tmp_path, task="other", datatype="meg", window_length=1.0).load()


def test_bids_dataset_subject_metadata_keeps_the_loader_coordinates(tmp_path):
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    raw = mne.io.RawArray(
        np.zeros((2, 200)), mne.create_info(["M1", "M2"], 100.0, "mag"), verbose=False
    )
    raw.set_meas_date(0)
    for subject in ("01", "02"):
        write_raw_bids(
            raw,
            BIDSPath(root=tmp_path, subject=subject, task="rest", datatype="meg"),
            allow_preload=True,
            format="FIF",
            verbose=False,
        )

    def load(meta, key="subject"):
        return BIDSDataset(
            tmp_path,
            task="rest",
            datatype="meg",
            window_length=1.0,
            subject_metadata_df=meta,
            subject_key=key,
        ).load()

    # keyed on a column named like the coordinate: the key only matches rows
    meta = pd.DataFrame({"subject": ["sub-01", "sub-02"], "age": [9, 10]})
    container = load(meta)
    assert container.coords["subject"].tolist() == ["01", "01", "02", "02"]
    assert container.coords["age"].tolist() == [9, 9, 10, 10]

    # any other such column would replace what the files say
    with pytest.raises(ValueError, match=r"\['run'\] have the names"):
        load(meta.assign(run=["9", "9"]))
    participants = tmp_path / "participants.tsv"
    table = pd.read_csv(participants, sep="\t").assign(session="x")
    _tsv(participants, table)
    with pytest.raises(ValueError, match=r"\['session'\] have the names"):
        load(None, key=None)


def test_bids_dataset_raw_fif_derivatives(tmp_path):
    import mne
    from mne_bids import BIDSPath

    # FIF is no BIDS format for EEG, but preprocessing derivatives are saved so
    data = np.arange(2 * 300, dtype=float).reshape(2, 300) * 1e-6
    for description, scale in (("base", 1.0), ("clean", 2.0)):
        path = BIDSPath(
            root=tmp_path,
            subject="01",
            task="rest",
            description=description,
            datatype="eeg",
            suffix="eeg",
            extension=".fif",
            check=False,
        ).mkdir()
        raw = mne.io.RawArray(
            data * scale, mne.create_info(["C1", "C2"], 100.0, "eeg"), verbose=False
        )
        raw.save(path.fpath, verbose=False)

    container = BIDSDataset(
        tmp_path, task="rest", description="base", window_length=1.0
    ).load()
    np.testing.assert_allclose(
        container.X, data.reshape(2, 3, 100).transpose(1, 0, 2), rtol=1e-6
    )
    # two versions of one recording: the loader does not pick one itself
    with pytest.raises(ValueError, match="Several files match recording"):
        BIDSDataset(tmp_path, task="rest", window_length=1.0).load()


def _save_split(inst, root, suffix):
    """Save ``inst`` under ``root`` in parts of 2 MB, as MNE splits big files."""
    from mne_bids import BIDSPath

    path = BIDSPath(
        root=root,
        subject="01",
        task="rest",
        datatype="eeg",
        suffix=suffix,
        extension=".fif",
        check=False,
    ).mkdir()
    inst.save(path.fpath, split_size="2MB", split_naming="bids", verbose=False)
    assert len(list(path.fpath.parent.glob("*_split-*"))) > 1


def test_bids_dataset_reads_a_raw_file_split_into_parts(tmp_path):
    import mne

    data = np.random.default_rng(0).normal(size=(4, 300_000)) * 1e-6
    raw = mne.io.RawArray(data, mne.create_info(4, 100.0, "eeg"), verbose=False)
    _save_split(raw, tmp_path, "eeg")

    container = BIDSDataset(tmp_path, task="rest", window_length=1.0).load()

    assert set(container.coords["recording_id"]) == {"sub-01_task-rest"}
    assert len(container.X) == 3000


def test_bids_dataset_reads_saved_epochs_split_into_parts(tmp_path, write_bids_epochs):
    import mne

    data = np.random.default_rng(0).normal(size=(4, 300_000)) * 1e-6
    raw = mne.io.RawArray(data, mne.create_info(4, 100.0, "eeg"), verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True)
    _save_split(epochs, tmp_path, "epo")

    container = _load_epochs(tmp_path)

    assert len(container.X) == 3000


def test_bids_dataset_reports_recordings_that_fail_to_load(
    tmp_path, write_bids_epochs, caplog
):
    write_bids_epochs(tmp_path, "01")
    write_bids_epochs(tmp_path, "02").write_bytes(b"not a FIF file")

    with caplog.at_level("WARNING"):
        container = _load_epochs(tmp_path)

    assert set(container.coords["subject"]) == {"01"}
    (failure,) = container.meta["load_failures"]
    assert failure["recording_id"] == "sub-02_task-rest"
    assert failure["error"]
    assert "Could not load sub-02_task-rest" in caplog.text

    assert _load_epochs(tmp_path, subjects="01").meta["load_failures"] == []

    # a recording with none of the requested events is skipped, not a failure
    write_bids_epochs(tmp_path, "03", event_id={"up": 1, "down": 2})
    caplog.clear()
    with caplog.at_level("WARNING"):
        left = _load_epochs(tmp_path, subjects=["01", "03"], event_id="left")
    assert left.meta["load_failures"] == []
    assert left.meta["skipped_no_matching_events"] == ["sub-03_task-rest"]
    assert "sub-03" not in caplog.text
    # so is one that lists the event but kept none of its epochs
    import mne

    path = write_bids_epochs(tmp_path, "04")
    epochs = mne.read_epochs(path, verbose=False).drop([0], verbose=False)
    epochs.save(path, overwrite=True, verbose=False)
    left = _load_epochs(tmp_path, subjects=["01", "04"], event_id="left")
    assert left.meta["skipped_no_matching_events"] == ["sub-04_task-rest"]
    assert set(left.coords["subject"]) == {"01"}
    with pytest.raises(
        RuntimeError, match=r"No valid data found in .*sub-02_task-rest"
    ):
        _load_epochs(tmp_path, subjects="02")
