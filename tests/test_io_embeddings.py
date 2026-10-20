import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from mne_bids import BIDSPath, get_entities_from_fname

from coco_pipe.io import EMBEDDING_CONFIG_KEYS
from coco_pipe.io import embeddings as embeddings_module
from coco_pipe.io.embeddings import (
    consolidate_embeddings,
    embedding_path,
    find_embeddings,
    load_embeddings,
    write_embedding_dataset,
    write_embedding_manifest,
    write_recording_embeddings,
)
from coco_pipe.utils import stable_hash
from tests.fixtures.embeddings import extract, source_path, write_recording


def test_embedding_path_names_recording_and_dataset_files(tmp_path):
    recording = embedding_path(
        tmp_path,
        "baseRandomEaCoral",
        "window",
        recording_id="sub-01_ses-02_task-rest_run-03",
    ).fpath
    assert recording == (
        tmp_path
        / "sub-01"
        / "ses-02"
        / "eeg"
        / "sub-01_ses-02_task-rest_run-03_desc-baseRandomEaCoral_windowembeddings.nc"
    )
    assert get_entities_from_fname(recording.name)["description"] == "baseRandomEaCoral"
    assert embedding_path(tmp_path, "base", "token", task="rest").fpath == (
        tmp_path / "task-rest_desc-base_tokenembeddings.nc"
    )
    assert embedding_path(tmp_path, "baseRa", "cov").fpath == (
        tmp_path / "desc-baseRa_covembeddings.nc"
    )


@pytest.mark.parametrize(
    "label, view, kwargs",
    [
        ("base_x", "window", {}),
        ("base x", "window", {}),
        ("baseRä", "window", {}),
        ("base", "tokens", {}),
        ("base", "window", {"recording_id": "sub-01_task-rest_eeg"}),
        ("base", "window", {"recording_id": "sub-01_task-rest_desc-base"}),
        ("base", "window", {"recording_id": "task-rest"}),
        ("base", "window", {"recording_id": "sub-01_task-rest", "task": "rest"}),
    ],
)
def test_embedding_path_refuses_names_bids_cannot_hold(tmp_path, label, view, kwargs):
    with pytest.raises(ValueError):
        embedding_path(tmp_path, label, view, **kwargs)


def test_find_embeddings_finds_the_recording_files_of_one_label_and_view(tmp_path):
    for recording_id in (
        "sub-10_task-rest",
        "sub-02_task-rest",
        "sub-01_ses-01_task-rest_run-02",
        "sub-01_ses-01_task-rest_run-01",
    ):
        for label in ("base", "baseRandom", "a+b", "aab"):
            for view in ("window", "token"):
                path = embedding_path(tmp_path, label, view, recording_id=recording_id)
                path.mkdir().fpath.touch()
    embedding_path(tmp_path, "base", "window", task="rest").fpath.touch()
    written = embedding_path(
        tmp_path, "base", "window", recording_id="sub-02_task-rest"
    ).fpath
    written.with_name(f".{written.name}.0123abcd.tmp").touch()

    def found(label, view="window", **kwargs):
        paths = find_embeddings(tmp_path, label, view, **kwargs)
        assert all(isinstance(path, BIDSPath) for path in paths)
        return [path.fpath.relative_to(tmp_path).as_posix() for path in paths]

    assert found("base") == [
        "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_run-01_desc-base_windowembeddings.nc",
        "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_run-02_desc-base_windowembeddings.nc",
        "sub-02/eeg/sub-02_task-rest_desc-base_windowembeddings.nc",
        "sub-10/eeg/sub-10_task-rest_desc-base_windowembeddings.nc",
    ]
    assert found("base", "token") == [
        path.replace("window", "token") for path in found("base")
    ]
    assert found("a+b") == [path.replace("base", "a+b") for path in found("base")]
    assert found("base", subjects=["01", "10"]) == [
        path for path in found("base") if not path.startswith("sub-02")
    ]
    assert found("base", subjects="1") == []
    assert found("baseEaMean") == []


@pytest.mark.parametrize(
    "label, view, error",
    [
        ("base_x", "window", ValueError),
        ("base", "tokens", ValueError),
        ("base", "window", FileNotFoundError),
    ],
)
def test_find_embeddings_refuses(tmp_path, label, view, error):
    with pytest.raises(error):
        find_embeddings(tmp_path / "missing", label, view)


def test_write_recording_embeddings_round_trip(tmp_path):
    results = {"EO": extract([1, 3], seed=1), "EC": extract([0, 2, 4])}
    paths = write_recording_embeddings(
        results,
        tmp_path / "out",
        label="base",
        source=source_path(tmp_path),
        source_dataset="preproc",
        bad_channels=["Cz"],
    )
    recording = "sub-01_ses-02_task-rest_run-03"
    assert {view: path.basename for view, path in paths.items()} == {
        "window": f"{recording}_desc-base_windowembeddings.nc",
        "token": f"{recording}_desc-base_tokenembeddings.nc",
    }
    # Windows in epoch order: EO's 1 and 3 between EC's 0, 2 and 4.
    order = [2, 0, 3, 1, 4]
    hashes = set()
    for view, dims in (
        ("window", ("window", "feature")),
        ("token", ("window", "channel", "time_patch", "feature")),
    ):
        with xr.open_dataset(paths[view].fpath, engine="h5netcdf") as ds:
            expected = np.concatenate(
                [getattr(results[c], f"{view}_embeddings") for c in ("EO", "EC")]
            )[order]
            assert ds["embeddings"].dims == dims
            assert ds["embeddings"].dtype == np.float32
            np.testing.assert_array_equal(ds["embeddings"], expected.astype(np.float32))
            assert ds["obs"].values.tolist() == [
                f"{recording}_epoch-{number:04d}" for number in range(5)
            ]
            assert ds["condition"].values.tolist() == ["EC", "EO", "EC", "EO", "EC"]
            assert ds["epoch"].values.tolist() == [0, 1, 2, 3, 4]
            assert ds["start"].values.tolist() == [0, 10, 20, 30, 40]
            assert ds["stop"].values.tolist() == [10, 20, 30, 40, 50]
            for name, value in (
                ("recording_id", recording),
                ("subject", "01"),
                ("session", "02"),
                ("task", "rest"),
                ("run", "03"),
            ):
                assert set(ds[name].values) == {value}
            assert ds["recording"].values.tolist() == [recording]
            assert ds["source"].values.tolist() == [
                f"bids:preproc:sub-01/ses-02/eeg/{recording}_desc-base_epo.fif"
            ]
            assert ds["sfreq"].values.tolist() == [200.0]
            assert ds["channels"].values.tolist() == ["T3,Cz"]
            assert ds["bad_channels"].values.tolist() == ["Cz"]
            assert ds["interpolated_channels"].values.tolist() == [""]
            assert ds.attrs["model_key"] == "cbramod"
            assert ds.attrs["description"] == "base"
            config = json.loads(ds.attrs["config"])
            assert config["ModelKey"] == "cbramod"
            assert config["NormalizeEmbeddings"] is True
            assert set(config) <= set(EMBEDDING_CONFIG_KEYS.values())
            assert stable_hash(config, length=16) == ds.attrs["config_hash"]
            hashes.add(ds.attrs["config_hash"])
            if view == "token":
                assert ds.attrs["token_source"] == "test_output"
                assert ds.attrs["token_feature_axis"] == "feature"
                assert ds["channel"].values.tolist() == ["T7", "Cz"]
            else:
                assert "channel" not in ds.coords
    assert len(hashes) == 1


def test_recordings_of_one_configuration_share_config_hash(tmp_path):
    def config_hash(subject, result):
        paths = write_recording_embeddings(
            {"EC": result},
            tmp_path / "out",
            label="base",
            source=source_path(tmp_path, subject=subject),
            source_dataset="preproc",
        )
        with xr.open_dataset(paths["window"].fpath, engine="h5netcdf") as ds:
            return ds.attrs["config_hash"]

    first = config_hash("01", extract([0, 1]))
    assert config_hash("02", extract([0], seed=1, channels=("Fz",))) == first
    assert config_hash("03", extract([0], normalize_embeddings=False)) != first


def _nan_at_first(result, view):
    array = np.array(getattr(result, f"{view}_embeddings"))
    array.flat[0] = np.nan
    return replace(result, **{f"{view}_embeddings": array})


@pytest.mark.parametrize(
    "results, message",
    [
        (
            lambda: {
                "EC": extract([0]),
                "EO": extract([1], normalize_embeddings=False),
            },
            "config_hash",
        ),
        (
            lambda: {"EC": extract([0]), "EO": extract([1], store_tokens=False)},
            "views",
        ),
        (
            lambda: {
                "EC": extract([0]),
                "EO": replace(
                    extract([1]),
                    window_embeddings=extract([1]).window_embeddings[:, :4],
                ),
            },
            "views",
        ),
        (
            lambda: {"EC": extract([0]), "EO": extract([1], channels=("T3", "Fz"))},
            "channels",
        ),
        (lambda: {"EC": _nan_at_first(extract([0]), "window")}, "non-finite"),
        (lambda: {"EC": _nan_at_first(extract([0]), "token")}, "non-finite"),
        (lambda: {"EC": extract([0, 1]), "EO": extract([1])}, "more than once"),
        (
            lambda: {
                "EC": replace(
                    extract([0]),
                    metadata={**extract([0]).metadata, "model_channels": ["Cz"]},
                )
            },
            "model_channels",
        ),
    ],
)
def test_write_recording_embeddings_refuses_before_writing(tmp_path, results, message):
    with pytest.raises(ValueError, match=message):
        write_recording_embeddings(
            results(),
            tmp_path / "out",
            label="base",
            source=source_path(tmp_path),
            source_dataset="preproc",
        )
    assert not (tmp_path / "out").exists()


def test_failed_write_leaves_the_previous_file(tmp_path, monkeypatch):
    def write(result):
        return write_recording_embeddings(
            {"EC": result},
            tmp_path / "out",
            label="base",
            source=source_path(tmp_path),
            source_dataset="preproc",
        )

    path = write(extract([0], store_tokens=False))["window"].fpath
    before = path.read_bytes()

    def fail(self, target, **kwargs):
        Path(target).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fail)
    with pytest.raises(OSError, match="disk full"):
        write(extract([0], seed=1, store_tokens=False))
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def _embedding_dataset(tmp_path, labels=("base",)):
    """A dataset with one recording's files and a top-level file per label."""
    root = tmp_path / "coco-pipe-cbramod"
    paths = write_recording_embeddings(
        {"EC": extract([0, 1])},
        root,
        label="base",
        source=source_path(tmp_path),
        source_dataset="preproc",
    )
    for label in labels:
        with xr.open_dataset(paths["window"].fpath, engine="h5netcdf") as ds:
            ds = ds.load()
        ds.attrs["description"] = label
        ds.to_netcdf(
            embedding_path(root, label, "window", task="rest").fpath,
            engine="h5netcdf",
        )
    return root, paths["window"]


def _write_dataset(root, **kwargs):
    return write_embedding_dataset(
        root,
        kwargs.pop("descriptions", {"base": "pretrained, every epoch"}),
        name="coco-pipe CBraMod embeddings",
        dataset_links=kwargs.pop("dataset_links", {"preproc": "../preproc"}),
        authors=["A. Author", "B. Author"],
    )


def test_write_embedding_dataset_writes_bids_files_and_sidecars(tmp_path):
    root, recording_file = _embedding_dataset(tmp_path)
    (root / ".bidsignore").write_text("extra/\n\n*.nc\n", encoding="utf-8")
    descriptions = {"base": "pretrained, every epoch", "baseRandom": "random weights"}
    written = _write_dataset(root, descriptions=descriptions)

    description = json.loads((root / "dataset_description.json").read_text())
    assert description["BIDSVersion"] == "1.10.1"
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"][0]["Name"] == "coco-pipe"
    assert description["DatasetLinks"] == {"preproc": "../preproc"}
    assert description["SourceDatasets"] == [{"URL": "../preproc"}]
    table = pd.read_csv(root / "descriptions.tsv", sep="\t")
    assert table.to_dict("list") == {
        "desc_id": ["desc-base", "desc-baseRandom"],
        "description": list(descriptions.values()),
    }
    assert "Description" in json.loads((root / "descriptions.json").read_text())
    assert (root / ".bidsignore").read_text().splitlines() == [
        "extra/",
        "*.nc",
        "*embeddings.nc",
        "*embeddings.json",
        "*.tar.zst",
        "*_manifest.json",
    ]
    assert "windowembeddings" in (root / "README.md").read_text()

    sidecar = root / "task-rest_desc-base_windowembeddings.json"
    with xr.open_dataset(recording_file.fpath, engine="h5netcdf") as ds:
        assert json.loads(sidecar.read_text()) == {
            "Description": "pretrained, every epoch: one vector per window.",
            **json.loads(ds.attrs["config"]),
            "ConfigHash": ds.attrs["config_hash"],
        }
    assert recording_file.find_matching_sidecar(extension=".json") == sidecar

    before = {path: path.read_bytes() for path in written}
    assert _write_dataset(root, descriptions=descriptions) == written
    assert {path: path.read_bytes() for path in written} == before


def _set_attrs(path, **attrs):
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        ds = ds.load()
    ds.attrs.update(attrs)
    ds.to_netcdf(path, engine="h5netcdf")


def _other_model(path):
    config = {"ModelKey": "reve"}
    _set_attrs(
        path, config=json.dumps(config), config_hash=stable_hash(config, length=16)
    )


@pytest.mark.parametrize(
    "change, kwargs, message",
    [
        (None, {"descriptions": {"baseX": "other"}}, "descriptions lacks"),
        (None, {"descriptions": {"base": "one\ttwo"}}, "one line"),
        (None, {"dataset_links": {"pre:proc": "../preproc"}}, "DatasetLinks"),
        (None, {"dataset_links": {"raw": ".."}}, "dataset_links must name"),
        (lambda path: _set_attrs(path, description="baseX"), {}, "holds the label"),
        (lambda path: _set_attrs(path, config_hash="0" * 16), {}, "hash"),
        (_other_model, {}, "one model"),
    ],
)
def test_write_embedding_dataset_refuses_before_writing(
    tmp_path, change, kwargs, message
):
    root, _ = _embedding_dataset(tmp_path, labels=("base", "baseRandom"))
    kwargs.setdefault("descriptions", {"base": "pretrained", "baseRandom": "random"})
    if change is not None:
        change(root / "task-rest_desc-baseRandom_windowembeddings.nc")
    with pytest.raises(ValueError, match=message):
        _write_dataset(root, **kwargs)
    assert not list(root.glob("*.json"))


def test_each_label_has_its_own_manifest(tmp_path):
    source = source_path(tmp_path)
    base = write_embedding_manifest(
        tmp_path, "base", [{"recording_id": "a", "status": "failed", "path": source}]
    )
    random = write_embedding_manifest(
        tmp_path, "baseRandom", [{"recording_id": "b", "status": "skipped"}]
    )
    assert base.name == "desc-base_manifest.json"
    assert random.name == "desc-baseRandom_manifest.json"
    write_embedding_manifest(tmp_path, "base", [])
    assert json.loads(base.read_text())["records"] == []
    assert json.loads(random.read_text())["records"] == [
        {"recording_id": "b", "status": "skipped"}
    ]
    write_embedding_manifest(tmp_path, "base", [{"path": source}])
    assert json.loads(base.read_text())["records"] == [
        {"path": source.fpath.as_posix()}
    ]
    assert {p.name for p in tmp_path.iterdir()} == {base.name, random.name}
    with pytest.raises(ValueError, match="BIDS label"):
        write_embedding_manifest(tmp_path, "base_x", [])


def test_load_embeddings_reads_windows_tokens_and_recordings(tmp_path, monkeypatch):
    # Blocks of 3 rows: the whole first file is read as two slices, its EO
    # windows (rows 1 and 3) as one list of rows.
    monkeypatch.setattr(embeddings_module, "_READ_ROWS", 3)
    first = {"EO": extract([1, 3], seed=1), "EC": extract([0, 2])}
    second = {"EC": extract([5], seed=2, channels=("Cz", "T3"))}
    write_recording(tmp_path, first, bad_channels=["Cz"])
    write_recording(tmp_path, second, subject="02", session=None, run=None)
    root = tmp_path / "out"
    first_id, second_id = "sub-01_ses-02_task-rest_run-03", "sub-02_task-rest"

    def expected(view, order=(0, 2, 1, 3, 4)):
        arrays = [
            getattr(first["EC"], f"{view}_embeddings"),
            getattr(first["EO"], f"{view}_embeddings"),
            getattr(second["EC"], f"{view}_embeddings"),
        ]
        if view == "token":
            # The second recording's channels are Cz, T7: put in T7, Cz order.
            arrays[2] = arrays[2][:, ::-1]
        return np.concatenate(arrays)[list(order)].astype(np.float32)

    windows = load_embeddings(find_embeddings(root, "base", "window"))
    assert windows.dims == ("obs", "feature")
    assert windows.X.dtype == np.float32
    np.testing.assert_array_equal(windows.X, expected("window"))
    assert windows.ids.tolist() == [
        *(f"{first_id}_epoch-{number:04d}" for number in range(4)),
        f"{second_id}_epoch-0005",
    ]
    for name, values in (
        ("condition", ["EC", "EO", "EC", "EO", "EC"]),
        ("epoch", [0, 1, 2, 3, 5]),
        ("subject", ["01"] * 4 + ["02"]),
        ("session", ["02"] * 4 + [""]),
        ("recording_id", [first_id] * 4 + [second_id]),
        ("bad_channels", ["Cz"] * 4 + [""]),
        ("start", [0, 10, 20, 30, 50]),
    ):
        assert windows.coords[name].tolist() == values
    meta = windows.meta
    assert (meta["model_key"], meta["description"], meta["view"]) == (
        "cbramod",
        "base",
        "window",
    )
    assert meta["config"]["ModelKey"] == "cbramod"
    assert stable_hash(meta["config"], length=16) == meta["config_hash"]
    assert [row["recording_id"] for row in meta["recordings"]] == [first_id, second_id]
    assert meta["recordings"][1]["source"] == (
        f"bids:preproc:sub-02/eeg/{second_id}_desc-base_epo.fif"
    )
    assert meta["recordings"][1]["channels"] == "Cz,T3"

    tokens = load_embeddings(find_embeddings(root, "base", "token"))
    assert tokens.dims == ("obs", "channel", "time_patch", "feature")
    assert tokens.coords["channel"].tolist() == ["T7", "Cz"]
    np.testing.assert_array_equal(tokens.X, expected("token"))
    assert tokens.meta["token_feature_axis"] == "feature"

    files = find_embeddings(root, "base", "window")
    eyes_open = load_embeddings(files, conditions="EO")
    np.testing.assert_array_equal(eyes_open.X, expected("window", [2, 3]))
    assert eyes_open.coords["epoch"].tolist() == [1, 3]
    assert load_embeddings(files, subjects=["02", "03"]).ids.tolist() == [
        f"{second_id}_epoch-0005"
    ]

    recordings = load_embeddings(files, representation="recording")
    assert recordings.ids.tolist() == [
        f"{first_id}_cond-EC",
        f"{first_id}_cond-EO",
        f"{second_id}_cond-EC",
    ]
    assert recordings.X.dtype == np.float32
    np.testing.assert_allclose(
        recordings.X,
        np.stack(
            [
                first["EC"].recording_embedding,
                first["EO"].recording_embedding,
                second["EC"].recording_embedding,
            ]
        ),
        rtol=1e-6,
    )
    assert recordings.coords["epoch_count"].tolist() == [2, 2, 1]
    assert recordings.coords["condition"].tolist() == ["EC", "EO", "EC"]
    assert "epoch" not in recordings.coords


def _rewrite(path, change):
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        dataset = dataset.load()
    change(dataset)
    dataset.to_netcdf(path, engine="h5netcdf")
    return path


def test_load_embeddings_upcasts_float16(tmp_path):
    path = write_recording(tmp_path, {"EC": extract([0, 1])})["token"].fpath

    def to_float16(dataset):
        dataset["embeddings"] = dataset["embeddings"].astype(np.float16)

    _rewrite(path, to_float16)
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        assert dataset["embeddings"].dtype == np.float16
        half = dataset["embeddings"].values
    loaded = load_embeddings(path)
    assert loaded.X.dtype == np.float32
    np.testing.assert_array_equal(loaded.X, half.astype(np.float32))


def _one(tmp_path, view="window", subject="01", label="base", **settings):
    paths = write_recording(
        tmp_path, {"EC": extract([0], **settings)}, subject=subject, label=label
    )
    return paths[view].fpath


def _set_nan(dataset):
    dataset["embeddings"][0, 0] = np.nan


@pytest.mark.parametrize(
    "source, kwargs, message",
    [
        (lambda tp: [], {}, "no embedding file"),
        (lambda tp: [tp / "sub-01_task-rest_epo.fif"], {}, "not an embedding file"),
        (_one, {"representation": "subject"}, "representation"),
        (
            lambda tp: [_one(tp), _one(tp, subject="02", normalize_embeddings=False)],
            {},
            "config_hash",
        ),
        (
            lambda tp: [_one(tp), _one(tp, subject="02", label="baseRandom")],
            {},
            "label",
        ),
        (lambda tp: [_one(tp), _one(tp, "token")], {}, "view"),
        (
            lambda tp: [
                _one(tp, "token"),
                _one(tp, "token", subject="02", channels=("Fz", "Cz")),
            ],
            {},
            "channels",
        ),
        (lambda tp: [_one(tp), _one(tp)], {}, "more than once"),
        (_one, {"conditions": ["EC", "EO"]}, "conditions"),
        (_one, {"subjects": "02"}, "No window"),
        (lambda tp: _rewrite(_one(tp), _set_nan), {}, "non-finite"),
    ],
)
def test_load_embeddings_refuses(tmp_path, source, kwargs, message):
    with pytest.raises(ValueError, match=message):
        load_embeddings(source(tmp_path), **kwargs)


def test_consolidated_file_reads_back_as_the_recording_files(tmp_path):
    write_recording(
        tmp_path,
        {"EO": extract([1, 3], seed=1), "EC": extract([0, 2])},
        bad_channels=["Cz"],
    )
    write_recording(
        tmp_path,
        {"EC": extract([5], seed=2, channels=("Cz", "T3"))},
        subject="02",
        session=None,
        run=None,
    )
    root = tmp_path / "out"
    for view in ("window", "token"):
        files = find_embeddings(root, "base", view)
        path = consolidate_embeddings(files, root)
        assert path.fpath == root / f"task-rest_desc-base_{view}embeddings.nc"
        assert find_embeddings(root, "base", view) == files
        combined, separate = load_embeddings(path), load_embeddings(files)
        assert combined.meta["files"] == [path.fpath.as_posix()]
        xr.testing.assert_identical(
            combined.data.assign_attrs(files=None),
            separate.data.assign_attrs(files=None),
        )
        with (
            xr.open_dataset(path.fpath, engine="h5netcdf") as whole,
            xr.open_dataset(files[0].fpath, engine="h5netcdf") as part,
        ):
            assert whole.attrs == part.attrs
            assert whole["embeddings"].dtype == np.float32
            assert whole["recording"].values.tolist() == [
                "sub-01_ses-02_task-rest_run-03",
                "sub-02_task-rest",
            ]


def test_consolidated_file_keeps_float16_and_drops_a_task_not_shared(tmp_path):
    def to_float16(dataset):
        dataset["embeddings"] = dataset["embeddings"].astype(np.float16)

    other = write_recording(tmp_path, {"EC": extract([0])}, subject="02", task="other")
    files = [
        _rewrite(_one(tmp_path, "token"), to_float16),
        _rewrite(other["token"].fpath, to_float16),
    ]
    path = consolidate_embeddings(files, tmp_path / "out")
    assert path.fpath.name == "desc-base_tokenembeddings.nc"
    with xr.open_dataset(path.fpath, engine="h5netcdf") as dataset:
        assert dataset["embeddings"].dtype == np.float16
    np.testing.assert_array_equal(load_embeddings(path).X, load_embeddings(files).X)


@pytest.mark.parametrize(
    "files, message",
    [
        (lambda tp: [], "no embedding file"),
        (
            lambda tp: [_one(tp), _one(tp, subject="02", normalize_embeddings=False)],
            "config_hash",
        ),
        (lambda tp: [_one(tp), _one(tp)], "more than once"),
        # Found while copying, after the file was begun.
        (
            lambda tp: [_one(tp), _rewrite(_one(tp, subject="02"), _set_nan)],
            "non-finite",
        ),
    ],
)
def test_consolidate_embeddings_refuses_and_leaves_no_file(tmp_path, files, message):
    root = tmp_path / "out"
    with pytest.raises(ValueError, match=message):
        consolidate_embeddings(files(tmp_path), root)
    assert not [path for path in root.glob("*") if path.is_file()]


def test_load_embeddings_attaches_subject_metadata(tmp_path):
    # The embeddings' source is preproc, whose source is raw: participants.tsv
    # is two datasets away, as in a real derivative tree.
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {"participant_id": ["sub-01", "sub-02"], "age": [9, 10], "group": ["a", "b"]}
    ).to_csv(raw / "participants.tsv", sep="\t", index=False)
    preproc = tmp_path / "preproc"
    preproc.mkdir()
    (preproc / "dataset_description.json").write_text(
        json.dumps({"DatasetType": "derivative", "SourceDatasets": [{"URL": "../raw"}]})
    )
    write_recording(tmp_path, {"EC": extract([0, 1]), "EO": extract([2], seed=1)})
    write_recording(tmp_path, {"EC": extract([0], seed=2)}, subject="03")
    root = tmp_path / "out"
    write_embedding_dataset(
        root,
        {"base": "Pretrained"},
        name="test",
        dataset_links={"preproc": "../preproc"},
    )
    files = find_embeddings(root, "base", "window")
    whole = consolidate_embeddings(files, root)

    for source in (files, whole):
        windows = load_embeddings(source)
        # sub-03 has no row in participants.tsv
        np.testing.assert_array_equal(windows.coords["age"], [9, 9, 9, np.nan])
        assert windows.coords["group"][:3].tolist() == ["a", "a", "a"]
        assert pd.isna(windows.coords["group"][3])

    meta = pd.DataFrame({"id": ["sub-03", "01"], "age": [11, 8], "iq": [100, 90]})
    windows = load_embeddings(whole, subject_metadata_df=meta, subject_key="id")
    assert windows.coords["age"].tolist() == [8, 8, 8, 11]
    assert windows.coords["iq"].tolist() == [90, 90, 90, 100]
    assert "id" not in windows.coords
    recordings = load_embeddings(
        whole, representation="recording", subject_metadata_df=meta, subject_key="id"
    )
    assert recordings.coords["age"].tolist() == [8, 8, 11]

    with pytest.raises(ValueError, match=r"\['condition'\] have the names"):
        load_embeddings(
            whole, subject_metadata_df=meta.assign(condition="x"), subject_key="id"
        )
    with pytest.raises(ValueError, match="subject_key must be provided"):
        load_embeddings(whole, subject_metadata_df=meta)
