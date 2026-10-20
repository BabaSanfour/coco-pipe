import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from mne_bids import BIDSPath, get_entities_from_fname

from coco_pipe.decoding._specs import SignalMetadata
from coco_pipe.decoding.foundation_models import FoundationEmbeddingExtractor
from coco_pipe.decoding.registry import get_foundation_model_spec
from coco_pipe.io import EMBEDDING_CONFIG_KEYS
from coco_pipe.io.embeddings import (
    embedding_path,
    write_embedding_dataset,
    write_embedding_manifest,
    write_recording_embeddings,
)
from coco_pipe.utils import stable_hash


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


class _TokenModel:
    """Stands in for CBraMod: tokens are (window, channel, time_patch, feature)."""

    def transform(self, X, *, return_tokens=False):
        tokens = np.stack([X[..., :5], X[..., 5:]], axis=2)
        pooled = tokens.mean(axis=(1, 2))
        return (pooled, tokens) if return_tokens else pooled

    def get_token_output_metadata(self):
        return {
            "token_source": "test_output",
            "token_axes": ["window", "channel", "time_patch", "feature"],
            "token_observation_axes": ["channel", "time_patch"],
            "token_feature_axis": "feature",
        }

    def get_embedding_info(self):
        return get_foundation_model_spec("cbramod")


def _extract(epochs, *, seed=0, channels=("T3", "Cz"), store_tokens=True, **settings):
    """The extractor's result for windows numbered ``epochs``, 10 samples each."""
    X = np.random.default_rng(seed).normal(size=(len(epochs), len(channels), 10))
    extractor = FoundationEmbeddingExtractor(
        "cbramod",
        model=_TokenModel(),
        resample=False,
        store_tokens=store_tokens,
        **settings,
    )
    starts = np.asarray(epochs) * 10
    return extractor.extract(
        X,
        signal_metadata=SignalMetadata(sfreq=200.0, ch_names=list(channels)),
        window_start=starts,
        window_stop=starts + 10,
        window_index=epochs,
    )


def _source(tmp_path, **entities):
    entities = {"subject": "01", "session": "02", "run": "03", **entities}
    return BIDSPath(
        root=tmp_path / "preproc",
        task="rest",
        description="base",
        suffix="epo",
        extension=".fif",
        datatype="eeg",
        check=False,
        **entities,
    )


def test_write_recording_embeddings_round_trip(tmp_path):
    results = {"EO": _extract([1, 3], seed=1), "EC": _extract([0, 2, 4])}
    paths = write_recording_embeddings(
        results,
        tmp_path / "out",
        label="base",
        source=_source(tmp_path),
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
    assert len(hashes) == 1


def test_recordings_of_one_configuration_share_config_hash(tmp_path):
    def config_hash(subject, result):
        paths = write_recording_embeddings(
            {"EC": result},
            tmp_path / "out",
            label="base",
            source=_source(tmp_path, subject=subject),
            source_dataset="preproc",
        )
        with xr.open_dataset(paths["window"].fpath, engine="h5netcdf") as ds:
            return ds.attrs["config_hash"]

    first = config_hash("01", _extract([0, 1]))
    assert config_hash("02", _extract([0], seed=1, channels=("Fz",))) == first
    assert config_hash("03", _extract([0], normalize_embeddings=False)) != first


def _nan_at_first(result, view):
    array = np.array(getattr(result, f"{view}_embeddings"))
    array.flat[0] = np.nan
    return replace(result, **{f"{view}_embeddings": array})


@pytest.mark.parametrize(
    "results, message",
    [
        (
            lambda: {
                "EC": _extract([0]),
                "EO": _extract([1], normalize_embeddings=False),
            },
            "config_hash",
        ),
        (
            lambda: {"EC": _extract([0]), "EO": _extract([1], store_tokens=False)},
            "views",
        ),
        (
            lambda: {
                "EC": _extract([0]),
                "EO": replace(
                    _extract([1]),
                    window_embeddings=_extract([1]).window_embeddings[:, :4],
                ),
            },
            "views",
        ),
        (
            lambda: {"EC": _extract([0]), "EO": _extract([1], channels=("T3", "Fz"))},
            "channels",
        ),
        (lambda: {"EC": _nan_at_first(_extract([0]), "window")}, "non-finite"),
        (lambda: {"EC": _nan_at_first(_extract([0]), "token")}, "non-finite"),
        (lambda: {"EC": _extract([0, 1]), "EO": _extract([1])}, "more than once"),
    ],
)
def test_write_recording_embeddings_refuses_before_writing(tmp_path, results, message):
    with pytest.raises(ValueError, match=message):
        write_recording_embeddings(
            results(),
            tmp_path / "out",
            label="base",
            source=_source(tmp_path),
            source_dataset="preproc",
        )
    assert not (tmp_path / "out").exists()


def test_failed_write_leaves_the_previous_file(tmp_path, monkeypatch):
    def write(result):
        return write_recording_embeddings(
            {"EC": result},
            tmp_path / "out",
            label="base",
            source=_source(tmp_path),
            source_dataset="preproc",
        )

    path = write(_extract([0], store_tokens=False))["window"].fpath
    before = path.read_bytes()

    def fail(self, target, **kwargs):
        Path(target).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fail)
    with pytest.raises(OSError, match="disk full"):
        write(_extract([0], seed=1, store_tokens=False))
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def _embedding_dataset(tmp_path, labels=("base",)):
    """A dataset with one recording's files and a top-level file per label."""
    root = tmp_path / "coco-pipe-cbramod"
    paths = write_recording_embeddings(
        {"EC": _extract([0, 1])},
        root,
        label="base",
        source=_source(tmp_path),
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
    source = _source(tmp_path)
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
