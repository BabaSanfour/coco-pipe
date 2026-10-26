import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from mne_bids import BIDSPath, get_entities_from_fname

from coco_pipe.decoding.foundation_models import FoundationEmbeddingExtractor
from coco_pipe.io import EMBEDDING_CONFIG_KEYS, DataContainer
from coco_pipe.io import embeddings as embeddings_module
from coco_pipe.io.embeddings import (
    consolidate_embeddings,
    embedding_label,
    embedding_path,
    embeddings_up_to_date,
    find_embeddings,
    load_embeddings,
    write_embedding_dataset,
    write_embedding_manifest,
    write_recording_embeddings,
    write_transformed_embeddings,
)
from coco_pipe.transforms.subject_alignment import EuclideanAlign, RiemannAlign
from coco_pipe.utils import stable_hash
from tests.fixtures.embeddings import TokenModel, extract, source_path, write_recording


def test_embedding_path_names_recording_and_dataset_files(tmp_path):
    recording = embedding_path(
        tmp_path,
        "baseRandom0EaCoral",
        "window",
        recording_id="sub-01_ses-02_task-rest_run-03",
    ).fpath
    assert recording == (
        tmp_path
        / "sub-01"
        / "ses-02"
        / "eeg"
        / "sub-01_ses-02_task-rest_run-03_desc-baseRandom0EaCoral_windowembeddings.nc"
    )
    entities = get_entities_from_fname(recording.name)
    assert entities["description"] == "baseRandom0EaCoral"
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
    assert {key: path.basename for key, path in paths.items()} == {
        ("base", view): f"{recording}_desc-base_{view}embeddings.nc"
        for view in ("window", "token", "channel", "patch")
    }
    # Windows in epoch order: EO's 1 and 3 between EC's 0, 2 and 4.
    order = [2, 0, 3, 1, 4]
    hashes = set()
    for view, dims in (
        ("window", ("window", "feature")),
        ("token", ("window", "channel", "time_patch", "feature")),
        ("channel", ("window", "channel", "feature")),
        ("patch", ("window", "time_patch", "feature")),
    ):
        with xr.open_dataset(paths["base", view].fpath, engine="h5netcdf") as ds:
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
            assert config["Weights"] == "pretrained" and "Seed" not in config
            assert set(config) <= set(EMBEDDING_CONFIG_KEYS.values())
            assert stable_hash(config, length=16) == ds.attrs["config_hash"]
            hashes.add(ds.attrs["config_hash"])
            if view == "token":
                assert ds.attrs["token_source"] == "test_output"
                assert ds.attrs["token_feature_axis"] == "feature"
            if "channel" in dims:
                assert ds["channel"].values.tolist() == ["T7", "Cz"]
            else:
                assert "channel" not in ds.coords
    assert len(hashes) == 1


def test_extra_poolings_get_window_files_of_their_own_label(tmp_path):
    results = {
        "EO": extract([1], seed=1, extra_poolings="max"),
        "EC": extract([0, 2], extra_poolings="max"),
    }
    paths = write_recording(tmp_path, results)
    recording = "sub-01_ses-02_task-rest_run-03"
    assert {key for key in paths if key[0] != "base"} == {("baseMax", "window")}
    assert paths["baseMax", "window"].basename == (
        f"{recording}_desc-baseMax_windowembeddings.nc"
    )
    pooled = load_embeddings(paths["baseMax", "window"].fpath)
    windows = load_embeddings(paths["base", "window"].fpath)
    expected = np.concatenate(
        [results[c].extra_window_embeddings["max"] for c in ("EO", "EC")]
    )[[1, 0, 2]]
    np.testing.assert_array_equal(pooled.X, expected.astype(np.float32))
    assert pooled.ids.tolist() == windows.ids.tolist()
    with (
        xr.open_dataset(paths["base", "window"].fpath, engine="h5netcdf") as base,
        xr.open_dataset(paths["baseMax", "window"].fpath, engine="h5netcdf") as ds,
    ):
        assert ds.attrs["description"] == "baseMax"
        config = json.loads(base.attrs["config"])
        assert config["WindowPooling"] == "mean"
        assert json.loads(ds.attrs["config"]) == {**config, "WindowPooling": "max"}
        assert ds.attrs["config_hash"] != base.attrs["config_hash"]
    plain = write_recording(
        tmp_path / "plain",
        {"EO": extract([1], seed=1), "EC": extract([0, 2])},
    )
    assert set(plain) == {key for key in paths if key[0] == "base"}
    np.testing.assert_array_equal(
        load_embeddings(plain["base", "window"].fpath).X, windows.X
    )


def test_tokens_can_be_written_as_float16(tmp_path):
    result = extract([0, 1])

    def write(results, token_dtype):
        return write_recording_embeddings(
            results,
            tmp_path / "out",
            label="base",
            source=source_path(tmp_path),
            source_dataset="preproc",
            token_dtype=token_dtype,
        )

    paths = write({"EC": result}, "float16")
    half = result.token_embeddings.astype(np.float16)
    with xr.open_dataset(paths["base", "token"].fpath, engine="h5netcdf") as ds:
        assert ds["embeddings"].dtype == np.float16
        np.testing.assert_array_equal(ds["embeddings"], half)
    for view in ("window", "channel", "patch"):
        with xr.open_dataset(paths["base", view].fpath, engine="h5netcdf") as ds:
            assert ds["embeddings"].dtype == np.float32
    loaded = load_embeddings(paths["base", "token"].fpath)
    assert loaded.X.dtype == np.float32
    np.testing.assert_array_equal(loaded.X, half.astype(np.float32))

    before = {path: path.fpath.read_bytes() for path in paths.values()}
    large = replace(result, token_embeddings=np.full_like(half, 1e5, dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite values as float16"):
        write({"EC": large}, "float16")
    assert {path: path.fpath.read_bytes() for path in paths.values()} == before
    with pytest.raises(ValueError, match="token_dtype must be"):
        write({"EC": result}, "float64")


def _aligned_tokens(tmp_path):
    write_recording(tmp_path, {"EC": extract([0, 2], seed=1), "EO": extract([1, 3])})
    write_recording(tmp_path, {"EC": extract([0, 1], seed=2)}, subject="02")
    tokens = load_embeddings(find_embeddings(tmp_path / "out", "base", "token"))
    aligner = EuclideanAlign("coral", channel_axis=1)
    aligned = aligner.fit_transform(tokens.X, groups=tokens.coords["subject"])
    return tokens, aligned, aligner


def test_transformed_embeddings_round_trip(tmp_path):
    tokens, aligned, aligner = _aligned_tokens(tmp_path)
    root = tmp_path / "out"
    path = write_transformed_embeddings(
        tokens, aligned, root, transform=aligner, fitted_on="every window"
    )
    assert path.basename == "task-rest_desc-baseEaCoralCh_tokenembeddings.nc"
    loaded = load_embeddings(path.fpath)
    np.testing.assert_array_equal(loaded.X, aligned)
    assert loaded.dims == tokens.dims
    for name in ("obs", "subject", "condition", "epoch", "start", "channel"):
        assert loaded.coords[name].tolist() == tokens.coords[name].tolist()
    assert loaded.meta["recordings"] == tokens.meta["recordings"]
    # The source's normalisation does not describe the transformed vectors.
    source = dict(tokens.meta["config"])
    normalized = source.pop("NormalizeEmbeddings")
    assert normalized is True
    assert loaded.meta["config"] == {
        **source,
        "Transform": {
            "name": "ea_coral",
            "label": "EaCoralCh",
            "channel_axis": 1,
            "mode": "coral",
            "shrinkage": True,
            "fitted_on": "every window",
        },
    }
    assert loaded.meta["config_hash"] != tokens.meta["config_hash"]
    with pytest.raises(ValueError):
        load_embeddings([tokens.meta["files"][0], path.fpath])

    tangent = np.random.default_rng(0).normal(size=(len(aligned), 15))
    cov = write_transformed_embeddings(
        tokens, tangent, root, transform=RiemannAlign(), view="cov"
    )
    assert cov.basename == "task-rest_desc-baseRa_covembeddings.nc"
    assert load_embeddings(cov.fpath).dims == ("obs", "feature")
    _write_dataset(root, descriptions={"base": "Every epoch"})
    table = pd.read_csv(root / "descriptions.tsv", sep="\t")
    ea = (
        "ea_coral transform (channel_axis=1, mode=coral, shrinkage=True, "
        "fitted_on=every window)"
    )
    assert table.to_dict("list") == {
        "desc_id": ["desc-baseEaCoralCh", "desc-baseRa"],
        "description": [
            f"Every epoch, pretrained weights, {ea}",
            "Every epoch, pretrained weights, ra transform (shrinkage=0.1)",
        ],
    }
    sidecar = json.loads(path.fpath.with_suffix(".json").read_text())
    assert sidecar["Transform"]["name"] == "ea_coral"


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda t, a: (t, a[:-1], {}), "rows"),
        (lambda t, a: (t, np.where(a > 0, np.nan, a), {}), "non-finite"),
        (lambda t, a: (t, a[:, 0], {"view": "cov"}), "expected axes"),
        (lambda t, a: (t, a, {"view": "window"}), "expected axes"),
        (lambda t, a: (t, a, {"token_dtype": "float64"}), "token_dtype"),
        (
            lambda t, a: (t, a, {"transform": EuclideanAlign("coral", channel_axis=2)}),
            "no channel axis",
        ),
    ],
)
def test_transformed_embeddings_refused(tmp_path, change, message):
    source, embeddings, settings = change(*_aligned_tokens(tmp_path)[:2])
    settings = {"transform": EuclideanAlign("coral", channel_axis=1), **settings}
    with pytest.raises(ValueError, match=message):
        write_transformed_embeddings(source, embeddings, tmp_path / "out", **settings)
    assert not find_embeddings(tmp_path / "out", "baseEaCoralCh", "token")


def test_transformed_embeddings_need_window_rows_of_a_plain_file(tmp_path):
    tokens, aligned, aligner = _aligned_tokens(tmp_path)
    root = tmp_path / "out"
    files = find_embeddings(root, "base", "window")
    by_recording = load_embeddings(files, representation="recording")
    with pytest.raises(ValueError, match="one row per window"):
        write_transformed_embeddings(
            by_recording, by_recording.X, root, transform=EuclideanAlign("mean")
        )
    path = write_transformed_embeddings(tokens, aligned, root, transform=aligner)
    again = load_embeddings(path.fpath)
    with pytest.raises(ValueError, match="already transformed"):
        write_transformed_embeddings(
            again, again.X, root, transform=EuclideanAlign("mean")
        )


def test_recordings_of_one_configuration_share_config_hash(tmp_path):
    def config_hash(subject, result):
        paths = write_recording_embeddings(
            {"EC": result},
            tmp_path / "out",
            label="base",
            source=source_path(tmp_path, subject=subject),
            source_dataset="preproc",
        )
        label = embedding_label("base", seed=result.metadata.get("seed"))
        assert {key[0] for key in paths} == {label}
        with xr.open_dataset(paths[label, "window"].fpath, engine="h5netcdf") as ds:
            return ds.attrs["config_hash"]

    first = config_hash("01", extract([0, 1]))
    assert config_hash("02", extract([0], seed=1, channels=("Fz",))) == first
    assert config_hash("03", extract([0], normalize_embeddings=False)) != first
    random = extract([0])
    random = replace(random, metadata={**random.metadata, "weights": "random"})
    seeds = {
        config_hash(
            f"0{seed + 4}", replace(random, metadata={**random.metadata, "seed": seed})
        )
        for seed in (0, 1)
    }
    assert len(seeds) == 2 and first not in seeds


def _recording(tmp_path, subject, windows):
    """The BIDS loader's rows of one recording, a (condition, epoch) per window."""
    source = source_path(tmp_path, subject=subject)
    recording_id = f"sub-{subject}_ses-02_task-rest_run-03"
    n_windows = len(windows)
    return DataContainer(
        X=np.random.default_rng(int(subject)).normal(size=(n_windows, 2, 10)),
        dims=("obs", "channel", "time"),
        coords={
            "channel": np.array(["T3", "Cz"], dtype=object),
            "time": np.arange(10) / 200.0,
            "condition": np.array([name for name, _ in windows], dtype=object),
            "epoch": np.array([epoch for _, epoch in windows]),
            "recording_id": np.full(n_windows, recording_id, dtype=object),
            "source": np.full(n_windows, str(source), dtype=object),
        },
        ids=np.array([f"{recording_id}_epoch-{e:04d}" for _, e in windows]),
        meta={"sfreq": 200.0},
    )


def test_embeddings_are_up_to_date_when_their_files_hold_windows_and_settings(
    tmp_path,
):
    def write(recording, skip_up_to_date=False, **settings):
        extractor = FoundationEmbeddingExtractor(
            "cbramod", model=TokenModel(), resample=False, **settings
        )
        return write_recording_embeddings(
            extractor.extract_recording(recording),
            tmp_path / "out",
            label="base",
            source=recording.coords["source"][0],
            source_dataset="preproc",
            bad_channels="T3,Cz",
            skip_up_to_date=skip_up_to_date,
        )

    first = _recording(tmp_path, "01", [("EO", 0), ("EC", 1), ("EO", 2)])
    written = write(first)
    with xr.open_dataset(written["base", "window"].fpath, engine="h5netcdf") as ds:
        assert ds["condition"].values.tolist() == ["EO", "EC", "EO"]
        assert ds["start"].values.tolist() == [0, 10, 20]
        assert ds["bad_channels"].values.tolist() == ["T3,Cz"]
    assert embeddings_up_to_date(first, written)
    assert not embeddings_up_to_date(first.isel(obs=[0, 2]), written)

    second = _recording(tmp_path, "02", [("EC", 0)])
    assert not embeddings_up_to_date(second, written)
    write(second, normalize_embeddings=False)
    assert not embeddings_up_to_date(second, written)
    window = write(second, skip_up_to_date=True)["base", "window"].fpath
    assert embeddings_up_to_date(second, written)

    # An up-to-date file is kept, another one replaced.
    kept = window.stat().st_mtime_ns
    write(second, skip_up_to_date=True)
    assert window.stat().st_mtime_ns == kept
    write(second, skip_up_to_date=True, normalize_embeddings=False)
    assert window.stat().st_mtime_ns != kept


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
        (
            lambda: {"EC": extract([0], extra_poolings="max"), "EO": extract([1])},
            "poolings",
        ),
        (lambda: {"EC": _nan_at_first(extract([0]), "window")}, "non-finite"),
        (
            lambda: {
                "EC": replace(
                    extract([0]),
                    extra_window_embeddings={
                        "max": np.full((1, 5), np.nan, dtype=np.float32)
                    },
                )
            },
            "non-finite",
        ),
        (
            lambda: {
                "EC": replace(
                    extract([0]),
                    extra_window_embeddings={"max-1": extract([0]).window_embeddings},
                )
            },
            "letters and digits",
        ),
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

    paths = [path.fpath for path in write(extract([0], store_tokens=False)).values()]
    before = [path.read_bytes() for path in paths]

    def fail(self, target, **kwargs):
        Path(target).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fail)
    with pytest.raises(OSError, match="disk full"):
        write(extract([0], seed=1, store_tokens=False))
    assert [path.read_bytes() for path in paths] == before
    assert sorted(paths[0].parent.iterdir()) == sorted(paths)


def _embedding_dataset(tmp_path):
    """A recording's window files, pretrained and random, and their top-level files."""
    root = tmp_path / "coco-pipe-cbramod"
    pretrained = extract([0, 1])
    random = replace(
        pretrained, metadata={**pretrained.metadata, "weights": "random", "seed": 0}
    )
    for result in (pretrained, random):
        paths = write_recording_embeddings(
            {"EC": result},
            root,
            label="base",
            source=source_path(tmp_path),
            source_dataset="preproc",
        )
        window = next(path for (_, view), path in paths.items() if view == "window")
        consolidate_embeddings([window], root)
    return root, find_embeddings(root, "base", "window")[0]


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
    descriptions = {"base": "Every epoch"}
    written = _write_dataset(root, descriptions=descriptions)

    description = json.loads((root / "dataset_description.json").read_text())
    assert description["BIDSVersion"] == "1.10.1"
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"][0]["Name"] == "coco-pipe"
    assert description["DatasetLinks"] == {"preproc": "../preproc"}
    assert description["SourceDatasets"] == [{"URL": "../preproc"}]
    table = pd.read_csv(root / "descriptions.tsv", sep="\t")
    assert table.to_dict("list") == {
        "desc_id": ["desc-base", "desc-baseRandom0"],
        "description": [
            "Every epoch, pretrained weights",
            "Every epoch, random weights (seed 0)",
        ],
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
            "Description": "Every epoch, pretrained weights: one vector per window.",
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
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        config = {**json.loads(ds.attrs["config"]), "ModelKey": "reve"}
    _set_attrs(
        path, config=json.dumps(config), config_hash=stable_hash(config, length=16)
    )


@pytest.mark.parametrize(
    "change, kwargs, message",
    [
        (None, {"descriptions": {"baseX": "other"}}, "no label of descriptions"),
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
    root, _ = _embedding_dataset(tmp_path)
    kwargs.setdefault("descriptions", {"base": "Every epoch"})
    if change is not None:
        change(root / "task-rest_desc-baseRandom0_windowembeddings.nc")
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
    assert json.loads(base.read_text())["records"] == [{"path": source.as_posix()}]
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
        if view in ("token", "channel"):
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

    for view, dims in (
        ("channel", ("obs", "channel", "feature")),
        ("patch", ("obs", "time_patch", "feature")),
    ):
        loaded = load_embeddings(find_embeddings(root, "base", view))
        assert loaded.dims == dims
        np.testing.assert_array_equal(loaded.X, expected(view))
        assert loaded.meta["view"] == view
        if "channel" in dims:
            assert loaded.coords["channel"].tolist() == ["T7", "Cz"]

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
                first["EC"].window_embeddings.mean(axis=0),
                first["EO"].window_embeddings.mean(axis=0),
                second["EC"].window_embeddings.mean(axis=0),
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
    path = write_recording(tmp_path, {"EC": extract([0, 1])})["base", "token"].fpath

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
    return paths[label, view].fpath


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
        _rewrite(other["base", "token"].fpath, to_float16),
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
