"""BIDS-compatible derivative I/O for foundation-model embeddings."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from mne_bids import BIDSPath, get_bids_path_from_fname, get_entities_from_fname

from coco_pipe.utils import get_package_version, stable_hash

from ._constants import EMBEDDING_CONFIG_KEYS, EMBEDDING_VIEWS
from ._serialization import atomic_path, write_json

_RECORDING_ENTITIES = frozenset({"subject", "session", "task", "acquisition", "run"})

# descriptions.tsv came in BIDS 1.10; the validator (schema 1.2.7) does not
# know 1.11 yet, and mne-bids 0.20 writes 1.9.0.
_BIDS_VERSION = "1.10.1"
_CODE_URL = "https://github.com/BabaSanfour/coco-pipe"
_BIDSIGNORE = ("*embeddings.nc", "*embeddings.json", "*.tar.zst", "*_manifest.json")

_VIEW_TEXT = {
    "token": "the model's token tensor of each window, nothing pooled",
    "window": "one vector per window",
    "channel": "one vector per channel of each window, averaged over time patches",
    "patch": "one vector per time patch of each window, averaged over channels",
    "cov": "the tangent-space covariance of each window's tokens",
}

_VIEW_DIMS = {
    "window": ("window", "feature"),
    "channel": ("window", "channel", "feature"),
    "patch": ("window", "time_patch", "feature"),
    "cov": ("window", "feature"),
}


def _check_label(label: str) -> None:
    """Refuse a ``desc`` label that is not a BIDS label."""
    if not re.fullmatch(r"[A-Za-z0-9+]+", label):
        raise ValueError(
            f"label must be letters, digits and '+' (a BIDS label), got {label!r}."
        )


def embedding_path(
    root: str | Path,
    label: str,
    view: str,
    *,
    recording_id: str | None = None,
    task: str | None = None,
    datatype: str = "eeg",
) -> BIDSPath:
    """Return the BIDS path of an embedding file.

    A file is named ``<entities>_desc-<label>_<view>embeddings.nc``. With
    ``recording_id`` it holds one recording and sits in its subject (and
    session) folder; without it, it holds every recording and sits at the
    dataset root, named by ``task`` when one is given.

    Parameters
    ----------
    root : path-like
        The derivative dataset, e.g. ``derivatives/coco-pipe-cbramod``.
    label : str
        The ``desc`` label: the processing chain, e.g. ``"base"`` or
        ``"baseRandomEaCoral"``. A BIDS label: letters, digits and ``+``.
    view : str
        One of :data:`~coco_pipe.io.EMBEDDING_VIEWS`.
    recording_id : str, optional
        A recording's BIDS entities, as the BIDS loader's ``recording_id``
        coordinate holds them, e.g. ``"sub-01_ses-01_task-rest_run-01"``.
    task : str, optional
        The task of a file holding every recording. A recording's file takes
        its task from ``recording_id``.
    datatype : str
        The BIDS datatype folder of a recording's file, e.g. ``"eeg"``.

    Returns
    -------
    BIDSPath
        The file's path, without checks for BIDS suffixes; ``.fpath`` is the
        file.

    Raises
    ------
    ValueError
        If ``label`` is not a BIDS label, ``view`` is not an embedding view,
        ``recording_id`` is not a recording's entities, or both
        ``recording_id`` and ``task`` are given.
    """
    _check_label(label)
    if view not in EMBEDDING_VIEWS:
        raise ValueError(f"view must be one of {EMBEDDING_VIEWS}, got {view!r}.")
    if recording_id is None:
        entities = {"task": task}
    elif task is not None:
        raise ValueError("A recording's file takes its task from recording_id.")
    else:
        parsed = get_entities_from_fname(recording_id, on_error="ignore")
        entities = {key: value for key, value in parsed.items() if value is not None}
        if (
            "subject" not in entities
            or not set(entities) <= _RECORDING_ENTITIES
            or BIDSPath(**entities).basename != recording_id
        ):
            raise ValueError(
                "recording_id must be a recording's BIDS entities (sub, ses, task, "
                "acq, run), e.g. 'sub-01_ses-01_task-rest_run-01'; "
                f"got {recording_id!r}."
            )
        entities["datatype"] = datatype
    return BIDSPath(
        root=root,
        **entities,
        description=label,
        suffix=f"{view}embeddings",
        extension=".nc",
        check=False,
    )


def _recording_id(bids_path: BIDSPath) -> str:
    """Name a recording by its entities, as the BIDS loader does."""
    return BIDSPath(
        subject=bids_path.subject,
        session=bids_path.session,
        task=bids_path.task,
        acquisition=bids_path.acquisition,
        run=bids_path.run,
    ).basename


def _channel_list(names: Sequence[str]) -> str:
    """Join channel names with commas, as the BIDS loader's bad_channels."""
    return ",".join(str(name) for name in names)


def _view_dims(view: str, array: np.ndarray, metadata: Mapping[str, Any]) -> tuple:
    """Return the dimension names of one view's array, checking its rank."""
    if view != "token":
        dims = _VIEW_DIMS[view]
    else:
        dims = tuple(metadata.get("token_axes") or ())
        feature_axis = metadata.get("token_feature_axis")
        if dims[:1] != ("window",) or len(set(dims)) != len(dims):
            raise ValueError(
                f"token_axes must be unique names starting with 'window', got {dims}."
            )
        if feature_axis not in dims[1:] or not metadata.get("token_source"):
            raise ValueError(
                "Token metadata must name token_source and a token_feature_axis "
                "among token_axes."
            )
    if array.ndim != len(dims):
        raise ValueError(
            f"{view} embeddings have shape {array.shape}, expected axes {dims}."
        )
    return dims


def _write_netcdf(dataset: xr.Dataset, path: Path) -> None:
    """Write a dataset to a temporary sibling, then move it into place."""
    with atomic_path(path) as tmp:
        dataset.to_netcdf(tmp, engine="h5netcdf")


def write_recording_embeddings(
    results: Mapping[str, Any],
    root: str | Path,
    *,
    label: str,
    source: BIDSPath,
    source_dataset: str,
    bad_channels: Sequence[str] = (),
) -> dict[str, BIDSPath]:
    """Write one recording's embeddings, one file per view.

    Each view the results hold (``window_embeddings``, ``token_embeddings``,
    and so on, see :data:`~coco_pipe.io.EMBEDDING_VIEWS`) goes to its own
    file, named by :func:`embedding_path` after the source recording. A file
    holds every condition, its windows in epoch order. Nothing is written if
    any check fails, and each file is written to a temporary name and then
    moved into place, so an existing file is replaced whole or not at all.

    Parameters
    ----------
    results : mapping of str to FoundationEmbeddingResult
        The extractor's result for each condition of the recording. Its
        ``window_index`` holds the epoch numbers.
    root : path-like
        The derivative dataset, e.g. ``derivatives/coco-pipe-cbramod``.
    label : str
        The ``desc`` label: the processing chain, e.g. ``"base"``.
    source : BIDSPath
        The file the epochs were read from, with its ``root`` and
        ``datatype``. Its entities name the recording, and the embedding
        files go in the same datatype folder.
    source_dataset : str
        The name of the source dataset in ``DatasetLinks``, e.g.
        ``"preproc"``, for the BIDS URI of ``source``.
    bad_channels : sequence of str
        The recording's bad channels.

    Returns
    -------
    dict of str to BIDSPath
        The path written for each view.

    Raises
    ------
    ValueError
        If the conditions come from different models or settings (their
        ``config_hash``), differ in views, feature size or channels, hold
        non-finite values or share epoch numbers, or if ``source`` lacks its
        root, subject or datatype.

    Notes
    -----
    The ``embeddings`` variable holds the view, as float32. Along ``window``
    are the coordinates ``obs`` (``<recording_id>_epoch-NNNN``, as the BIDS
    loader names observations), ``recording_id``, ``subject``, ``session``,
    ``task``, ``run``, ``condition``, ``epoch``, ``start`` and ``stop``.
    Along ``recording`` are its ``source`` (a BIDS URI), sampling rate,
    channels, bad channels and the channels the model interpolated,
    zero-filled or dropped. The attributes are ``model_key``, ``config``
    (the extraction settings that are the same for every recording, as JSON
    text, see :data:`~coco_pipe.io.EMBEDDING_CONFIG_KEYS`), ``config_hash``
    (their hash) and ``description`` (the label); a token file adds
    ``token_source`` and ``token_feature_axis``, the other token axes being
    the observation axes. Other metadata in the results is not stored.
    """
    if not results:
        raise ValueError("results must hold at least one condition.")
    if source.root is None or source.subject is None or source.datatype is None:
        raise ValueError("source must be a BIDSPath with root, subject and datatype.")
    if not source_dataset or ":" in source_dataset:
        raise ValueError(
            f"source_dataset must be a DatasetLinks name, got {source_dataset!r}."
        )
    recording_id = _recording_id(source)
    relative = source.fpath.relative_to(Path(source.root)).as_posix()

    views: dict[str, list[np.ndarray]] = {}
    windows: dict[str, list[np.ndarray]] = {
        "condition": [],
        "epoch": [],
        "start": [],
        "stop": [],
    }
    reference: dict[str, Any] | None = None
    for condition, result in results.items():
        if not isinstance(condition, str) or not condition:
            raise ValueError(f"Conditions must be names, got {condition!r}.")
        metadata = dict(result.metadata or {})
        if not metadata.get("model_key"):
            raise ValueError(f"Condition {condition!r} has no model_key in metadata.")
        config = {
            field: _json_value(metadata[key])
            for key, field in EMBEDDING_CONFIG_KEYS.items()
            if key in metadata
        }
        adaptation = metadata.get("channel_adaptation") or {}
        try:
            recording = {
                "sfreq": float(metadata["input_sfreq"]),
                "channels": _channel_list(metadata["original_channels"]),
            }
        except KeyError as exc:
            raise ValueError(
                f"Condition {condition!r} has no {exc} in metadata."
            ) from None
        for kind in ("interpolated", "zero_filled", "dropped"):
            recording[f"{kind}_channels"] = _channel_list(
                adaptation.get(f"{kind}_channels", ())
            )
        arrays = {
            view: np.asarray(array, dtype=np.float32)
            for view in EMBEDDING_VIEWS
            if (array := getattr(result, f"{view}_embeddings", None)) is not None
        }
        if "window" not in arrays:
            raise ValueError(f"Condition {condition!r} has no window embeddings.")
        current = {
            "config_hash": stable_hash(config, length=16),
            "config": config,
            "recording": recording,
            "views": {
                view: (_view_dims(view, array, metadata), array.shape[1:])
                for view, array in arrays.items()
            },
            "token": {
                key: metadata[key]
                for key in ("token_source", "token_feature_axis")
                if "token" in arrays
            },
        }
        if reference is None:
            reference = current
        else:
            for key, what in (
                ("config_hash", "models or settings (config_hash)"),
                ("views", "views or their shapes"),
                ("recording", "channels or sampling rates"),
                ("token", "token layouts"),
            ):
                if current[key] != reference[key]:
                    raise ValueError(
                        f"Conditions differ in {what}: {condition!r} has "
                        f"{current[key]}, the first has {reference[key]}."
                    )
        n_windows = len(arrays["window"])
        epochs = np.asarray(result.window_index, dtype=np.int64)
        starts = np.asarray(result.window_start, dtype=np.int64)
        stops = np.asarray(result.window_stop, dtype=np.int64)
        for view, array in arrays.items():
            if len(array) != n_windows:
                raise ValueError(
                    f"Condition {condition!r} has {len(array)} {view} rows for "
                    f"{n_windows} windows."
                )
            if not np.isfinite(array).all():
                raise ValueError(
                    f"The {view} embeddings of condition {condition!r} hold "
                    "non-finite values."
                )
            views.setdefault(view, []).append(array)
        if not len(epochs) == len(starts) == len(stops) == n_windows:
            raise ValueError(
                f"Condition {condition!r} needs one epoch number, start and stop "
                "per window."
            )
        windows["condition"].append(np.full(n_windows, condition, dtype=object))
        windows["epoch"].append(epochs)
        windows["start"].append(starts)
        windows["stop"].append(stops)

    columns = {key: np.concatenate(parts) for key, parts in windows.items()}
    numbers, counts = np.unique(columns["epoch"], return_counts=True)
    if (counts > 1).any():
        raise ValueError(
            f"Epoch numbers appear more than once in {recording_id}: "
            f"{numbers[counts > 1].tolist()}."
        )
    order = np.argsort(columns["epoch"], kind="stable")
    columns = {key: column[order] for key, column in columns.items()}
    n_windows = len(order)
    entities = {
        "subject": source.subject,
        "session": source.session or "",
        "task": source.task or "",
        "run": source.run or "",
    }
    coords = {
        "obs": (
            "window",
            [f"{recording_id}_epoch-{number:04d}" for number in columns["epoch"]],
        ),
        "recording_id": ("window", np.full(n_windows, recording_id, dtype=object)),
        **{
            name: ("window", np.full(n_windows, value, dtype=object))
            for name, value in entities.items()
        },
        **{key: ("window", column) for key, column in columns.items()},
        "recording": [recording_id],
        "source": ("recording", [f"bids:{source_dataset}:{relative}"]),
        "bad_channels": ("recording", [_channel_list(bad_channels)]),
        **{
            key: ("recording", [value]) for key, value in reference["recording"].items()
        },
    }
    attrs = {
        "model_key": reference["config"]["ModelKey"],
        "config_hash": reference["config_hash"],
        "config": json.dumps(reference["config"]),
        "description": label,
    }
    paths = {}
    datasets = {}
    for view, parts in views.items():
        dims = reference["views"][view][0]
        paths[view] = embedding_path(
            root, label, view, recording_id=recording_id, datatype=source.datatype
        )
        datasets[view] = xr.Dataset(
            {"embeddings": (dims, np.concatenate(parts)[order])},
            coords=coords,
            attrs={**attrs, **(reference["token"] if view == "token" else {})},
        )
    for view, dataset in datasets.items():
        _write_netcdf(dataset, paths[view].mkdir().fpath)
    return paths


def _dataset_readme(name: str, views: Sequence[str]) -> str:
    """The README of an embedding dataset: what its files hold and how."""
    suffixes = "\n".join(
        f"  - `{view}embeddings`: {_VIEW_TEXT[view]}." for view in views
    )
    return f"""# {name}

Foundation-model embeddings of EEG epochs, written by coco-pipe
({_CODE_URL}).

## Files

Files are named `<entities>_desc-<label>_<view>embeddings.nc`.

- `desc` is the processing chain: the epochs' own label, then the weights
  (`Random` for random weights), the pooling and the alignment, e.g.
  `baseRandomEaCoral`. `descriptions.tsv` lists the labels.
- The suffix is the representation:
{suffixes}
- Files under `sub-*/` hold one recording. Files at the top level hold every
  recording; the JSON file of the same name gives the extraction settings,
  which apply to every file of that label and suffix.
- `desc-<label>_manifest.json` lists the recordings skipped or failed for a
  label.

## Format

NetCDF-4 files, read with xarray (engine `h5netcdf`) or coco-pipe's
`load_embeddings`. The variable `embeddings` holds the view as float32, along
`window` and the view's own axes. Along `window`, the coordinates `obs`,
`recording_id`, `subject`, `session`, `task`, `run`, `condition`, `epoch`,
`start` and `stop` identify each epoch. Along `recording`, `source` gives
each recording's source file as a BIDS URI (resolved through `DatasetLinks`
in `dataset_description.json`), with its sampling rate and channels. The
attribute `config` holds the extraction settings as JSON and `config_hash`
their hash: files whose hashes differ were not made the same way.
"""


def write_embedding_dataset(
    root: str | Path,
    descriptions: Mapping[str, str],
    *,
    name: str,
    dataset_links: Mapping[str, str],
    authors: Sequence[str] = (),
) -> list[Path]:
    """Write the dataset files of an embedding derivative and its sidecars.

    Writes ``dataset_description.json``, ``README.md``, ``descriptions.tsv``
    and its ``descriptions.json``, and ``.bidsignore`` (keeping lines already
    there). Each top-level embedding file (one holding every recording) gets
    a JSON sidecar of the same name, holding the file's extraction settings
    (its ``config`` attribute), its ``config_hash`` as ``ConfigHash`` and a
    ``Description``. By BIDS inheritance a sidecar also applies to the
    recording files of the same label and suffix. Every file is replaced
    atomically and comes out the same whoever calls, so several jobs may
    call this on one dataset.

    Parameters
    ----------
    root : path-like
        The derivative dataset, e.g. ``derivatives/coco-pipe-cbramod``.
    descriptions : mapping of str to str
        What each ``desc`` label means, one row of ``descriptions.tsv`` each.
        Every top-level embedding file's label must be here; a label without
        files yet is listed all the same.
    name : str
        The dataset's ``Name``.
    dataset_links : mapping of str to str
        Each source dataset's name and location, e.g.
        ``{"preproc": "../preproc"}``: the ``DatasetLinks`` that resolve the
        BIDS URIs of the recordings' sources, also listed as
        ``SourceDatasets``.
    authors : sequence of str
        The dataset's ``Authors``.

    Returns
    -------
    list of Path
        The files written.

    Raises
    ------
    ValueError
        If a label or link name is not valid, a description holds a tab or a
        line break, or a top-level embedding file has a label not in
        ``descriptions`` or other than the one it holds, an unknown view, no
        ``config`` or one that does not match its ``config_hash``, sources in
        a dataset ``dataset_links`` does not name, or another model than the
        other files.
    """
    root = Path(root)
    for label, text in descriptions.items():
        _check_label(label)
        if not text or any(char in text for char in "\t\r\n"):
            raise ValueError(
                f"The description of {label!r} must be one line of text, got {text!r}."
            )
    for link in dataset_links:
        if not link or ":" in link:
            raise ValueError(
                f"DatasetLinks names cannot be empty or hold ':': {link!r}."
            )

    sidecars: dict[Path, dict[str, Any]] = {}
    models = set()
    for path in sorted(root.glob("*embeddings.nc")):
        bids_path = get_bids_path_from_fname(path.name, check=False).update(root=root)
        label = bids_path.description
        view = bids_path.suffix.removesuffix("embeddings")
        if view not in EMBEDDING_VIEWS:
            raise ValueError(f"{path.name} names no embedding view.")
        if label not in descriptions:
            raise ValueError(
                f"{path.name} has the label {label!r}, which descriptions lacks."
            )
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            attrs = dict(dataset.attrs)
            links = {str(uri).split(":")[1] for uri in dataset["source"].values}
        if not {"config", "config_hash"} <= set(attrs):
            raise ValueError(f"{path.name} has no config or config_hash attribute.")
        if not links <= set(dataset_links):
            raise ValueError(
                f"The sources of {path.name} are in {sorted(links)}, which "
                "dataset_links must name."
            )
        if attrs.get("description") != label:
            raise ValueError(
                f"{path.name} holds the label {attrs.get('description')!r}."
            )
        config = json.loads(attrs["config"])
        if stable_hash(config, length=16) != attrs["config_hash"]:
            raise ValueError(f"The config of {path.name} does not match its hash.")
        models.add(config["ModelKey"])
        sidecars[bids_path.copy().update(extension=".json").fpath] = {
            "Description": f"{descriptions[label]}: {_VIEW_TEXT[view]}.",
            **config,
            "ConfigHash": attrs["config_hash"],
        }
    if len(models) > 1:
        raise ValueError(
            f"One dataset holds one model, but {root} has files of {sorted(models)}."
        )

    description = {
        "Name": name,
        "BIDSVersion": _BIDS_VERSION,
        "DatasetType": "derivative",
        **({"Authors": list(authors)} if authors else {}),
        "GeneratedBy": [
            {
                "Name": "coco-pipe",
                "Version": get_package_version("coco-pipe"),
                "CodeURL": _CODE_URL,
            }
        ],
        "SourceDatasets": [{"URL": url} for url in dataset_links.values()],
        "DatasetLinks": dict(dataset_links),
    }
    table = "".join(f"desc-{label}\t{text}\n" for label, text in descriptions.items())
    ignore = root / ".bidsignore"
    kept = ignore.read_text(encoding="utf-8").splitlines() if ignore.exists() else []
    texts = {
        root / "README.md": _dataset_readme(name, EMBEDDING_VIEWS),
        root / "descriptions.tsv": f"desc_id\tdescription\n{table}",
        ignore: "\n".join(dict.fromkeys(filter(None, [*kept, *_BIDSIGNORE]))) + "\n",
    }
    written = [write_json(root / "dataset_description.json", description)]
    written.append(
        write_json(
            root / "descriptions.json",
            {
                "Description": "The desc labels of this dataset's files.",
                "desc_id": {"Description": "The desc entity: a processing chain."},
                "description": {"Description": "What the processing chain does."},
            },
        )
    )
    for path, text in texts.items():
        with atomic_path(path) as tmp:
            tmp.write_text(text, encoding="utf-8")
        written.append(path)
    for path, sidecar in sidecars.items():
        written.append(write_json(path, sidecar))
    return written


def _json_value(value: Any) -> Any:
    if isinstance(value, BIDSPath):
        return value.fpath.as_posix()
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__module__}.{type(value).__name__}: {value!s}>"


def write_embedding_manifest(
    root: str | Path,
    label: str,
    records: Iterable[Mapping[str, Any]],
) -> Path:
    """Write the run records of one label: its skipped and failed recordings.

    Each label has its own file, ``desc-<label>_manifest.json`` at the
    dataset root, written by the job that makes the label, so jobs writing
    other labels of the dataset never replace it. Nothing reads it to find
    embedding files. The file is replaced atomically.

    Parameters
    ----------
    root : path-like
        The derivative dataset, e.g. ``derivatives/coco-pipe-cbramod``.
    label : str
        The ``desc`` label the records belong to.
    records : iterable of mapping
        One record per recording, e.g. its ``recording_id``, ``status`` and
        ``reason``.

    Returns
    -------
    Path
    """
    _check_label(label)
    path = BIDSPath(
        root=root,
        description=label,
        suffix="manifest",
        extension=".json",
        check=False,
    ).fpath
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "records": [_json_value(dict(record)) for record in records],
    }
    return write_json(path, payload, sort_keys=True)
