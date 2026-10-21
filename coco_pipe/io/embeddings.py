"""BIDS-compatible derivative I/O for foundation-model embeddings."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5netcdf
import numpy as np
import pandas as pd
import xarray as xr
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_bids_path_from_fname,
    get_entities_from_fname,
)

from coco_pipe.utils import get_package_version, stable_hash

from ._constants import EMBEDDING_CONFIG_KEYS, EMBEDDING_VIEWS, LOADED_LEVELS
from ._serialization import atomic_path, write_json
from .structures import DataContainer, aggregate_recordings
from .utils import (
    bids_recording_id,
    load_subject_metadata,
    normalize_subject_value,
)

_RECORDING_ENTITIES = frozenset({"subject", "session", "task", "acquisition", "run"})

# descriptions.tsv came in BIDS 1.10; the validator (schema 1.2.7) does not
# know 1.11 yet, and mne-bids 0.20 writes 1.9.0.
_BIDS_VERSION = "1.10.1"
_CODE_URL = "https://github.com/BabaSanfour/coco-pipe"
_BIDSIGNORE = ("*embeddings.nc", "*embeddings.json", "*.tar.zst", "*_manifest.json")

_VIEW_TEXT = {
    "token": "the model's token tensor of each window, nothing pooled",
    "window": "one vector per window",
    "channel": (
        "one vector per channel of each window, averaged over time patches, "
        "not normalised"
    ),
    "patch": (
        "one vector per time patch of each window, averaged over channels, "
        "not normalised"
    ),
    "cov": "the tangent-space covariance of each window's tokens",
}

_VIEW_DIMS = {
    "window": ("window", "feature"),
    "channel": ("window", "channel", "feature"),
    "patch": ("window", "time_patch", "feature"),
    "cov": ("window", "feature"),
}

# h5py reads a long list of indices slowly: 43,741 rows of an 87,000-window
# file took 0.97 s as one list and 0.08 s as lists of 1,024 (2026-10-20).
_READ_ROWS = 1024


def _check_label(label: str) -> None:
    """Refuse a ``desc`` label that is not a BIDS label."""
    if not re.fullmatch(r"[A-Za-z0-9+]+", label):
        raise ValueError(
            f"label must be letters, digits and '+' (a BIDS label), got {label!r}."
        )


def _check_view(view: str) -> None:
    """Refuse a view that is not one of the embedding views."""
    if view not in EMBEDDING_VIEWS:
        raise ValueError(f"view must be one of {EMBEDDING_VIEWS}, got {view!r}.")


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
        ``"baseRandom0EaCoral"``. A BIDS label: letters, digits and ``+``.
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
    _check_view(view)
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
    token_dtype: str = "float32",
) -> dict[tuple[str, str], BIDSPath]:
    """Write one recording's embeddings, one file per view.

    Each view the results hold (``window_embeddings``, ``token_embeddings``,
    and so on, see :data:`~coco_pipe.io.EMBEDDING_VIEWS`) goes to its own
    file, named by :func:`embedding_path` after the source recording. The
    window vectors of each other pooling (``extra_window_embeddings``, e.g.
    REVE's attention read-out) go to a window file of their own label, the
    label followed by the pooling's name (``baseAttention``), whose settings
    name that pooling. A file holds every condition, its windows in epoch
    order. Nothing is written if any check fails, and each file is written to
    a temporary name and then moved into place, so an existing file is
    replaced whole or not at all.

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
    token_dtype : {"float32", "float16"}
        The dtype of the token file; ``"float16"`` halves its size. The other
        views are float32.

    Returns
    -------
    dict of (str, str) to BIDSPath
        The path written for each label and view, e.g.
        ``paths["base", "window"]``.

    Raises
    ------
    ValueError
        If the conditions come from different models or settings (their
        ``config_hash``), differ in views, poolings, feature size or channels,
        hold non-finite values or share epoch numbers, if a channel axis does
        not match ``model_channels``, if a pooling's name is not letters and
        digits, if ``source`` lacks its root, subject or datatype, or if
        ``token_dtype`` is not ``"float32"`` or ``"float16"``.

    Notes
    -----
    The ``embeddings`` variable holds the view, as float32 (tokens as
    ``token_dtype``; values beyond float16's range are refused). Along ``window``
    are the coordinates ``obs`` (``<recording_id>_epoch-NNNN``, as the BIDS
    loader names observations), ``recording_id``, ``subject``, ``session``,
    ``task``, ``run``, ``condition``, ``epoch``, ``start`` and ``stop``.
    Along ``recording`` are its ``source`` (a BIDS URI), sampling rate,
    channels, bad channels and the channels the model interpolated,
    zero-filled or dropped. A view with a ``channel`` axis names it with the
    ``channel`` coordinate: the channels the model saw (``model_channels``).
    The attributes are ``model_key``, ``config``
    (the extraction settings that are the same for every recording, as JSON
    text, see :data:`~coco_pipe.io.EMBEDDING_CONFIG_KEYS`), ``config_hash``
    (their hash) and ``description`` (the label); a token file adds
    ``token_source`` and ``token_feature_axis``, the other token axes being
    the observation axes. Other metadata in the results is not stored.
    """
    if not results:
        raise ValueError("results must hold at least one condition.")
    if token_dtype not in ("float32", "float16"):
        raise ValueError(
            f"token_dtype must be 'float32' or 'float16', got {token_dtype!r}."
        )
    if source.root is None or source.subject is None or source.datatype is None:
        raise ValueError("source must be a BIDSPath with root, subject and datatype.")
    if not source_dataset or ":" in source_dataset:
        raise ValueError(
            f"source_dataset must be a DatasetLinks name, got {source_dataset!r}."
        )
    recording_id = bids_recording_id(source)
    relative = source.fpath.relative_to(Path(source.root)).as_posix()

    views: dict[str, list[np.ndarray]] = {}
    poolings: dict[str, list[np.ndarray]] = {}
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
        with np.errstate(over="ignore"):
            arrays = {
                view: np.asarray(
                    array, dtype=token_dtype if view == "token" else "float32"
                )
                for view in EMBEDDING_VIEWS
                if (array := getattr(result, f"{view}_embeddings", None)) is not None
            }
        if "window" not in arrays:
            raise ValueError(f"Condition {condition!r} has no window embeddings.")
        pooled = {
            pooling: np.asarray(array, dtype=np.float32)
            for pooling, array in (
                getattr(result, "extra_window_embeddings", None) or {}
            ).items()
        }
        for pooling in pooled:
            if not (
                isinstance(pooling, str) and pooling.isascii() and pooling.isalnum()
            ):
                raise ValueError(
                    f"Pooling names must be letters and digits, got {pooling!r}."
                )
        current = {
            "config_hash": stable_hash(config, length=16),
            "config": config,
            "recording": recording,
            "model_channels": [
                str(name) for name in metadata.get("model_channels", ())
            ],
            "views": {
                view: (_view_dims(view, array, metadata), array.shape[1:])
                for view, array in arrays.items()
            },
            "poolings": {
                pooling: (_view_dims("window", array, metadata), array.shape[1:])
                for pooling, array in pooled.items()
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
                ("poolings", "poolings or their shapes"),
                ("recording", "channels or sampling rates"),
                ("model_channels", "the channels the model saw"),
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
        named = {
            **arrays,
            **{f"{pooling} window": array for pooling, array in pooled.items()},
        }
        for view, array in named.items():
            if len(array) != n_windows:
                raise ValueError(
                    f"Condition {condition!r} has {len(array)} {view} rows for "
                    f"{n_windows} windows."
                )
            if not np.isfinite(array).all():
                raise ValueError(
                    f"The {view} embeddings of condition {condition!r} hold "
                    "non-finite values"
                    + (
                        f" as {token_dtype}"
                        if view == "token" and token_dtype == "float16"
                        else ""
                    )
                    + "."
                )
        for view, array in arrays.items():
            views.setdefault(view, []).append(array)
        for pooling, array in pooled.items():
            poolings.setdefault(pooling, []).append(array)
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
        dims, shape = reference["views"][view]
        view_coords = coords
        if "channel" in dims:
            channels = reference["model_channels"]
            if len(channels) != shape[dims.index("channel") - 1]:
                raise ValueError(
                    f"The {view} embeddings have {shape[dims.index('channel') - 1]} "
                    f"channels, but model_channels names {len(channels)}."
                )
            view_coords = {**coords, "channel": channels}
        paths[label, view] = embedding_path(
            root, label, view, recording_id=recording_id, datatype=source.datatype
        )
        datasets[label, view] = xr.Dataset(
            {"embeddings": (dims, np.concatenate(parts)[order])},
            coords=view_coords,
            attrs={**attrs, **(reference["token"] if view == "token" else {})},
        )
    for pooling, parts in poolings.items():
        pooled_label = label + pooling[0].upper() + pooling[1:]
        config = {**reference["config"], "WindowPooling": pooling}
        paths[pooled_label, "window"] = embedding_path(
            root,
            pooled_label,
            "window",
            recording_id=recording_id,
            datatype=source.datatype,
        )
        datasets[pooled_label, "window"] = xr.Dataset(
            {"embeddings": (_VIEW_DIMS["window"], np.concatenate(parts)[order])},
            coords=coords,
            attrs={
                **attrs,
                "config_hash": stable_hash(config, length=16),
                "config": json.dumps(config),
                "description": pooled_label,
            },
        )
    for key, dataset in datasets.items():
        _write_netcdf(dataset, paths[key].mkdir().fpath)
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
  (`Random<seed>` for random weights drawn from that seed), the pooling and
  the alignment, e.g. `baseRandom0EaCoral`. `descriptions.tsv` lists the
  labels.
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
        header = _read_header(path)
        label = header["description"]
        if label not in descriptions:
            raise ValueError(
                f"{path.name} has the label {label!r}, which descriptions lacks."
            )
        links = {str(uri).split(":")[1] for uri in header["recordings"]["source"]}
        if not links <= set(dataset_links):
            raise ValueError(
                f"The sources of {path.name} are in {sorted(links)}, which "
                "dataset_links must name."
            )
        config = json.loads(header["config"])
        models.add(config["ModelKey"])
        sidecar = header["name"].copy().update(root=root, extension=".json")
        sidecars[sidecar.fpath] = {
            "Description": f"{descriptions[label]}: {_VIEW_TEXT[header['view']]}.",
            **config,
            "ConfigHash": header["config_hash"],
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


def find_embeddings(
    root: str | Path,
    label: str,
    view: str,
    *,
    subjects: str | Sequence[str] | None = None,
) -> list[BIDSPath]:
    """Return the per-recording embedding files of one label and view.

    The files are found on disk with :func:`mne_bids.find_matching_paths`,
    never through a manifest. The files at the dataset root, which hold every
    recording, are not returned.

    Parameters
    ----------
    root : path-like
        The derivative dataset, e.g. ``derivatives/coco-pipe-cbramod``.
    label : str
        The ``desc`` label, matched whole: ``"base"`` does not find the
        ``"baseRandom0"`` files.
    view : str
        One of :data:`~coco_pipe.io.EMBEDDING_VIEWS`.
    subjects : str or sequence of str, optional
        The subject labels to find, without ``sub-``; every subject when
        omitted.

    Returns
    -------
    list of BIDSPath
        The files, sorted by path; empty when none match.

    Raises
    ------
    FileNotFoundError
        If ``root`` is not a directory.
    ValueError
        If ``label`` is not a BIDS label or ``view`` is not an embedding view.
    """
    _check_label(label)
    _check_view(view)
    if not Path(root).is_dir():
        raise FileNotFoundError(f"No embedding dataset at {root}.")
    return find_matching_paths(
        root,
        subjects=subjects,
        descriptions=re.escape(label),
        suffixes=f"{view}embeddings",
        extensions=".nc",
        ignore_nosub=True,
        check=False,
    )


def _read_header(path: Path) -> dict[str, Any]:
    """Return what :func:`load_embeddings` checks of a file, without its data."""
    try:
        name = get_bids_path_from_fname(path.name, check=False)
    except KeyError:
        name = None
    suffix = (name.suffix if name else None) or ""
    view = suffix.removesuffix("embeddings")
    if not suffix.endswith("embeddings") or view not in EMBEDDING_VIEWS:
        raise ValueError(
            f"{path.name} is not an embedding file "
            "(<entities>_desc-<label>_<view>embeddings.nc)."
        )
    # A recording's file sits below sub-<label>/ in its dataset; a file holding
    # every recording, at the dataset root.
    root = path.parent
    if name.subject is not None:
        folders = [f for f in path.parents if f.name == f"sub-{name.subject}"]
        root = folders[0].parent if folders else None
    with xr.open_dataset(path, engine="h5netcdf", cache=False) as dataset:
        attrs = dict(dataset.attrs)
        variable = dataset["embeddings"]
        dims = variable.dims
        header = {
            "path": path,
            "name": name,
            "root": root,
            "view": view,
            "axes": dims,
            "shape": variable.shape[1:],
            "dtype": variable.dtype,
            "attrs": attrs,
            "token": {
                key: attrs.get(key)
                for key in ("token_source", "token_feature_axis")
                if view == "token"
            },
            "windows": {
                name: coord.values
                for name, coord in dataset.coords.items()
                if coord.dims == ("window",)
            },
            "recordings": {
                name: coord.values.tolist()
                for name, coord in dataset.coords.items()
                if coord.dims == ("recording",)
            },
            "channel": (
                dataset["channel"].values.tolist()
                if "channel" in dataset.coords
                else None
            ),
        }
    for key in ("model_key", "config_hash", "description", "config"):
        if not attrs.get(key):
            raise ValueError(f"{path.name} has no {key} attribute.")
        header[key] = attrs[key]
    if stable_hash(json.loads(header["config"]), length=16) != header["config_hash"]:
        raise ValueError(f"The config of {path.name} does not match its hash.")
    if name.description != header["description"]:
        raise ValueError(f"{path.name} holds the label {header['description']!r}.")
    token = header["token"]
    if view == "token":
        valid = (
            dims[:1] == ("window",)
            and token["token_source"]
            and token["token_feature_axis"] in dims[1:]
        )
    else:
        valid = dims == _VIEW_DIMS[view]
    if not valid:
        raise ValueError(f"{path.name} has axes {dims}, which its view cannot have.")
    if "channel" in dims and header["channel"] is None:
        raise ValueError(f"{path.name} has a channel axis without channel names.")
    if not {"obs", "recording_id", "subject", "condition"} <= set(header["windows"]):
        raise ValueError(f"{path.name} lacks coordinates along window.")
    return header


def _read_headers(
    source: str | Path | BIDSPath | Sequence[str | Path | BIDSPath],
) -> list[dict[str, Any]]:
    """Read the headers of files to combine, checking they can be combined.

    Each header's ``order`` puts its channels in the first file's order, or
    is None when they already are.
    """
    items = [source] if isinstance(source, str | Path | BIDSPath) else list(source)
    if not items:
        raise ValueError("source names no embedding file.")
    headers = [
        _read_header(Path(item.fpath if isinstance(item, BIDSPath) else item))
        for item in items
    ]
    first = headers[0]
    first["order"] = None
    for header in headers[1:]:
        for key, what in (
            ("model_key", "model"),
            ("config_hash", "settings (config_hash)"),
            ("description", "label"),
            ("view", "view"),
            ("axes", "axes"),
            ("shape", "shape"),
            ("token", "token layout"),
        ):
            if header[key] != first[key]:
                raise ValueError(
                    f"{header['path'].name} differs from {first['path'].name} in "
                    f"{what}: {header[key]} against {first[key]}."
                )
        for table in ("windows", "recordings"):
            if set(header[table]) != set(first[table]):
                raise ValueError(
                    f"{header['path'].name} has other coordinates than "
                    f"{first['path'].name}."
                )
        header["order"] = None
        if header["channel"] != first["channel"]:
            if sorted(header["channel"]) != sorted(first["channel"]):
                raise ValueError(
                    f"{header['path'].name} has the channels {header['channel']}, "
                    f"{first['path'].name} has {first['channel']}."
                )
            header["order"] = [header["channel"].index(c) for c in first["channel"]]
    return headers


def _check_unique(obs: np.ndarray) -> None:
    """Refuse observations that appear more than once."""
    names, counts = np.unique(obs, return_counts=True)
    if (counts > 1).any():
        raise ValueError(
            f"Observations appear more than once: {names[counts > 1][:5].tolist()}."
        )


def _copy_rows(
    header: Mapping[str, Any], rows: np.ndarray, out: Any, start: int
) -> None:
    """Copy rows of a file's embeddings to ``out[start:]``, a block at a time.

    Refuses non-finite values and puts the channels in the first file's
    order.
    """
    if not len(rows):
        return
    axes = header["axes"]
    with xr.open_dataset(header["path"], engine="h5netcdf", cache=False) as dataset:
        variable = dataset["embeddings"]
        for begin in range(0, len(rows), _READ_ROWS):
            block = rows[begin : begin + _READ_ROWS]
            index = (
                slice(block[0], block[-1] + 1)
                if block[-1] - block[0] == len(block) - 1
                else block
            )
            values = variable.isel(window=index).values
            if not np.isfinite(values).all():
                raise ValueError(f"{header['path'].name} holds non-finite values.")
            if header["order"] is not None:
                values = np.take(values, header["order"], axis=axes.index("channel"))
            out[start + begin : start + begin + len(block)] = values


def load_embeddings(
    source: str | Path | BIDSPath | Sequence[str | Path | BIDSPath],
    *,
    representation: str = "epoch",
    conditions: str | Sequence[str] | None = None,
    subjects: str | Sequence[str] | None = None,
    subject_metadata_df: pd.DataFrame | None = None,
    subject_key: str | None = None,
) -> DataContainer:
    """Load embedding files into a DataContainer.

    The files are read one at a time into one float32 array, so memory holds
    the result and a block of 1,024 windows of one file. Only the windows of
    ``conditions`` and ``subjects`` are read.

    Parameters
    ----------
    source : path-like or BIDSPath, or a sequence of them
        Embedding files of one model, label and view: one file holding every
        recording, or recordings' files (see :func:`find_embeddings`).
    representation : {"epoch", "recording"}
        ``"epoch"``: one row per window. ``"recording"``: one row per
        recording and condition, the mean of its windows.
    conditions : str or sequence of str, optional
        The conditions to load; every condition when omitted.
    subjects : str or sequence of str, optional
        The subjects to load, labels without ``sub-``; every subject when
        omitted. A subject no file holds is not an error.
    subject_metadata_df : DataFrame, optional
        Subject-level metadata to attach, in addition to ``participants.tsv``.
    subject_key : str, optional
        The column of ``subject_metadata_df`` holding the subject ID. It
        matches rows to subjects and is not attached itself.

    Returns
    -------
    DataContainer
        ``X`` is float32, with the file's axes and ``window`` renamed
        ``obs``: ``("obs", "feature")`` for window and cov files, and
        ``channel``, ``time_patch`` or the model's token axes before
        ``feature`` for the others. ``ids`` is ``obs``. Along ``obs`` are
        the file's window coordinates (``recording_id``, ``subject``,
        ``session``, ``task``, ``run``, ``condition``, ``epoch``, ``start``,
        ``stop``), each recording's ``bad_channels`` and each subject's
        metadata: the columns of ``participants.tsv``, found in the files'
        dataset or, following ``SourceDatasets``, in a dataset it comes from
        (see :func:`~coco_pipe.io.utils.load_participants_tsv`), and those of
        ``subject_metadata_df``. ``session`` and ``run`` are dropped when no
        recording has one. A channel axis keeps
        its ``channel`` names. ``meta`` holds ``model_key``, ``config``,
        ``config_hash``, ``description`` (the label), ``view``,
        ``representation``, ``files``, ``recordings`` (each loaded
        recording's row of the files' recording table) and, for tokens,
        ``token_source`` and ``token_feature_axis``.

    Raises
    ------
    ValueError
        If ``source`` names no file or a file that is not an embedding file;
        if the files differ in model, settings (``config_hash``), label,
        view, axes, shape, token layout or channel names; if a condition is
        in no file, no window is selected, an observation appears twice or a
        value is not finite; if ``subject_metadata_df`` comes without
        ``subject_key``, or a subject metadata column is named like a
        coordinate the loader sets.

    Notes
    -----
    The recording level is the mean of the window vectors as stored, which
    are L2-normalised when the extractor normalises them; its ``obs`` is
    ``<recording_id>_cond-<condition>`` and ``epoch_count`` holds the number
    of windows. Channel axes that hold the same channels in another order
    are put in the first file's order.
    """
    if representation not in LOADED_LEVELS:
        raise ValueError(
            f"representation must be one of {LOADED_LEVELS}, got {representation!r}."
        )
    wanted = {}
    for name, values in (("condition", conditions), ("subject", subjects)):
        if isinstance(values, str):
            values = [values]
        wanted[name] = None if values is None else set(values)

    headers = _read_headers(source)
    first = headers[0]
    held = {str(c) for header in headers for c in header["windows"]["condition"]}
    if wanted["condition"] is not None and wanted["condition"] - held:
        raise ValueError(
            f"No file holds the conditions {sorted(wanted['condition'] - held)}; "
            f"they hold {sorted(held)}."
        )
    for header in headers:
        keep = np.ones(len(header["windows"]["obs"]), dtype=bool)
        for name, values in wanted.items():
            if values is not None:
                keep &= np.isin(header["windows"][name], list(values))
        header["rows"] = np.flatnonzero(keep)
    total = sum(len(header["rows"]) for header in headers)
    if not total:
        raise ValueError(
            f"No window of the files matches conditions={conditions!r} and "
            f"subjects={subjects!r}."
        )

    columns = {
        name: np.concatenate([h["windows"][name][h["rows"]] for h in headers])
        for name in first["windows"]
    }
    obs = columns.pop("obs")
    _check_unique(obs)
    loaded = set(columns["recording_id"].tolist())
    recordings = [
        {
            "recording_id": recording_id,
            **{
                name: values[index]
                for name, values in header["recordings"].items()
                if name != "recording"
            },
        }
        for header in headers
        for index, recording_id in enumerate(header["recordings"]["recording"])
        if recording_id in loaded
    ]
    bad_channels = {row["recording_id"]: row["bad_channels"] for row in recordings}
    columns["bad_channels"] = np.array(
        [bad_channels[recording_id] for recording_id in columns["recording_id"]]
    )
    for name in ("session", "run"):
        if name in columns and not columns[name].astype(bool).any():
            del columns[name]
    lookup = load_subject_metadata(
        dict.fromkeys(h["root"] for h in headers if h["root"] is not None),
        subject_metadata_df,
        subject_key,
        reserved={*first["windows"], "bad_channels", "epoch_count", "ids", "y"},
    )
    if lookup:
        subject_labels, index = np.unique(columns["subject"], return_inverse=True)
        table = pd.DataFrame(
            [lookup.get(normalize_subject_value(s), {}) for s in subject_labels]
        )
        for name in table.columns:
            columns[name] = table[name].to_numpy()[index]

    X = np.empty((total, *first["shape"]), dtype=np.float32)
    start = 0
    for header in headers:
        _copy_rows(header, header["rows"], X, start)
        start += len(header["rows"])

    meta = {
        "model_key": first["model_key"],
        "config": json.loads(first["config"]),
        "config_hash": first["config_hash"],
        "description": first["description"],
        "view": first["view"],
        "representation": representation,
        "files": [header["path"].as_posix() for header in headers],
        "recordings": recordings,
        **first["token"],
    }
    coords = {
        "obs": obs,
        "ids": ("obs", obs),
        **{name: ("obs", values) for name, values in columns.items()},
        **({"channel": first["channel"]} if first["channel"] else {}),
    }
    container = DataContainer(
        xr.DataArray(X, dims=("obs", *first["axes"][1:]), coords=coords, attrs=meta)
    )
    if representation == "recording":
        container = aggregate_recordings(container)
    return container


def consolidate_embeddings(
    files: Sequence[str | Path | BIDSPath],
    root: str | Path,
) -> BIDSPath:
    """Write recordings' embedding files as one file at the dataset root.

    The file is named by :func:`embedding_path` from the files' label and
    view, with their task when they all have the same one. It holds their
    windows in the order of ``files``, their recording tables and their
    attributes, and reads back with :func:`load_embeddings` as the files
    do. The files are copied one at a time, so memory holds one of them; the
    embeddings keep their dtype. The file is written to a temporary name and
    then moved into place, so nothing is left if a check fails.

    Parameters
    ----------
    files : sequence of path-like or BIDSPath
        Embedding files of one model, label and view, e.g. from
        :func:`find_embeddings`.
    root : path-like
        The derivative dataset, e.g. ``derivatives/coco-pipe-cbramod``.

    Returns
    -------
    BIDSPath
        The file written.

    Raises
    ------
    ValueError
        As :func:`load_embeddings`: if ``files`` is empty or names a file
        that is not an embedding file, if the files cannot be combined, if
        an observation appears twice or a value is not finite.
    """
    headers = _read_headers(files)
    first = headers[0]
    windows = {
        name: np.concatenate([header["windows"][name] for header in headers])
        for name in first["windows"]
    }
    _check_unique(windows["obs"])
    recordings = {
        name: [value for header in headers for value in header["recordings"][name]]
        for name in first["recordings"]
    }
    tasks = set(windows["task"].tolist()) if "task" in windows else set()
    task = tasks.pop() if len(tasks) == 1 else None
    path = embedding_path(root, first["description"], first["view"], task=task or None)
    coords = {
        **{name: ("window", values) for name, values in windows.items()},
        "recording": recordings.pop("recording"),
        **{name: ("recording", values) for name, values in recordings.items()},
        **({"channel": first["channel"]} if first["channel"] else {}),
    }
    Path(root).mkdir(parents=True, exist_ok=True)
    with atomic_path(path.fpath) as tmp:
        xr.Dataset(coords=coords, attrs=first["attrs"]).to_netcdf(
            tmp, engine="h5netcdf"
        )
        with h5netcdf.File(tmp, "a") as file:
            for dim, size in zip(first["axes"][1:], first["shape"], strict=True):
                if dim not in file.dimensions:
                    file.dimensions[dim] = size
            out = file.create_variable(
                "embeddings",
                first["axes"],
                np.result_type(*(header["dtype"] for header in headers)),
            )
            start = 0
            for header in headers:
                rows = np.arange(len(header["windows"]["obs"]))
                _copy_rows(header, rows, out, start)
                start += len(rows)
    return path
