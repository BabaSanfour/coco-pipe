"""
coco_pipe/io/dataset.py
-----------------------
Specialized Dataset classes that produce standardized DataContainer objects.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, find_matching_paths, read_epochs_bids, read_raw_bids

from ._serialization import smart_reader
from .structures import DataContainer
from .utils import load_participants_tsv, normalize_subject_value, split_column

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    @abstractmethod
    def load(self) -> DataContainer:
        pass


class TabularDataset(BaseDataset):
    """
    Dataset for loading tabular feature data (CSV, TSV, Excel).

    This class handles loading, optional clearing, and reshaping of 2D tabular data
    into multi-dimensional DataContainers.

    Parameters
    ----------
    path : str or Path
        Path to the tabular file (csv, tsv, txt, xls, xlsx).
    target_col : str, optional
        Name of the column to extract as target `y`. Removed from features `X`.
    index_col : str or int, optional
        Column to use as index (observation IDs).
    sep : str, default='\\t'
        Separator for text files.
    header : int or list of int, default=0
        Row number(s) to use as column names.
    sheet_name : str or int, default=0
        Sheet name or index for Excel files.
    columns_to_dims : list of str, optional
        If provided, attempts to reshape the 2D feature columns into N-D dimensions.
        Columns must follow the naming convention: `dim1_dim2_..._feature`.
    col_sep : str, default='_'
        Separator used in column names for reshaping.
    meta_columns : list of str, optional
        List of columns to extract as metadata coordinates instead of features.
    clean : bool, default=False
        Whether to perform automated cleaning (drop NaNs/Infs).
    clean_kwargs : dict, optional
        Arguments passed to `TabularDataset.clean`.
    select_kwargs : dict, optional
        Arguments for feature selection (not yet implemented in load directly).

    Examples
    --------
    >>> # Load a simple CSV
    >>> ds = TabularDataset("data.csv", target_col="label")
    >>> container = ds.load()

    >>> # Load and reshape wide data (e.g. time series in columns)
    >>> # Columns: T0_F1, T0_F2, T1_F1... -> dims=('time', 'freq')
    >>> ds = TabularDataset("wide.csv", columns_to_dims=["time", "freq"], col_sep="_")
    """

    def __init__(
        self,
        path: str | Path,
        target_col: str | None = None,
        index_col: str | int | None = None,
        sep: str = "\t",
        header: int | list[int] | None = 0,
        sheet_name: str | int | None = 0,
        columns_to_dims: list[str] | None = None,
        col_sep: str = "_",
        meta_columns: list[str] | None = None,
        clean: bool = False,
        clean_kwargs: dict[str, Any] | None = None,
        select_kwargs: dict[str, Any] | None = None,
    ):
        self.path = Path(path)
        self.target_col = target_col
        self.index_col = index_col
        self.sep = sep
        self.header = header
        self.sheet_name = sheet_name

        self.columns_to_dims = columns_to_dims
        self.col_sep = col_sep
        self.meta_columns = meta_columns or []

        self.do_clean = clean
        self.clean_kwargs = clean_kwargs or {}
        self.select_kwargs = select_kwargs or {}
        self.strict_reshaping = True

    def load(self) -> DataContainer:
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        # 1. Load DataFrame
        if self.path.suffix in [".csv", ".tsv", ".txt"]:
            df = pd.read_csv(
                self.path, sep=self.sep, index_col=self.index_col, header=self.header
            )
        elif self.path.suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(
                self.path,
                index_col=self.index_col,
                header=self.header,
                sheet_name=self.sheet_name,
            )
        else:
            raise ValueError(f"Unsupported file extension: {self.path.suffix}")

        # Dtype Check
        non_numeric = df.select_dtypes(include=["object", "string", "category"])
        if not non_numeric.empty and self.target_col not in non_numeric.columns:
            logger.warning(
                f"Tabular data contains non-numeric columns: "
                f"{non_numeric.columns.tolist()}. "
                "These might cause issues during processing if not intended as "
                "metadata."
            )

        if self.target_col and self.target_col in df.columns:
            y = df[self.target_col].values
            X_df = df.drop(columns=[self.target_col])
        else:
            y = None
            X_df = df

        # 3. Handle Demographics / Meta Columns
        covariates = {}
        if self.meta_columns:
            found_meta = [c for c in self.meta_columns if c in X_df.columns]
            if found_meta:
                meta_df = X_df[found_meta]
                for col in meta_df.columns:
                    covariates[col] = meta_df[col].values
                X_df = X_df.drop(columns=found_meta)

        # 4. Cleaning
        if self.do_clean:
            X_df, report = self.clean(X_df, **self.clean_kwargs)
        else:
            report = None

        ids = df.index.astype(str).values if self.index_col is not None else None

        # 5. Reshaping Logic (2D -> ND)
        coords = {}
        if ids is not None:
            coords["obs"] = np.array(ids)

        # Add covariates to coords (Auxiliary Coords)
        coords.update(covariates)

        dims = ["obs", "feature"]
        X_final = X_df.values

        if self.columns_to_dims:
            parsed_cols = []
            valid_cols = []
            failed_cols = []

            for col in X_df.columns:
                parts = str(col).split(self.col_sep)
                if len(parts) == len(self.columns_to_dims):
                    parsed_cols.append(tuple(parts))
                    valid_cols.append(col)
                else:
                    failed_cols.append(col)

            if failed_cols:
                msg = (
                    f"{len(failed_cols)} columns failed reshaping pattern "
                    f"(sep='{self.col_sep}', expected {len(self.columns_to_dims)} "
                    f"parts). Examples: {failed_cols[:5]}"
                )
                if self.strict_reshaping:
                    logger.error(msg)
                    # raise ValueError(msg) # Optional: strict mode could raise
                logger.warning(msg)

            if not parsed_cols:
                raise ValueError(
                    f"No columns matched the reshaping pattern with sep="
                    f"'{self.col_sep}'."
                )

            # Create MultiIndex to sort and structure
            mi = pd.MultiIndex.from_tuples(parsed_cols, names=self.columns_to_dims)

            # Reorder columns to match sorted MultiIndex (Cartesian Product order)
            # This ensures reshape works correctly
            X_subset = X_df[valid_cols]

            X_subset.columns = mi
            X_sorted = X_subset.sort_index(axis=1)  # Sorts lexically by levels

            # Extract Levels to Coords
            for i, dim_name in enumerate(self.columns_to_dims):
                unique_vals = X_sorted.columns.unique(level=i)
                coords[dim_name] = unique_vals.values

            # Verify Shape
            n_obs = X_sorted.shape[0]
            dim_sizes = [len(coords[d]) for d in self.columns_to_dims]
            expected_total = np.prod(dim_sizes)

            if X_sorted.shape[1] != expected_total:
                raise ValueError(
                    f"Reshaping failed. Found {X_sorted.shape[1]} columns, expected "
                    f"full product {expected_total} ({dim_sizes}). Missing "
                    f"combinations?"
                )

            # Reshape: (N_obs, Dim1, Dim2, ...)
            new_shape = (n_obs, *tuple(dim_sizes))
            X_final = X_sorted.values.reshape(new_shape)
            dims = ("obs", *self.columns_to_dims)

        else:
            # Default 2D
            coords["feature"] = np.array(X_df.columns.tolist())

        return DataContainer(
            X=X_final,
            y=y,
            ids=np.array(ids) if ids is not None else None,
            dims=tuple(dims),
            coords=coords,
            meta={"filename": self.path.name, "cleaning_report": report},
        )

    @staticmethod
    def clean(
        X: pd.DataFrame,
        mode: str = "any",
        sep: str = "_",
        reverse: bool = False,
        verbose: bool = False,
        min_abs_value: float | None = None,
        min_abs_fraction: float = 0.0,
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Remove invalid feature columns containing NaN, ±Inf, and optionally very
        small values.
        """
        if X.shape[1] == 0:
            return X.copy(), {
                "dropped_columns": [],
                "dropped_features": [],
                "mode": mode,
                "n_before": 0,
                "n_after": 0,
            }

        # Identify columns with NaN/Inf
        num = X.select_dtypes(include=[np.number])
        other = X.drop(columns=num.columns, errors="ignore")

        bad_cols = []
        if not num.empty:
            arr = num.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                inf_mask = np.isinf(arr)
            bad_mask = num.isna().to_numpy() | inf_mask
            bad_any = bad_mask.any(axis=0)
            bad_cols.extend(num.columns[bad_any].tolist())

            if min_abs_value is not None:
                with np.errstate(invalid="ignore"):
                    tiny_mask = np.abs(arr) < float(min_abs_value)
                if min_abs_fraction <= 0.0:
                    tiny_cols = num.columns[tiny_mask.any(axis=0)].tolist()
                else:
                    frac = tiny_mask.mean(axis=0)
                    tiny_cols = num.columns[(frac >= min_abs_fraction)].tolist()
                bad_cols.extend(tiny_cols)

        if not other.empty:
            obj_bad = other.isna().all(axis=0)
            bad_cols.extend(other.columns[obj_bad].tolist())

        dropped_columns = []
        dropped_features = []
        if mode == "any":
            dropped_columns = sorted(set(bad_cols))
            X_clean = X.drop(columns=dropped_columns, errors="ignore")
        elif mode == "sensor_wide":
            feature_to_cols = {}
            for col in X.columns:
                _, feat = split_column(col, sep=sep, reverse=reverse)
                feature_to_cols.setdefault(feat, []).append(col)

            bad_features = set()
            for col in bad_cols:
                _, feat = split_column(col, sep=sep, reverse=reverse)
                bad_features.add(feat)

            for feat in sorted(bad_features):
                dropped_columns.extend(feature_to_cols.get(feat, []))
            dropped_columns = sorted(set(dropped_columns))
            dropped_features = sorted(bad_features)
            X_clean = X.drop(columns=dropped_columns, errors="ignore")
        else:
            raise ValueError("mode must be one of {'any','sensor_wide'}")

        report = {
            "mode": mode,
            "dropped_columns": dropped_columns,
            "dropped_features": dropped_features,
            "n_before": X.shape[1],
            "n_after": X_clean.shape[1],
        }
        return X_clean, report


class EmbeddingDataset(BaseDataset):
    r"""
    Generic Dataset for loading embedding files (Pickle, NPY, JSON, H5).

    This class decouples file discovery (via patterns and IDs) from content reading.
    It supports structured formats (e.g., Layers x Features) and user-supplied
    metadata coordinates.

    Parameters
    ----------
    path : str or Path
        Root directory containing the embedding files.
    pattern : str, default=r'\*.pkl'
        Glob pattern to match files (e.g., ``*.npy``, ``sub-*_emb.pkl``).
    dims : tuple of str, default=('obs', 'feature')
        Dimension labels for the data arrays (excluding the observation dimension if
        implicit). Typically ('feature',) or ('layer', 'feature').
    coords : dict, optional
        Dictionary of coordinates for dimensions. E.g., {'layer': ['L1', 'L2']}.
    reader : callable, optional
        Custom function to read a Path and return a numpy array or dict.
        If None, uses `smart_reader` based on file extension.
    id_fn : callable, optional
        Function returning the subject ID of a Path. Defaults to the file stem.
    task : str, optional
        (Legacy BIDS) Task name to construct search pattern.
    run : str, optional
        (Legacy BIDS) Run name to construct search pattern.
    processing : str, optional
        (Legacy BIDS) Processing label.
    subjects : int or list, optional
        If int, loads first N subjects. If list, loads specific subjects
        (matched by `id_fn`).

    Examples
    --------
    >>> # Load loose numpy files
    >>> ds = EmbeddingDataset("./embeddings", pattern="*.npy", dims=("feature",))
    >>> container = ds.load()
    """

    def __init__(
        self,
        path: str | Path,
        pattern: str = "*.pkl",
        dims: tuple[str, ...] = ("obs", "feature"),
        coords: dict[str, list | np.ndarray] | None = None,
        reader: Any | None = None,
        id_fn: Any | None = None,
        task: str | None = None,
        run: str | None = None,
        processing: str | None = None,
        subjects: int | list[int] | None = None,
    ):
        self.path = Path(path)
        self.subjects = subjects
        self.dims = dims
        self.coords_in = coords or {}

        # 1. Determine Pattern
        if any([task, run, processing]):
            # Legacy BIDS-like construction
            p_parts = ["sub-*"]
            if task:
                p_parts.append(f"task-{task}")
            if run:
                p_parts.append(f"run-{run}")
            p_parts.append(f"embeddings{processing or ''}.pkl")
            self.pattern = "_".join(p_parts)
        else:
            self.pattern = pattern

        # 2. Set Reader
        self.reader = reader if reader else smart_reader
        self.id_fn = id_fn or (lambda path: path.stem)

    def load(self) -> DataContainer:
        # Find files
        files = sorted(self.path.rglob(self.pattern))

        if not files:
            raise FileNotFoundError(
                f"No files matched pattern '{self.pattern}' in {self.path}"
            )

        # Filter by subjects
        if self.subjects is not None:
            if isinstance(self.subjects, int):
                files = files[: self.subjects]
            else:
                target_ids = {str(s) for s in self.subjects}
                files = [f for f in files if self.id_fn(f) in target_ids]

        data_list = []
        ids_list = []

        logger.info(f"Loading {len(files)} embedding files...")

        for fpath in files:
            try:
                # Reader returns (N, ...) or Dict
                content = self.reader(fpath)
                sid = self.id_fn(fpath)

                if isinstance(content, dict):
                    # Dict {segment_id: array}
                    for seg_k in sorted(content.keys()):
                        arr = np.array(content[seg_k])  # Ensure array

                        if arr.ndim == len(self.dims) + 1:
                            data_list.append(arr)
                            ids_list.extend(
                                [f"{sid}_{seg_k}_{i}" for i in range(len(arr))]
                            )
                        elif arr.ndim == len(self.dims):
                            data_list.append(arr[np.newaxis, ...])
                            ids_list.append(f"{sid}_{seg_k}")
                        else:
                            logger.warning(
                                f"Shape mismatch in {fpath.name} key {seg_k}: "
                                f"{arr.shape} vs dims {self.dims}"
                            )

                else:
                    # Single Array or List
                    arr = np.array(content)

                    if arr.ndim == len(self.dims) + 1:
                        data_list.append(arr)
                        ids_list.extend([f"{sid}_{i}" for i in range(len(arr))])
                    elif arr.ndim == len(self.dims):
                        data_list.append(arr[np.newaxis, ...])
                        ids_list.append(sid)
                    else:
                        logger.warning(
                            f"Shape mismatch in {fpath.name}: {arr.shape} vs dims "
                            f"{self.dims}"
                        )

            except Exception as e:
                logger.error(f"Failed to read {fpath.name}: {e}")
                continue

        if not data_list:
            raise RuntimeError("No valid data loaded.")

        try:
            X_final = np.concatenate(data_list, axis=0)
        except ValueError as e:
            shapes = [d.shape for d in data_list[:5]]
            raise ValueError(f"Concatenation failed. Shapes vary? {shapes}") from e

        # Obs Dim + User Dims
        final_dims = ("obs", *self.dims)

        # Build coordinates
        coords = {}
        if ids_list:
            coords["obs"] = np.array(ids_list)

        # Add user-provided coords (e.g. {'layer': ['L1', 'L2']})
        coords.update(self.coords_in)

        return DataContainer(
            X=X_final, dims=final_dims, coords=coords, ids=np.array(ids_list)
        )


class BIDSDataset(BaseDataset):
    """
    Dataset for loading epoched M/EEG data from a BIDS dataset.

    Recordings are found and read with MNE-BIDS, as :class:`mne.Epochs`, and
    stacked with :func:`mne.concatenate_epochs`, so they must share their
    channels, sampling rate and epoch length. Metadata from ``participants.tsv``
    is attached to every observation.

    Parameters
    ----------
    root : str or Path
        The root directory of the BIDS dataset.
    task : str, optional
        The task name (e.g., 'rest', 'audiovisual').
    session : str or List[str], optional
        The session ID(s) to load. If None, all sessions are loaded.
    datatype : str, default='eeg'
        The data type to load (e.g., 'eeg', 'meg').
    suffix : str, optional
        The suffix of the files to load. Defaults to `datatype`, or to 'epo'
        for pre-computed epochs.
    mode : str, default='epochs'
        - 'epochs': cut raw recordings into epochs.
        - 'load_existing': read pre-computed epochs (``*_epo.fif``).
    target_col : str, optional
        Metadata column to expose as the target ``y``. By default ``y`` holds
        the event codes.
    window_length : float, optional
        Length in seconds of the fixed-length epochs cut from raw recordings.
    stride : float, optional
        Stride between windows in seconds. Defaults to `window_length`.
    subjects : str or List[str], optional
        Subject IDs to load (without 'sub-' prefix). If None, all subjects.
    runs : str or List[str], optional
        Run IDs to load. If None, all runs.
    event_id : dict, str or list of str, optional
        For raw recordings, the events to epoch around (see
        :func:`mne.events_from_annotations`), in place of fixed-length windows.
        For pre-computed epochs, the event names to keep.
    subject_metadata_df : DataFrame, optional
        Subject-level metadata to attach, in addition to ``participants.tsv``.
    subject_key : str, optional
        Column of `subject_metadata_df` holding the subject ID.
    tmin, tmax : float
        Epoch limits in seconds around each event, with `event_id` on raw
        recordings.
    baseline : tuple, optional
        Baseline interval, with `event_id` on raw recordings.
    units : str, optional
        Units of the returned data (see :meth:`mne.Epochs.get_data`).
    description : str, optional
        The ``desc`` entity of the files to load.

    Notes
    -----
    The ``recording_id`` coordinate is the BIDS entity string of a recording,
    e.g. ``sub-0001_ses-01_task-rest_run-01``. Observation IDs are
    ``<recording_id>_epoch-<i>``.

    Examples
    --------
    >>> # Load resting state EEG for all subjects, sliced into 1s windows
    >>> ds = BIDSDataset(root="/data/bids", task="rest", window_length=1.0)
    >>> container = ds.load()
    """

    def __init__(
        self,
        root: str | Path,
        task: str | None = None,
        session: str | list[str] | None = None,
        datatype: str = "eeg",
        suffix: str | None = None,
        mode: str = "epochs",
        target_col: str | None = None,
        window_length: float | None = None,
        stride: float | None = None,
        subjects: str | list[str] | None = None,
        runs: str | list[str] | None = None,
        event_id: dict[str, int] | str | list[str] | None = None,
        subject_metadata_df: pd.DataFrame | None = None,
        subject_key: str | None = None,
        tmin: float = -0.2,
        tmax: float = 0.5,
        baseline: tuple[float | None, float | None] | None = None,
        units: str | None = None,
        description: str | None = None,
    ):
        self.root = Path(root)
        self.task = task
        self.session = session
        self.datatype = datatype
        self.suffix = suffix
        self.mode = mode
        self.target_col = target_col
        self.window_length = window_length
        self.stride = stride
        self.subjects = subjects
        self.runs = runs
        self.event_id = event_id
        self.subject_metadata_df = subject_metadata_df
        self.subject_key = subject_key
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.units = units
        self.description = description

    def _find_recordings(self, saved: bool) -> dict[str, BIDSPath]:
        """Return the ``BIDSPath`` of each recording to load, by recording ID."""
        paths = find_matching_paths(
            self.root,
            subjects=self.subjects,
            sessions=self.session,
            tasks=self.task,
            runs=self.runs,
            descriptions=self.description,
            datatypes=self.datatype,
            suffixes=self.suffix or ("epo" if saved else self.datatype),
            extensions=".fif" if saved else None,
            ignore_json=True,
        )
        recordings = {}
        for path in paths:
            recording_id = BIDSPath(
                subject=path.subject,
                session=path.session,
                task=path.task,
                acquisition=path.acquisition,
                run=path.run,
            ).basename
            if not saved:
                # Raw formats can have several files per recording (.vhdr, .eeg).
                recordings.setdefault(recording_id, path.copy().update(extension=None))
            elif recording_id in recordings:
                raise ValueError(
                    f"Several files match recording {recording_id}: "
                    f"{recordings[recording_id].basename} and {path.basename}. "
                    "Pass `description` to choose one."
                )
            else:
                recordings[recording_id] = path
        return dict(sorted(recordings.items()))

    def _read_epochs(self, bids_path: BIDSPath, saved: bool) -> mne.BaseEpochs:
        """Read one recording as epochs."""
        if saved:
            # Derivative epochs have no sidecars to warn about.
            epochs = read_epochs_bids(bids_path, verbose="error")
            if self.event_id is None:
                return epochs
            names = [self.event_id] if isinstance(self.event_id, str) else self.event_id
            names = [name for name in names if name in epochs.event_id]
            if not names:
                raise ValueError(f"No epochs with event_id={self.event_id}.")
            return epochs[names]

        raw = read_raw_bids(bids_path, verbose=False).pick(["eeg", "meg"])
        if self.event_id is None:
            return mne.make_fixed_length_epochs(
                raw,
                duration=self.window_length,
                overlap=self.window_length - (self.stride or self.window_length),
                preload=True,
                verbose=False,
            )
        events, event_id = mne.events_from_annotations(
            raw, event_id=self.event_id, verbose=False
        )
        return mne.Epochs(
            raw,
            events,
            event_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            preload=True,
            verbose=False,
        )

    def load(self) -> DataContainer:
        """
        Load the BIDS dataset into a DataContainer.

        Returns
        -------
        DataContainer
            A container with:
            - X: Data array of shape (N_obs, N_channels, N_time).
            - y: The event code of each observation, or `target_col`.
            - ids: Unique identifiers for each observation.
            - coords: 'channel', 'time', and one entry per observation for
              'obs', 'recording_id', 'subject', 'session', 'run' and the
              subject metadata.
            - dims: ('obs', 'channel', 'time').
            - meta: 'sfreq', 'source', 'units', and 'load_failures' (the
              'recording_id' and 'error' of each recording that failed to load).

        Raises
        ------
        RuntimeError
            If no recording could be loaded.
        ValueError
            If the recordings cannot be stacked (see
            :func:`mne.concatenate_epochs`).
        """
        saved = self.mode == "load_existing" or "epo" in (self.suffix or "")
        if not saved and self.window_length is None and self.event_id is None:
            raise ValueError("Raw recordings need `window_length` or `event_id`.")

        meta_lookup = load_participants_tsv(self.root)
        if self.subject_metadata_df is not None:
            if self.subject_key is None:
                raise ValueError(
                    "subject_key must be provided when subject_metadata_df is used."
                )
            if self.subject_key not in self.subject_metadata_df.columns:
                raise ValueError(
                    f"subject_key '{self.subject_key}' not found "
                    "in subject_metadata_df."
                )
            for _, row in self.subject_metadata_df.iterrows():
                sub = normalize_subject_value(row[self.subject_key])
                meta_lookup.setdefault(sub, {}).update(row.to_dict())

        epochs_list = []
        load_failures: list[dict[str, str]] = []
        with mne.use_log_level("WARNING"):
            for recording_id, bids_path in self._find_recordings(saved).items():
                try:
                    epochs = self._read_epochs(bids_path, saved)
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    logger.warning(f"Could not load {recording_id}: {error}")
                    load_failures.append({"recording_id": recording_id, "error": error})
                    continue
                epochs.metadata = pd.DataFrame(
                    {
                        "obs": [
                            f"{recording_id}_epoch-{i:04d}" for i in range(len(epochs))
                        ],
                        "recording_id": recording_id,
                        "subject": bids_path.subject,
                        "session": bids_path.session or "",
                        "run": bids_path.run or "",
                        **meta_lookup.get(
                            normalize_subject_value(bids_path.subject), {}
                        ),
                    }
                )
                if epochs_list and set(epochs.ch_names) == set(epochs_list[0].ch_names):
                    epochs.reorder_channels(epochs_list[0].ch_names)
                epochs_list.append(epochs)

            if not epochs_list:
                message = f"No valid data found in {self.root}"
                if load_failures:
                    message += (
                        f": {len(load_failures)} recording(s) could not be loaded, "
                        f"the first being {load_failures[0]['recording_id']} "
                        f"({load_failures[0]['error']})"
                    )
                raise RuntimeError(message)

            epochs = mne.concatenate_epochs(epochs_list, verbose=False)

        coords = {
            "channel": np.array(epochs.ch_names),
            "time": epochs.times,
            **{name: epochs.metadata[name].to_numpy() for name in epochs.metadata},
        }
        for name in ("session", "run"):
            if not coords[name].astype(bool).any():
                del coords[name]

        y = epochs.events[:, 2]
        if self.target_col is not None:
            if self.target_col not in coords:
                raise ValueError(
                    f"target_col '{self.target_col}' not found in BIDS coords."
                )
            y = coords[self.target_col]

        return DataContainer(
            X=epochs.get_data(units=self.units, copy=False),
            y=y,
            ids=coords["obs"],
            dims=("obs", "channel", "time"),
            coords=coords,
            meta={
                "sfreq": epochs.info["sfreq"],
                "source": str(self.root),
                "units": self.units,
                "load_failures": load_failures,
            },
        )
