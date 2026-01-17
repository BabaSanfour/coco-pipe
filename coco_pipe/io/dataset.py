"""
coco_pipe/io/dataset.py
-----------------------
Specialized Dataset classes that produce standardized DataContainer objects.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath

from .structures import DataContainer
from .utils import (
    default_id_extractor,
    detect_sessions,
    detect_subjects,
    load_participants_tsv,
    read_bids_entry,
    smart_reader,
    split_column,
)

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
    >>> ds = TabularDataset("wide.csv", columns_to_dims=['time', 'freq'], col_sep='_')
    """

    def __init__(
        self,
        path: Union[str, Path],
        target_col: Optional[str] = None,
        index_col: Optional[Union[str, int]] = None,
        sep: str = "\t",
        header: Optional[Union[int, List[int]]] = 0,
        sheet_name: Optional[Union[str, int]] = 0,
        columns_to_dims: Optional[List[str]] = None,
        col_sep: str = "_",
        meta_columns: Optional[List[str]] = None,
        clean: bool = False,
        clean_kwargs: Optional[Dict[str, Any]] = None,
        select_kwargs: Optional[Dict[str, Any]] = None,
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
                f"Tabular data contains non-numeric columns: {non_numeric.columns.tolist()}. "
                "These might cause issues during processing if not intended as metadata."
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
                msg = f"{len(failed_cols)} columns failed reshaping pattern (sep='{self.col_sep}', expected {len(self.columns_to_dims)} parts). Examples: {failed_cols[:5]}"
                if self.strict_reshaping:
                    logger.error(msg)
                    # raise ValueError(msg) # Optional: strict mode could raise
                logger.warning(msg)

            if not parsed_cols:
                raise ValueError(
                    f"No columns matched the reshaping pattern with sep='{self.col_sep}'."
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
                    f"Reshaping failed. Found {X_sorted.shape[1]} columns, expected full product {expected_total} ({dim_sizes}). Missing combinations?"
                )

            # Reshape: (N_obs, Dim1, Dim2, ...)
            new_shape = (n_obs,) + tuple(dim_sizes)
            X_final = X_sorted.values.reshape(new_shape)
            dims = tuple(["obs"] + self.columns_to_dims)

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
        min_abs_value: Optional[float] = None,
        min_abs_fraction: float = 0.0,
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Remove invalid feature columns containing NaN, ±Inf, and optionally very small values.
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
    """
    Generic Dataset for loading embedding files (Pickle, NPY, JSON, H5).

    This class decouples file discovery (via patterns and IDs) from content reading.
    It supports structured formats (e.g., Layers x Features) and user-supplied
    metadata coordinates.

    Parameters
    ----------
    path : str or Path
        Root directory containing the embedding files.
    pattern : str, default='*.pkl'
        Glob pattern to match files (e.g., "*.npy", "sub-*_emb.pkl").
    dims : tuple of str, default=('obs', 'feature')
        Dimension labels for the data arrays (excluding the observation dimension if implicit).
        Typically ('feature',) or ('layer', 'feature').
    coords : dict, optional
        Dictionary of coordinates for dimensions. E.g., {'layer': ['L1', 'L2']}.
    reader : callable, optional
        Custom function to read a Path and return a numpy array or dict.
        If None, uses `smart_reader` based on file extension.
    id_fn : callable, optional
        Custom function to extract subject ID from a Path.
        If None, uses `default_id_extractor`.
    task : str, optional
        (Legacy BIDS) Task name to construct search pattern.
    run : str, optional
        (Legacy BIDS) Run name to construct search pattern.
    processing : str, optional
        (Legacy BIDS) Processing label.
    subjects : int or list, optional
        If int, loads first N subjects. If list, loads specific subjects (matched by `id_fn`).

    Examples
    --------
    >>> # Load loose numpy files
    >>> ds = EmbeddingDataset("./embeddings", pattern="*.npy", dims=('feature',))
    >>> container = ds.load()
    """

    def __init__(
        self,
        path: Union[str, Path],
        pattern: str = "*.pkl",
        dims: Tuple[str, ...] = ("obs", "feature"),
        coords: Optional[Dict[str, Union[List, np.ndarray]]] = None,
        reader: Optional[Any] = None,
        id_fn: Optional[Any] = None,
        task: Optional[str] = None,
        run: Optional[str] = None,
        processing: Optional[str] = None,
        subjects: Optional[Union[int, List[int]]] = None,
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
        self.id_fn = id_fn if id_fn else default_id_extractor

    def load(self) -> DataContainer:
        # Find files
        files = sorted(list(self.path.rglob(self.pattern)))

        if not files:
            raise FileNotFoundError(
                f"No files matched pattern '{self.pattern}' in {self.path}"
            )

        # Filter by subjects
        if self.subjects is not None:
            if isinstance(self.subjects, int):
                files = files[: self.subjects]
            else:
                target_ids = set(str(s) for s in self.subjects)
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
                                f"Shape mismatch in {fpath.name} key {seg_k}: {arr.shape} vs dims {self.dims}"
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
                            f"Shape mismatch in {fpath.name}: {arr.shape} vs dims {self.dims}"
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
        final_dims = ("obs",) + self.dims

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
    Dataset for loading M/EEG data formatted according to the BIDS standard.

    This class supports loading valid BIDS structures, handling multiple subjects,
    sessions, and data types (Raw, Epoched, Evoked). It automatically extracts
    metadata from `participants.tsv` and aligns it with the loaded data.

    Parameters
    ----------
    root : str or Path
        The root directory of the BIDS dataset.
    task : str, optional
        The task name (e.g., 'rest', 'audiovisual').
    session : str or List[str], optional
        The session ID(s) to load. If None, detects all available sessions.
    datatype : str, default='eeg'
        The data type to load (e.g., 'eeg', 'meg').
    suffix : str, optional
        The suffix of the files to load.
        - If None, defaults to `datatype`.
        - Use 'epo' to load pre-computed epochs.
        - Use 'ave' to load evoked data.
    mode : str, default='epochs'
        The loading mode:
        - 'epochs': Splices raw continuous data into fixed-length windows.
        - 'continuous': Loads raw data as single continuous segments (1 epoch per run).
        - 'load_existing': treated as pre-computed epochs (requires `suffix='epo'`).
    window_length : float, optional
        Length of window in seconds for 'epochs' mode.
    stride : float, optional
        Stride between windows in seconds. If None, defaults to `window_length` (no overlap).
    subjects : str or List[str], optional
        Specific subject IDs to load (without 'sub-' prefix). If None, detects all subjects.

    Examples
    --------
    >>> # Load resting state EEG for all subjects, sliced into 1s windows
    >>> ds = BIDSDataset(root="/data/bids", task="rest", window_length=1.0)
    >>> container = ds.load()
    """

    def __init__(
        self,
        root: Union[str, Path],
        task: Optional[str] = None,
        session: Optional[Union[str, List[str]]] = None,
        datatype: str = "eeg",
        suffix: Optional[str] = None,
        mode: str = "epochs",
        window_length: Optional[float] = None,
        stride: Optional[float] = None,
        subjects: Optional[Union[str, List[str]]] = None,
    ):
        self.root = Path(root)
        self.task = task
        self.session = session
        self.datatype = datatype
        self.suffix = suffix
        self.mode = mode
        self.window_length = window_length
        self.stride = stride
        self.subjects = subjects

        if mne is None:
            raise ImportError("mne and mne-bids are required.")

    def load(self) -> DataContainer:
        """
        Load the BIDS dataset into a DataContainer.

        Returns
        -------
        DataContainer
            A container with:
            - X: Data array of shape (N_obs, N_channels, N_time).
            - ids: Unique identifiers for each observation.
            - coords: Dictionary containing 'channel', 'time', 'obs', and metadata.
            - dims: ('obs', 'channel', 'time').
        """
        # Resolve subjects
        if self.subjects is None:
            subjects = detect_subjects(self.root)
            subjects = sorted(subjects)
        elif isinstance(self.subjects, str):
            subjects = [self.subjects]
        else:
            subjects = sorted(self.subjects)

        # Load participants.tsv metadata
        meta_lookup = load_participants_tsv(self.root)

        data_list = []
        ids_list = []
        meta_columns = (
            {k: [] for k in next(iter(meta_lookup.values())).keys()}
            if meta_lookup
            else {}
        )

        ch_names = None
        times = None
        sfreq = None

        # Determine Loading Strategy
        # If suffix implies pre-computed data
        is_pre_epoched = (self.suffix and "epo" in self.suffix) or (
            self.mode == "load_existing"
        )
        is_evoked = (self.suffix and "ave" in self.suffix) or (self.datatype == "ave")

        for sub in subjects:
            # Resolve sessions
            if self.session is None:
                sessions = detect_sessions(self.root, sub)
                if not sessions:
                    sessions = [None]
            elif isinstance(self.session, str):
                sessions = [self.session]
            else:
                sessions = self.session

            sub_meta = meta_lookup.get(sub, {})

            for ses in sessions:
                bids_path = BIDSPath(
                    subject=sub,
                    session=ses,
                    task=self.task,
                    datatype=self.datatype,
                    root=self.root,
                    suffix=self.suffix or self.datatype,
                )

                try:
                    # --- LOAD STRATEGY (Delegated) ---
                    data, current_times, current_ch, current_sfreq = read_bids_entry(
                        bids_path,
                        is_pre_epoched=is_pre_epoched,
                        is_evoked=is_evoked,
                        mode=self.mode,
                        window_length=self.window_length,
                        stride=self.stride,
                    )

                    # --- CONSISTENCY CHECKS ---
                    if ch_names is None:
                        ch_names = current_ch
                        sfreq = current_sfreq
                        times = current_times
                    else:
                        # 1. Channel Consistency
                        if list(current_ch) != list(ch_names):
                            diff = set(current_ch) ^ set(ch_names)
                            logger.warning(
                                f"Channel mismatch for sub-{sub} ses-{ses}. "
                                f"Expected {len(ch_names)}, got {len(current_ch)}. "
                                f"Differing channels: {list(diff)[:5]}..."
                            )

                        # 2. Time/Length Consistency
                        if len(current_times) != len(times):
                            logger.warning(
                                f"Time length mismatch for sub-{sub} ses-{ses}. "
                                f"Expected {len(times)}, got {len(current_times)}. "
                                "This may cause concatenation failure."
                            )
                        elif not np.allclose(current_times, times, atol=1e-5):
                            # Often simple jitter in start times, but important if rigorous
                            pass

                    # --- APPEND DATA ---
                    data_list.append(data)

                    # --- GENERATE IDs & METADATA ---
                    # data shape is (N_epochs, C, T)
                    n_epochs = data.shape[0]

                    sid_base = f"{sub}"
                    if ses:
                        sid_base += f"_{ses}"

                    new_ids = [f"{sid_base}_{i}" for i in range(n_epochs)]
                    ids_list.extend(new_ids)

                    # Repeatedly append subject metadata for each epoch
                    for k, v in sub_meta.items():
                        meta_columns.setdefault(k, []).extend([v] * n_epochs)

                except Exception as e:
                    logger.debug(f"Failed to load subject {sub} session {ses}: {e}")
                    continue

        if not data_list:
            raise RuntimeError(f"No valid data found in {self.root}")

        # --- CONCATENATE ---
        try:
            # data_list contains (N_i, C, T)
            X_out = np.concatenate(data_list, axis=0)
        except ValueError as e:
            shapes = [d.shape for d in data_list[:5]]
            raise ValueError(f"Concatenation failed. Shapes vary? {shapes}") from e

        coords = {}
        if ch_names is not None and len(ch_names) > 0:
            coords["channel"] = np.array(ch_names)
        if times is not None:
            coords["time"] = times
        if ids_list:
            coords["obs"] = np.array(ids_list)

        # Add metadata coords
        for k, v in meta_columns.items():
            if len(v) == len(ids_list):
                coords[k] = np.array(v)

        dims = ("obs", "channel", "time")

        return DataContainer(
            X=X_out,
            y=None,
            ids=np.array(ids_list),
            dims=dims,
            coords=coords,
            meta={"sfreq": sfreq, "source": str(self.root)},
        )
