"""
coco_pipe/io/load.py
--------------------
High-level data loading factory.

Author: Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import BIDSConfig, DatasetConfig, EmbeddingConfig, TabularConfig
from .structures import DataContainer

logger = logging.getLogger(__name__)

__all__ = ["load_data"]


def __getattr__(name):
    if name in {"TabularDataset", "BIDSDataset", "EmbeddingDataset"}:
        from . import dataset as dataset_mod

        return getattr(dataset_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_dataset_class(name: str):
    """Resolve dataset classes lazily while remaining patch-friendly in tests."""
    return globals().get(name) or __getattr__(name)


def load_data(
    path: Optional[Union[str, Path]] = None,
    mode: str = "auto",
    # --- Tabular Arguments ---
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
    # --- BIDS Arguments ---
    task: Optional[str] = None,
    session: Optional[Union[str, List[str]]] = None,
    runs: Optional[Union[str, List[str]]] = None,
    datatype: str = "eeg",
    suffix: Optional[str] = None,
    loading_mode: str = "epochs",  # Maps to BIDSDataset `mode`
    window_length: Optional[float] = None,
    stride: Optional[float] = None,
    event_id: Optional[Union[Dict[str, int], str, List[str]]] = None,
    tmin: float = -0.2,
    tmax: float = 0.5,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    drop_short_epochs: bool = True,
    subject_metadata_df: Optional[Any] = None,
    subject_key: Optional[str] = None,
    # --- Embedding Arguments ---
    pattern: str = "*.pkl",
    dims: Tuple[str, ...] = ("obs", "feature"),
    coords: Optional[Dict[str, Union[List, np.ndarray]]] = None,
    run: Optional[str] = None,
    processing: Optional[str] = None,
    reader: Optional[Any] = None,
    id_fn: Optional[Any] = None,
    # --- Common Arguments ---
    subjects: Optional[Union[str, List[str], int, List[int]]] = None,
    config: Optional[
        Union[DatasetConfig, BIDSConfig, TabularConfig, EmbeddingConfig]
    ] = None,
    **kwargs,
) -> DataContainer:
    """
    Universal data loader factory.
    Dispatches to `BIDSDataset`, `TabularDataset`, or `EmbeddingDataset` based on
    `mode`.

    Parameters
    ----------
    path : str or Path, optional
        Path to data source (file or directory). Required unless ``config`` is
        given (in which case ``config.path`` is used).
    mode : {"auto", "tabular", "bids", "embedding"}, default="auto"
        Type of data to load.
        - "auto": Infers type from file extension or directory structure. A
          directory with ``dataset_description.json`` or ``sub-*`` entries is
          treated as ``"bids"``; ``.csv``/``.tsv``/``.xls``/``.xlsx``/``.txt``
          files as ``"tabular"``; everything else as ``"embedding"``.
        - "tabular": uses `TabularDataset` (CSV, TSV, Excel, TXT).
        - "bids": uses `BIDSDataset` (BIDS-compliant directories).
        - "embedding": uses `EmbeddingDataset` (NPY, PKL, H5, JSON).
    config : DatasetConfig or {Tabular,BIDS,Embedding}Config, optional
        A pre-validated configuration object (see :mod:`coco_pipe.io.config`).
        When provided, its fields drive the load and ``mode`` is taken from the
        config; the matching keyword arguments below are ignored. When omitted,
        the relevant keyword arguments are assembled into a config and validated
        before dispatch. The non-serializable parameters ``reader``, ``id_fn``,
        ``subject_metadata_df``, and ``subject_key`` are always passed through
        directly and are never part of the config schema.

    Tabular Arguments (mode="tabular")
    ----------------------------------
    target_col : str, optional
        Name of the column to extract as target `y`. Removed from features `X`.
    index_col : str or int, optional
        Column to use as index (observation IDs).
    sep : str, default='\\t'
        Separator for text files (e.g. ',' for CSV).
    header : int or list of int, default=0
        Row number(s) to use as column names.
    sheet_name : str or int, default=0
        Sheet name or index for Excel files.
    columns_to_dims : list of str, optional
        If provided, attempts to reshape 2D feature columns into N-D dimensions.
        Columns must follow: `dim1_dim2_..._feature`.
    col_sep : str, default='_'
        Separator used in column names for reshaping.
    meta_columns : list of str, optional
        Columns to extract as metadata coordinates instead of features.
    clean : bool, default=False
        Whether to perform automated cleaning (drop NaNs/Infs).
    clean_kwargs : dict, optional
        Arguments passed to `TabularDataset.clean`.

    BIDS Arguments (mode="bids")
    ----------------------------
    task : str, optional
        BIDS task name (e.g., 'rest', 'audiovisual').
    session : str or List[str], optional
        Session ID(s) to load. Defaults to all available.
    datatype : str, default='eeg'
        Data type folder (e.g., 'eeg', 'meg', 'ieeg').
    suffix : str, optional
        File suffix to load (e.g., 'eeg', 'epo', 'ave').
    loading_mode : str, default='epochs'
        How to process the data. Renamed to ``loading_mode`` here (and in
        ``BIDSConfig``) to avoid colliding with this function's ``mode``
        argument; it is passed through as ``mode`` to ``BIDSDataset``.
        - 'epochs': Splices continuous data into fixed-length windows.
        - 'continuous': Loads as single continuous segments.
        - 'load_existing': Loads pre-computed epochs.
    window_length : float, optional
        Window length in seconds (for 'epochs' mode).
    stride : float, optional
        Stride in seconds (for 'epochs' mode).
    subject_metadata_df : DataFrame, optional
        External subject-level metadata to merge by subject during BIDS loading.
    subject_key : str, optional
        Column in `subject_metadata_df` containing the BIDS subject identifier.
    subjects : str or List[str], optional
        Specific subject IDs to load (without 'sub-').

    Embedding Arguments (mode="embedding")
    --------------------------------------
    pattern : str, default='*.pkl'
        Glob pattern to match files.
    dims : tuple of str, default=('obs', 'feature')
        Dimension labels for the data arrays.
    coords : dict, optional
        Dictionary of coordinates for dimensions.
    reader : callable, optional
        Custom file reader function.
    id_fn : callable, optional
        Custom subject ID extraction function.
    subjects : int or list, optional
        If int, loads first N subjects. If list, filters by ID.

    Returns
    -------
    DataContainer
        Standardized data container with attributes:
        - X: (N_obs, ...) data array
        - y: Targets (if available)
        - ids: Observation identifiers
        - coords: Coordinate metadata

    Examples
    --------
    Two equivalent ways to load. The keyword form is convenient for quick,
    interactive use:

    >>> container = load_data("features.csv", mode="tabular", target_col="y")

    The **config-first** form is recommended for pipelines and reproducible
    runs: a :class:`~coco_pipe.io.config.TabularConfig` /
    :class:`~coco_pipe.io.config.BIDSConfig` /
    :class:`~coco_pipe.io.config.EmbeddingConfig` is validated once and can be
    serialized, version-controlled, and reused. It also keeps each mode's
    options self-contained instead of mixing all three modes' keywords:

    >>> from coco_pipe.io.config import TabularConfig
    >>> cfg = TabularConfig(path="features.csv", target_col="y")
    >>> container = load_data(config=cfg)

    BIDS loading uses ``loading_mode`` (not ``mode``) to choose the windowing
    strategy:

    >>> container = load_data(
    ...     "/data/bids", mode="bids", task="rest", loading_mode="epochs",
    ...     window_length=2.0,
    ... )
    """
    # 1. Resolve the validated dataset config (either passed in or built from
    #    kwargs). Building it through pydantic validates types up front.
    if config is not None:
        dataset_cfg = config.dataset if isinstance(config, DatasetConfig) else config
        path = Path(dataset_cfg.path)
    else:
        if path is None:
            raise ValueError("`path` is required when `config` is not provided.")
        path = Path(path)
        if mode == "auto":
            mode = _infer_mode(path)
        dataset_cfg = _build_config(
            path=path,
            mode=mode,
            target_col=target_col,
            index_col=index_col,
            sep=sep,
            header=header,
            sheet_name=sheet_name,
            columns_to_dims=columns_to_dims,
            col_sep=col_sep,
            meta_columns=meta_columns,
            clean=clean,
            clean_kwargs=clean_kwargs,
            task=task,
            session=session,
            runs=runs,
            datatype=datatype,
            suffix=suffix,
            loading_mode=loading_mode,
            window_length=window_length,
            stride=stride,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            drop_short_epochs=drop_short_epochs,
            pattern=pattern,
            dims=dims,
            coords=coords,
            run=run,
            processing=processing,
            subjects=subjects,
        )

    mode = dataset_cfg.mode
    logger.info(f"Loading data from {path} using mode='{mode}'")

    # 2. Dispatch using validated config fields. Non-serializable params
    #    (reader/id_fn/subject_metadata_df/subject_key) are passed through
    #    directly since they are not part of the config schema.
    if mode == "tabular":
        return _resolve_dataset_class("TabularDataset")(
            path=path,
            target_col=dataset_cfg.target_col,
            index_col=dataset_cfg.index_col,
            sep=dataset_cfg.sep,
            header=dataset_cfg.header,
            sheet_name=dataset_cfg.sheet_name,
            columns_to_dims=dataset_cfg.columns_to_dims,
            col_sep=dataset_cfg.col_sep,
            meta_columns=dataset_cfg.meta_columns,
            clean=dataset_cfg.clean,
            clean_kwargs=dataset_cfg.clean_kwargs,
            select_kwargs=dataset_cfg.select_kwargs,
            **kwargs,
        ).load()

    elif mode == "bids":
        # Note: config.loading_mode maps to BIDSDataset's `mode`.
        return _resolve_dataset_class("BIDSDataset")(
            root=path,
            mode=dataset_cfg.loading_mode,
            task=dataset_cfg.task,
            session=dataset_cfg.session,
            runs=dataset_cfg.runs,
            datatype=dataset_cfg.datatype,
            suffix=dataset_cfg.suffix,
            target_col=dataset_cfg.target_col,
            window_length=dataset_cfg.window_length,
            stride=dataset_cfg.stride,
            event_id=dataset_cfg.event_id,
            tmin=dataset_cfg.tmin,
            tmax=dataset_cfg.tmax,
            baseline=dataset_cfg.baseline,
            drop_short_epochs=dataset_cfg.drop_short_epochs,
            subject_metadata_df=subject_metadata_df,
            subject_key=subject_key,
            subjects=dataset_cfg.subjects,
            **kwargs,
        ).load()

    elif mode == "embedding":
        return _resolve_dataset_class("EmbeddingDataset")(
            path=path,
            pattern=dataset_cfg.pattern,
            dims=dataset_cfg.dims,
            coords=dataset_cfg.coords,
            task=dataset_cfg.task,
            run=dataset_cfg.run,
            processing=dataset_cfg.processing,
            reader=reader,
            id_fn=id_fn,
            subjects=dataset_cfg.subjects,
            **kwargs,
        ).load()

    else:
        raise ValueError(
            f"Unknown mode: '{mode}'. Must be 'tabular', 'bids', or 'embedding'."
        )


def _infer_mode(path: Path) -> str:
    """Infer the dataset mode from a path's structure or extension."""
    if path.is_dir():
        if (path / "dataset_description.json").exists() or any(path.glob("sub-*")):
            return "bids"
        return "embedding"
    if path.suffix.lower() in [".csv", ".tsv", ".xls", ".xlsx", ".txt"]:
        return "tabular"
    return "embedding"


def _build_config(
    path: Path,
    mode: str,
    target_col,
    index_col,
    sep,
    header,
    sheet_name,
    columns_to_dims,
    col_sep,
    meta_columns,
    clean,
    clean_kwargs,
    task,
    session,
    runs,
    datatype,
    suffix,
    loading_mode,
    window_length,
    stride,
    event_id,
    tmin,
    tmax,
    baseline,
    drop_short_epochs,
    pattern,
    dims,
    coords,
    run,
    processing,
    subjects,
) -> Union[TabularConfig, BIDSConfig, EmbeddingConfig]:
    """Construct and validate the mode-appropriate config from raw kwargs."""
    if mode == "tabular":
        return TabularConfig(
            path=path,
            subjects=subjects,
            target_col=target_col,
            index_col=index_col,
            sep=sep,
            header=header,
            sheet_name=sheet_name,
            columns_to_dims=columns_to_dims,
            col_sep=col_sep,
            meta_columns=meta_columns,
            clean=clean,
            clean_kwargs=clean_kwargs or {},
        )
    if mode == "bids":
        return BIDSConfig(
            path=path,
            subjects=subjects,
            task=task,
            session=session,
            runs=runs,
            datatype=datatype,
            suffix=suffix,
            loading_mode=loading_mode,
            target_col=target_col,
            window_length=window_length,
            stride=stride,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            drop_short_epochs=drop_short_epochs,
        )
    if mode == "embedding":
        return EmbeddingConfig(
            path=path,
            subjects=subjects,
            pattern=pattern,
            dims=dims,
            coords=coords,
            task=task,
            run=run,
            processing=processing,
        )
    raise ValueError(
        f"Unknown mode: '{mode}'. Must be 'tabular', 'bids', or 'embedding'."
    )
