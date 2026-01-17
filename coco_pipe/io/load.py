"""
coco_pipe/io/load.py
--------------------
High-level data loading factory.

Author: Hamza Abdelhedi <hamza.abdelhedii@gmail.com>
Date: 2026-01-14
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .dataset import BIDSDataset, EmbeddingDataset, TabularDataset
from .structures import DataContainer

logger = logging.getLogger(__name__)


def load_data(
    path: Union[str, Path],
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
    datatype: str = "eeg",
    suffix: Optional[str] = None,
    loading_mode: str = "epochs",  # Maps to BIDSDataset `mode`
    window_length: Optional[float] = None,
    stride: Optional[float] = None,
    # --- Embedding Arguments ---
    pattern: str = "*.pkl",
    dims: Tuple[str, ...] = ("obs", "feature"),
    coords: Optional[Dict[str, Union[List, np.ndarray]]] = None,
    reader: Optional[Any] = None,
    id_fn: Optional[Any] = None,
    # --- Common Arguments ---
    subjects: Optional[Union[str, List[str], int, List[int]]] = None,
    **kwargs,
) -> DataContainer:
    """
    Universal data loader factory.
    Dispatches to `BIDSDataset`, `TabularDataset`, or `EmbeddingDataset` based on `mode`.

    Parameters
    ----------
    path : str or Path
        Path to data source (file or directory).
    mode : {"auto", "tabular", "bids", "embedding"}, default="auto"
        Type of data to load.
        - "auto": Infers type from file extension or directory structure.
        - "tabular": uses `TabularDataset` (CSV, TSV, Excel, TXT).
        - "bids": uses `BIDSDataset` (BIDS-compliant directories).
        - "embedding": uses `EmbeddingDataset` (NPY, PKL, H5, JSON).

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
        How to process the data. passed as `mode` to BIDSDataset.
        - 'epochs': Splices continuous data into fixed-length windows.
        - 'continuous': Loads as single continuous segments.
        - 'load_existing': Loads pre-computed epochs.
    window_length : float, optional
        Window length in seconds (for 'epochs' mode).
    stride : float, optional
        Stride in seconds (for 'epochs' mode).
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
    """
    path = Path(path)

    # 1. Simple Inference
    if mode == "auto":
        if path.is_dir():
            if (path / "dataset_description.json").exists() or any(path.glob("sub-*")):
                mode = "bids"
            else:
                mode = "embedding"
        else:
            suffix = path.suffix.lower()
            if suffix in [".csv", ".tsv", ".xls", ".xlsx", ".txt"]:
                mode = "tabular"
            else:
                mode = "embedding"

    logger.info(f"Loading data from {path} using mode='{mode}'")

    # 2. Dispatch
    if mode == "tabular":
        return TabularDataset(
            path=path,
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
            **kwargs,
        ).load()

    elif mode == "bids":
        # Note: mapping loading_mode -> mode
        return BIDSDataset(
            root=path,
            mode=loading_mode,
            task=task,
            session=session,
            datatype=datatype,
            suffix=suffix,
            window_length=window_length,
            stride=stride,
            subjects=subjects,
            **kwargs,
        ).load()

    elif mode == "embedding":
        return EmbeddingDataset(
            path=path,
            pattern=pattern,
            dims=dims,
            coords=coords,
            reader=reader,
            id_fn=id_fn,
            subjects=subjects,
            **kwargs,
        ).load()

    else:
        raise ValueError(
            f"Unknown mode: '{mode}'. Must be 'tabular', 'bids', or 'embedding'."
        )
