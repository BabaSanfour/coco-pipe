"""
coco_pipe/io/load.py
--------------------
High-level data loading from a dataset configuration.

Author: Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>
"""

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

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
    config: DatasetConfig | TabularConfig | BIDSConfig | EmbeddingConfig,
    *,
    subject_metadata_df: pd.DataFrame | None = None,
    subject_key: str | None = None,
    reader: Callable[[Any], Any] | None = None,
    id_fn: Callable[[Any], str] | None = None,
) -> DataContainer:
    """
    Load a dataset described by a configuration into a DataContainer.

    The configuration says what to load and how:
    :class:`~coco_pipe.io.config.TabularConfig` for a feature table,
    :class:`~coco_pipe.io.config.BIDSConfig` for a BIDS M/EEG dataset and
    :class:`~coco_pipe.io.config.EmbeddingConfig` for embedding files, each
    validated when it is built. Only what a configuration cannot hold, tables
    and functions, is passed here.

    Parameters
    ----------
    config : DatasetConfig or {Tabular,BIDS,Embedding}Config
        The dataset to load.
    subject_metadata_df : DataFrame, optional
        BIDS only: subject-level metadata attached to each observation, in
        addition to ``participants.tsv``.
    subject_key : str, optional
        BIDS only: the column of `subject_metadata_df` holding the subject ID.
    reader : callable, optional
        Embeddings only: reads one file and returns its array or dict.
    id_fn : callable, optional
        Embeddings only: returns the subject ID of a file path.

    Returns
    -------
    DataContainer

    Raises
    ------
    TypeError
        If ``config`` is not a dataset configuration, e.g. a path or a dict.

    Examples
    --------
    >>> from coco_pipe.io import BIDSConfig, TabularConfig, load_data
    >>> table = load_data(TabularConfig(path="features.csv", target_col="y"))
    >>> epochs = load_data(
    ...     BIDSConfig(path="/data/bids", task="rest", window_length=2.0)
    ... )
    """
    cfg = config.dataset if isinstance(config, DatasetConfig) else config
    if not isinstance(cfg, TabularConfig | BIDSConfig | EmbeddingConfig):
        raise TypeError(
            "load_data takes a dataset configuration (TabularConfig, BIDSConfig, "
            f"EmbeddingConfig or a DatasetConfig holding one), got "
            f"{type(config).__name__}; e.g. load_data(TabularConfig(path=...))."
        )
    logger.info(f"Loading data from {cfg.path} using mode='{cfg.mode}'")

    if cfg.mode == "tabular":
        return _resolve_dataset_class("TabularDataset")(
            path=cfg.path,
            target_col=cfg.target_col,
            index_col=cfg.index_col,
            sep=cfg.sep,
            header=cfg.header,
            sheet_name=cfg.sheet_name,
            columns_to_dims=cfg.columns_to_dims,
            col_sep=cfg.col_sep,
            meta_columns=cfg.meta_columns,
            clean=cfg.clean,
            clean_kwargs=cfg.clean_kwargs,
        ).load()

    if cfg.mode == "bids":
        return _resolve_dataset_class("BIDSDataset")(
            root=cfg.path,
            mode=cfg.loading_mode,
            task=cfg.task,
            session=cfg.session,
            runs=cfg.runs,
            datatype=cfg.datatype,
            suffix=cfg.suffix,
            target_col=cfg.target_col,
            window_length=cfg.window_length,
            stride=cfg.stride,
            event_id=cfg.event_id,
            tmin=cfg.tmin,
            tmax=cfg.tmax,
            baseline=cfg.baseline,
            units=cfg.units,
            description=cfg.description,
            dtype=cfg.dtype,
            n_jobs=cfg.n_jobs,
            subject_metadata_df=subject_metadata_df,
            subject_key=subject_key,
            subjects=cfg.subjects,
        ).load()

    return _resolve_dataset_class("EmbeddingDataset")(
        path=cfg.path,
        pattern=cfg.pattern,
        dims=cfg.dims,
        coords=cfg.coords,
        reader=reader,
        id_fn=id_fn,
        subjects=cfg.subjects,
    ).load()
