"""
Configuration Schemas for IO
============================

Pydantic models for verifying dataset configurations.

Classes
-------
TabularConfig
    Configuration for tabular data (CSV, Excel).
BIDSConfig
    Configuration for BIDS-compliant datasets.
EmbeddingConfig
    Configuration for pre-computed embeddings.
DatasetConfig
    Union container for any dataset configuration.

These schemas mirror the keyword arguments accepted by
:func:`coco_pipe.io.load.load_data` and the dataset classes in
:mod:`coco_pipe.io.dataset`. Only serializable, scalar parameters live here;
callables and in-memory tables (``reader``, ``id_fn``, ``subject_metadata_df``,
``subject_key``) are passed to ``load_data`` directly and are intentionally not
part of the config schema.

Author: Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class BaseDatasetConfig(BaseModel):
    path: Path = Field(..., description="Path to the data source (file or directory).")
    subjects: int | list[str | int] | None = Field(
        None, description="Specific subjects to load (int for top N, list for IDs)."
    )


class TabularConfig(BaseDatasetConfig):
    """Configuration for TabularDataset."""

    mode: Literal["tabular"] = "tabular"
    target_col: str | None = Field(None, description="Column to use as target `y`.")
    index_col: str | int | None = Field(None, description="Column to use as index.")
    sep: str = Field("\t", description="Separator for text files.")
    header: int | list[int] | None = 0
    sheet_name: str | int | None = 0
    columns_to_dims: list[str] | None = Field(
        None, description="Reshape columns into dimensions."
    )
    col_sep: str = "_"
    meta_columns: list[str] | None = Field(
        None, description="Columns to extract as metadata."
    )
    clean: bool = False
    clean_kwargs: dict[str, Any] = Field(default_factory=dict)
    select_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Arguments for feature selection."
    )


class BIDSConfig(BaseDatasetConfig):
    """Configuration for BIDSDataset."""

    mode: Literal["bids"] = "bids"
    task: str | None = None
    session: str | list[str] | None = None
    runs: str | list[str] | None = None
    datatype: str = "eeg"
    suffix: str | None = None
    loading_mode: str = Field(
        "epochs",
        description="Loading strategy: 'epochs', 'continuous', 'load_existing'.",
    )
    target_col: str | None = Field(
        None, description="Metadata coordinate to expose as target `y`."
    )
    window_length: float | None = None
    stride: float | None = None
    event_id: dict[str, int] | str | list[str] | None = Field(
        None, description="Event selection for annotation-based epoching."
    )
    tmin: float = -0.2
    tmax: float = 0.5
    baseline: tuple[float | None, float | None] | None = None
    drop_short_epochs: bool = True
    units: str | None = Field(None, description="Units to load EEG data in.")


class EmbeddingConfig(BaseDatasetConfig):
    """Configuration for EmbeddingDataset."""

    mode: Literal["embedding"] = "embedding"
    pattern: str = "*.pkl"
    dims: tuple[str, ...] = ("obs", "feature")
    coords: dict[str, list | Any] | None = None
    task: str | None = None
    run: str | None = None
    processing: str | None = None


class DatasetConfig(BaseModel):
    """Master configuration container for IO."""

    dataset: TabularConfig | BIDSConfig | EmbeddingConfig = Field(
        ..., discriminator="mode"
    )
