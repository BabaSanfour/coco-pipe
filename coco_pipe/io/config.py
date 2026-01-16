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

Author: Antigravity
Date: 2026-01-16
"""

from typing import Dict, Any, List, Optional, Union, Literal, Tuple
from pathlib import Path
from pydantic import BaseModel, Field, validator

class BaseDatasetConfig(BaseModel):
    path: Path = Field(..., description="Path to the data source (file or directory).")
    subjects: Optional[Union[int, List[Union[str, int]]]] = Field(
        None, description="Specific subjects to load (int for top N, list for IDs)."
    )

class TabularConfig(BaseDatasetConfig):
    """Configuration for TabularDataset."""
    mode: Literal["tabular"] = "tabular"
    target_col: Optional[str] = Field(None, description="Column to use as target `y`.")
    index_col: Optional[Union[str, int]] = Field(None, description="Column to use as index.")
    sep: str = Field("\t", description="Separator for text files.")
    header: Optional[Union[int, List[int]]] = 0
    sheet_name: Union[str, int] = 0
    columns_to_dims: Optional[List[str]] = Field(None, description="Reshape columns into dimensions.")
    col_sep: str = "_"
    meta_columns: Optional[List[str]] = Field(None, description="Columns to extract as metadata.")
    clean: bool = False
    clean_kwargs: Dict[str, Any] = Field(default_factory=dict)
    select_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Arguments for feature selection.")

class BIDSConfig(BaseDatasetConfig):
    """Configuration for BIDSDataset."""
    mode: Literal["bids"] = "bids"
    task: Optional[str] = None
    session: Optional[Union[str, List[str]]] = None
    datatype: str = "eeg"
    suffix: Optional[str] = None
    loading_mode: str = Field("epochs", description="Loading strategy: 'epochs', 'continuous', 'load_existing'.")
    window_length: Optional[float] = None
    stride: Optional[float] = None

class EmbeddingConfig(BaseDatasetConfig):
    """Configuration for EmbeddingDataset."""
    mode: Literal["embedding"] = "embedding"
    pattern: str = "*.pkl"
    dims: Tuple[str, ...] = ("obs", "feature")
    coords: Optional[Dict[str, Union[List, Any]]] = None
    task: Optional[str] = None
    run: Optional[str] = None
    processing: Optional[str] = None


class DatasetConfig(BaseModel):
    """Master configuration container for IO."""
    dataset: Union[TabularConfig, BIDSConfig, EmbeddingConfig] = Field(..., discriminator='mode')
