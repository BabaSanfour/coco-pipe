#!/usr/bin/env python3
"""
coco_pipe/io/tabular.py

Load and process tabular data (CSV, Excel, TSV. Supports ML pipeline usage.
"""
import logging
from pathlib import Path
from typing import Union, Optional, Tuple

import pandas as pd
import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_tabular(
    data_path: Union[str, Path],
    target_col: Optional[str] = None,
    header: Union[int, None] = 0,
    index_col: Optional[Union[int, str]] = None,
    sheet_name: Optional[str] = None,
    sep: Optional[str] = None,
) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load tabular data from various formats (CSV, Excel, TSV) with flexible configuration.

    Args:
        data_path: Path to the tabular data file
        target_col: Target column name for ML pipeline usage
        header: Row number to use as column names (None if no header)
        index_col: Column to use as index
        sheet_name: Sheet name for Excel files
        sep: Separator for CSV/TSV files (auto-detected if None)
        sensorwise: Whether to return data in sensorwise format

    Returns:
        If target_col is specified:
            X: Features DataFrame
            y: Target Series
        If BIDS format:
            data_array: shape (n_samples, features)
            subjects_array: shape (n_samples,)
            segments_array: shape (n_samples,)
    """
    data_path = Path(data_path)
    
    # Determine file type and load accordingly
    if data_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(
            data_path,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col
        )
    else:  # CSV, TSV, or similar
        if sep is None:
            # Auto-detect separator
            if data_path.suffix.lower() == '.tsv':
                sep = '\t'
            else:
                sep = ','
        
        df = pd.read_csv(
            data_path,
            sep=sep,
            header=header,
            index_col=index_col
        )
    
    # Handle ML pipeline usage if target column is specified
    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y
    
    return df