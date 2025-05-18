#!/usr/bin/env python3
"""
coco_pipe/io/tabular.py
----------------
Load and process tabular data (CSV, Excel, TSV). Supports ML pipeline usage.

Author: Hamza Abdelhedi <hamza.abdelhedii@gmail.com>
Date: 2025-05-18
Version: 0.0.1
License: TBD
"""
import logging
from pathlib import Path
from typing import Union, Optional, Tuple, List

import pandas as pd
import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# TODO: add test units

def load_tabular(
    data_path: Union[str, Path],
    target_cols: Optional[Union[str, List[str]]] = None,
    header: Union[int, None] = 0,
    index_col: Optional[Union[int, str]] = None,
    sheet_name: Optional[str] = None,
    sep: Optional[str] = None,
) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load tabular data from various formats (CSV, Excel, TSV) with flexible configuration.

    Parameters:
        :data_path: Union[str, Path], Path to the tabular data file
        :target_cols: Optional[Union[str, List[str]]], Target column name for ML pipeline usage
        :header: Union[int, None], Row number to use as column names (None if no header)
        :index_col: Optional[Union[int, str]], Column to use as index (None if no index)
        :sheet_name: Optional[str], Sheet name for Excel files
        :sep: Optional[str], Separator for CSV/TSV files (auto-detected if None)

    Returns:
        If target_cols is specified:
            :X: Union[pd.DataFrame, np.ndarray], Features DataFrame
            :y: Union[pd.Series, np.ndarray], Target Series
        Otherwise:
            :df: Union[pd.DataFrame, np.ndarray], Dataframe or array
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
    if target_cols is not None:
        if isinstance(target_cols, str):
            target_cols = [target_cols]
        for target_col in target_cols:
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")
        y = df[target_cols]
        X = df.drop(columns=target_cols)
        logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features and {len(y)} target columns")
        return X, y
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df