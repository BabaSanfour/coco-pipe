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

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def load_tabular(
    data_path: Union[str, Path],
    target_cols: Optional[Union[str, List[str]]] = None,
    header: Union[int, None] = 0,
    index_col: Optional[Union[int, str]] = None,
    sheet_name: Optional[str] = None,
    sep: Optional[str] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load tabular data from CSV, Excel, or TSV formats.

    Supports both general loading and machine learning workflows where
    features (X) and target(s) (y) are separated.

    Parameters
    ----------
    data_path : str or Path
        Path to the data file.
    target_cols : str or list of str, optional
        Name(s) of the target column(s). If provided, returns (X, y).
    header : int or None, optional
        Row number to use as column names. None if no header.
    index_col : int or str, optional
        Column to set as index.
    sheet_name : str, optional
        Sheet name to load from an Excel file. If None, loads the first sheet.
    sep : str, optional
        Separator for CSV/TSV files. Auto-detected if None.

    Returns
    -------
    df : pd.DataFrame
        Entire loaded DataFrame if `target_cols` is None.
    X, y : tuple of (pd.DataFrame, pd.DataFrame)
        Feature matrix and target(s) if `target_cols` is provided.

    Raises
    ------
    ValueError
        If any target column specified in `target_cols` is not found in the DataFrame.
    """
    data_path = Path(data_path)
    ext = data_path.suffix.lower()

    # 1) Load raw DataFrame
    if ext in ['.xlsx', '.xls']:
        # default to first sheet if none specified
        actual_sheet = sheet_name if sheet_name is not None else 0
        df = pd.read_excel(
            data_path,
            sheet_name=actual_sheet,
            header=header,
            index_col=index_col,
        )
    else:
        # CSV/TSV loading
        if sep is None:
            sep = '\t' if ext == '.tsv' else ','
        df = pd.read_csv(
            data_path,
            sep=sep,
            header=header,
            index_col=index_col,
        )

    # 2) Split features and targets if requested
    if target_cols is not None:
        if isinstance(target_cols, str):
            target_cols = [target_cols]
        missing = [col for col in target_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Target column(s) not found in data: {missing}")
        y = df[target_cols]
        X = df.drop(columns=target_cols)
        logger.info(
            f"Loaded {len(X)} samples with {len(X.columns)} features and "
            f"{len(y.columns)} target column(s)."
        )
        return X, y

    # 3) Return full DataFrame
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns.")
    return df
