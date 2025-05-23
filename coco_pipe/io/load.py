"""
coco_pipe/io/load.py
----------------
Load data from various formats (CSV, Excel, TSV). Supports ML pipeline usage.

Author: Hamza Abdelhedi <hamza.abdelhedii@gmail.com>
Date: 2025-05-18
Version: 0.0.1
License: TBD
"""
import numpy as np
import pandas as pd
from typing import Union, Optional, List, Tuple

# TODO: add embeddings, meeg loaders
# TODO: add function description
# TODO: add docstrings
# TODO: add test units

def load(
    type: str,
    data_path: str,
    task: Optional[str] = None,
    run: Optional[str] = None,
    processing: Optional[str] = None,
    subjects: Optional[List[str]] = None,
    max_seg: Optional[int] = None,
    flatten: bool = False,
    sensorwise: bool = False,
    target_cols: Optional[Union[str, List[str]]] = None,
    header: Union[int, None] = 0,
    index_col: Optional[Union[int, str]] = None,
    sheet_name: Optional[str] = None,
    sep: Optional[str] = None,
) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]:
    if type == "embeddings":
        raise NotImplementedError("Embeddings loading not implemented yet")
    elif type in ["meeg", "meg", "eeg"]:
        raise NotImplementedError("M/EEG loading not implemented yet")
    elif type in ["tabular", "csv", "excel", "tsv"]:
        from coco_pipe.io.tabular import load_tabular
        return load_tabular(
            data_path=data_path,
            target_cols=target_cols,
            header=header,
            index_col=index_col,
            sheet_name=sheet_name,
            sep=sep,
        )
    else:
        raise ValueError(f"Unknown data type '{type}', choose from 'embeddings', 'csv', 'meg', 'eeg', or 'meeg'")