import numpy as np
import pandas as pd
from typing import Union, Optional, List, Tuple

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
    target_col: Optional[str] = None,
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
            target_col=target_col,
            header=header,
            index_col=index_col,
            sheet_name=sheet_name,
            sep=sep,
        )
    else:
        raise ValueError(f"Unknown data type '{type}', choose from 'embeddings', 'csv', 'meg', 'eeg', or 'meeg'")