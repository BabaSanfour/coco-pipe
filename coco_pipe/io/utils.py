#!/usr/bin/env python3
"""
coco_pipe/io/utils.py
---------------------
Generic I/O utilities for data quality assessment and row selection.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def row_quality_score(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    count_zero: bool = True,
) -> pd.Series:
    """Compute a per-row quality score: lower is cleaner (fewer NaN/inf/zeros).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    exclude_cols : list of str, optional
        Columns to exclude from the assessment (e.g., target, covariates).
    count_zero : bool, default=True
        Whether to count exact zeros as undesirable.

    Returns
    -------
    pd.Series
        Quality score per row (int). Lower values indicate cleaner rows.
    """
    use_df = df.drop(columns=exclude_cols, errors="ignore") if exclude_cols else df
    # Consider only numeric columns for NaN/inf/zero assessment
    num = use_df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return pd.Series(0, index=df.index)

    nan_cnt = num.isna().sum(axis=1)
    # Convert to numpy for isinf check; treat NaN as not inf (already counted)
    arr = num.to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        inf_mask = np.isinf(arr)
    inf_cnt = pd.Series(inf_mask.sum(axis=1), index=num.index)
    zero_cnt = num.eq(0).sum(axis=1) if count_zero else 0 ## This to check the validity as some values could be legitly 0!

    score = nan_cnt.add(inf_cnt, fill_value=0).add(zero_cnt, fill_value=0)
    return score.astype(int)


def select_cleanest_rows(
    df: pd.DataFrame,
    k: int,
    exclude_cols: Optional[List[str]] = None,
    count_zero: bool = True,
) -> pd.DataFrame:
    """Return top-k cleanest rows by minimal NaN/inf/zero counts.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    k : int
        Number of rows to return.
    exclude_cols : list of str, optional
        Columns to exclude from the assessment (e.g., target, covariates).
    count_zero : bool, default=True
        Whether to count exact zeros as undesirable.

    Returns
    -------
    pd.DataFrame
        Subset with top-k cleanest rows.
    """
    q = row_quality_score(df, exclude_cols=exclude_cols, count_zero=count_zero)
    # stable sort; return top k
    return df.loc[q.sort_values(kind='mergesort').index[:k]]


__all__ = ["row_quality_score", "select_cleanest_rows"]

