"""
IO Utilities
============

Helper functions for IO operations.
"""
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

def row_quality_score(df: 'pd.DataFrame', exclude_cols: Optional[List[str]] = None, count_zero: bool = True) -> 'pd.Series':
    """
    Calculate a 'badness' score for each row (NaNs + Infs + Zeros).
    Lower is better.
    """
    use_df = df.drop(columns=exclude_cols, errors="ignore") if exclude_cols else df
    num = use_df.select_dtypes(include=[np.number])
    if num.shape[1] == 0: return np.zeros(len(df), dtype=int)
    
    nan_cnt = num.isna().sum(axis=1)
    arr = num.to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        inf_mask = np.isinf(arr)
    inf_cnt = inf_mask.sum(axis=1)
    zero_cnt = num.eq(0).sum(axis=1) if count_zero else 0
    return (nan_cnt + inf_cnt + zero_cnt).astype(int)

def make_strata(df: 'pd.DataFrame', covariates: List[str], n_bins: int=5, binning: str="quantile") -> 'pd.Series':
    """
    Create a single stratification label from multiple covariates.
    Numeric covariates are binned.
    """
    labels = []
    for cov in covariates:
        s = df[cov]
        if is_numeric_dtype(s) or is_datetime64_any_dtype(s):
            if binning == "uniform":
                b = pd.cut(s, bins=n_bins)
            else:
                try: b = pd.qcut(s, q=n_bins, duplicates="drop")
                except: b = pd.cut(s, bins=n_bins)
            labels.append(b.astype(str).fillna("NA"))
        else:
            labels.append(s.astype(str).fillna("NA"))
    
    if len(labels) == 1: return labels[0].astype("category")
    return pd.concat(labels, axis=1).astype(str).agg("|".join, axis=1).astype("category")

def sample_indices(df: 'pd.DataFrame', target: str, size_map: Dict[Any, int], rng, replace: bool, prefer_clean: bool, exclude: List[str]) -> 'pd.Index':
    """
    Sample indices for each class based on size_map.
    """
    indices = []
    for cls, n in size_map.items():
        sub = df[df[target] == cls]
        if n <= 0: continue
        
        if prefer_clean:
            q = row_quality_score(sub, exclude_cols=exclude)
            if not replace:
                sub_shuf = sub.sample(frac=1.0, random_state=rng.integers(0, 1<<32))
                idx_top = q.loc[sub_shuf.index].sort_values(kind='mergesort').index[:n]
                indices.append(idx_top)
            else:
                w = (1.0 / (1.0 + q)).astype(float)
                sampled = sub.sample(n=n, replace=True, weights=w, random_state=rng.integers(0, 1<<32))
                indices.append(sampled.index)
        else:
            if n <= len(sub) and not replace:
                sampled = sub.sample(n=n, replace=False, random_state=rng.integers(0, 1<<32))
                indices.append(sampled.index)
            else:
                sampled = sub.sample(n=n, replace=True, random_state=rng.integers(0, 1<<32))
                indices.append(sampled.index)
    
    if not indices: return pd.Index([])
    # Concat the indices, shuffle, and return the VALUES as the new Index
    combined = pd.concat([pd.Series(i) for i in indices]).sample(frac=1.0, random_state=rng)
    return pd.Index(combined.values)
