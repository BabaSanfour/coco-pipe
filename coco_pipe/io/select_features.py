"""
coco_pipe/io/select_features.py
----------------
Select features from a DataFrame.
Author: Hamza Abdelhedi <hamza.abdelhedii@gmail.com>
Date: 2025-05-18
Version: 0.0.1
License: TBD
"""
import re
import difflib
import pandas as pd
from typing import Any, List, Optional, Union, Tuple, Dict

__all__ = ["select_features"]

def _get_col_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a case-insensitive mapping from lowercase column names to actual DataFrame column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    dict
        Mapping from lowercase column names to actual column names.
    """
    return {col.lower(): col for col in df.columns}

def _suggest_closest(name: str, choices: List[str], n: int = 3) -> List[str]:
    """
    Suggest closest matches for a string from a list of choices.

    Parameters
    ----------
    name : str
        Name to match.
    choices : list of str
        List of candidate names.
    n : int, optional
        Maximum number of suggestions to return, by default 3.

    Returns
    -------
    list of str
        List of suggested close matches.
    """
    return difflib.get_close_matches(name, choices, n)

def _apply_row_filters(
    df: pd.DataFrame,
    row_filters: Union[Dict[str, Any], List[Dict[str, Any]]],
    col_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Apply row filters to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    row_filters : dict or list of dict
        Filtering conditions. Each dict must include:
            - 'column': column name (case-insensitive)
            - 'values': list or scalar for filtering
            - 'operator': comparison operator (optional)
              One of ['>', '<', '>=', '<=', '==', '!=']
              Defaults to 'isin' if not provided.
    col_map : dict
        Case-insensitive mapping of column names.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.

    Raises
    ------
    ValueError
        If a required key is missing or column is not found.
    """
    if not isinstance(row_filters, list):
        row_filters = [row_filters]
    op_map = {
        ">": lambda col, val: col > val,
        "<": lambda col, val: col < val,
        ">=": lambda col, val: col >= val,
        "<=": lambda col, val: col <= val,
        "==": lambda col, val: col == val,
        "!=": lambda col, val: col != val,
    }
    mask = pd.Series(True, index=df.index)
    for filt in row_filters:
        if 'column' not in filt:
            raise ValueError("Row filter missing 'column' key")
        col_key = filt['column']
        lower = col_key.lower()
        if lower not in col_map:
            sugg = _suggest_closest(lower, list(col_map))
            raise ValueError(f"Row filter column '{col_key}' not found. Did you mean {sugg}?")
        actual_col = col_map[lower]
        operator = filt.get('operator')
        if 'values' not in filt:
            raise ValueError(f"Row filter for column '{col_key}' must include 'values'.")
        values = filt['values']
        if operator in op_map:
            val = values if not isinstance(values, list) else values[0]
            mask &= op_map[operator](df[actual_col], val)
        else:
            val_list = values if isinstance(values, list) else [values]
            mask &= df[actual_col].isin(val_list)
    return df[mask]

def select_features(
    df: pd.DataFrame,
    target_columns: Union[str, List[str]],
    covariates: Optional[List[str]] = None,
    spatial_units: Optional[Union[str, List[str], Dict[str, List[str]]]] = None,
    feature_names: Union[str, List[str]] = "all",
    sep: str = "_",
    reverse: bool = False,
    row_filter: Optional[Union[dict, List[dict]]] = None,
) -> Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]:
    """
    Select covariates and spatial-feature columns from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_columns : str or list of str
        Column(s) to be used as target variable(s).
    covariates : list of str, optional
        List of additional covariates to include.
    spatial_units : str, list of str, dict, or None
        Spatial units to extract (e.g., sensors or brain regions).
        If dict, keys are used as unit names.
        If 'all', all available units are used.
    feature_names : str or list of str
        Features to extract (e.g., 'alpha', 'beta'). Use 'all' to select all.
    sep : str, optional
        Separator between spatial unit and feature, by default '_'.
    reverse : bool, optional
        If True, expects columns in format '<feature><sep><unit>' instead of '<unit><sep><feature>'.
    row_filter : dict or list of dict, optional
        Row filtering conditions. See `_apply_row_filters` for format.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or pd.DataFrame
        Target variable(s).

    Raises
    ------
    ValueError
        If required columns are missing or selection fails.
    """
    col_map = _get_col_map(df)
    if row_filter:
        df = _apply_row_filters(df, row_filter, col_map)
    parts: List[pd.DataFrame] = []
    if covariates:
        matched = []
        for cov in covariates:
            low = cov.lower()
            if low not in col_map:
                sugg = _suggest_closest(low, list(col_map))
                raise ValueError(f"Covariate '{cov}' not found. Did you mean {sugg}?")
            matched.append(col_map[low])
        parts.append(df[matched])
    all_cols = df.columns.tolist()
    pairs = []
    for col in all_cols:
        if sep not in col:
            continue
        left, right = col.split(sep, 1)
        feat, su = (left, right) if reverse else (right, left)
        su, feat = (right, left) if reverse else (left, right)
        pairs.append({'col': col, 'su': su, 'feat': feat})
    actual_su = sorted({p['su'] for p in pairs})
    actual_feat = sorted({p['feat'] for p in pairs})
    sel_su: List[str] = []
    if spatial_units == 'all':
        sel_su = actual_su
    elif spatial_units is None:
        sel_su = []
    elif isinstance(spatial_units, dict):
        keys = list(spatial_units.keys())
        for key in keys:
            matches = [su for su in actual_su if su.lower() == key.lower()]
            if matches:
                sel_su.extend(matches)
    else:
        for su_in in spatial_units:
            matches = [su for su in actual_su if su.lower() == su_in.lower()]
            if matches:
                sel_su.extend(matches)
    sel_feat: List[str] = []
    if feature_names == 'all':
        sel_feat = actual_feat
    else:
        feat_list = feature_names if isinstance(feature_names, list) else [feature_names]
        for fn in feat_list:
            matches = [f for f in actual_feat if f.lower() == fn.lower()]
            if matches:
                sel_feat.extend(matches)
    sel_cols: List[str] = []
    for p in pairs:
        if p['su'] in sel_su and p['feat'] in sel_feat:
            sel_cols.append(p['col'])
    if sel_cols:
        parts.append(df[sel_cols])
    if not parts:
        raise ValueError("No features selected: check your spatial_units and feature_names.")
    X = pd.concat(parts, axis=1)
    tgt_list = target_columns if isinstance(target_columns, list) else [target_columns]
    tcols: List[str] = []
    for tgt in tgt_list:
        low = tgt.lower()
        if low not in col_map:
            sugg = _suggest_closest(low, list(col_map))
            raise ValueError(f"Target '{tgt}' not found. Did you mean {sugg}?")
        tcols.append(col_map[low])
    y = df[tcols[0]] if len(tcols) == 1 else df[tcols]
    return X, y