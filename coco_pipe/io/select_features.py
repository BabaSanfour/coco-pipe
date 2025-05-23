import re
import difflib
import pandas as pd
from typing import Any, List, Optional, Union, Tuple, Dict

__all__ = ["select_features"]


def _get_col_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Map lowercase column names to actual column names for case-insensitive lookup.
    """
    return {col.lower(): col for col in df.columns}


def _suggest_closest(name: str, choices: List[str], n: int = 3) -> List[str]:
    """
    Suggest close matches for a given name from choices.
    """
    return difflib.get_close_matches(name, choices, n)


def _apply_row_filters(
    df: pd.DataFrame,
    row_filters: Union[Dict[str, Any], List[Dict[str, Any]]],
    col_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Apply one or more row filters to df.

    Each filter dict must have:
      - 'column': column name (case-insensitive)
      - 'values': list or single value for comparison or isin
      - optional 'operator': one of ['>', '<', '>=', '<=', '==', '!=']
    Without operator, defaults to isin.
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
        col_key = filt.get("column")
        if not col_key:
            raise ValueError("Row filter missing 'column' key")
        lower = col_key.lower()
        if lower not in col_map:
            sugg = _suggest_closest(lower, list(col_map))
            raise ValueError(f"Row filter column '{col_key}' not found. Did you mean {sugg}?")
        actual_col = col_map[lower]
        values = filt.get("values")
        operator = filt.get("operator")

        if operator in op_map:
            val = values if not isinstance(values, list) else values[0]
            mask &= op_map[operator](df[actual_col], val)
        elif values is not None:
            val_list = values if isinstance(values, list) else [values]
            mask &= df[actual_col].isin(val_list)
        else:
            raise ValueError(f"Row filter for column '{col_key}' must include 'values'.")

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
    Select feature columns (brain-feature combinations) and covariates from df,
    and extract target(s).

    Parameters
    ----------
        :df: pd.DataFrame, DataFrame containing the data to select features from
        :target_columns: Union[str, List[str]], Target variable(s) to predict
        :covariates: Optional[List[str]], Covariates to include in the model, e.g. age, sex, clinical scores, behavioral metrics
        :spatial_units: Optional[Union[str, List[str], Dict[str, List[str]]]], List of brain regions, sensors, or groups of regions/sensors, e.g. ["left_frontal", "right_frontal", "T1", "C3"]
        :feature_names: Union[str, List[str]], Names of features computed, e.g. ["alpha", "beta", "gamma", "entropy", "slope"]
        :sep: str, Separator between spatial unit and feature name (default '_')
        :reverse: bool, If True, expects naming '<feature>{sep}<spatial_unit>' instead of '<spatial_unit>{sep}<feature>'
        :row_filter: Optional[dict], {"column": ..., "values": [...]} to subset rows, e.g. {"column": "subject", "values": ["sub-01", "sub-02"]}
    
    Returns
    -------
        X: pd.DataFrame of selected features
        y: pd.Series or DataFrame of target(s)
    """
    col_map = _get_col_map(df)

    # 1) Apply row filters
    if row_filter:
        df = _apply_row_filters(df, row_filter, col_map)

    parts: List[pd.DataFrame] = []

    # 2) Covariates
    if covariates:
        matched = []
        for cov in covariates:
            low = cov.lower()
            if low not in col_map:
                sugg = _suggest_closest(low, list(col_map))
                raise ValueError(f"Covariate '{cov}' not found. Did you mean {sugg}?")
            matched.append(col_map[low])
        parts.append(df[matched])

    # 3) Parse columns into (su, feat)
    all_cols = df.columns.tolist()
    pairs = []
    for col in all_cols:
        if sep not in col:
            continue
        left, right = col.split(sep, 1)
        if reverse:
            feat, su = left, right
        else:
            su, feat = left, right
        pairs.append({'col': col, 'su': su, 'feat': feat})

    actual_su = sorted({p['su'] for p in pairs})
    actual_feat = sorted({p['feat'] for p in pairs})

    # 4) Normalize spatial_units
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

    # 5) Normalize feature_names
    sel_feat: List[str] = []
    if feature_names == 'all':
        sel_feat = actual_feat
    else:
        feat_list = feature_names if isinstance(feature_names, list) else [feature_names]
        for fn in feat_list:
            matches = [f for f in actual_feat if f.lower() == fn.lower()]
            if not matches:
                sel_feat = []
                break
            sel_feat.extend(matches)

    # 6) Select feature columns
    sel_cols: List[str] = []
    for p in pairs:
        if p['su'] in sel_su and p['feat'] in sel_feat:
            sel_cols.append(p['col'])
    if sel_cols:
        parts.append(df[sel_cols])

    # 7) Error if nothing selected
    if not parts:
        raise ValueError("No features selected: check your spatial_units and feature_names.")
    X = pd.concat(parts, axis=1)

    # 8) Select targets
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